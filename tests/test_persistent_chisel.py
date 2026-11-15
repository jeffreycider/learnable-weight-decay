"""Current-architecture batching, full-panel penalty, and optimizer boundaries."""
import copy
import json
import pytest
import torch
from baseline.model import GPT
from rewriter.transformer_chisel import TransformerChisel
from rewriter.function_preservation import FunctionalPreservation, gpt_logits
from rewriter.persistent_gpt import PersistentGPTVJP
from rewriter.persistent_mlp import CooperativeTrainingKernel
from rewriter.persistent_offline import PersistentOfflineTrainer, PersistentActorOptimizer
from scripts.benchmark_chisel_kernels import synthetic_snapshot
from scripts.qualify_persistent_offline_chisel import reference_step

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='cooperative CUDA kernel')

def relative(a,b):return float((a-b).double().norm()/a.double().norm().clamp_min(1e-30))


@pytest.fixture
def configuration(monkeypatch):
    monkeypatch.setenv('CHISEL_STREAM_ATTENTION','3')
    monkeypatch.setenv('CHISEL_HEAD_CHUNK_TOKENS','8')
    monkeypatch.setenv('CHISEL_FUSED_HEAD_KL','1')
    monkeypatch.setenv('CHISEL_PERSISTENT_MAX_REGISTERS','128')
    monkeypatch.setenv('CHISEL_PERSISTENT_BLOCK_MULTIPLIER','4')
    torch.backends.cuda.matmul.allow_tf32=False


def test_batched_gpt_preserves_sequence_boundaries_and_vjp(configuration):
    torch.manual_seed(198)
    model=GPT(257,1,128).cuda()
    inputs=torch.randint(0,257,(4,37),device='cuda')
    logits=gpt_logits(model,inputs)
    cotangent=torch.randn_like(logits)/logits.numel()
    logits.backward(cotangent)
    program=PersistentGPTVJP(model,inputs,cotangent.flatten(0,1))
    program.step();torch.cuda.synchronize()
    assert relative(logits.detach().flatten(0,1),program.logits)<.01
    expected=dict(model.named_parameters())
    wanted=torch.cat([expected[n].grad.float().flatten() for n in program.gradients])
    actual=torch.cat([g.flatten() for g in program.gradients.values()])
    assert relative(wanted,actual)<.03


class NoOptimizer:
    def step(self):pass
    def zero_grad(self,**kwargs):pass


@pytest.mark.parametrize('state_weight,tokens',[(.5,16),(1/200,17)])
def test_full_panel_microbatch_gradient_accumulation_then_one_adam(configuration,state_weight,tokens):
    torch.manual_seed(199)
    model=GPT(257,1,128).cuda().requires_grad_(False)
    with torch.no_grad():
        for name,value in model.named_parameters():
            if name.endswith('.bias'):value.normal_(std=.01)
    reference=TransformerChisel(model_dim=128,num_layers=1,patch_size=8).cuda()
    with torch.no_grad():
        for head in reference.output_heads:head.weight.normal_(std=.01)
    actual=copy.deepcopy(reference)
    before={n:p.detach().clone() for n,p in actual.named_parameters()}
    accumulated={n:torch.zeros_like(p) for n,p in actual.named_parameters()}
    expected={n:torch.zeros_like(p) for n,p in reference.named_parameters()}
    engine=FunctionalPreservation(model,teacher_cache_device='cuda',kl_dtype=torch.float64)
    kernel=CooperativeTrainingKernel()
    for state_index in range(2):
        if state_index:
            model=copy.deepcopy(model)
            with torch.no_grad():
                for value in model.parameters():value.mul_(1.07)
            engine=FunctionalPreservation(model,teacher_cache_device='cuda',kl_dtype=torch.float64)
        snapshot=synthetic_snapshot(model)
        inputs=torch.randint(0,257,(4,tokens),device='cuda')
        with torch.no_grad():teacher=torch.cat([gpt_logits(model,inputs[i:i+2]).double().log_softmax(-1) for i in (0,2)])
        reference.zero_grad(set_to_none=True)
        values,gradients,_,_=reference_step(reference,snapshot,engine,teacher,inputs,NoOptimizer(),
            radius_fraction=.1,budget=1e-4,clip=float('inf'),microbatch_sequences=2,state_weight=state_weight)
        for name,g in gradients.items():expected[name].add_(g)
        if state_index==0:
            program=PersistentOfflineTrainer(actual,model,snapshot,inputs,teacher,radius_fraction=.1,
                kl_budget=1e-4,microbatch_sequences=2,state_weight=state_weight,gradient_accumulator=accumulated,
                apply_optimizer=False,kernel=kernel)
            pointers=[value.data_ptr() for value in program.plan.storage]
            stage_pointer=program.plan.stages.data_ptr()
        else:
            program.load_checkpoint(model,snapshot);program.load_panel(inputs,teacher)
            assert pointers==[value.data_ptr() for value in program.plan.storage]
            assert stage_pointer==program.plan.stages.data_ptr()
        program.step();torch.cuda.synchronize()
        assert abs(float(program.kl)-values['kl'])<max(5e-6,.03*values['kl'])
        assert all(torch.equal(before[n],p) for n,p in actual.named_parameters())
        assert program.plan.temporary_storage
        wanted=torch.cat([g.flatten() for g in gradients.values()])
        obtained=torch.cat([g.flatten() for g in program.gradients.values()])
        error=relative(wanted,obtained)
        print(json.dumps(dict(scope='checkpoint_gradient',state_weight=state_weight,state_index=state_index,
            relative_gradient_error=error,reference_kl=values['kl'],literal_kl=float(program.kl))),flush=True)
        assert error<.05
    assert relative(torch.cat([g.flatten() for g in expected.values()]),torch.cat([g.flatten() for g in accumulated.values()]))<.05
    rejected=copy.deepcopy(model)
    with torch.no_grad():rejected.proj.bias.zero_()
    original_bias=program.base['proj.bias'].clone();original_stages=program.plan.stages.clone()
    with pytest.raises(ValueError,match='zero-reference'):
        program.load_checkpoint(rejected,snapshot)
    assert torch.equal(program.base['proj.bias'],original_bias)
    assert torch.equal(program.plan.stages,original_stages)
    optimizer=torch.optim.Adam(reference.parameters(),lr=1e-5,betas=(.9,.99),eps=1e-8)
    for name,p in reference.named_parameters():p.grad=expected[name]
    torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
    optimizer.step()
    update=PersistentActorOptimizer(actual,accumulated,lr=1e-5,kernel=kernel)
    update.step();torch.cuda.synchronize()
    assert int(update.optimizer_state()['step'])==1
    error=relative(torch.cat([p.detach().flatten() for p in reference.parameters()]),
                    torch.cat([p.detach().flatten() for p in actual.parameters()]))
    print(json.dumps(dict(scope='two_checkpoints_one_adam',state_weight=state_weight,updated_parameter_relative_error=error,
                         optimizer_updates=int(update.optimizer_state()['step']))),flush=True)
    assert error<.001
