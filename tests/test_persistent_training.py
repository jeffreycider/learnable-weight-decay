"""Reference-gradient tests for the genuine cooperative CUDA interpreter."""
import copy
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend

from rewriter.persistent_mlp import CooperativeTrainingKernel,PersistentMLPTrainer
from rewriter.persistent_attention import PersistentAttentionVJP
from rewriter.persistent_transformer import PersistentTransformerStackTrainer


pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='cooperative CUDA kernel')


@pytest.fixture(scope='module')
def kernel():return CooperativeTrainingKernel()


def relative(a,b):return (a-b).double().norm()/a.double().norm().clamp_min(1e-30)


def test_persistent_mlp_gradients_and_repeated_adam(kernel):
    torch.manual_seed(77)
    x=torch.randn(1025,55,device='cuda');dy=torch.randn(1025,64,device='cuda')/1025
    weights=[torch.randn(64,55,device='cuda')*.03,torch.randn(64,device='cuda')*.01,
             torch.randn(64,64,device='cuda')*.03,torch.randn(64,device='cuda')*.01]
    expected=[v.clone().requires_grad_() for v in weights];rx=x.clone().requires_grad_()
    actual=PersistentMLPTrainer(x,dy,*weights,kernel=kernel)
    optimizer=torch.optim.Adam(expected,lr=2e-4,betas=(.9,.99),eps=1e-8)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True);rx.grad=None
        output=F.linear(F.gelu(F.linear(rx,*expected[:2])),*expected[2:])
        output.backward(dy);optimizer.step();actual.step();torch.cuda.synchronize()
        assert relative(output.detach(),actual.output)<5e-5
        assert relative(rx.grad,actual.input_cotangent)<5e-5
        for wanted,gradient,value in zip(expected,actual.gradients,weights):
            assert relative(wanted.grad,gradient)<5e-5
            assert relative(wanted.detach(),value)<5e-5


@pytest.mark.parametrize('causal',[False,True])
def test_persistent_attention_all_vjps(kernel,causal):
    torch.manual_seed(31)
    q=torch.randn(4,37,32,device='cuda',requires_grad=True)
    k=torch.randn(4,53,32,device='cuda',requires_grad=True);v=torch.randn_like(k,requires_grad=True)
    dy=torch.randn_like(q)
    scores=q@k.transpose(-1,-2)/32**.5
    if causal:
        mask=torch.arange(53,device='cuda')[None,:]>torch.arange(37,device='cuda')[:,None]
        scores=scores.masked_fill(mask,float('-inf'))
    output=scores.softmax(-1)@v;output.backward(dy)
    actual=PersistentAttentionVJP(q.detach(),k.detach(),v.detach(),dy,causal=causal,kernel=kernel)
    actual.step();torch.cuda.synchronize()
    for wanted,value in ((output.detach(),actual.output),(q.grad,actual.query_gradient),
                         (k.grad,actual.key_gradient),(v.grad,actual.value_gradient)):
        assert relative(wanted,value)<5e-5


def test_persistent_three_block_composed_reverse(kernel):
    torch.manual_seed(914)
    reference=nn.Sequential(*[nn.TransformerEncoderLayer(128,4,512,dropout=0.,activation='gelu',
        batch_first=True,norm_first=True).cuda() for _ in range(3)])
    actual=copy.deepcopy(reference);x=torch.randn(64,128,device='cuda');rx=x.clone().requires_grad_()
    dy=torch.randn_like(x)/64
    program=PersistentTransformerStackTrainer(actual,x,dy,kernel=kernel)
    with sdpa_kernel(SDPBackend.MATH):output=reference(rx[None])[0]
    output.backward(dy)
    expected_grad=torch.cat([p.grad.flatten() for p in reference.parameters()])
    program.step();torch.cuda.synchronize()
    actual_grad=torch.cat([g.flatten() for g in program.gradients.values()])
    assert relative(output.detach(),program.output)<1e-4
    assert relative(rx.grad,program.input_gradient)<1e-4
    assert relative(expected_grad,actual_grad)<1e-4
    assert int(program.plan.step_number)==1


def test_persistent_fp64_kl_same_logits(kernel):
    from rewriter.persistent_plan import PersistentTrainingPlan
    from rewriter.persistent_offline import append_kl
    from rewriter.function_preservation import chunked_teacher_kl
    torch.manual_seed(713)
    logits=torch.randn(19,509,device='cuda',requires_grad=True)
    teacher=(logits.detach().double()+torch.randn_like(logits)*.02).log_softmax(-1)
    reference=chunked_teacher_kl(logits,teacher);reference.backward()
    plan=PersistentTrainingPlan(logits.device,kernel=kernel)
    actual,gradient=append_kl(plan,logits.detach(),teacher);plan.finalize();plan.step();torch.cuda.synchronize()
    torch.testing.assert_close(actual[0],reference.detach(),atol=2e-14,rtol=2e-10)
    assert relative(logits.grad,gradient)<1e-6


@pytest.mark.parametrize('radius_fraction',[.0001,.1])
def test_persistent_group_objective_and_global_radius_vjp(kernel,radius_fraction):
    import math
    from baseline.model import GPT
    from rewriter.algebra import Factors,RawAction
    from rewriter.function_preservation import FunctionalPreservation
    from rewriter.grouped_norm import GroupedNormObjective
    from rewriter.edit_scope import editable_names,editable_weight_norm_sq,keep_readonly_parameters
    from rewriter.persistent_plan import PersistentTrainingPlan
    from rewriter.persistent_offline import append_edits,append_group_objective
    from scripts.benchmark_chisel_kernels import synthetic_snapshot
    torch.manual_seed(812)
    model=GPT(257,1,128).cuda().requires_grad_(False)
    # An exactly zero functional group must remain frozen and retain its place
    # in the same fifteen-group mean.
    model.blocks[0].attn.q.bias.zero_()
    engine=FunctionalPreservation(model);grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
    editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
    snapshot=synthetic_snapshot(model,rank=2);matrices={};vectors={}
    for name,obs in snapshot.matrices.items():
        p=torch.randn_like(obs.anchors.A)*.02;q=torch.randn_like(obs.anchors.B)*.02
        p=p-obs.anchors.A@(obs.anchors.A.T@p)
        matrices[name]=Factors(p.contiguous().requires_grad_(),q.contiguous().requires_grad_(),obs.anchors)
    for name,obs in snapshot.vectors.items():vectors[name]=(torch.randn_like(obs.weight)*.02).requires_grad_()
    action=RawAction(matrices,vectors)
    radius=float(editable_weight_norm_sq(engine.parameters,editable).sqrt().float()*radius_fraction)
    scales={}
    for group,names in grouped.groups.items():
        rms=(grouped.reference_norms[group]/math.sqrt(sum(engine.parameters[n].numel() for n in names))).float()
        for name in names:
            value=engine.parameters[name];dimension=2*(sum(value.shape)-2) if value.ndim==2 else value.numel()
            scales[name]=rms*math.sqrt(value.numel()/dimension)
    scaled=RawAction({n:Factors(f.P*scales.get(n,0),f.Q*scales.get(n,0),f.anchors) for n,f in matrices.items()},
        {n:v*scales.get(n,0) for n,v in vectors.items()})
    norm=scaled.norm_sq().clamp_min(1e-24).sqrt();scale=torch.minimum(norm.new_ones(()),radius/(norm+1e-12))
    replacements=keep_readonly_parameters(engine.edited_parameters(scaled,scale),engine.parameters,editable)
    result=grouped.evaluate(replacements);regularizer=100*result.mean_relative_change+.01*result.edit_norm_sq/radius**2
    fixed={n:torch.randn_like(replacements[n])*.001 for n in editable}
    torch.autograd.backward([regularizer]+[replacements[n] for n in fixed],
                            [torch.ones_like(regularizer)]+list(fixed.values()))
    plan=PersistentTrainingPlan('cuda',kernel=kernel);base={n:p.float() for n,p in engine.parameters.items()}
    edited,info,back=append_edits(plan,action,base,grouped,editable,radius)
    values,group_ids=append_group_objective(plan,edited,base,grouped)
    coefficients=torch.tensor([1.,.02/radius**2],device='cuda',dtype=torch.float64)
    gpt={n:g.float() for n,g in fixed.items()};cotangents={}
    for n in editable:
        out=plan.empty(base[n].shape);cotangents[n]=out
        plan.append([30,base[n].data_ptr(),edited[n].data_ptr(),gpt[n].data_ptr(),values.data_ptr(),coefficients.data_ptr(),
                     out.numel(),group_ids[n],int(n=='embed.weight'),out.data_ptr()])
    actual=back(cotangents);plan.finalize();plan.step();torch.cuda.synchronize()
    expected=torch.cat([v.grad.flatten() if v.grad is not None else torch.zeros_like(v).flatten()
                        for f in matrices.values() for v in (f.P,f.Q)]+[v.grad.flatten() if v.grad is not None else torch.zeros_like(v).flatten() for v in vectors.values()])
    obtained=torch.cat([v.flatten() for f in actual.matrices.values() for v in (f.P,f.Q)]+[v.flatten() for v in actual.vectors.values()])
    assert relative(expected,obtained)<3e-4
    torch.testing.assert_close(info[1],scale.detach(),rtol=2e-6,atol=1e-8)
    assert torch.equal(edited['blocks.0.attn.q.bias'],base['blocks.0.attn.q.bias'])


def test_persistent_adam_history_survives_new_checkpoint_plan(kernel):
    from rewriter.persistent_plan import PersistentTrainingPlan
    torch.manual_seed(137)
    weight=torch.randn(11,7,device='cuda')*.02;bias=torch.zeros(11,device='cuda')
    ref_weight=weight.clone().requires_grad_();ref_bias=bias.clone().requires_grad_()
    optimizer=torch.optim.Adam((ref_weight,ref_bias),lr=2e-4,betas=(.9,.99),eps=1e-8)
    state=None
    for checkpoint in range(3):
        inputs=torch.randn(23,7,device='cuda');cotangent=torch.randn(23,11,device='cuda')/23
        optimizer.zero_grad(set_to_none=True)
        F.linear(inputs,ref_weight,ref_bias).backward(cotangent);optimizer.step()
        plan=PersistentTrainingPlan(inputs.device,kernel=kernel)
        output,_=plan.linear(inputs,weight,bias)
        _,dw,db=plan.linear_vjp(inputs,weight,cotangent)
        plan.adam((weight,bias),(dw,db),state=state);plan.finalize();plan.step();torch.cuda.synchronize()
        state=plan.optimizer_state()
        assert int(state['step'])==checkpoint+1
        assert relative(ref_weight.detach(),weight)<5e-5
        assert relative(ref_bias.detach(),bias)<5e-5
