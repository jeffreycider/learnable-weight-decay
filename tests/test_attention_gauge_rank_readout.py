from types import SimpleNamespace
import pytest
import torch
from baseline.model import GPT
from rewriter.exact_gauges import uniform_attention_group_control
from rewriter.function_preservation import FunctionalPreservation
from rewriter.rank_preservation import topk_preservation_flags
from scripts.evaluate_uniform_attention_gauge import rank_readout


def test_rank_readout_matches_direct_finite_logits_and_counts_without_backward():
    torch.set_num_threads(1);torch.manual_seed(983)
    model=GPT(32,2,128).float().requires_grad_(False)
    engine=FunctionalPreservation(model,microbatch_sequences=2,teacher_cache_device='cpu',kl_dtype=torch.float64,gpt_precision='fp32')
    inputs=torch.randint(32,(2,5));teacher=engine.cache_teacher(inputs,data_role='past_training',sequence_ids=[11,12])[0]
    panel=SimpleNamespace(inputs=inputs,sequence_ids=[11,12],teacher_batch=lambda e,a,b:teacher)
    control=uniform_attention_group_control(engine.parameters,model)
    params={**engine.parameters,**control.replacements}
    zero=rank_readout(engine,engine.parameters,panel,2,backend='reference')
    actual=rank_readout(engine,params,panel,2,backend='reference')
    with torch.no_grad():
        logits=engine._forward(params,inputs)
        kl=(teacher.log_probs.exp()*(teacher.log_probs-logits.double().log_softmax(-1))).sum(-1)
        flags=topk_preservation_flags(teacher.log_probs,logits,k=32)
    assert actual['max_token_kl']==pytest.approx(float(kl.max()),abs=1e-14)
    assert actual['mean_kl']==pytest.approx(float(kl.mean()),abs=1e-14)
    assert actual['strict_topk_changes']==int(flags['strict_topk_changed'].sum())
    assert actual['literal_top100_order_changes']==int(flags['literal_order_changed'].sum())
    assert zero['strict_topk_changes']==0 and zero['max_token_kl']==0
    assert actual['tokens']==10 and actual['sequence_ids']==[11,12]
    assert engine.compute.student_forwards==2 and engine.compute.student_backwards==0
    with pytest.raises(ValueError,match='microbatches'):rank_readout(engine,params,panel,1,backend='reference')


@pytest.mark.parametrize('scale',[0.,-1.,float('inf'),float('nan')])
def test_gauge_rejects_invalid_scale(scale):
    model=GPT(32,1,128).float()
    with pytest.raises(ValueError,match='value scale'):
        uniform_attention_group_control(dict(model.named_parameters()),model,value_scale=scale)


def test_selected_single_channel_binary_gauge_exact_group_energy_and_function():
    import copy
    from rewriter.exact_gauges import single_channel_binary_attention_control
    from rewriter.grouped_norm import GroupedNormObjective
    torch.manual_seed(984);model=GPT(32,2,128).double()
    parameters=dict(model.named_parameters());control=single_channel_binary_attention_control(parameters,model)
    edited={**parameters,**control.replacements}
    actual=GroupedNormObjective(parameters).evaluate(edited).metrics()['mean_group_l2_percentage_reduction']
    assert actual==pytest.approx(control.diagnostics['expected_mean_group_l2_percentage_reduction'],abs=1e-12)
    assert control.diagnostics['value_scale'] in (.5,2.) and control.diagnostics['candidate_count']==512
    changed=copy.deepcopy(model);changed.load_state_dict({**model.state_dict(),**edited})
    inputs=torch.randint(32,(2,7))
    def logits(m):
        x=m.norm1(m.embed(inputs))
        for block in m.blocks:x=block(x)
        return m.proj(m.norm2(x))
    torch.testing.assert_close(logits(model),logits(changed),atol=1e-12,rtol=1e-12)
    for name,weight in control.replacements.items():
        if weight.ndim==2:assert int(torch.linalg.matrix_rank(weight-parameters[name]))==1
