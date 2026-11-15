import copy
import dataclasses
import json
import pytest
import torch
from rewriter.soap46_action_diagnostic import compare_basis_action
from rewriter.soap46_map import committed_weight,soap_direction
from tests.test_soap46_map import make_record


@pytest.mark.parametrize('variant',['inner-outer','outer-only'])
@pytest.mark.parametrize('shape',[(4,7),(7,4)])
def test_zero_x_identity_and_readonly_inputs(variant,shape):
    r=dataclasses.replace(make_record(shape),variant=variant)
    left=torch.zeros_like(r.q_left,requires_grad=True);right=torch.zeros_like(r.q_right,requires_grad=True)
    before=copy.deepcopy(r);left.grad=torch.ones_like(left);before_grad=left.grad.clone();rng=torch.get_rng_state().clone()
    out=compare_basis_action(r,left,right,parameter_name='blocks.0.attn.q.weight')
    json.dumps(out,allow_nan=False)
    assert all(s['changed_elements']==0 and s['difference_norm']==0 for s in out['comparisons'].values())
    for f in dataclasses.fields(r):
        a,b=getattr(r,f.name),getattr(before,f.name)
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert torch.equal(left.grad,before_grad) and right.grad is None
    assert torch.equal(rng,torch.get_rng_state())


def test_nonzero_action_matches_executed_committed_displacement():
    r=make_record();left=torch.randn_like(r.q_left)*.01;right=torch.randn_like(r.q_right)*.01
    out=compare_basis_action(r,left,right,parameter_name='matrix')
    a=committed_weight(r,left,right).double()-r.weight.double()
    b=committed_weight(r,left*0,right*0).double()-r.weight.double()
    assert out['residuals']['left_x_relative_basis_norm']==(left.double().norm()/r.q_left.double().norm()).item()
    row=out['comparisons']['committed_displacement']
    assert row['changed_elements']==int((a!=b).sum())>0
    assert row['difference_norm']==(a-b).norm().item()
    assert row['relative_difference_vs_zero_norm']==(a-b).norm().item()/b.norm().item()


def test_nonzero_tiny_x_changes_fp32_direction_but_is_erased_at_bf16_cast():
    r=make_record((4,4),attention=False,output=False)
    r=dataclasses.replace(r,variant='outer-only',q_left=torch.eye(4),q_right=torch.eye(4),
        direction=torch.diag(torch.tensor([.8,.7,.6,.5])),variance=torch.ones(4,4))
    left=torch.diag(torch.tensor([1e-5,-1e-5,1e-5,-1e-5]));right=torch.zeros(4,4)
    out=compare_basis_action(r,left,right,parameter_name='toy')
    assert out['residuals']['left_x_norm']>0
    assert out['comparisons']['pre_muon_direction']['changed_elements']>0
    for key in ('muon_cast_input','muon_normalized_input','finite_muon_output','committed_displacement'):
        assert out['comparisons'][key]['changed_elements']==0
    # Cast adjoint can remain nonzero inside this finite-forward deadband.
    left.requires_grad_()
    cot=torch.arange(16,dtype=torch.float32).reshape(4,4)
    grad=torch.autograd.grad((committed_weight(r,left,right)*cot).sum(),left)[0]
    assert torch.isfinite(grad).all() and grad.norm()>0


def test_zero_stage_norm_is_explicit_not_epsilon_invented():
    r=make_record();r=dataclasses.replace(r,direction=torch.zeros_like(r.direction),lr=0.,cwd=0.,
                                        weight=torch.zeros_like(r.weight))
    out=compare_basis_action(r,torch.zeros_like(r.q_left),torch.zeros_like(r.q_right),parameter_name='zero')
    for row in out['comparisons'].values():
        assert row['zero_reference_norm'] and row['relative_difference_vs_zero_norm'] is None
        assert row['cosine'] is None
    json.dumps(out,allow_nan=False)


def test_reject_missing_basis_and_shape():
    r=make_record()
    with pytest.raises(ValueError,match='basis-free'):
        compare_basis_action(dataclasses.replace(r,q_left=None),torch.zeros_like(r.q_left),torch.zeros_like(r.q_right),parameter_name='x')
    with pytest.raises(ValueError,match='shape'):
        compare_basis_action(r,torch.zeros(2,2),torch.zeros_like(r.q_right),parameter_name='x')


def test_autocast_and_wrong_residual_precision_reject():
    r=make_record();left=torch.zeros_like(r.q_left);right=torch.zeros_like(r.q_right)
    with torch.autocast('cpu',dtype=torch.bfloat16),pytest.raises(ValueError,match='autocast'):
        compare_basis_action(r,left,right,parameter_name='x')
    with pytest.raises(ValueError,match='dtype'):
        compare_basis_action(r,left.double(),right,parameter_name='x')
