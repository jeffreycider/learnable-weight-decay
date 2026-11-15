import torch
from chisel.harness import RunState
from chisel.preservation import prediction_terms
from chisel.preservation_fit import logits,preservation_gradient


def test_preservation_vjp_matches_full_autograd():
    torch.manual_seed(3)
    model=torch.nn.Linear(4,5).double()
    original={name:p.detach().clone() for name,p in model.named_parameters()}
    effective={'weight':(original['weight']+torch.randn_like(original['weight'])*.01).requires_grad_()}
    edited=original|effective;batch=dict(inputs=torch.randn(7,4,dtype=torch.float64))
    cfg=dict(microbatch_size=3,runtime=dict(autocast=None),preservation=dict(top_k=4,margin_fraction=.5,logit_mse_weight=10.,ranking_weight=3.))
    state=RunState(extras=dict(logits_forward=logits,device=torch.device('cpu')))
    gradients,metrics=preservation_gradient(model,original,edited,effective,batch,cfg,state,4.)
    terms=prediction_terms(logits(model,edited,batch['inputs']),logits(model,original,batch['inputs']),4,.5)
    expected=torch.autograd.grad(10*terms['mse'].mean()+4*terms['kl'].mean()+3*terms['ranking'].mean(),effective['weight'])[0]
    torch.testing.assert_close(gradients['weight'],expected,rtol=1e-10,atol=1e-12)
    assert metrics['evaluated_positions']==7 and state.extras['training_forwards']==6
