import torch
from rewriter.algebra import Anchors, Factors, RawAction
from rewriter.controller import ControllerSnapshot, MatrixObservation, VectorObservation
from rewriter.direction_warmup import direction_targets, direction_loss, direction_metrics


def test_direction_target_is_canonical_projected_negative_weight():
    torch.manual_seed(42)
    w=torch.randn(9,6);v=torch.randn(9);rank=2
    a=torch.linalg.qr(torch.randn(9,rank)).Q;b=torch.linalg.qr(torch.randn(6,rank)).Q
    anchors=Anchors(a,b)
    snapshot=ControllerSnapshot({'weight':MatrixObservation(anchors,torch.empty(9,25),torch.empty(6,25),(w@b,w.T@a))},
        {'bias':VectorObservation(torch.empty(9,18),v)},torch.empty(2,4),w.norm(),w.norm()*.01)
    groups={'matrix':['weight'],'vector':['bias']};refs={'matrix':w.square().sum(),'vector':v.square().sum()}
    target=direction_targets(snapshot,groups,refs,{'weight':w.shape,'bias':v.shape})
    scale=(refs['matrix']/w.numel()).sqrt()*(w.numel()/(rank*(sum(w.shape)-rank)))**.5
    f=target.action.matrices['weight']
    dense=(f.P@b.T+a@f.Q.T)*scale
    projection=w@b@b.T+a@a.T@w-a@a.T@w@b@b.T
    torch.testing.assert_close(dense,-projection,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(a.T@f.P,torch.zeros(rank,rank),atol=1e-6,rtol=0)
    assert direction_loss(target.action,target)==0
    metrics=direction_metrics(target.action,target)
    assert abs(metrics['matrix']['cosine']-1)<1e-6
    zero=RawAction({'weight':Factors(torch.zeros_like(f.P,requires_grad=True),torch.zeros_like(f.Q,requires_grad=True),anchors)},
                   {'bias':torch.zeros_like(v,requires_grad=True)})
    loss=direction_loss(zero,target);loss.backward()
    torch.testing.assert_close(loss,torch.tensor(1.))
    assert zero.matrices['weight'].P.grad.norm()>0


def test_zero_reference_group_has_no_loss_or_gradient():
    a=torch.eye(3)[:,:1];b=torch.eye(4)[:,:1];anchors=Anchors(a,b)
    snapshot=ControllerSnapshot({'w':MatrixObservation(anchors,torch.empty(3,20),torch.empty(4,20),(torch.zeros(3,1),torch.zeros(4,1)))},
        {},torch.empty(1,2),torch.tensor(0.),torch.tensor(0.))
    target=direction_targets(snapshot,{'zero':['w']},{'zero':torch.tensor(0.)},{'w':(3,4)})
    p=torch.ones(3,1,requires_grad=True);q=torch.ones(4,1,requires_grad=True)
    loss=direction_loss(RawAction({'w':Factors(p,q,anchors)},{}),target);loss.backward()
    assert loss==0 and p.grad.count_nonzero()==0 and q.grad.count_nonzero()==0
    from rewriter.direction_warmup import aggregate_direction_metrics
    metrics=direction_metrics(RawAction({'w':Factors(p,q,anchors)},{}),target)
    assert metrics['zero']==dict(active_target=False,cosine=None,norm_ratio=None)
    combined=aggregate_direction_metrics([metrics,{'zero':dict(active_target=True,cosine=.9,norm_ratio=1.1)}])
    assert combined['zero']==dict(active_states=1,inactive_states=1,cosine=.9,norm_ratio=1.1)
