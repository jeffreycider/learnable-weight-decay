from types import SimpleNamespace
import pytest
import torch
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual,logit_terms_and_cotangent
from rewriter.image_tokenwise_panel import image_tokenwise_panel

class Engine:
    device=torch.device('cpu');chunk_tokens=2;teacher_id='immutable-cnn'
    def __init__(self,micro):self.microbatch_sequences=micro;self.calls=0
    def _validate_replacements(self,p):assert set(p)=={'conv','head'}
    def _forward(self,p,x):
        self.calls+=1
        h=torch.nn.functional.conv2d(x,p['conv'].reshape(2,1,3,3)).relu().mean((2,3))
        return h@p['head'].T

@pytest.mark.parametrize('micro',[1,2,5])
def test_image_normalization_and_single_backward_match_full_cnn_autograd(micro):
    torch.set_num_threads(1);torch.manual_seed(32)
    e=Engine(micro);images=torch.randn(5,1,5,5);labels=torch.arange(5)
    base={'conv':torch.randn(2,9),'head':torch.randn(10,2)}
    teacher=e._forward(base,images).detach().double().log_softmax(-1);e.calls=0
    direction=torch.randn(10,2);theta=torch.tensor(.01,requires_grad=True)
    params=dict(base,head=base['head']+theta*direction);reg=.3*theta.square()
    config=TokenwiseConfig(.001,.002,.01,.03,.4,ranking_k=100,ranking_tie_policy='exact_teacher_ties')
    dual=TokenwiseDual([1],config=config,dual_lr=.1);dual.multipliers[1]=.2
    def provider(a,b):return SimpleNamespace(inputs=images[a:b],log_probs=teacher[a:b],teacher_id=e.teacher_id,data_role='past_training')
    result=image_tokenwise_panel(e,params,reg,images,labels,config=config,dual=dual,state_id=1,teacher_provider=provider,state_weight=.25)
    actual=theta.grad.clone();assert e.calls==(5+micro-1)//micro
    other=torch.tensor(.01,requires_grad=True);logits=e._forward(dict(base,head=base['head']+other*direction),images)
    g,_,_=logit_terms_and_cotangent(logits,teacher,config=config,multiplier=.2,panel_tokens=5)
    torch.autograd.backward([logits,.3*other.square()],[g*.25,torch.tensor(.25)])
    torch.testing.assert_close(actual,other.grad,rtol=3e-5,atol=1e-6)
    assert result['context_count']==5 and result['ranking_k']==10
    assert result['compute']['teacher_forwards']==0 and result['compute']['student_images']==5
    assert dual.updates==0 and dual.multipliers[1]==.2


def test_audit_images_cannot_update_actor():
    e=Engine(1);x=torch.zeros(1,1,5,5);p={'conv':torch.zeros(2,9),'head':torch.zeros(10,2)}
    c=TokenwiseConfig(1e-8,1e-8,1e-8,.01,ranking_k=100,ranking_tie_policy='exact_teacher_ties');d=TokenwiseDual([1],config=c,dual_lr=.1)
    def teacher(a,b):return SimpleNamespace(inputs=x,log_probs=torch.zeros(1,10,dtype=torch.float64),teacher_id=e.teacher_id,data_role='withheld_training')
    with pytest.raises(ValueError,match='audit-only'):image_tokenwise_panel(e,p,torch.tensor(0.),x,torch.tensor([0]),config=c,dual=d,state_id=1,teacher_provider=teacher)
    assert e.calls==0


def test_legacy_full_vocabulary_rejected_before_student_forward():
    e=Engine(1);x=torch.zeros(1,1,5,5);p={'conv':torch.zeros(2,9),'head':torch.zeros(10,2)}
    c=TokenwiseConfig(1e-8,1e-8,1e-8,.01);d=TokenwiseDual([1],config=c,dual_lr=.1)
    def teacher(a,b):return SimpleNamespace(inputs=x,log_probs=torch.zeros(1,10,dtype=torch.float64),teacher_id=e.teacher_id,data_role='past_training')
    with pytest.raises(ValueError,match='ranking_tie_policy'):image_tokenwise_panel(e,p,torch.tensor(0.),x,torch.tensor([0]),config=c,dual=d,state_id=1,teacher_provider=teacher)
    assert e.calls==0
