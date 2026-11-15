import torch
from chisel.teacher_cache import TeacherCache
from chisel.preservation_fit import preservation_gradient,logits
from chisel.harness import RunState


def test_cache_owns_storage_and_evicts_by_bytes():
    cache=TeacherCache(16);value=torch.arange(4.)
    cache.put('a',value);value.zero_()
    assert cache.get('a').sum()==6
    cache.put('b',value)
    assert cache.get('a') is None and cache.bytes==16


def test_cached_teacher_does_not_hide_edits_or_change_gradients():
    model=torch.nn.Linear(4,3);base={n:p.detach().clone() for n,p in model.named_parameters()}
    state=RunState(extras=dict(device=torch.device('cpu'),logits_forward=logits))
    config=dict(runtime=dict(autocast=None),microbatch_size=4,teacher_cache_bytes=1024,preservation_cache_key=('checkpoint',3),preservation=dict(top_k=3,margin_fraction=.5,logit_mse_weight=1.,ranking_weight=0.))
    batch=dict(inputs=torch.randn(8,4))
    def calculate():
        effective={'weight':(base['weight']+.1*torch.randn_like(base['weight'])).requires_grad_()}
        return effective
    effective=calculate()
    first,metrics=preservation_gradient(model,base,base|effective,effective,batch,config,state,1.)
    effective={n:v.detach().clone().requires_grad_() for n,v in effective.items()}
    second,repeated=preservation_gradient(model,base,base|effective,effective,batch,config,state,1.)
    assert metrics['kl']>0 and repeated==metrics
    torch.testing.assert_close(first['weight'],second['weight'])
    assert state.extras['training_forwards']==6
    assert state.extras['teacher_cache'].hits==2
