import pytest
import torch
from chisel.harness import RunState
from chisel.preservation_fit import logits, preservation_gradient, audit
from test_compiled_logit_loss import config


def test_audit_matches_fitting_and_leaves_no_gradient_state():
    model=torch.nn.Linear(4,5).double()
    base={name:value.detach().clone() for name,value in model.named_parameters()}
    effective={'weight':(base['weight']+.01).requires_grad_()}
    batch=dict(inputs=torch.randn(7,4,dtype=torch.float64))
    cfg=config() | dict(compile_preservation_loss=False)
    state=RunState(extras=dict(device=torch.device('cpu'),logits_forward=logits))
    _,expected=preservation_gradient(model,base,base|effective,effective,batch,cfg,state,0.)
    actual=audit(model,base,base|effective,batch,cfg,state)
    for name in ['logit_mse','maximum_logit_change','evaluated_positions']:
        assert actual[name]==expected[name]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert effective['weight'].grad is None


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_audit_reuses_compiled_training_forward():
    from chisel.models import CifarGPT
    from torch._dynamo.utils import counters
    torch.manual_seed(17)
    torch.set_float32_matmul_precision('highest')
    model=CifarGPT(32,2,4,'mean',head_dim=8,causal=False,learned_qk_scale=True,expansion=2).cuda().float()
    torch.nn.init.normal_(model.proj.weight,std=.1)
    base={name:value.detach().clone() for name,value in model.named_parameters()}
    effective={name:(value+.001).requires_grad_() for name,value in base.items()
               if name.startswith('blocks.') and value.ndim==2}
    state=RunState(extras=dict(device=torch.device('cuda'),
        logits_forward=torch.compile(logits,fullgraph=True,dynamic=False)))
    batch=dict(inputs=torch.ones(4,3,32,32,device='cuda'))
    cfg=config() | dict(microbatch_size=4,compile_preservation_loss=False)
    _,expected=preservation_gradient(model,base,base|effective,effective,batch,cfg,state,0.)
    count=counters['stats']['unique_graphs']
    model.eval()
    for _ in range(3):
        actual=audit(model,base,base|effective,batch,cfg,state)
        assert actual['logit_mse']==expected['logit_mse']
        assert actual['maximum_logit_change']==expected['maximum_logit_change']
    assert counters['stats']['unique_graphs']==count
