"""Fusion must preserve the FP32 softcap arithmetic and raw KL cotangent."""
import pytest
import torch
from rewriter.persistent_mlp import CooperativeTrainingKernel
from rewriter.persistent_plan import PersistentTrainingPlan
from rewriter.persistent_offline import append_kl
from rewriter.persistent_gpt import unary

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='literal CUDA kernel')

@pytest.fixture(scope='module')
def kernel():return CooperativeTrainingKernel()

@pytest.mark.parametrize('rows,cols',[(8,128),(17,50304)])
@pytest.mark.parametrize('scale',[1.,1/200/64])
def test_fused_fp32_softcap_keeps_loss_and_gradient_bits(kernel,rows,cols,scale):
    torch.manual_seed(899)
    raw=torch.randn(rows,cols,device='cuda')*3
    teacher=torch.log_softmax(torch.randn(rows,cols,device='cuda',dtype=torch.float64),-1)
    plan=PersistentTrainingPlan(raw.device,kernel=kernel)
    logits=unary(plan,raw,3)
    loss,grad=append_kl(plan,logits,teacher)
    grad=unary(plan,grad,5,scale=scale)
    grad=unary(plan,raw,8,grad)
    fused_loss,fused_grad=append_kl(plan,raw,teacher,softcap_gradient_scale=scale,softcap_fp32=True)
    plan.finalize();plan.step();torch.cuda.synchronize()
    torch.testing.assert_close(fused_loss,loss,rtol=0,atol=0)
    torch.testing.assert_close(fused_grad,grad,rtol=0,atol=0)
