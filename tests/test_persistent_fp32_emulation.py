import os
import pytest
import torch
from rewriter.persistent_mlp import CooperativeTrainingKernel
from rewriter.persistent_plan import PersistentTrainingPlan

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get('CHISEL_FP32_EMULATION')!='bf16x9',reason='optional FP32 emulation qualification')

@pytest.fixture(scope='module')
def kernel():
    torch.backends.cuda.matmul.allow_tf32=False
    return CooperativeTrainingKernel()

def test_three_residual_components_reconstruct_normal_fp32_values(kernel):
    torch.manual_seed(840)
    x=torch.randn(128,64,device='cuda')*torch.logspace(-20,20,64,device='cuda')
    plan=PersistentTrainingPlan(x.device,kernel=kernel)
    packed=plan.empty((3*x.numel()//2,))
    plan.append([49,x.data_ptr(),packed.data_ptr(),0,0,0,128,64,64,0])
    plan.finalize();plan.step();torch.cuda.synchronize()
    parts=packed.view(torch.bfloat16).view(3,128,64).float()
    torch.testing.assert_close(parts[0]+parts[1]+parts[2],x,rtol=0,atol=0)

def test_all_nine_products_keep_small_cancellation_residual(kernel):
    a=torch.zeros(128,64,device='cuda');b=torch.zeros(64,64,device='cuda')
    a[:,0]=-1.;b[0,:]=1.;a[:,1]=1.+2**-13;b[1,:]=1.-2**-13
    plan=PersistentTrainingPlan(a.device,kernel=kernel);out=plan.gemm(a,b,ieee=True)
    assert any(row[0]==50 for row in plan.instructions)
    plan.finalize();plan.step();torch.cuda.synchronize()
    torch.testing.assert_close(out,torch.full_like(out,-2**-26),rtol=0,atol=0)

@pytest.mark.parametrize('ta,tb',[(False,False),(False,True),(True,False),(True,True)])
def test_output_and_small_finite_edit_response(kernel,ta,tb):
    torch.manual_seed(841);m,n,k=256,768,768
    a=torch.randn((k,m) if ta else (m,k),device='cuda')*.03
    b=torch.randn((n,k) if tb else (k,n),device='cuda')*.03
    plan=PersistentTrainingPlan(a.device,kernel=kernel);out=plan.gemm(a,b,ta=ta,tb=tb,ieee=True);plan.finalize()
    def ref():return (a.T if ta else a)@(b.T if tb else b)
    expected=ref();plan.step();torch.cuda.synchronize();base=out.clone()
    assert float((out-expected).norm()/expected.norm())<3e-6
    b.add_(torch.randn_like(b)*3e-7)
    changed=ref();plan.step();torch.cuda.synchronize()
    response=changed-expected
    assert float(((out-base)-response).norm()/response.norm())<.3


def test_immutable_operand_cache_recomputes_on_next_launch(kernel,monkeypatch):
    monkeypatch.setenv('CHISEL_EMULATION_CACHE_WEIGHTS','1')
    torch.manual_seed(842)
    a=torch.randn(128,128,device='cuda')*.03
    b=torch.randn(128,64,device='cuda')*.03
    plan=PersistentTrainingPlan(a.device,kernel=kernel)
    a2=a*2
    one=plan.gemm(a,b,ieee=True,immutable_b=True)
    two=plan.gemm(a2,b,ieee=True,immutable_b=True)
    assert sum(row[0]==49 for row in plan.instructions)==3
    plan.finalize()
    for _ in range(2):
        plan.step();torch.cuda.synchronize()
        expected=a@b
        assert float((one-expected).norm()/expected.norm())<3e-6
        torch.testing.assert_close(two,one*2,rtol=0,atol=0)
        b.add_(torch.randn_like(b)*3e-7)
