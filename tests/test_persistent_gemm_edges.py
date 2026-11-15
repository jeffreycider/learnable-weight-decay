"""Predicated CUTLASS tiles must preserve unaligned actor token dimensions."""
import os
import pytest
import torch
from rewriter.persistent_mlp import CooperativeTrainingKernel
from rewriter.persistent_plan import PersistentTrainingPlan

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('CHISEL_CUTLASS_INCLUDE'),reason='CUTLASS CUDA kernel')

@pytest.fixture(scope='module')
def kernel():return CooperativeTrainingKernel()


@pytest.mark.skipif(not os.environ.get('CHISEL_CUTLASS_IEEE'),reason='IEEE CUTLASS path')
def test_ieee_pipeline_uses_fused_fp32_rounding(kernel):
    a=torch.zeros(128,8,device='cuda');b=torch.zeros(8,64,device='cuda')
    a[:,0]=-1.;b[0,:]=1.;a[:,1]=1.+2**-13;b[1,:]=1.-2**-13
    plan=PersistentTrainingPlan(a.device,kernel=kernel);actual=plan.gemm(a,b,ieee=True)
    plan.finalize();plan.step();torch.cuda.synchronize()
    # fma(a,b,-1) retains this residual; separately rounded a*b then -1 loses it.
    torch.testing.assert_close(actual,torch.full_like(actual,-2**-26),rtol=0,atol=0)

@pytest.mark.parametrize('shape',[(37,41,40),(13228,512,128),(257,131,1040)])
@pytest.mark.parametrize('transpose,rounded',[(False,False),(True,True)])
def test_predicated_bf16_edges(kernel,shape,transpose,rounded):
    torch.manual_seed(821)
    m,n,k=shape
    a=torch.randn((k,m) if transpose else (m,k),device='cuda')*.03
    b=torch.randn((n,k) if transpose else (k,n),device='cuda')*.03
    bias=torch.randn(n,device='cuda')*.01
    plan=PersistentTrainingPlan(a.device,kernel=kernel)
    actual=plan.gemm(a,b,ta=transpose,tb=transpose,bf16=True,bias=bias,
                     epilogue=1,factor=.7,round_output=rounded)
    assert any(row[0]==44 for row in plan.instructions)
    plan.finalize();plan.step();torch.cuda.synchronize()
    av=(a.T if transpose else a).bfloat16().float()
    bv=(b.T if transpose else b).bfloat16().float()
    expected=(av@bv)*.7+bias
    if rounded:expected=expected.bfloat16().float()
    torch.testing.assert_close(actual,expected,rtol=.008 if rounded else 2e-5,atol=2e-6)


@pytest.mark.parametrize('shape',[(67,8,33),(257,16,1040),(50304,8,128)])
@pytest.mark.parametrize('ta,tb',[(False,False),(False,True),(True,False),(True,True)])
def test_ieee_rank_eight_contractions(kernel,shape,ta,tb):
    torch.manual_seed(822)
    m,n,k=shape
    a=torch.randn((k,m) if ta else (m,k),device='cuda')*.03
    b=torch.randn((n,k) if tb else (k,n),device='cuda')*.03
    plan=PersistentTrainingPlan(a.device,kernel=kernel)
    actual=plan.gemm(a,b,ta=ta,tb=tb,ieee=True)
    plan.finalize();plan.step();torch.cuda.synchronize()
    expected=(a.T if ta else a)@(b.T if tb else b)
    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=2e-7)


@pytest.mark.skipif(not os.environ.get('CHISEL_CUTLASS_TF32X3'),reason='optional compensated pipeline')
@pytest.mark.parametrize('shape',[(132,68,40),(256,768,3072),(256,768,50304)])
@pytest.mark.parametrize('ta,tb',[(False,False),(False,True),(True,False),(True,True)])
def test_compensated_pipeline_contractions(kernel,shape,ta,tb):
    torch.manual_seed(823)
    m,n,k=shape
    a=torch.randn((k,m) if ta else (m,k),device='cuda')*.03
    b=torch.randn((n,k) if tb else (k,n),device='cuda')*.03
    plan=PersistentTrainingPlan(a.device,kernel=kernel)
    actual=plan.gemm(a,b,ta=ta,tb=tb)
    assert any(row[0]==48 for row in plan.instructions)
    plan.finalize();plan.step();torch.cuda.synchronize()
    expected=(a.T if ta else a)@(b.T if tb else b)
    relative=float((actual-expected).norm()/expected.norm())
    assert relative<1e-5,relative


@pytest.mark.skipif(not os.environ.get('CHISEL_CUTLASS_TF32X3'),reason='optional compensated pipeline')
def test_explicit_forward_emulation_dispatch_and_finite_response(kernel,monkeypatch):
    monkeypatch.setenv('CHISEL_FP32_EMULATION','tf32x3')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(824)
    a=torch.randn(256,768,device='cuda')*.03
    b=torch.randn(768,768,device='cuda')*.03
    plan=PersistentTrainingPlan(a.device,kernel=kernel)
    output=plan.gemm(a,b,ieee=True)
    assert any(row[0]==48 for row in plan.instructions)
    plan.finalize();plan.step();torch.cuda.synchronize()
    before=output.clone();expected=a@b
    assert float((before-expected).norm()/expected.norm())<3e-6
    b.add_(torch.randn_like(b)*3e-7)
    response=a@b-expected;plan.step();torch.cuda.synchronize()
    assert float(((output-before)-response).norm()/response.norm())<.3
