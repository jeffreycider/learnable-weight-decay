"""Numerical tests for literal FP32 and thin-coordinate reverse mode."""
import pytest
import torch

from baseline.model import GPT
from rewriter.algebra import Anchors, Factors
from rewriter.function_preservation import FunctionalPreservation
from rewriter.persistent_fp32 import append_gpt
from rewriter.persistent_mlp import CooperativeTrainingKernel
from rewriter.persistent_offline import constants, typed_empty
from rewriter.persistent_plan import PersistentTrainingPlan
from rewriter.primal_dual import PrimalDual

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='cooperative CUDA FP32 kernel')


@pytest.fixture(scope='module')
def kernel():
    torch.backends.cuda.matmul.allow_tf32=False
    return CooperativeTrainingKernel()


@pytest.mark.parametrize('thin',[False,True])
@pytest.mark.parametrize('tokens',[17,128])
def test_nonzero_gpt_forward_and_thin_or_dense_vjp(kernel,thin,tokens):
    torch.manual_seed(719)
    model=GPT(257,2,128).cuda().float()
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.ndim==2:p.normal_(std=.03)
            elif 'bias' in name:p.normal_(std=.005)
    engine=FunctionalPreservation(model,gpt_precision='fp32',kl_dtype=torch.float64)
    params={n:p.detach().clone().requires_grad_(p.ndim in (1,2) and 'gains' not in n)
            for n,p in engine.parameters.items()}
    inputs=torch.randint(257,(2,tokens),device='cuda')
    expected=engine._forward(params,inputs)
    cotangent=torch.randn_like(expected)/expected.numel()
    (expected*cotangent).sum().backward()
    factors={}
    generator=torch.Generator().manual_seed(720)
    if thin:
        for name,p in params.items():
            if p.ndim!=2:continue
            out,inside=p.T.shape if name=='embed.weight' else p.shape
            # Fixed orthonormal anchors; zero factors do not change this VJP.
            anchors=Anchors.random(out,inside,2,generator=generator,device='cuda')
            factors[name]=Factors(torch.zeros_like(anchors.A),torch.zeros_like(anchors.B),anchors)
    plan=PersistentTrainingPlan(inputs.device,kernel=kernel)
    output,back=append_gpt(plan,{n:p.detach() for n,p in params.items()},engine.buffers,inputs,
                           num_layers=2,thin_factors=factors if thin else None)
    gradients=back(cotangent.flatten(0,1));plan.finalize();plan.step();torch.cuda.synchronize()
    torch.testing.assert_close(output.view_as(expected),expected,atol=2e-6,rtol=2e-5)
    for name,actual in gradients.items():
        dense=params[name].grad
        if isinstance(actual,Factors):
            a,b=actual.anchors.A,actual.anchors.B
            dp=dense.T@b if name=='embed.weight' else dense@b
            dq=dense@a if name=='embed.weight' else dense.T@a
            pairs=((actual.P,dp),(actual.Q,dq))
        else:pairs=((actual,dense),)
        for got,want in pairs:
            relative=float((got-want).double().norm()/want.double().norm().clamp_min(1e-30))
            assert relative<2e-5,(name,relative)


@pytest.mark.parametrize('kl',[-1e-12,8e-7,1.2e-6])
@pytest.mark.parametrize('raw',[False,True])
def test_current_primal_dual_and_raw_kl_controls(kernel,kl,raw):
    plan=PersistentTrainingPlan(torch.device('cuda'),kernel=kernel)
    target=constants(plan,[kl],torch.float64)
    groups=constants(plan,[[-.001,.02,.3],[-.003,.01,.5]],torch.float64)
    budget=constants(plan,1e-6,torch.float64);radius=constants(plan,.1,torch.float64)
    controls=constants(plan,[0.,.1,.01,2. if raw else 1.],torch.float64)
    output=typed_empty(plan,(5,),torch.float64);coefficient=typed_empty(plan,(2,),torch.float64)
    plan.append([29,target.data_ptr(),groups.data_ptr(),output.data_ptr(),coefficient.data_ptr(),
                 budget.data_ptr(),2,0,radius.data_ptr(),controls.data_ptr()])
    plan.finalize();plan.step();torch.cuda.synchronize()
    dual=PrimalDual([1],tolerance=1e-6,rho=.01,dual_lr=.1);dual.multipliers[1]=.1
    terms=dual.terms(kl,1)
    assert float(coefficient[0])==pytest.approx(1. if raw else terms['kl_coefficient'],rel=1e-14)
    assert float(coefficient[1])==0.
    assert float(output[4])==pytest.approx(kl if raw else -.2+terms['kl_penalty'],abs=1e-14)
