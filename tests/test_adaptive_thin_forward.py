"""Whole tiny GPT oracle for the default-off adaptive thin derivative path."""
import pytest
import torch
from rewriter.adaptive_action import AdaptiveAction, AdaptiveFactors
from rewriter.factorized_fp32 import factorized_logits
from tests.test_factorized_fp32 import setup


@pytest.mark.parametrize('factor_dtype', [torch.float32, torch.float64])
def test_stored_forward_exact_and_full_gpt_factor_gradient(factor_dtype):
    torch.set_num_threads(1)
    engine, template, _, inputs = setup()
    generator = torch.Generator().manual_seed(91)
    values = {}
    for name, f in template.matrices.items():
        values[name] = [torch.randn(f.P.shape[0], 2, generator=generator, dtype=factor_dtype)*.01,
                       torch.randn(2, 2, generator=generator, dtype=factor_dtype)*.01,
                       torch.randn(f.Q.shape[0], 2, generator=generator, dtype=factor_dtype)*.01]
    vectors = {n: v.detach().to(factor_dtype)*.01 for n,v in template.vectors.items()}
    outcomes, gradients = [], []
    for thin in (False, True):
        mats = {n: AdaptiveFactors(*[x.clone().requires_grad_() for x in factors]) for n,factors in values.items()}
        vec = {n:v.clone().requires_grad_() for n,v in vectors.items()}
        action = AdaptiveAction(mats,vec)
        scale = torch.tensor(.13, dtype=factor_dtype, requires_grad=True)
        realized = {}
        for n, base in engine.parameters.items():
            delta = mats[n].dense() if n in mats else vec[n]
            if engine.transpose_storage.get(n,False): delta=delta.T
            realized[n] = (base.to(factor_dtype)+scale*delta).float()
        # Normalization parameters are fixed in production. Exclude their write
        # derivatives here too; adaptive thin norm uses the unchanged base.
        for n in vec:
            if n.endswith('.gains'):
                realized[n] = engine.parameters[n]
                vec[n] = torch.zeros_like(vec[n])
        action=AdaptiveAction(mats,vec)
        if thin:
            out=factorized_logits(engine.view.gpt,engine.parameters,engine.buffers,inputs,
                action,scale,realized_parameters={n:p.detach() for n,p in realized.items()},dx_backend='ieee')
        else:
            out=engine._forward(realized,inputs)
        outcomes.append(out.detach())
        leaves=[v for f in mats.values() for v in (f.U,f.C,f.V)]
        leaves += [v for v in vec.values() if v.requires_grad]+[scale]
        gradients.append(torch.autograd.grad(out.square().mean(),leaves))
    torch.testing.assert_close(outcomes[0],outcomes[1],rtol=0,atol=0)
    for dense,thin in zip(*gradients):
        torch.testing.assert_close(dense,thin,rtol=5e-4,atol=2e-6)
