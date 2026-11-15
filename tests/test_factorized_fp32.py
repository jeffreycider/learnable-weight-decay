import pytest
import torch

from baseline.model import GPT
from rewriter.algebra import Anchors, Factors, RawAction
from rewriter.factorized_fp32 import factorized_logits
from rewriter.function_preservation import FunctionalPreservation


def setup(zero=False):
    torch.manual_seed(552)
    engine = FunctionalPreservation(GPT(32, 1, 128), gpt_precision='fp32', kl_dtype=torch.float64)
    matrices, vectors = {}, {}
    generator = torch.Generator().manual_seed(91)
    leaves = []
    for name, parameter in engine.parameters.items():
        if parameter.ndim == 2:
            weight = parameter.T if name == 'embed.weight' else parameter
            anchors = Anchors.random(*weight.shape, 2, generator=generator)
            p = (torch.randn_like(anchors.A) * (0 if zero else .002)).requires_grad_()
            q = (torch.randn_like(anchors.B) * (0 if zero else .002)).requires_grad_()
            leaves.extend((p, q)); matrices[name] = Factors.canonical(p, q, anchors)
        else:
            value = (torch.randn_like(parameter) * (0 if zero else .002)).requires_grad_()
            vectors[name] = value; leaves.append(value)
    return engine, RawAction(matrices, vectors), leaves, torch.randint(32, (2, 5))


def test_zero_edit_matches_dense_teacher_exactly():
    engine, action, _, inputs = setup(zero=True)
    expected = engine._forward(engine.parameters, inputs)
    actual = factorized_logits(engine.view.gpt, engine.parameters, engine.buffers, inputs, action, 1.)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dense_base_parameter_gradients_are_rejected():
    engine, action, _, inputs = setup()
    parameters = dict(engine.parameters)
    parameters['proj.weight'] = parameters['proj.weight'].clone().requires_grad_()
    with pytest.raises(ValueError, match='frozen FP32'):
        factorized_logits(engine.view.gpt, parameters, engine.buffers, inputs, action, 1.)


@pytest.mark.parametrize('saturated', [False, True])
@pytest.mark.parametrize('dense_forward',[False,True])
def test_thin_factor_gradients_match_dense_edit_including_global_scale(saturated,dense_forward):
    engine, action, leaves, inputs = setup()
    scale = .01 / action.norm_sq().sqrt() if saturated else action.norm_sq().new_ones(())
    dense = engine._forward(engine.edited_parameters(action, scale), inputs)
    thin = factorized_logits(engine.view.gpt, engine.parameters, engine.buffers, inputs, action, scale,realized_parameters=engine.edited_parameters(action,scale) if dense_forward else None)
    if dense_forward:torch.testing.assert_close(thin,dense,rtol=0,atol=0)
    torch.testing.assert_close(thin, dense, rtol=2e-5, atol=2e-6)
    cotangent = torch.randn_like(dense)
    expected = torch.autograd.grad((dense * cotangent).sum(), leaves, retain_graph=True)
    observed = torch.autograd.grad((thin * cotangent).sum(), leaves)
    for actual, reference in zip(observed, expected):
        torch.testing.assert_close(actual, reference, rtol=3e-4, atol=5e-6)
    assert all(p.grad is None and not p.requires_grad for p in engine.parameters.values())


@pytest.mark.parametrize('dense_forward',[False,True])
def test_accumulated_thin_actor_gradient_matches_dense_production_chain(dense_forward):
    from types import SimpleNamespace
    from rewriter.grouped_norm import GroupedNormObjective
    from rewriter.edit_scope import editable_names, scoped_action, keep_readonly_parameters
    from rewriter.primal_dual import PrimalDual
    from scripts.train_multistate_chisel import group_action_scales, mean_kl_and_backward
    from scripts.benchmark_factorized_fp32 import thin_step
    torch.set_num_threads(1)
    engine, template, _, inputs = setup()
    engine.microbatch_sequences = 1
    grouped = GroupedNormObjective(engine.parameters, zero_reference_policy='freeze')
    editable = editable_names(engine.parameters, 'grouped') - grouped.frozen_parameter_names
    scales = group_action_scales(grouped, 2)
    phi = torch.tensor(.3, requires_grad=True)
    def actor(_):
        return RawAction({n: Factors(f.P.detach()*phi.sin(), f.Q.detach()*phi.square(), f.anchors)
                          for n,f in template.matrices.items()},
                         {n:v.detach()*phi for n,v in template.vectors.items()})
    snapshot = SimpleNamespace(radius=torch.tensor(.001))  # exercise saturated guard
    class Panel:
        def teacher_batch(self, engine, start, stop):
            return engine.cache_teacher(inputs[start:stop], data_role='past_training')[0]
    panel=Panel();dual=PrimalDual([7], tolerance=1e-6, rho=.01, dual_lr=.1)
    dual.multipliers[7]=.1
    result=thin_step(actor,snapshot,engine,grouped,inputs,panel,dual,7,dense_forward=dense_forward)
    observed=phi.grad.clone();phi.grad=None
    raw=scoped_action(actor(snapshot),editable)
    action=RawAction({n:Factors(f.P*scales[n],f.Q*scales[n],f.anchors) for n,f in raw.matrices.items()},
                     {n:v*scales[n] for n,v in raw.vectors.items()})
    scale=torch.minimum(torch.ones(()),snapshot.radius/(action.norm_sq().clamp_min(1e-24).sqrt()+1e-12))
    params=keep_readonly_parameters(engine.edited_parameters(action,scale),engine.parameters,editable)
    expected=mean_kl_and_backward(engine,params,100*grouped.evaluate(params).mean_relative_change,inputs,
        kl_budget=1e-6,teacher_provider=lambda a,b:panel.teacher_batch(engine,a,b),primal_dual=dual,state_id=7)
    torch.testing.assert_close(observed,phi.grad,rtol=2e-3,atol=1e-6)
    assert abs(result['mean_kl']-expected['mean_kl'])<1e-9
