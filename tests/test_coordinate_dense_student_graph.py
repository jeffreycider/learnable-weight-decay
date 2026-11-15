"""CPU ownership tests, not evidence that CUDA capture is qualified."""
import pytest
import torch
from dataclasses import asdict
import copy
from rewriter.coordinate_dense_student_graph import DenseStudentGraph
from rewriter.tokenwise_reference import TokenwiseConfig, TokenwiseDual, tokenwise_panel
from tests.test_factorized_fp32 import setup


@pytest.fixture(autouse=True)
def strict_precision():
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def cpu_contract(engine, parameters, inputs):
    graph = object.__new__(DenseStudentGraph)
    def factory(call, samples, **options):
        assert options == dict(num_warmup_iters=2, allow_unused_input=True)
        assert all(a is not b for a, b in zip(samples, (*parameters.values(), *engine.buffers.values(), inputs)))
        return call
    graph._initialize(engine, parameters, inputs, factory, warmups=2)
    return graph


def test_multiple_microbatch_dense_accumulation_and_shared_private_credit():
    config = TokenwiseConfig(mean_tolerance=1e-10, tail_tolerance=1e-10, scale=1e-5, rho=.01,
                            ranking_weight=.01, ranking_k=10,
                            ranking_tie_policy='exact_teacher_ties', ranking_loss_mode='relative_gap')
    dual = TokenwiseDual([3], config=config, dual_lr=.01)
    dual.multipliers[3] = .1
    results, grads = [], []
    for use_wrapper in (False, True):
        engine, action, factors, inputs = setup()
        engine.microbatch_sequences = 1
        teacher = [engine.cache_teacher(x[None], data_role='synthetic')[0] for x in inputs]
        parameters = engine.edited_parameters(action, 1.)
        regularizer = sum(v.double().square().mean() for v in parameters.values()) * .001
        wrapper = cpu_contract(engine, parameters, inputs[:1]) if use_wrapper else engine
        results.append(tokenwise_panel(wrapper, parameters, regularizer, inputs,
            config=config, dual=dual, state_id=3, state_weight=1/16,
            teacher_provider=lambda a,b: teacher[a]))
        grads.append([None if v.grad is None else v.grad.clone() for v in factors])
        if use_wrapper:
            assert not wrapper.pending and wrapper.replays == 2
    assert results[0] == results[1]
    for a,b in zip(*grads):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert all(v.grad is None for v in engine.parameters.values())


def test_pending_layout_identity_and_buffer_guards():
    engine, action, _, inputs = setup()
    parameters = {n:v.detach().requires_grad_(v.requires_grad)
                  for n,v in engine.edited_parameters(action,1.).items()}
    graph = cpu_contract(engine, parameters, inputs)
    y = graph._forward(parameters, inputs)
    with pytest.raises(RuntimeError, match='previous'):
        graph._forward(parameters, inputs)
    with pytest.raises(RuntimeError, match='previous'):
        graph.bind_checkpoint()
    y.sum().backward()
    assert not graph.pending
    with pytest.raises(ValueError, match='microbatch'):
        graph._forward(parameters, inputs[:1])
    with torch.no_grad(), pytest.raises(RuntimeError, match='inference'):
        graph._forward(parameters, inputs)
    key = next(n for n,v in parameters.items() if v.requires_grad)
    changed = dict(parameters);changed[key] = changed[key].detach()
    with pytest.raises(ValueError, match='editable mask'):
        graph._forward(changed, inputs)
    name = next(iter(engine.buffers))
    engine.buffers[name] = engine.buffers[name].clone()
    with pytest.raises(RuntimeError, match='identity/buffers'):
        graph._forward(parameters, inputs)
    engine.teacher_id = 'verified-new-checkpoint'
    with torch.no_grad():
        engine.buffers[name].add_(.01)
    graph.bind_checkpoint()
    expected = engine._forward(parameters, inputs)
    observed = graph._forward(parameters, inputs)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    observed.sum().backward()
    assert graph.teacher_id == engine.teacher_id
    engine.view.gpt_precision = 'baseline'
    with pytest.raises(RuntimeError, match='implementation'):
        graph.bind_checkpoint()


def test_public_constructor_rejects_cpu_and_precision_changes():
    engine, action, _, inputs = setup()
    parameters = engine.edited_parameters(action,1.)
    with pytest.raises(ValueError, match='requires CUDA'):
        DenseStudentGraph(engine, parameters, inputs)
    with pytest.raises(ValueError, match='fixed_anchored'):
        DenseStudentGraph(engine, parameters, inputs, action_family='adaptive_ucv_v1')
    torch.backends.cuda.matmul.allow_tf32 = True
    with pytest.raises(ValueError, match='TF32'):
        cpu_contract(engine, parameters, inputs)


def test_stale_backward_cannot_consume_new_pending_replay_or_change_gradients():
    engine,action,_,inputs=setup()
    parameters={n:v.detach().requires_grad_(v.requires_grad)
                for n,v in engine.edited_parameters(action,1.).items()}
    graph=cpu_contract(engine,parameters,inputs)
    old=graph._forward(parameters,inputs)
    old.sum().backward(retain_graph=True)
    new=graph._forward(parameters,inputs)
    before={n:v.grad.clone() for n,v in parameters.items() if v.grad is not None}
    with pytest.raises(RuntimeError,match='stale or repeated'):
        old.sum().backward(retain_graph=True)
    assert graph.pending
    for n,value in before.items():
        torch.testing.assert_close(parameters[n].grad,value,rtol=0,atol=0)
    new.sum().backward(retain_graph=True)
    assert not graph.pending
    with pytest.raises(RuntimeError,match='stale or repeated'):
        new.sum().backward(retain_graph=True)


def test_failed_bind_invalidates_previous_checkpoint_until_valid_rebind():
    engine,action,_,inputs=setup()
    parameters={n:v.detach().requires_grad_(v.requires_grad)
                for n,v in engine.edited_parameters(action,1.).items()}
    graph=cpu_contract(engine,parameters,inputs)
    name=next(iter(engine.buffers));original=engine.buffers[name]
    engine.buffers[name]=original.flatten()[:1]
    with pytest.raises(ValueError,match='buffer layout'):
        graph.bind_checkpoint()
    assert not graph.bound
    engine.buffers[name]=original
    with pytest.raises(RuntimeError,match='not bound'):
        graph._forward(parameters,inputs)
    graph.bind_checkpoint()
    graph._forward(parameters,inputs).sum().backward()
    assert not graph.pending


def test_replay_exception_poisoned_capture_cannot_be_rebound():
    engine,action,_,inputs=setup()
    parameters={n:v.detach().requires_grad_(v.requires_grad)
                for n,v in engine.edited_parameters(action,1.).items()}
    graph=cpu_contract(engine,parameters,inputs)
    def failure(*args):raise RuntimeError('injected replay failure')
    graph.call=failure
    with pytest.raises(RuntimeError,match='injected'):
        graph._forward(parameters,inputs)
    assert not graph.usable and not graph.pending
    with pytest.raises(RuntimeError,match='new capture'):
        graph.bind_checkpoint()


def test_qualifier_explicitly_rejects_new_action_or_loss_family_before_cuda():
    from scripts.qualify_coordinate_dense_student_graph import validate_candidate_recipe
    config=TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-8,rho=.01,
        ranking_weight=100.,ranking_k=100,ranking_tie_policy='exact_teacher_ties',
        ranking_loss_mode='relative_gap',loss_backend='tiled_relative_fp64_v1')
    recipe=dict(architecture={'frame_mode':'tied_orthonormal'},
                objectives={d:asdict(config) for d in ('gpt','cifar10')})
    validate_candidate_recipe(recipe)
    changed=copy.deepcopy(recipe);changed['action_family']='adaptive_ucv_v1'
    with pytest.raises(ValueError,match='anchored family'):validate_candidate_recipe(changed)
    changed=copy.deepcopy(recipe);changed['objectives']['gpt']['ranking_k']=10
    with pytest.raises(ValueError,match='relative100'):validate_candidate_recipe(changed)


def test_private_only_then_joint_preserves_prior_shared_gradient_and_adam():
    from rewriter.algebra import RawAction,Factors
    from rewriter.distributed_probe_store import _digest
    results=[]
    for use_graph in (False,True):
        engine,template,_,inputs=setup();engine.microbatch_sequences=1
        teachers=[engine.cache_teacher(x[None],data_role='synthetic')[0] for x in inputs]
        shared=torch.tensor(.3,requires_grad=True);private=torch.tensor(.7,requires_grad=True)
        shared.grad=torch.tensor(7.)  # already accumulated other-domain credit
        so=torch.optim.Adam([shared],lr=1e-4);po=torch.optim.Adam([private],lr=1e-4)
        config=TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-5,rho=.01,
            ranking_weight=.01,ranking_k=10,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap')
        dual=TokenwiseDual([3],config=config,dual_lr=.01);dual.multipliers[3]=.1
        graph=None;rows=[]
        for joint in (False,True):
            shared.requires_grad_(joint)
            action=RawAction({n:Factors(f.P.detach()*shared.sin(),f.Q.detach()*private.square(),f.anchors)
                              for n,f in template.matrices.items()},
                             {n:v.detach()*shared*private for n,v in template.vectors.items()})
            parameters=engine.edited_parameters(action,1.)
            regularizer=sum(v.double().square().mean() for v in parameters.values())*.001
            if use_graph and graph is None:graph=cpu_contract(engine,parameters,inputs[:1])
            weight=.5 if joint else 1.
            row=tokenwise_panel(graph if use_graph else engine,parameters,regularizer,inputs,
                config=config,dual=dual,state_id=3,state_weight=weight,teacher_provider=lambda a,b:teachers[a])
            rows.append(row)
            if not joint:assert shared.grad.item()==7.
            private.grad.div_(weight);po.step();po.zero_grad(set_to_none=True)
            dual.step([(3,row['mean_kl'])],expected_ids=[3])
        so.step()
        results.append(dict(shared=shared.detach(),private=private.detach(),shared_optimizer=so.state_dict(),
                            private_optimizer=po.state_dict(),dual=dual.state_dict(),rows=rows))
    assert _digest(results[0])==_digest(results[1])


def test_mutation_during_pending_backward_rejected_before_gradient_accumulation():
    engine,action,_,inputs=setup()
    parameters={n:v.detach().requires_grad_(v.requires_grad)
                for n,v in engine.edited_parameters(action,1.).items()}
    graph=cpu_contract(engine,parameters,inputs)
    y=graph._forward(parameters,inputs)
    with torch.no_grad():next(v for v in parameters.values() if v.requires_grad).add_(.001)
    with pytest.raises(RuntimeError,match='weights changed'):
        y.sum().backward()
    assert all(v.grad is None for v in parameters.values())
    assert not graph.pending and not graph.usable  # incomplete VJP requires recapture


def test_autocast_cannot_silently_change_captured_strict_precision():
    engine,action,_,inputs=setup()
    parameters={n:v.detach().requires_grad_(v.requires_grad)
                for n,v in engine.edited_parameters(action,1.).items()}
    with torch.autocast('cpu',dtype=torch.bfloat16),pytest.raises(ValueError,match='autocast'):
        cpu_contract(engine,parameters,inputs)
    graph=cpu_contract(engine,parameters,inputs)
    with torch.autocast('cpu',dtype=torch.bfloat16),pytest.raises(RuntimeError,match='precision'):
        graph._forward(parameters,inputs)
    assert not graph.pending


def test_capture_rejects_sample_clone_that_changes_parameter_strides():
    engine,action,_,inputs=setup()
    parameters={n:v.detach().requires_grad_(v.requires_grad)
                for n,v in engine.edited_parameters(action,1.).items()}
    name=next(n for n,v in parameters.items() if v.ndim==2)
    original=parameters[name];wide=torch.empty(original.shape[0],original.shape[1]*2)
    strided=wide[:,::2];strided.copy_(original.detach());strided.requires_grad_(original.requires_grad)
    parameters[name]=strided
    with pytest.raises(ValueError,match='cloning changed tensor layout'):
        cpu_contract(engine,parameters,inputs)


def test_capture_rejects_no_grad_and_inference_before_factory():
    engine,action,_,inputs=setup();parameters=engine.edited_parameters(action,1.)
    with torch.no_grad(),pytest.raises(ValueError,match='gradient-enabled'):
        cpu_contract(engine,parameters,inputs)
    with torch.inference_mode(),torch.enable_grad(),pytest.raises(ValueError,match='gradient-enabled'):
        cpu_contract(engine,parameters,inputs)
    half=object.__new__(DenseStudentGraph)
    with pytest.raises(AttributeError):getattr(half,'engine')


def test_failed_external_backward_can_be_terminally_discarded():
    engine,action,_,inputs=setup()
    parameters={n:v.detach().requires_grad_(v.requires_grad)
                for n,v in engine.edited_parameters(action,1.).items()}
    graph=cpu_contract(engine,parameters,inputs)
    def fail(gradient):raise RuntimeError('injected gradient failure')
    handle=next(v for v in parameters.values() if v.requires_grad).register_hook(fail)
    y=graph._forward(parameters,inputs)
    with pytest.raises(RuntimeError,match='injected gradient'):
        y.sum().backward()
    graph.discard();handle.remove()
    assert not graph.pending and not graph.usable and graph._leaves is None
    with pytest.raises(RuntimeError,match='new capture'):
        graph.bind_checkpoint()


# Fable review regression: imitate CUDA graph gradient-storage aliasing on CPU.
class _StaticReplayFactory:
    """Eager forward whose backward hands back REUSED static gradient storage, like Graphed."""
    def __init__(self):
        self.static_grad_inputs = None
        self.replays = 0

    def __call__(self, call, samples, **options):
        factory = self
        names = [i for i, s in enumerate(samples) if s.requires_grad]

        class Static(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *values):
                factory.replays += 1
                with torch.enable_grad():
                    leaves = [v.detach().requires_grad_(v.requires_grad) for v in values]
                    y = call(*leaves)
                    ctx.leaves = leaves
                    ctx.y = y
                return y.detach()

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, g):
                grads = torch.autograd.grad(ctx.y, [ctx.leaves[i] for i in names], g, allow_unused=True)
                if factory.static_grad_inputs is None:
                    factory.static_grad_inputs = [None if x is None else x.clone() for x in grads]
                else:
                    for s, x in zip(factory.static_grad_inputs, grads):
                        if s is not None:
                            s.copy_(x)
                result = [None] * len(values_len[0])
                for i, s in zip(names, factory.static_grad_inputs):
                    result[i] = None if s is None else s.detach()
                return tuple(result)

        values_len = [samples]
        return lambda *values: Static.apply(*values)


def test_static_gradient_storage_accumulates_without_aliasing():
    """Two microbatches through tokenwise_panel: true accumulation vs aliasing of the static buffer."""
    config = TokenwiseConfig(mean_tolerance=1e-10, tail_tolerance=1e-10, scale=1e-5, rho=.01,
                             ranking_weight=.01, ranking_k=10, ranking_tie_policy='exact_teacher_ties',
                             ranking_loss_mode='relative_gap')
    dual = TokenwiseDual([3], config=config, dual_lr=.01); dual.multipliers[3] = .1
    results = {}
    for mode in ('eager', 'static_with_clone', 'static_without_clone'):
        engine, action, factors, inputs = setup()
        engine.microbatch_sequences = 1
        teacher = [engine.cache_teacher(x[None], data_role='synthetic')[0] for x in inputs]
        parameters = engine.edited_parameters(action, 1.)
        regularizer = sum(v.double().square().mean() for v in parameters.values()) * .001
        if mode == 'eager':
            student = engine
        else:
            student = object.__new__(DenseStudentGraph)
            student._initialize(engine, parameters, inputs[:1], _StaticReplayFactory(), warmups=2)
            if mode == 'static_without_clone':
                student._backward_complete = lambda: (setattr(student, '_leaves', None), setattr(student, 'pending', False), setattr(student, '_backward_scheduled', False))
        row = tokenwise_panel(student, parameters, regularizer, inputs, config=config, dual=dual, state_id=3,
                              state_weight=1 / 16, teacher_provider=lambda a, b: teacher[a])
        results[mode] = [None if v.grad is None else v.grad.clone() for v in factors]
    def diff(a, b):
        return max(float((x - y).abs().max()) for x, y in zip(a, b) if x is not None)
    assert diff(results['eager'], results['static_with_clone']) == 0.
    assert diff(results['eager'], results['static_without_clone']) > .01
