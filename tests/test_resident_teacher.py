import copy

import pytest
import torch

from baseline.model import GPT
from rewriter.function_preservation import FunctionalPreservation
from rewriter.offline_systems import select_panel_blocks, BlockSelectedTeacherPanel
from rewriter.resident_teacher import ResidentTeacherFeatures
from rewriter.teacher_cache import TeacherCacheReader, TeacherPanel, bind_engine, write_panel


def fixture(tmp_path, device='cpu'):
    torch.set_num_threads(1)
    torch.manual_seed(981)
    engine = FunctionalPreservation(GPT(17, 1, 128).float().to(device).requires_grad_(False),
        microbatch_sequences=2, teacher_cache_device=device, kl_dtype=torch.float64, gpt_precision='fp32')
    initial = {n: t.clone() for n, t in engine.parameters.items()}
    buffers = {n: t.clone() for n, t in engine.buffers.items()}
    states = {1: initial, 2: {n: t+.001 if n == 'proj.bias' else t.clone() for n, t in initial.items()}}
    inputs = torch.randint(17, (6, 4))
    for state_id, state in states.items():
        engine.load_teacher_state(state, buffers)
        write_panel(tmp_path, state_id, 'train', engine, inputs, range(6),
                    state_sha256=str(state_id)*64, execution={'purpose': 'resident teacher test'})
    reader = TeacherCacheReader(tmp_path, require_complete=False, verify_once=True, mmap=True)
    def load(state_id, blocks=(2, 0)):
        engine.load_teacher_state(states[state_id], buffers)
        bind_engine(engine, str(state_id)*64)
        return select_panel_blocks(reader.read(state_id, 'train'), blocks)
    return engine, load


def test_reconstruction_and_rebinding_match_original_without_cpu_selection(tmp_path, monkeypatch):
    engine, load = fixture(tmp_path)
    panel = load(1)
    original = [panel.teacher_batch(engine, i, i+2) for i in (0, 2)]
    def forbidden(_):
        raise AssertionError('lazy features must not be materialized on CPU')
    monkeypatch.setattr(BlockSelectedTeacherPanel, 'features', property(forbidden))
    before = engine.cached_teacher_head_forwards
    resident = ResidentTeacherFeatures(panel, engine)
    assert engine.cached_teacher_head_forwards == before  # binding adds no head forward
    pointers = resident.inputs.data_ptr(), resident.features.data_ptr()
    assert resident.sequence_ids == panel.sequence_ids
    for i, expected in zip((0, 2), original):
        actual = resident.teacher_batch(engine, i, i+2)
        assert torch.equal(actual.inputs, expected.inputs)
        assert torch.equal(actual.log_probs, expected.log_probs)
        assert actual.sequence_ids == expected.sequence_ids
    assert resident.resident_bytes == 4*4*128*4 + 4*4*8
    for state_id, blocks in ((2, (1, 2)), (1, (0, 2))):
        panel = load(state_id, blocks)
        with pytest.raises(ValueError, match='rebound'):
            resident.check_binding(engine)
        resident.bind(panel, engine)
        assert pointers == (resident.inputs.data_ptr(), resident.features.data_ptr())
        for start in (0, 2):
            expected = panel.teacher_batch(engine, start, start+2)
            actual = resident.teacher_batch(engine, start, start+2)
            assert torch.equal(actual.log_probs, expected.log_probs)
            assert actual.sequence_ids == expected.sequence_ids


def test_guards_reject_mixed_precision_head_mutation_and_stale_storage(tmp_path):
    engine, load = fixture(tmp_path)
    panel = load(1)
    resident = ResidentTeacherFeatures(panel, engine)
    with pytest.raises(ValueError, match='aligned'):
        resident.teacher_batch(engine, 1, 3)
    resident.features.add_(1.)
    with pytest.raises(ValueError, match='mutated'):
        resident.check_binding(engine)
    resident.bind(panel, engine)
    engine.parameters['proj.bias'].add_(1.)
    with pytest.raises(ValueError, match='rebound'):
        resident.check_binding(engine)
    with pytest.raises(ValueError, match='identity/version'):
        resident.bind(panel, engine)
    panel = load(1)
    wrong = copy.deepcopy(panel.metadata)
    wrong['precision']['rms_epsilon'] = 2**-7
    bad = TeacherPanel(panel.inputs, panel.parent.features[:4], panel.sequence_ids, wrong)
    with pytest.raises(ValueError, match='normalization epsilon'):
        resident.bind(bad, engine)
    with pytest.raises(ValueError, match='rebound'):
        resident.check_binding(engine)
    panel = load(1, (0, 1, 2))
    with pytest.raises(ValueError, match='schema changed'):
        resident.bind(panel, engine)


def test_partial_copy_failure_invalidates_and_engine_identity_is_checked(tmp_path, monkeypatch):
    engine, load = fixture(tmp_path)
    panel = load(1)
    resident = ResidentTeacherFeatures(panel, engine)
    # Same tensors, UUID and cache metadata, but a different engine object.
    other_engine = copy.copy(engine)
    with pytest.raises(ValueError, match='rebound'):
        resident.check_binding(other_engine)
    replacement = load(1, (0, 1))
    original_copy = torch.Tensor.copy_
    copies = 0

    def interrupted_copy(target, source, *args, **kwargs):
        nonlocal copies
        copies += 1
        if copies == 3:  # input and first feature block have already changed
            raise RuntimeError('injected second-block copy failure')
        return original_copy(target, source, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'copy_', interrupted_copy)
        with pytest.raises(RuntimeError, match='second-block'):
            resident.bind(replacement, engine)
    assert copies == 3
    with pytest.raises(ValueError, match='rebound'):
        resident.check_binding(engine)
    resident.bind(replacement, engine)
    resident.check_binding(engine)
    assert torch.equal(resident.teacher_batch(engine, 0, 2).log_probs,
                       replacement.teacher_batch(engine, 0, 2).log_probs)


def test_actual_gpt_credit_and_three_adam_steps_are_exact(tmp_path):
    from scripts.train_multistate_chisel import mean_kl_and_backward
    from rewriter.primal_dual import PrimalDual
    engine, load = fixture(tmp_path)
    def run(use_resident):
        actor = torch.nn.Parameter(torch.tensor([.03, -.02]))
        optimizer = torch.optim.Adam([actor], lr=2e-6)
        dual = PrimalDual([1, 2], tolerance=1e-6, rho=.01, dual_lr=.1)
        resident = None
        history = []
        direction = torch.linspace(-1., 1., 17)
        for state_id in (1, 2, 1):
            panel = load(state_id)
            if use_resident:
                if resident is None:resident = ResidentTeacherFeatures(panel, engine)
                else:resident.bind(panel, engine)
                provider = lambda a, b: resident.teacher_batch(engine, a, b)
            else:
                provider = lambda a, b: panel.teacher_batch(engine, a, b)
            optimizer.zero_grad(set_to_none=True)
            output = actor*1.
            leaf = output.detach().requires_grad_(True)
            params = dict(engine.parameters)
            params['proj.bias'] = params['proj.bias'] + leaf[0]*direction + leaf[1]*direction.square()
            measured = mean_kl_and_backward(engine, params, leaf.square().sum(), panel.inputs,
                kl_budget=1e-6, state_weight=1/200, actor_credit=([output], [leaf]),
                teacher_provider=provider, primal_dual=dual, state_id=state_id)
            gradient = actor.grad.clone()
            torch.nn.utils.clip_grad_norm_([actor], 1., error_if_nonfinite=True)
            optimizer.step()
            dual.step([(state_id, measured['mean_kl'])], expected_ids=[state_id])
            history.append((measured, gradient, actor.detach().clone()))
        return history, copy.deepcopy(optimizer.state_dict()), dual.state_dict()
    a, opt_a, dual_a = run(False)
    b, opt_b, dual_b = run(True)
    for left, right in zip(a, b):
        assert left[0] == right[0]
        assert torch.equal(left[1], right[1]) and torch.equal(left[2], right[2])
    for name, value in opt_a['state'][0].items():
        assert torch.equal(value, opt_b['state'][0][name])
    assert dual_a == dual_b


@pytest.mark.skipif(not torch.cuda.is_available(), reason='owner-coordinated CUDA qualification')
def test_captured_reconstruction_rebinds_state_and_panel_with_fixed_storage(tmp_path):
    engine, load = fixture(tmp_path, 'cuda')
    panel = load(1)
    resident = ResidentTeacherFeatures(panel, engine)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):resident.teacher_batch(engine, 0, 2)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = resident.teacher_batch(engine, 0, 2).log_probs
    for state_id, blocks in ((1, (2, 0)), (2, (1, 0)), (1, (0, 1))):
        panel = load(state_id, blocks)
        resident.bind(panel, engine)
        resident.check_binding(engine)  # required outside every graph replay
        graph.replay()
        expected = panel.teacher_batch(engine, 0, 2)
        assert torch.equal(output, expected.log_probs)
        assert resident.sequence_ids[:2] == expected.sequence_ids
