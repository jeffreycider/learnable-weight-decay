import dataclasses

import pytest
import torch

from rewriter.diagnostic_action import DiagnosticAnchoredAction
from scripts.train_multistate_chisel import detached_action_credit
from tests.test_controller import make_snapshot


def test_zero_control_is_trainable_in_the_original_anchored_family():
    snapshot = make_snapshot()
    control = DiagnosticAnchoredAction(snapshot)
    action = control(snapshot)
    assert action.norm_sq() == 0
    action.contraction(snapshot.weights, snapshot.vector_weights).backward()
    assert all(p.grad is not None and p.grad.norm() > 0 for p in control.parameters())
    with torch.no_grad():
        for p in control.parameters():
            p.add_(p.grad, alpha=-.01)
    action = control(snapshot)
    for f in action.matrices.values():
        torch.testing.assert_close(f.anchors.A.T @ f.P, torch.zeros(2, 2), atol=1e-6, rtol=0)
        torch.testing.assert_close(f.norm_sq(), f.dense().square().sum())
        assert torch.linalg.matrix_rank(f.dense(), atol=1e-6) <= 4


def test_credit_bridge_matches_direct_free_factor_gradient():
    snapshot = make_snapshot()
    direct = DiagnosticAnchoredAction(snapshot)
    bridged = DiagnosticAnchoredAction(snapshot)
    direct(snapshot).contraction(snapshot.weights, snapshot.vector_weights).backward()
    leaves, (original, detached) = detached_action_credit(bridged(snapshot))
    leaves.contraction(snapshot.weights, snapshot.vector_weights).backward()
    torch.autograd.backward(original, [p.grad for p in detached])
    for a, b in zip(direct.parameters(), bridged.parameters()):
        torch.testing.assert_close(a.grad, b.grad)


def test_control_rejects_anchor_transfer_and_schema_changes():
    snapshot = make_snapshot()
    control = DiagnosticAnchoredAction(snapshot)
    with pytest.raises(ValueError, match="another snapshot"):
        control(make_snapshot())
    with pytest.raises(ValueError, match="schema"):
        control(dataclasses.replace(snapshot, vectors={}))


def test_saved_factors_resume_on_the_verified_fixed_snapshot():
    snapshot = make_snapshot()
    first = DiagnosticAnchoredAction(snapshot)
    with torch.no_grad():
        for p in first.parameters():
            p.normal_(std=.01)
    second = DiagnosticAnchoredAction(snapshot)
    second.load_state_dict(first.state_dict())
    a, b = first(snapshot), second(snapshot)
    for name in a.matrices:
        torch.testing.assert_close(a.matrices[name].dense(), b.matrices[name].dense())
    assert not any(p.data_ptr() == a.data_ptr() for p in first.parameters()
                   for anchor in first.anchors for a in (anchor.A, anchor.B))


def test_free_factor_control_uses_actual_fp32_gpt_gradient_and_readonly_scope():
    from baseline.model import GPT
    from rewriter.algebra import Anchors
    from rewriter.controller import ControllerSnapshot, MatrixObservation, VectorObservation
    from rewriter.function_preservation import FunctionalPreservation
    from rewriter.grouped_norm import GroupedNormObjective
    from rewriter.primal_dual import PrimalDual
    from scripts.train_multistate_chisel import build_proposal, mean_kl_and_backward

    torch.set_num_threads(1)
    torch.manual_seed(312)
    model = GPT(16, 1, 128).float().requires_grad_(False)
    engine = FunctionalPreservation(model, microbatch_sequences=1,
                                    gpt_precision='fp32', kl_dtype=torch.float64)
    generator = torch.Generator().manual_seed(40)
    matrices, vectors = {}, {}
    for name, value in engine.parameters.items():
        if value.ndim == 2:
            weight = value.T if name == 'embed.weight' else value
            anchors = Anchors.random(*weight.shape, 2, generator=generator)
            matrices[name] = MatrixObservation(anchors, torch.zeros(weight.shape[0], 25),
                torch.zeros(weight.shape[1], 25), (weight @ anchors.B, weight.T @ anchors.A))
        else:
            vectors[name] = VectorObservation(torch.zeros(value.numel(), 18), value)
    snapshot = ControllerSnapshot(matrices, vectors, torch.zeros(1, 12),
                                  torch.tensor(1.), torch.tensor(.01))
    control = DiagnosticAnchoredAction(snapshot)
    groups = GroupedNormObjective(engine.parameters, zero_reference_policy='freeze')
    dual = PrimalDual([1], tolerance=1e-6, rho=.01, dual_lr=.1)
    optimizer = torch.optim.Adam(control.parameters(), lr=1e-6, betas=(.9, .99))
    inputs = torch.randint(0, 16, (2, 4), generator=generator)
    teacher = engine.cache_teacher(inputs, data_role='past_training')
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        parameters, loss, stats, editable, credit = build_proposal(
            control, snapshot, engine, groups, radius_fraction=.01)
        measured = mean_kl_and_backward(engine, parameters, loss, inputs, kl_budget=1e-6,
            primal_dual=dual, state_id=1, actor_credit=credit,
            teacher_provider=lambda start, stop: teacher[start])
        assert torch.isfinite(torch.tensor(measured['mean_kl']))
        if step == 0:
            assert measured['mean_kl'] == 0 and stats['realized_edit_norm'] == 0
        else:
            assert stats['mean_group_l2_percentage_reduction'] > 0
            assert stats['realized_edit_norm'] > 0
        for name in set(engine.parameters) - editable:
            assert torch.equal(parameters[name], engine.parameters[name])
        assert all(torch.isfinite(p.grad).all() for p in control.parameters() if p.grad is not None)
        optimizer.step()
        dual.step([(1, measured['mean_kl'])], expected_ids=[1])
    assert engine.compute.student_backwards == 4


def test_bounded_runner_and_interruption_resume_match(tmp_path, monkeypatch):
    import json
    from baseline.model import GPT
    from rewriter.function_preservation import FunctionalPreservation
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import write_panel
    import scripts.diagnose_fixed_action_fit as runner
    from tests.test_multistate_chisel import _tiny_state_bank

    torch.set_num_threads(1)
    bank, cache = tmp_path/'bank', tmp_path/'cache'
    cfg = _tiny_state_bank(bank)
    reader = StateBankReader(bank)
    engine = FunctionalPreservation(GPT(**cfg['model']).requires_grad_(False),
        gpt_precision='fp32', microbatch_sequences=16, kl_dtype=torch.float64)
    generator = torch.Generator().manual_seed(61)
    rows = []
    for state in range(1, 6):
        record = reader.read(state)
        engine.load_teacher_state({n: record.model[n] for n in engine.parameters},
                                  {n: record.model[n] for n in engine.buffers})
        for role, count, offset in [('train', 128, 0), ('audit', 64, 128)]:
            rows.append(write_panel(cache, state, role, engine,
                torch.randint(0, 16, (count, 4), generator=generator),
                list(range(state*1000+offset, state*1000+offset+count)),
                state_sha256=record.assert_verified(), execution={'mode': 'unit-test-fp32'}))
    (cache/'manifest.json').write_text(json.dumps(dict(complete=True, args={'states': 5}, panels=rows)))
    common = ['--state-bank', str(bank), '--teacher-cache', str(cache), '--state-id', '1',
              '--updates', '2', '--audit-every', '1', '--device', 'cpu', '--lr', '1e-6']
    direct, resumed = tmp_path/'direct', tmp_path/'resumed'
    runner.main([*common, '--output', str(direct)])
    original = runner.mean_kl_and_backward
    calls = 0

    def interrupt_on_second_update(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('simulated worker interruption')
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, 'mean_kl_and_backward', interrupt_on_second_update)
    with pytest.raises(RuntimeError, match='simulated'):
        runner.main([*common, '--output', str(resumed)])
    monkeypatch.setattr(runner, 'mean_kl_and_backward', original)
    runner.main([*common, '--output', str(resumed), '--resume', str(resumed/'action_latest.pt')])
    left = torch.load(direct/'action_latest.pt', weights_only=False)
    right = torch.load(resumed/'action_latest.pt', weights_only=False)
    assert left['update'] == right['update'] == 2
    assert left['primal_dual'] == right['primal_dual']
    assert left['compute'] == right['compute']
    for name in left['action']:
        torch.testing.assert_close(left['action'][name], right['action'][name], rtol=0, atol=0)
    for index, state in left['optimizer']['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(value, right['optimizer']['state'][index][key], rtol=0, atol=0)
    finished = json.loads((resumed/'metrics.jsonl').read_text().splitlines()[-1])
    assert finished['event'] == 'finished' and not finished['deployable']
    assert finished['compute']['teacher_forwards'] == 0
    assert finished['selected_withheld']['role'] == 'selected_withheld_sequences'
    with pytest.raises(SystemExit):
        runner.main([*common, '--state-id', '5', '--output', str(tmp_path/'forbidden')])
