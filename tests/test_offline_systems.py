import copy
import json

import pytest
import torch

from baseline.model import GPT
from rewriter.function_preservation import FunctionalPreservation
from rewriter.offline_systems import select_panel_blocks, CachedBaseNormPreservation
from rewriter.teacher_cache import TeacherCacheReader, TeacherPanel, write_panel
from scripts.train_multistate_chisel import mean_kl_and_backward


def test_lazy_selection_does_not_read_or_copy_features():
    class ForbiddenFeatures:
        def index_select(self, *args):
            raise AssertionError('feature selection must remain lazy')
    panel = TeacherPanel(torch.arange(24).reshape(6, 4), ForbiddenFeatures(),
                         tuple(range(6)), {'microbatch_sequences': 2})
    selected = select_panel_blocks(panel, [2, 0])
    assert selected.parent is panel
    assert selected.sequence_ids == (4, 5, 0, 1)
    assert torch.equal(selected.inputs, panel.inputs[[4, 5, 0, 1]])
    with pytest.raises(ValueError, match='unique'):
        select_panel_blocks(panel, [1, 1])
    with pytest.raises(ValueError, match='unique'):
        select_panel_blocks(panel, [3])
    with pytest.raises(ValueError, match='aligned'):
        selected.teacher_batch(None, 1, 3)


def test_full_update_batch_keeps_eight_rank_state_ownership_and_cursor():
    from scripts.benchmark_chisel_systems import production_batch
    training = [s for s in range(1, 1001) if s % 5]
    saved = dict(metadata=dict(world_size=8, options=dict(fixed_state_shards=True, data_seed=400001)),
                 next_epoch=25, next_batch=2)
    states, epoch = production_batch(saved, training, 8)
    assert epoch == 25 and len(states) == len(set(states)) == 200
    for rank in range(8):
        local = states[rank::8]
        assert len(local) == 25 and set(local) <= set(training[rank::8])
    assert production_batch(saved, training, 8)[0] == states
    previous = dict(saved, next_batch=1)
    assert not set(production_batch(previous, training, 8)[0]) & set(states)
    with pytest.raises(ValueError, match='eight ranks'):
        production_batch(saved, training, 1)


def test_lazy_fp32_targets_actor_credit_and_adam_are_exact(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    model = GPT(vocab_size=32, num_layers=1, model_dim=128).float().requires_grad_(False)
    engine = FunctionalPreservation(model, kl_dtype=torch.float64,
                                   microbatch_sequences=2, gpt_precision='fp32')
    inputs = torch.randint(32, (8, 4))
    write_panel(tmp_path, 1, 'train', engine, inputs, list(range(8)),
                state_sha256='a'*64, execution={'mode': 'eager'})
    panel = TeacherCacheReader(tmp_path, require_complete=False, mmap=True).read(1, 'train')
    eager = panel.select_blocks([3, 0, 2])
    lazy = select_panel_blocks(panel, [3, 0, 2])
    assert lazy.metadata == eager.metadata
    assert lazy.sequence_ids == eager.sequence_ids
    assert torch.equal(lazy.features, eager.features)
    nested = lazy.select_blocks([2, 0])
    eager_nested = eager.select_blocks([2, 0])
    assert torch.equal(nested.features, eager_nested.features)
    assert nested.metadata == eager_nested.metadata
    for start in range(0, 6, 2):
        a, b = eager.teacher_batch(engine, start, start+2), lazy.teacher_batch(engine, start, start+2)
        assert torch.equal(a.log_probs, b.log_probs)
        assert torch.equal(a.inputs, b.inputs)
        assert a.sequence_ids == b.sequence_ids

    def optimize(selected):
        # Three nonzero updates: test accumulated credit and Adam moment state,
        # not merely zero-head loss or identical input values.
        actor = torch.nn.Parameter(torch.tensor([.03, -.02]))
        optimizer = torch.optim.Adam([actor], lr=.001, betas=(.9, .99))
        history = []
        direction = torch.linspace(-1., 1., 32)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            output = actor*1.
            factor = output.detach().requires_grad_(True)
            params = dict(engine.parameters)
            params['proj.bias'] = params['proj.bias'] + factor[0]*direction + factor[1]*direction.square()
            measured = mean_kl_and_backward(
                engine, params, factor.square().sum(), selected.inputs,
                kl_budget=1e-6, state_weight=.125,
                actor_credit=([output], [factor]),
                teacher_provider=lambda a, b: selected.teacher_batch(engine, a, b))
            grad = actor.grad.clone()
            torch.nn.utils.clip_grad_norm_([actor], 1., error_if_nonfinite=True)
            optimizer.step()
            history.append((measured, grad, actor.detach().clone()))
        return history, copy.deepcopy(optimizer.state_dict())

    left, left_opt = optimize(eager)
    right, right_opt = optimize(lazy)
    for (lm, lg, lp), (rm, rg, rp) in zip(left, right):
        assert lm == rm
        assert torch.equal(lg, rg)
        assert torch.equal(lp, rp)
    for key in left_opt['state'][0]:
        assert torch.equal(left_opt['state'][0][key], right_opt['state'][0][key])
    # Lazy dispatch must not bypass the original cache invalidation contract.
    engine.load_teacher_state(engine.parameters, engine.buffers)
    with pytest.raises(ValueError, match='bound'):
        lazy.teacher_batch(engine, 0, 2)


def test_verified_base_norm_cache_preserves_reload_and_mutation_guards(tmp_path, monkeypatch):
    from rewriter.state_bank import StateBankReader
    from tests.test_multistate_chisel import _tiny_state_bank
    torch.set_num_threads(1)
    bank = tmp_path/'bank'
    config = _tiny_state_bank(bank)
    records = StateBankReader(bank, verify_once=True, mmap_model=True)
    engine = CachedBaseNormPreservation(GPT(**config['model']).float().requires_grad_(False),
                                       kl_dtype=torch.float64, gpt_precision='fp32')
    panels = tmp_path/'panels'
    rows = []
    for state_id in (1, 2):
        record = records.read(state_id)
        engine.load_teacher_state({n: record.model[n] for n in engine.parameters},
                                  {n: record.model[n] for n in engine.buffers})
        for role in ('train', 'audit'):
            rows.append(write_panel(panels, state_id, role, engine, torch.zeros(1, 4, dtype=torch.long),
                                    [state_id], state_sha256=record.assert_verified(), execution={'mode': 'eager'}))
    (panels/'manifest.json').write_text(json.dumps(dict(complete=True, args={'states': 2}, panels=rows)))
    cache = TeacherCacheReader(panels, verify_once=True, mmap=True)
    calls = []
    refresh = FunctionalPreservation._refresh_base_norm
    def counted(self):
        calls.append(self.teacher_id)
        return refresh(self)
    monkeypatch.setattr(FunctionalPreservation, '_refresh_base_norm', counted)
    record = records.read(1)
    engine.load_verified_teacher(cache, record)
    expected = engine.base_weight_norm_sq.clone()
    old_id = engine.teacher_id
    teacher = cache.read(1, 'train').teacher_batch(engine, 0, 1)
    assert len(calls) == 1
    engine.base_weight_norm_sq.zero_()  # The public output must not alias cache storage.
    engine.load_verified_teacher(cache, record)
    assert len(calls) == 1 and engine.teacher_id != old_id
    assert torch.equal(engine.base_weight_norm_sq, expected)
    assert engine.compute.student_forwards == 0
    with pytest.raises(ValueError, match='different frozen checkpoint'):
        engine.student_parameter_loss(engine.parameters, teacher)
    assert torch.equal(cache.read(1, 'train').teacher_batch(engine, 0, 1).log_probs, teacher.log_probs)
    engine.load_verified_teacher(cache, records.read(2))
    assert len(calls) == 2
    engine.load_verified_teacher(cache, records.read(1))
    assert len(calls) == 2 and torch.equal(engine.base_weight_norm_sq, expected)
    # Direct, unverified loads must always compute, never use a previous SHA key.
    engine.load_teacher_state(engine.parameters, engine.buffers)
    assert len(calls) == 3
    record.model['proj.bias'].add_(1.)
    with pytest.raises(ValueError, match='mutated'):
        engine.load_verified_teacher(cache, record)
    assert engine._loading_verified_sha is None
    bad = records.read(1)
    original = cache.expected[(1, 'train')]['state_sha256']
    cache.expected[(1, 'train')]['state_sha256'] = 'f'*64
    with pytest.raises(ValueError, match='differs'):
        engine.load_verified_teacher(cache, bad)
    assert engine._loading_verified_sha is None
    cache.expected[(1, 'train')]['state_sha256'] = original


def test_all_state_constants_and_real_chisel_optimizer_are_exact(tmp_path, monkeypatch):
    from rewriter.offline_systems import VerifiedStateConstants, build_proposal_with_radius
    from rewriter.grouped_norm import GroupedNormObjective
    from rewriter.edit_scope import editable_names, editable_weight_norm_sq
    from rewriter.state_bank import StateBankReader
    from rewriter.transformer_chisel import TransformerChisel
    from scripts.train_multistate_chisel import build_proposal
    from tests.test_multistate_chisel import _tiny_state_bank
    torch.set_num_threads(1)
    bank = tmp_path/'bank'
    config = _tiny_state_bank(bank)
    reader = StateBankReader(bank, verify_once=True, mmap_model=True)
    engine = CachedBaseNormPreservation(GPT(**config['model']).float().requires_grad_(False),
                                       kl_dtype=torch.float64, gpt_precision='fp32')
    panels = tmp_path/'panels'
    panel_rows = []
    for state_id in (1, 2):
        record = reader.read(state_id)
        engine.load_teacher_state({n: record.model[n] for n in engine.parameters},
                                  {n: record.model[n] for n in engine.buffers})
        for role in ('train', 'audit'):
            panel_rows.append(write_panel(panels, state_id, role, engine, torch.arange(4).reshape(1, 4),
                                          [state_id], state_sha256=record.assert_verified(), execution={'mode': 'eager'}))
    (panels/'manifest.json').write_text(json.dumps(dict(complete=True, args={'states': 2}, panels=panel_rows)))
    cache = TeacherCacheReader(panels, verify_once=True, mmap=True)
    constants = VerifiedStateConstants(engine, GroupedNormObjective)
    for state_id in (1, 2, 1, 2):
        record = reader.read(state_id)
        constants.load(cache, record)
        group = constants.grouped()
        reference = GroupedNormObjective(engine.parameters, zero_reference_policy='freeze')
        for name in reference.groups:
            assert torch.equal(group.reference_squared[name], reference.reference_squared[name])
            assert torch.equal(group.reference_norms[name], reference.reference_norms[name])
        assert group.zero_reference_groups == reference.zero_reference_groups
        assert group.frozen_parameter_names == reference.frozen_parameter_names
        radius = constants.radius_squared(group)
        expected = editable_weight_norm_sq(engine.parameters, editable_names(engine.parameters, 'grouped')-group.frozen_parameter_names)
        assert torch.equal(radius, expected)
        radius.zero_()
        assert torch.equal(constants.radius_squared(group), expected)
    assert constants.hits == dict(groups=2, radius=6)

    actor = TransformerChisel(rank=2, trajectory_dim=12, model_dim=128, num_layers=1, patch_size=8)
    # Nonzero heads exercise the complete factor credit path, not identity output.
    with torch.no_grad():
        for head in actor.output_heads:
            head.weight.normal_(std=.0001)
    initial = copy.deepcopy(actor.state_dict())
    def run(cached):
        actor.load_state_dict(initial)
        optimizer = torch.optim.Adam(actor.parameters(), lr=2e-6, betas=(.9, .99))
        result = []
        for state_id in (1, 2, 1):
            record = reader.read(state_id)
            if cached:
                constants.load(cache, record)
                grouped = constants.grouped()
            else:
                cache.load_engine(engine, record)
                grouped = GroupedNormObjective(engine.parameters, zero_reference_policy='freeze')
            optimizer.zero_grad(set_to_none=True)
            args = dict(radius_fraction=.01)
            function = build_proposal
            if cached:
                args['norm_squared'] = constants.radius_squared(grouped)
                function = build_proposal_with_radius
            params, regularizer, stats, editable, credit = function(actor, record.snapshot, engine, grouped, **args)
            panel = cache.read(state_id, 'train')
            measured = mean_kl_and_backward(engine, params, regularizer, panel.inputs, kl_budget=1e-6,
                                            state_weight=.125, actor_credit=credit,
                                            teacher_provider=lambda a, b: panel.teacher_batch(engine, a, b))
            gradients = [p.grad.clone() if p.grad is not None else None for p in actor.parameters()]
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            result.append((stats, measured, gradients, copy.deepcopy(actor.state_dict())))
        return result, copy.deepcopy(optimizer.state_dict())
    baseline, baseline_opt = run(False)
    candidate, candidate_opt = run(True)
    for a, b in zip(baseline, candidate):
        assert a[:2] == b[:2]
        for x, y in zip(a[2], b[2]):
            assert (x is None and y is None) or torch.equal(x, y)
        for name in a[3]:
            assert torch.equal(a[3][name], b[3][name])
    for parameter, state in baseline_opt['state'].items():
        for name, value in state.items():
            assert torch.equal(value, candidate_opt['state'][parameter][name])
    # Cached setup must reject mutated weights, a direct engine reload, and a
    # grouped object from a previous load. Reference evaluation keeps its guard.
    constants.load(cache, reader.read(1))
    stale_group = constants.grouped()
    constants.load(cache, reader.read(2))
    with pytest.raises(RuntimeError, match='previous state'):
        constants.radius_squared(stale_group)
    engine.parameters['proj.bias'].add_(1.)
    with pytest.raises(RuntimeError, match='unchanged'):
        constants.grouped()
    constants.load(cache, reader.read(1))
    engine.load_teacher_state(engine.parameters, engine.buffers)
    with pytest.raises(RuntimeError, match='unchanged'):
        constants.radius_squared(constants._template)
