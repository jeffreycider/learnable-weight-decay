"""Fable-owned independent checks for the replay-KL maintenance design (root 05:49 UTC approval).

Scope: the seed-pure replay sampler (rewriter/online_replay.py) against the ordinary
training stream across a shard boundary and against the trainer's data manifest; CPU
oracles for the distributed reduction on the REAL tiny-GPT path (FunctionalPreservation,
prepare_transformer_action through the live adapter, GroupedNormObjective, PrimalDual),
both the dense-leaf route and the production thin-leaf + scale route
(online_preservation._broadcast_action / factorized_logits with realized parameters).
None of this is GPU, Gloo transport, or a production protocol claim.
"""
import hashlib
import json
import struct

import numpy as np
import pytest
import torch

from baseline.data import FinewebStream
from rewriter.online_replay import OnlineReplaySampler
from scripts.train_multistate_chisel import heldout_sequence

SEQ, PER_STEP = 4, 4          # seq_len 4, batch_tokens 16 -> 4 sequences per global batch
BATCH_TOKENS = SEQ * PER_STEP


def _write_shard(path, tokens):
    tokens = np.asarray(tokens, dtype=np.uint16)
    path.write_bytes(struct.pack('<iii', 20240520, 1, len(tokens)) + bytes(1012) + tokens.tobytes())


def _trainer_manifest(tmp_path, paths, *, batch_tokens=BATCH_TOKENS):
    """Rows in the exact shape baseline.verify_data.data_manifest writes (train rows only here)."""
    rows = []
    for p in sorted(paths):
        tokens = struct.unpack('<iii', p.read_bytes()[:12])[2]
        rows.append(dict(kind='train', path=str(p), tokens=tokens, bytes=p.stat().st_size,
                         consumed_batches=(tokens - 2) // batch_tokens, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    manifest = tmp_path / 'data_manifest.json'
    manifest.write_text(json.dumps(rows, indent=2) + '\n')
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _two_shards(tmp_path):
    # Shard 0 holds exactly three 16-token batches (positions 0, 16, 32; 48+17 > 50
    # forces the one-advance rule), shard 1 holds the rest: the prefix crosses a boundary.
    a = tmp_path / 'fineweb_train_000000.bin'
    b = tmp_path / 'fineweb_train_000001.bin'
    _write_shard(a, np.arange(50) + 1000)
    _write_shard(b, np.arange(120) + 5000)
    return [a, b]


def _sampler(files, manifest, sha, **kw):
    settings = dict(batch_tokens=BATCH_TOKENS, seq_len=SEQ, seed=11, split_seed=3, manifest_sha256=sha, manifest_path=manifest)
    settings.update(kw)
    return OnlineReplaySampler(files, **settings)


def test_sampler_spans_equal_training_stream_rows_across_shard_boundary(tmp_path):
    files = _two_shards(tmp_path)
    manifest, sha = _trainer_manifest(tmp_path, files)
    steps = 7  # 3 batches in shard 0, 4 in shard 1
    stream = FinewebStream(str(tmp_path / 'fineweb_train_*.bin'), BATCH_TOKENS, SEQ, rank=0, world_size=1, device='cpu')
    batches = [next(stream)[0] for _ in range(steps)]
    sampler = _sampler(files, manifest, sha)
    eligible = [i for i in range(steps * PER_STEP) if not heldout_sequence(i, 3)]
    drawn = sampler.sample(steps, len(eligible))
    assert sorted(drawn['ids']) == eligible  # every training-role id of the completed prefix, none twice
    for row, span in zip(drawn['provenance'], drawn['inputs']):
        assert row['id'] == row['step'] * PER_STEP + row['sequence'] and row['step'] < steps
        torch.testing.assert_close(span.to(torch.int32), batches[row['step']][row['sequence']], rtol=0, atol=0)
    assert any(r['shard_index'] == 1 for r in drawn['provenance']) and any(r['shard_index'] == 0 for r in drawn['provenance'])
    assert drawn['input_sha256'] == hashlib.sha256(drawn['inputs'].numpy().tobytes()).hexdigest()
    assert drawn['manifest_sha256'] == sha


def test_sampler_is_seed_pure_bounded_to_completed_prefix_and_fails_closed(tmp_path):
    files = _two_shards(tmp_path)
    manifest, sha = _trainer_manifest(tmp_path, files)
    make = lambda: _sampler(files, manifest, sha)
    one, two = make(), make()
    first = one.sample(5, 6)
    assert one.sample(5, 6)['ids'] == first['ids'] == two.sample(5, 6)['ids']  # no hidden sampler state
    assert first['id_sha256'] == two.sample(5, 6)['id_sha256']
    assert first['ids'] != make().sample(6, 6)['ids']  # the clock is part of the key
    assert max(first['ids']) < 5 * PER_STEP  # the just-completed batch (step 5) is excluded
    assert all(not heldout_sequence(i, 3) for i in first['ids'])
    small = make().sample(1, 2)
    assert all(i < PER_STEP for i in small['ids'])  # step 0 only
    with pytest.raises(ValueError, match='insufficient'):
        make().sample(1, 5)
    with pytest.raises(ValueError, match='manifest SHA256'):
        _sampler(files, manifest, '0' * 64)
    bad = tmp_path / 'fineweb_val_000000.bin'
    _write_shard(bad, np.arange(40))
    with pytest.raises(ValueError, match='prohibited'):
        _sampler(sorted(files + [bad]), manifest, sha)


def test_sampler_binds_full_shard_content_not_just_headers(tmp_path):
    files = _two_shards(tmp_path)
    manifest, sha = _trainer_manifest(tmp_path, files)
    _sampler(files, manifest, sha)  # pristine data accepted
    # Flip one token in shard 1 with the header (magic, version, count) and the size unchanged.
    raw = bytearray(files[1].read_bytes())
    offset = 1024 + 2 * 7
    raw[offset:offset + 2] = struct.pack('<H', struct.unpack('<H', bytes(raw[offset:offset + 2]))[0] ^ 1)
    files[1].write_bytes(bytes(raw))
    assert struct.unpack('<iii', files[1].read_bytes()[:12]) == (20240520, 1, 120)
    with pytest.raises(ValueError, match='content SHA256'):
        _sampler(files, manifest, sha)
    # A manifest listing a different shard order/set than the given files is rejected too.
    other_manifest, other_sha = _trainer_manifest(tmp_path / 'other', [files[0]]) if (tmp_path / 'other').mkdir() is None else (None, None)
    with pytest.raises(ValueError, match='shard order'):
        _sampler(files, other_manifest, other_sha)


def _live_action_and_engine():
    """A real prepared action from the live adapter path, plus a preservation engine on its base."""
    from baseline.model import GPT
    from rewriter.function_preservation import FunctionalPreservation
    from tests.test_transformer_online import _adapter_fixture, _ordinary_step
    model, opt, adapter, cfg = _adapter_fixture('learned')
    _ordinary_step(model, opt, adapter, 0)
    action = adapter.lifecycle.action
    assert action is not None
    base = action.parameters
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    gpt = GPT(vocab_size=32, num_layers=1, model_dim=128).float().requires_grad_(False)
    engine = FunctionalPreservation(gpt, parameter_state=base, buffer_state=buffers,
                                    microbatch_sequences=2, kl_dtype=torch.float64, gpt_precision='fp32')
    return model, adapter, action, engine


def _rank_teachers(engine, panel, rank_sizes):
    locals_, start = [], 0
    for size in rank_sizes:
        part = panel[start:start + size]; start += size
        teachers = []
        for mb in range(0, size, engine.microbatch_sequences):
            teachers.extend(engine.cache_teacher(part[mb:mb + engine.microbatch_sequences], data_role='past_training'))
        locals_.append(teachers)
    return locals_


def _actor_grads(params):
    return [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]


def _clear(params):
    for p in params:
        p.grad = None


def test_summed_rank_cotangents_with_global_coefficient_match_centralized_actor_gradient():
    from rewriter.grouped_norm import GroupedNormObjective
    from rewriter.primal_dual import PrimalDual
    _, adapter, action, engine = _live_action_and_engine()
    actor_params = [p for _, p in adapter.lifecycle.actor.named_parameters()]
    torch.manual_seed(5)
    panel = torch.randint(32, (6, 4))
    rank_sizes = (1, 2, 3)              # unequal panels; rank 2 needs two microbatches
    n_global = panel.numel()
    replacements = action.materialize()
    editable = [n for n in action.editable if replacements[n].requires_grad]
    assert editable, 'the lifecycle must keep the final action graph alive for credit'
    locals_ = _rank_teachers(engine, panel, rank_sizes)

    def kl_terms(params):
        return [(engine.student_parameter_loss(params, t), t.inputs.numel() / n_global) for ts in locals_ for t in ts]

    # Tolerance between the rank-local means so rank-local penalties would straddle the hinge.
    with torch.no_grad():
        rank_means = []
        for ts in locals_:
            toks = sum(t.inputs.numel() for t in ts)
            rank_means.append(sum(float(engine.student_parameter_loss(replacements, t)) * t.inputs.numel() / toks for t in ts))
    tau = (min(rank_means) + max(rank_means)) / 2
    dual = PrimalDual([0], tolerance=tau, rho=.3, dual_lr=.1)
    dual.multipliers[0] = .2

    # Centralized reference: one global objective, direct autodiff to the actor.
    grouped = GroupedNormObjective(action.parameters, zero_reference_policy='freeze')
    kl_global = sum(kl * w for kl, w in kl_terms(replacements))
    violation = kl_global.clamp_min(0) / tau - 1
    loss = 100 * grouped.evaluate(replacements).mean_relative_change + .2 * violation + .15 * violation.clamp_min(0) ** 2
    reference = torch.autograd.grad(loss, actor_params, retain_graph=True, allow_unused=True)

    # Distributed route: per rank, dense detached leaves, local raw-KL backward weighted by
    # local_tokens / N_global; SUM across ranks; global coefficient from the SUMMED mean; one
    # actor VJP with the norm cotangent once.
    summed = {n: torch.zeros_like(replacements[n]) for n in editable}
    per_rank_coefficient = {n: torch.zeros_like(replacements[n]) for n in editable}
    mean = 0.
    for ts in locals_:
        leaves = {n: (v.detach().requires_grad_(n in editable)) for n, v in replacements.items()}
        local_mean = 0.
        for t in ts:
            kl = engine.student_parameter_loss(leaves, t)
            w = t.inputs.numel() / n_global
            (kl * w).backward()
            local_mean += float(kl.detach()) * w
        mean += local_mean
        local_coefficient = dual.terms(local_mean * n_global / sum(t.inputs.numel() for t in ts), 0)['kl_coefficient']
        for n in editable:
            g = leaves[n].grad if leaves[n].grad is not None else torch.zeros_like(leaves[n])
            summed[n] += g
            per_rank_coefficient[n] += local_coefficient * g
    assert mean == pytest.approx(float(kl_global.detach()), rel=1e-9)
    terms = dual.terms(mean, 0)
    assert terms['kl_penalty'] == pytest.approx(float((.2 * violation + .15 * violation.clamp_min(0) ** 2).detach()), rel=1e-9)
    coefficient = terms['kl_coefficient']

    _clear(actor_params)
    norm_once = 100 * GroupedNormObjective(action.parameters, zero_reference_policy='freeze').evaluate(replacements).mean_relative_change
    torch.autograd.backward([norm_once] + [replacements[n] for n in editable],
                            [torch.ones_like(norm_once)] + [coefficient * summed[n] for n in editable], retain_graph=True)
    worst = 0.
    for p, ref in zip(actor_params, reference):
        got = p.grad if p.grad is not None else torch.zeros_like(p)
        ref = torch.zeros_like(p) if ref is None else ref
        if ref.norm() > 0:
            worst = max(worst, float((got - ref).norm() / ref.norm()))
        torch.testing.assert_close(got, ref, rtol=2e-5, atol=1e-12)
    print('dense route: max relative actor-gradient error vs centralized:', worst)

    # Wrong reductions are measurably different: a per-rank coefficient (hinge straddled), and
    # the norm cotangent added once per rank instead of once.
    _clear(actor_params)
    torch.autograd.backward([replacements[n] for n in editable], [per_rank_coefficient[n] for n in editable], retain_graph=True)
    wrong = _actor_grads(actor_params)
    _clear(actor_params)
    torch.autograd.backward([replacements[n] for n in editable], [coefficient * summed[n] for n in editable], retain_graph=True)
    right_kl_only = _actor_grads(actor_params)
    assert any(not torch.allclose(a, b, rtol=1e-4, atol=1e-12) for a, b in zip(wrong, right_kl_only))
    _clear(actor_params)
    torch.autograd.backward([norm_once], [torch.full_like(norm_once, float(len(rank_sizes)))], retain_graph=True)
    norm_times_ranks = _actor_grads(actor_params)
    _clear(actor_params)
    torch.autograd.backward([norm_once], [torch.ones_like(norm_once)], retain_graph=True)
    norm_once_grad = _actor_grads(actor_params)
    assert any(not torch.allclose(a, b, rtol=1e-4, atol=1e-12) for a, b in zip(norm_times_ranks, norm_once_grad))
    adapter.close()


def _thin_route_actor_gradient(model, action, engine, locals_, n_global, actor_params, *, drop_scale=False):
    """Production wire: detached UNSCALED factors + separate scale leaf; thin backward through
    factorized_logits with realized parameters; SUM of weighted leaf cotangents; one root VJP."""
    from rewriter.function_preservation import chunked_teacher_kl
    from rewriter.factorized_fp32 import factorized_logits
    from rewriter.online_preservation import _broadcast_action, _stored_parameters
    wire_action, scale_leaf, leaves, keys = _broadcast_action(action, torch.device('cpu'))
    editable = set(n for n, r in keys if r != 'scale')
    buffers = dict(model.named_buffers())
    with torch.no_grad():
        stored = _stored_parameters(action.parameters, wire_action, scale_leaf, editable, action.embedding_names)
    total = 0.
    for ts in locals_:
        for t in ts:
            logits = factorized_logits(model, action.parameters, buffers, t.inputs, wire_action, scale_leaf,
                                       realized_parameters=stored, linear_backend='ieee')
            kl = chunked_teacher_kl(logits, t.log_probs, engine.chunk_tokens)
            w = t.inputs.numel() / n_global
            (kl * w).backward()
            total += float(kl.detach()) * w
    outputs, cotangents = [], []
    for (name, role), leaf in zip(keys, leaves):
        value = action.scale if role == 'scale' else action.action.vectors[name] if role == 'v' else getattr(action.action.matrices[name], role)
        grad = torch.zeros_like(leaf) if leaf.grad is None else leaf.grad
        if role == 'scale' and drop_scale:
            grad = torch.zeros_like(grad)
        if value.requires_grad:
            outputs.append(value); cotangents.append(grad)
    _clear(actor_params)
    torch.autograd.backward(outputs, cotangents, retain_graph=True)
    scale_grad = next(leaf.grad for (n, r), leaf in zip(keys, leaves) if r == 'scale')
    return total, _actor_grads(actor_params), float(scale_grad) if scale_grad is not None else None, float(scale_leaf)


def test_thin_leaf_and_scale_route_matches_dense_centralized_gradient_including_saturated_cap():
    from rewriter.transformer_action import prepare_transformer_action
    model, adapter, action, engine = _live_action_and_engine()
    actor = adapter.lifecycle.actor
    actor_params = [p for _, p in actor.named_parameters()]
    torch.manual_seed(7)
    panel = torch.randint(32, (6, 4))
    n_global = panel.numel()
    locals_ = _rank_teachers(engine, panel, (1, 2, 3))

    def dense_reference(prepared):
        replacements = prepared.materialize()
        kl = sum(engine.student_parameter_loss(replacements, t) * (t.inputs.numel() / n_global) for ts in locals_ for t in ts)
        grads = torch.autograd.grad(kl, actor_params, retain_graph=True, allow_unused=True)
        return float(kl.detach()), [torch.zeros_like(p) if g is None else g for p, g in zip(actor_params, grads)]

    def saturated_case():
        # Force the hard radius to bind: a much larger raw proposal from the same actor. The live
        # action's version guard forbids touching the actor earlier, so this runs second.
        with torch.no_grad():
            for p in actor_params:
                p.mul_(200.)
        return prepare_transformer_action(actor, action.snapshot, action.parameters, radius_fraction=1e-4)

    seen_active_cap = False
    for label, make in (('live action', lambda: action), ('saturated cap', saturated_case)):
        prepared = make()
        ref_kl, ref = dense_reference(prepared)
        kl, got, dscale, scale_value = _thin_route_actor_gradient(model, prepared, engine, locals_, n_global, actor_params)
        assert kl == pytest.approx(ref_kl, rel=1e-6)
        worst = max(float((g - r).norm() / r.norm()) for g, r in zip(got, ref) if r.norm() > 0)
        print(label, 'scale', scale_value, 'dscale', dscale, 'max relative actor-gradient error thin vs dense:', worst)
        assert worst < 1e-4
        if scale_value < 1:
            seen_active_cap = True
            assert dscale is not None and dscale != 0.
            # Dropping the scale cotangent (as TransformerAction.contraction-style detachment would)
            # changes the actor gradient when the cap binds.
            _, dropped, _, _ = _thin_route_actor_gradient(model, prepared, engine, locals_, n_global, actor_params, drop_scale=True)
            assert any(not torch.allclose(a, b, rtol=1e-6, atol=1e-14) for a, b in zip(dropped, got))
    assert seen_active_cap, 'the saturated case must exercise a binding hard radius'
    adapter.close()


def test_base_motion_budget_keeps_thin_and_dense_gradients_equal_and_is_exact_when_slack():
    """Root's optional action budget (rewriter/action_budget.py) applied to a real prepared action:
    the budgeted scale must stay connected through the production thin wire (scale leaf cotangent)
    and match dense centralized autodiff; a slack budget must leave the scale object's value exact."""
    from dataclasses import replace
    from rewriter.action_budget import base_motion_radius, restrict_global_scale
    model, adapter, action, engine = _live_action_and_engine()
    actor_params = [p for _, p in adapter.lifecycle.actor.named_parameters()]
    torch.manual_seed(9)
    panel = torch.randint(32, (6, 4))
    n_global = panel.numel()
    locals_ = _rank_teachers(engine, panel, (1, 2, 3))
    raw_norm = float(action.action.norm_sq().detach().sqrt())
    s0 = float(action.scale.detach())
    committed = s0 * raw_norm

    def dense_reference(prepared):
        replacements = prepared.materialize()
        kl = sum(engine.student_parameter_loss(replacements, t) * (t.inputs.numel() / n_global) for ts in locals_ for t in ts)
        grads = torch.autograd.grad(kl, actor_params, retain_graph=True, allow_unused=True)
        return float(kl.detach()), [torch.zeros_like(p) if g is None else g for p, g in zip(actor_params, grads)]

    # Slack budget (twice the committed edit): scale value unchanged bit-for-bit, graph still connected.
    slack = replace(action, scale=restrict_global_scale(action.action.norm_sq(), action.scale,
                                                       base_motion_radius(torch.tensor(2 * committed / .25), fraction=.25)))
    assert float(slack.scale.detach()) == s0 and slack.scale.requires_grad
    # Binding budget (half the committed edit): scale halves, edit norm obeys the new radius.
    radius = base_motion_radius(torch.tensor(.5 * committed / .25), fraction=.25)
    tight = replace(action, scale=restrict_global_scale(action.action.norm_sq(), action.scale, radius))
    assert float(tight.scale.detach()) == pytest.approx(.5 * s0, rel=1e-5)
    edit = sum((v - action.parameters[n]).double().square().sum() for n, v in tight.materialize().items() if n in action.editable).sqrt()
    assert float(edit) <= float(radius) * (1 + 1e-6)
    for label, prepared in (('slack', slack), ('binding', tight)):
        ref_kl, ref = dense_reference(prepared)
        kl, got, dscale, scale_value = _thin_route_actor_gradient(model, prepared, engine, locals_, n_global, actor_params)
        assert kl == pytest.approx(ref_kl, rel=1e-6)
        worst = max(float((g - r).norm() / r.norm()) for g, r in zip(got, ref) if r.norm() > 0)
        print(label, 'budget scale', scale_value, 'dscale', dscale, 'max relative actor-gradient error thin vs dense:', worst)
        assert worst < 1e-4 and dscale is not None and dscale != 0.
    # Dropping the scale cotangent under a binding budget changes the gradient (budget graph matters).
    _, dropped, _, _ = _thin_route_actor_gradient(model, tight, engine, locals_, n_global, actor_params, drop_scale=True)
    _, kept, _, _ = _thin_route_actor_gradient(model, tight, engine, locals_, n_global, actor_params)
    assert any(not torch.allclose(a, b, rtol=1e-6, atol=1e-14) for a, b in zip(dropped, kept))
    adapter.close()
