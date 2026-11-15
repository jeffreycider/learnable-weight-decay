"""Separate tiny-model correctness runs; these extra passes are never benchmark work."""
import copy
import io
import unittest

import torch

from baseline.model import GPT, initialize_parameters_like_record
from rewriter.integration import RewriterAdapter


def small_config():
    return dict(model=dict(num_layers=1), batch_tokens=12, seq_len=3,
                max_steps=3, schedule_steps=3,
                rewriter=dict(probe_rank=2, coordinate_width=8, global_width=16,
                              global_latents=4, global_blocks=1, attention_heads=2,
                              ff_width=32, pool_queries=2, auxiliary_steps=1,
                              warmup_steps=0, ramp_steps=1, anchor_refresh_period=1,
                              trajectory_positions=6))


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=.002, betas=(.9, .95))


def batches(seed=15):
    gen = torch.Generator().manual_seed(seed)
    return [(torch.randint(32, (4, 3), generator=gen),
             torch.randint(32, (4, 3), generator=gen)) for _ in range(3)]


def train_step(t, model, opt, batch, adapter=None):
    x, y = batch
    if adapter:
        adapter.before_step(t, model, [opt])
    losses = []
    for start in (0, 2):
        if adapter:
            adapter.before_microbatch(t, torch.arange(start, start+2), x[start:start+2])
        loss = model(x[start:start+2], y[start:start+2])
        loss.backward()
        losses.append(loss.detach())
    if adapter:
        adapter.after_backward(t, model)
    opt.step()
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    if adapter:
        adapter.after_step(t, model, [opt])
    opt.zero_grad(set_to_none=True)
    return torch.stack(losses).sum(), base


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.model = GPT(vocab_size=32, num_layers=1, model_dim=128).float()
        initialize_parameters_like_record(self.model)

    def test_hidden_scope_preserves_post_base_embedding_head_and_scales_radius(self):
        config=small_config()
        config['rewriter'].update(editable_scope='hidden',checkpoint_train_measurements=True)
        opt=optimizer(self.model)
        adapter=RewriterAdapter(self.model,[opt],config)
        before={n:p.detach().clone() for n,p in self.model.named_parameters()}
        _,base=train_step(0,self.model,opt,batches()[0],adapter)
        readonly={'embed.weight','proj.weight','proj.bias'}
        for name in readonly:
            self.assertTrue(torch.equal(adapter.params[name],base[name]))
        self.assertTrue(any(not torch.equal(adapter.params[n],base[n]) for n in adapter.editable))
        snapshot=adapter.online.pending.snapshot
        self.assertIn('embed.weight',snapshot.matrices)
        self.assertIn('proj.weight',snapshot.matrices)
        self.assertIn('proj.bias',snapshot.vectors)
        norm=sum(v.float().square().sum() for n,v in base.items() if n not in readonly).sqrt()
        motion=sum((v.float()-before[n].float()).square().sum() for n,v in base.items() if n not in readonly).sqrt()
        torch.testing.assert_close(snapshot.weight_norm,norm)
        torch.testing.assert_close(snapshot.radius,.25*motion)
        self.assertTrue(adapter.metrics['read_only_parameters_unchanged'])
        saved=copy.deepcopy(adapter.state_dict())
        self.assertEqual(saved['editable_scope'],'hidden')
        adapter.load_state_dict(saved)
        pending=adapter.online.pending
        self.assertEqual(float(pending.action.matrices['embed.weight'].norm_sq()),0.)
        self.assertEqual(float(pending.action.matrices['proj.weight'].norm_sq()),0.)
        self.assertEqual(float(pending.action.vectors['proj.bias'].abs().sum()),0.)
        # Restored pending graph consumes ordinary next-pass credit normally.
        train_step(1,self.model,opt,batches()[1],adapter)
        self.assertEqual(adapter.online.updates,3)
        legacy=dict(saved);legacy.pop('editable_scope')
        with self.assertRaisesRegex(ValueError,'editable scope'):
            adapter.load_state_dict(legacy)
        adapter.collector.close()

    def test_explicit_all_matches_legacy_missing_scope_exactly(self):
        legacy,explicit=copy.deepcopy(self.model),copy.deepcopy(self.model)
        legacy_opt,explicit_opt=optimizer(legacy),optimizer(explicit)
        config=small_config();all_config=copy.deepcopy(config)
        all_config['rewriter']['editable_scope']='all'
        left=RewriterAdapter(legacy,[legacy_opt],config)
        right=RewriterAdapter(explicit,[explicit_opt],all_config)
        for t,batch in enumerate(batches()):
            a,_=train_step(t,legacy,legacy_opt,batch,left)
            b,_=train_step(t,explicit,explicit_opt,batch,right)
            self.assertTrue(torch.equal(a,b))
            for p,q in zip(legacy.parameters(),explicit.parameters()):self.assertTrue(torch.equal(p,q))
        left.collector.close();right.collector.close()

    def test_shadow_preserves_parameters_loss_and_rng_exactly(self):
        baseline, shadow = copy.deepcopy(self.model), copy.deepcopy(self.model)
        ob, os = optimizer(baseline), optimizer(shadow)
        rng_before = torch.random.get_rng_state().clone()
        adapter = RewriterAdapter(shadow, [os], small_config(), mode="shadow")
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        for t, batch in enumerate(batches()):
            lb, _ = train_step(t, baseline, ob, batch)
            ls, _ = train_step(t, shadow, os, batch, adapter)
            self.assertTrue(torch.equal(lb, ls))
            for a, b in zip(baseline.parameters(), shadow.parameters()):
                self.assertTrue(torch.equal(a, b))
            self.assertIsNone(adapter.online.pending)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        self.assertEqual(adapter.online.updates, 2)  # aux only; terminal consumes none
        adapter.collector.close()

    def test_delayed_credit_old_anchors_counts_and_terminal_identity(self):
        model = self.model
        opt = optimizer(model)
        adapter = RewriterAdapter(model, [opt], small_config())
        counts = dict(forward=0, backward=0, credits=0)
        def count_forward(module, args, out):
            counts['forward'] += 1
            def count_backward(gradient):
                counts['backward'] += 1
            out.register_hook(count_backward)
        handle = model.register_forward_hook(count_forward)
        original = adapter.online.consume_credit
        def check_credit(matrix, vector):
            pending = adapter.online.pending
            dense = sum((adapter.oriented(n, adapter.params[n].grad).float() * f.dense()).sum()
                        for n, f in pending.action.matrices.items())
            dense += sum((adapter.params[n].grad.float() * v).sum()
                         for n, v in pending.action.vectors.items())
            measured = pending.action.contraction(matrix, vector)
            torch.testing.assert_close(measured, dense / 12, atol=1e-7, rtol=1e-5)
            self.assertTrue(any(not torch.equal(f.anchors.A, adapter.anchors[n].A)
                                for n, f in pending.action.matrices.items()))
            counts['credits'] += 1
            original(matrix, vector)
        adapter.online.consume_credit = check_credit
        for t, batch in enumerate(batches()):
            _, base = train_step(t, model, opt, batch, adapter)
            if t < 2:
                self.assertGreater(adapter.metrics['realized_edit_norm'], 0)
                self.assertLess(adapter.metrics['allones_relative_error'], 1e-5)
                self.assertIsNotNone(adapter.online.pending)
            else:
                self.assertTrue(adapter.metrics['terminal_identity'])
                self.assertIsNone(adapter.online.pending)
                for n, p in model.named_parameters():
                    self.assertTrue(torch.equal(p, base[n]))
        self.assertEqual(counts, dict(forward=6, backward=6, credits=2))
        self.assertEqual(adapter.online.updates, 4)  # 2 aux + 2 delayed task
        handle.remove()
        adapter.collector.close()

    def test_checkpoint_replays_pending_controller_graph(self):
        model, config = self.model, small_config()
        config['rewriter']['checkpoint_train_measurements'] = True
        opt = optimizer(model)
        adapter = RewriterAdapter(model, [opt], config)
        data = batches()
        train_step(0, model, opt, data[0], adapter)
        buffer = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), adapter=adapter.state_dict()), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=False)
        self.assertIs(saved['adapter']['diagnostic_training']['snapshot'],
                      saved['adapter']['pending']['snapshot'])
        resumed = copy.deepcopy(model)
        resumed.load_state_dict(saved['model'])
        resumed_opt = optimizer(resumed)
        resumed_opt.load_state_dict(saved['optimizer'])
        resumed_adapter = RewriterAdapter(resumed, [resumed_opt], config)
        # Simulate a different reconstruction on the destination host. Resume
        # must restore actual stored values, not trust seed reconstruction.
        resumed_adapter.collector.omega.mul_(-1)
        resumed_adapter.collector.positions = resumed_adapter.collector.positions.roll(1)
        resumed_adapter.load_state_dict(saved['adapter'])
        restored = resumed_adapter.state_dict()
        self.assertIs(restored['diagnostic_training']['snapshot'], restored['pending']['snapshot'])
        torch.testing.assert_close(resumed_adapter.collector.omega, adapter.collector.omega, atol=0, rtol=0)
        self.assertTrue(torch.equal(resumed_adapter.collector.positions, adapter.collector.positions))
        for t in (1, 2):
            a, _ = train_step(t, model, opt, data[t], adapter)
            b, _ = train_step(t, resumed, resumed_opt, data[t], resumed_adapter)
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            for p, q in zip(model.parameters(), resumed.parameters()):
                torch.testing.assert_close(p, q, atol=0, rtol=0)
            for p, q in zip(adapter.controller.parameters(), resumed_adapter.controller.parameters()):
                torch.testing.assert_close(p, q, atol=0, rtol=0)
        adapter.collector.close()
        resumed_adapter.collector.close()

    def test_optional_train_measurement_export_is_detached_and_behavior_neutral(self):
        plain, exported = copy.deepcopy(self.model), copy.deepcopy(self.model)
        plain_opt, exported_opt = optimizer(plain), optimizer(exported)
        config = small_config()
        adapter = RewriterAdapter(plain, [plain_opt], config)
        export_config = copy.deepcopy(config)
        export_config['rewriter']['checkpoint_train_measurements'] = True
        export_adapter = RewriterAdapter(exported, [exported_opt], export_config)
        batch = batches()[0]
        a, _ = train_step(0, plain, plain_opt, batch, adapter)
        b, _ = train_step(0, exported, exported_opt, batch, export_adapter)
        self.assertTrue(torch.equal(a, b))
        for p, q in zip(plain.parameters(), exported.parameters()):
            self.assertTrue(torch.equal(p, q))
        for p, q in zip(adapter.controller.parameters(), export_adapter.controller.parameters()):
            self.assertTrue(torch.equal(p, q))
        self.assertIsNone(adapter.state_dict()['diagnostic_training'])
        state = export_adapter.state_dict()
        diagnostic = state['diagnostic_training']
        self.assertEqual(diagnostic['step'], 0)
        self.assertEqual(diagnostic['loss_units'], 'sequence mean CE in nats/token')
        self.assertTrue(diagnostic['snapshot_shared_with_pending'])
        self.assertIs(diagnostic['snapshot'], state['pending']['snapshot'])
        measurements = diagnostic['measurements']
        expected_elements = 4 * (sum((p.shape[0] + p.shape[1]) * 2
                                    for p in plain.parameters() if p.ndim == 2)
                                 + sum(p.numel() for p in plain.parameters() if p.ndim == 1))
        self.assertEqual(diagnostic['measurements_tensor_bytes'], expected_elements * 4)
        self.assertEqual(diagnostic['additional_checkpoint_tensor_bytes'], expected_elements * 4)
        for name, pair in measurements.matrices.items():
            for i, tensor in enumerate(pair):
                self.assertEqual(tensor.device.type, 'cpu')
                self.assertFalse(tensor.requires_grad)
                self.assertEqual(tensor.shape[0], 4)
                source = export_adapter._diagnostic_training['measurements'].matrices[name][i]
                self.assertNotEqual(tensor.data_ptr(), source.data_ptr())
                self.assertNotEqual(tensor.data_ptr(), state['pending']['audit'].matrices[name][i].data_ptr())
                original = source.clone()
                tensor.zero_()
                self.assertTrue(torch.equal(source, original))
        for tensor in measurements.vectors.values():
            self.assertEqual(tensor.shape[0], 4)
            self.assertFalse(tensor.requires_grad)
        adapter.collector.close()
        export_adapter.collector.close()

    def test_warmup_export_accounts_for_unshared_snapshot_and_retains_decision_step(self):
        opt = optimizer(self.model)
        config = small_config()
        config['rewriter'].update(checkpoint_train_measurements=True, warmup_steps=64)
        adapter = RewriterAdapter(self.model, [opt], config)
        for t, batch in enumerate(batches()):
            train_step(t, self.model, opt, batch, adapter)
        state = adapter.state_dict()
        self.assertIsNone(state['pending'])
        record = state['diagnostic_training']
        self.assertEqual(record['step'], 1)  # terminal step 2 has no new decision
        self.assertFalse(record['snapshot_shared_with_pending'])
        self.assertGreater(record['snapshot_tensor_bytes'], 0)
        self.assertEqual(record['additional_checkpoint_tensor_bytes'],
                         record['snapshot_tensor_bytes'] + record['measurements_tensor_bytes'])
        adapter.collector.close()

    def test_legacy_sensor_resume_warns_and_invalid_positions_are_rejected(self):
        opt = optimizer(self.model)
        adapter = RewriterAdapter(self.model, [opt], small_config())
        state = adapter.state_dict()
        legacy = dict(state)
        del legacy['trajectory_sensors']
        with self.assertWarnsRegex(RuntimeWarning, 'legacy checkpoint'):
            adapter.load_state_dict(legacy)
        corrupt = copy.deepcopy(state)
        corrupt['trajectory_sensors']['positions'][1] = corrupt['trajectory_sensors']['positions'][0]
        with self.assertRaisesRegex(ValueError, 'globally sampled token positions'):
            adapter.load_state_dict(corrupt)
        adapter.collector.close()

    def test_bfloat16_projection_rounding_is_measured(self):
        # The real model's BF16 forward casts round parameter gradients. FP32
        # sensor contractions represent the underlying derivative before each
        # cast's rounding, so exact equality is inappropriate for this path.
        model = self.model
        model.embed.bfloat16()
        opt = optimizer(model)
        adapter = RewriterAdapter(model, [opt], small_config(), mode="shadow")
        train_step(0, model, opt, batches()[0], adapter)
        self.assertLess(adapter.metrics['allones_relative_error'], .03)
        self.assertGreaterEqual(adapter.metrics['allones_relative_error'], 0)
        adapter.collector.close()


if __name__ == "__main__":
    unittest.main()
