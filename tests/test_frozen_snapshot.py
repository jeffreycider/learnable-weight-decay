import copy
import math
import unittest
from unittest.mock import patch

import torch

from baseline.model import GPT
from baseline.optim import Muon
from rewriter.frozen_snapshot import observe_frozen_checkpoint
from rewriter.sensors import sequence_signs


def assert_tree_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(list(left), list(right))
        for key in left:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    elif hasattr(left, '__dataclass_fields__'):
        for name in left.__dataclass_fields__:
            assert_tree_equal(test, getattr(left, name), getattr(right, name))
    else:
        test.assertEqual(left, right)


class FrozenSnapshotTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(114)
        self.model = GPT(vocab_size=16, num_layers=1, model_dim=128).float()
        self.inputs, self.targets = torch.randint(16, (3, 4)), torch.randint(16, (3, 4))
        self.config = dict(batch_tokens=1200, max_steps=3250, schedule_steps=3250,
                           rewriter=dict(probe_rank=2, anchor_seed=1001, signs_seed=2001,
                                         trajectory_positions=6))
        hidden = [p for p in self.model.blocks.parameters() if p.ndim == 2]
        hidden_ids = {id(p) for p in hidden}
        remaining = [p for p in self.model.parameters() if id(p) not in hidden_ids]
        adam = torch.optim.AdamW(remaining, lr=.012, betas=(.8, .95))
        muon = Muon(hidden, lr=.025, mu=.95)
        self.optimizers = [adam, muon]
        generator = torch.Generator().manual_seed(515)
        self.momenta = {}
        for name, parameter in self.model.named_parameters():
            # Deliberately different original/calibration token counts. Merely
            # comparing normalized coordinate values would miss wrong scaling;
            # tests also check the retained absolute log-scale fields.
            momentum = torch.randn(parameter.shape, generator=generator) * self.config['batch_tokens']
            self.momenta[name] = momentum.clone()
            if id(parameter) in hidden_ids:
                muon.state[parameter] = dict(momentum=momentum)
            else:
                adam.state[parameter] = dict(step=torch.tensor(25.), exp_avg=momentum,
                                             exp_avg_sq=torch.rand(parameter.shape, generator=generator))

    def test_no_optimizer_or_parameter_mutation_and_exact_compute_counts(self):
        parameters = copy.deepcopy(self.model.state_dict())
        states = copy.deepcopy([o.state_dict() for o in self.optimizers])
        counters = dict(forward=0, backward=0)
        def forward_hook(module, args, output):
            counters['forward'] += 1
            def backward_hook(gradient):
                counters['backward'] += 1
            output.register_hook(backward_hook)
        handle = self.model.register_forward_hook(forward_hook)
        rng = torch.random.get_rng_state().clone()
        # Calling either base step would fail before any NCCL dependency.
        with (patch.object(self.optimizers[0], 'step', side_effect=AssertionError('Adam step forbidden')),
              patch.object(self.optimizers[1], 'step', side_effect=AssertionError('Muon step forbidden'))):
            snapshot, measurements, accounting = observe_frozen_checkpoint(
                self.model, self.optimizers, self.config, self.inputs, self.targets,
                step=37, radius_fraction=.01, microbatch_sequences=2)
        handle.remove()
        self.assertEqual(counters, dict(forward=2, backward=2))
        self.assertEqual(accounting['calibration_forwards'], 2)
        self.assertEqual(accounting['calibration_backwards'], 2)
        self.assertEqual(accounting['calibration_tokens'], 12)
        self.assertEqual(accounting['base_steps'], 0)
        self.assertEqual(accounting['momentum_normalization_tokens'], 1200)
        self.assertEqual(accounting['task_gradient_normalization_tokens'], 12)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        assert_tree_equal(self, parameters, self.model.state_dict())
        assert_tree_equal(self, states, [o.state_dict() for o in self.optimizers])
        torch.testing.assert_close(snapshot.radius, .01 * snapshot.weight_norm, atol=0, rtol=0)
        self.assertEqual(snapshot.trajectories.shape, (6, 12))
        for observation in snapshot.matrices.values():
            self.assertEqual(torch.count_nonzero(observation.rows[:, 6:8]).item(), 0)
            self.assertEqual(torch.count_nonzero(observation.columns[:, 6:8]).item(), 0)
        for observation in snapshot.vectors.values():
            self.assertEqual(torch.count_nonzero(observation.coordinates[:, 6]).item(), 0)

    def test_gradient_probes_and_momentum_units_match_independent_autograd(self):
        reference = copy.deepcopy(self.model)
        snapshot, train, accounting = observe_frozen_checkpoint(
            self.model, self.optimizers, self.config, self.inputs, self.targets,
            step=37, microbatch_sequences=2)
        grads = {name: [] for name, _ in reference.named_parameters()}
        losses = []
        for sequence in range(3):
            reference.zero_grad(set_to_none=True)
            loss = reference(self.inputs[sequence:sequence+1], self.targets[sequence:sequence+1]) / 4
            losses.append(loss.detach())
            loss.backward()
            for name, parameter in reference.named_parameters():
                grads[name].append(parameter.grad.clone())
        signs = sequence_signs(torch.arange(3), 4, 2001, step=37)
        torch.testing.assert_close(torch.tensor(accounting['calibration_mean_ce']), torch.stack(losses).mean(), atol=1e-6, rtol=1e-6)
        for name, sequence_grads in grads.items():
            gradient = torch.stack(sequence_grads).mean(0)
            signed = torch.einsum('kb,b...->k...', signs, torch.stack(sequence_grads)) / math.sqrt(3)
            momentum = self.momenta[name] / 1200
            if name in snapshot.matrices:
                if name == 'embed.weight':
                    gradient, momentum, signed = gradient.T, momentum.T, signed.transpose(-1, -2)
                obs = snapshot.matrices[name]
                a, b = obs.anchors.A, obs.anchors.B
                torch.testing.assert_close(train.matrices[name][0], signed @ b, atol=2e-6, rtol=3e-5)
                torch.testing.assert_close(train.matrices[name][1], signed.transpose(-1, -2) @ a, atol=2e-6, rtol=3e-5)
                for features, expected_g, expected_m in [(obs.rows, gradient @ b, momentum @ b),
                                                         (obs.columns, gradient.T @ a, momentum.T @ a)]:
                    for bank_index, expected in [(1, expected_g), (2, expected_m)]:
                        scale = expected.square().mean().sqrt().clamp_min(1e-12)
                        torch.testing.assert_close(features[:, 2*bank_index:2*(bank_index+1)],
                                                   expected / scale, atol=2e-5, rtol=3e-5)
                        torch.testing.assert_close(features[:, 11+bank_index],
                                                   scale.log().expand(features.shape[0]), atol=1e-6, rtol=1e-6)
            else:
                obs = snapshot.vectors[name]
                torch.testing.assert_close(train.vectors[name], signed, atol=2e-6, rtol=3e-5)
                for bank_index, expected in [(1, gradient), (2, momentum)]:
                    scale = expected.square().mean().sqrt().clamp_min(1e-12)
                    torch.testing.assert_close(obs.coordinates[:, 2*bank_index], expected / scale, atol=2e-5, rtol=3e-5)
                    torch.testing.assert_close(obs.coordinates[:, 2*bank_index+1],
                                               scale.log().expand(expected.numel()), atol=1e-6, rtol=1e-6)

    def test_observations_do_not_alias_model_optimizer_or_replay_buffers(self):
        snapshot, train, _ = observe_frozen_checkpoint(
            self.model, self.optimizers, self.config, self.inputs, self.targets, step=37)
        original_snapshot, original_train = copy.deepcopy((snapshot, train))
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(10)
            for optimizer in self.optimizers:
                for state in optimizer.state.values():
                    for value in state.values():
                        if isinstance(value, torch.Tensor):
                            value.zero_()
            self.inputs.zero_()
            self.targets.zero_()
        assert_tree_equal(self, snapshot, original_snapshot)
        assert_tree_equal(self, train, original_train)

    def test_preexisting_gradients_are_rejected_without_consuming_them(self):
        parameter = next(self.model.parameters())
        parameter.grad = torch.ones_like(parameter)
        with self.assertRaisesRegex(ValueError, 'cleared gradients'):
            observe_frozen_checkpoint(self.model, self.optimizers, self.config,
                                      self.inputs, self.targets, step=37)
        self.assertTrue(torch.equal(parameter.grad, torch.ones_like(parameter)))


if __name__ == '__main__':
    unittest.main()
