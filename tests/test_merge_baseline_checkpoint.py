import copy
from pathlib import Path
import tempfile
import unittest

import torch

from baseline.checkpoint import restore_checkpoint
from baseline.model import GPT
from baseline.train import make_optimizers
from scripts.merge_baseline_checkpoint import FORMAT, merge_checkpoints


class MergeBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        torch.manual_seed(104)
        self.model = GPT(vocab_size=8, num_layers=1, model_dim=128)
        optimizers = make_optimizers(self.model)
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for i, parameter in enumerate(group['params']):
                    if optimizer is optimizers[0]:
                        optimizer.state[parameter] = dict(step=torch.tensor(25.),
                            exp_avg=torch.full_like(parameter, .125), exp_avg_sq=torch.full_like(parameter, .25))
                    else:
                        optimizer.state[parameter] = dict(momentum=torch.full_like(parameter, i + .5))
        self.full_states = [o.state_dict() for o in optimizers]
        config = dict(model=dict(vocab_size=8, num_layers=1, model_dim=128), mode='baseline',
                      world_size=8, max_steps=3250, seed=0, batch_tokens=64,
                      seq_len=4, microbatch_sequences=2)
        self.paths = []
        for rank in range(8):
            states = copy.deepcopy(self.full_states)
            states[1]['state'] = {i: value for i, value in states[1]['state'].items() if i % 8 == rank}
            cursor = dict(rank=rank, world_size=8, batch_size=64, seq_len=4,
                          file_names=['fineweb_train_000001.bin'], file_index=0, pos=1600, batches=25)
            state = dict(format_version=1, model=self.model.state_dict(), optimizers=states,
                         controller=None, config=config, data_cursor=cursor, step=25,
                         counters=dict(forward=25, backward=25), elapsed_seconds=rank + .25,
                         rng=dict(cpu=torch.tensor([rank], dtype=torch.uint8)))
            path = self.root / f'rank{rank}.pt'
            torch.save(state, path)
            self.paths.append(path)
        self.output = self.root / 'merged.pt'

    def mutate(self, rank, function):
        state = torch.load(self.paths[rank], weights_only=False)
        function(state)
        torch.save(state, self.paths[rank])

    def test_eight_rank_merge_loads_complete_single_device_observation_state(self):
        rng = torch.random.get_rng_state().clone()
        metadata = merge_checkpoints(self.paths[::-1], self.output)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        merged = torch.load(self.output, weights_only=False)
        self.assertEqual(merged['format_version'], FORMAT)
        self.assertEqual(metadata['source_world_size'], 8)
        self.assertFalse(metadata['resumable'])
        self.assertEqual([x['rank'] for x in metadata['sources']], list(range(8)))
        self.assertTrue(all(len(x['sha256']) == 64 for x in metadata['sources']))
        self.assertEqual(merged['config']['world_size'], 8)
        self.assertEqual(merged['data_cursor']['rank'], 0)
        self.assertEqual(merged['elapsed_seconds'], .25)
        self.assertEqual(merged['rng']['cpu'].item(), 0)
        self.assertIsNone(merged['controller'])
        model = GPT(**merged['config']['model'])
        model.load_state_dict(merged['model'])
        optimizers = make_optimizers(model)
        for optimizer, saved in zip(optimizers, merged['optimizers']):
            optimizer.load_state_dict(saved)
        for index, parameter in enumerate(optimizers[1].param_groups[0]['params']):
            self.assertTrue(torch.equal(optimizers[1].state[parameter]['momentum'],
                                        torch.full_like(parameter, index + .5)))
        # The artifact is intentionally unusable by the baseline resume API.
        with self.assertRaisesRegex(ValueError, 'checkpoint format'):
            restore_checkpoint(merged, model, optimizers, None, None, merged['config'])
        # No input file is rewritten, including ranks with zero owned matrices.
        for rank, path in enumerate(self.paths):
            source = torch.load(path, weights_only=False)
            self.assertEqual(set(source['optimizers'][1]['state']), {i for i in range(6) if i % 8 == rank})

    def test_rejects_missing_rank_and_duplicate_rank_even_when_no_owned_state(self):
        with self.assertRaisesRegex(ValueError, 'missing source ranks'):
            merge_checkpoints(self.paths[:-1], self.output)
        duplicate = self.root / 'copy.pt'
        duplicate.write_bytes(self.paths[0].read_bytes())
        with self.assertRaisesRegex(ValueError, 'duplicate source rank'):
            merge_checkpoints(self.paths[:-1] + [duplicate], self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_missing_or_foreign_momentum(self):
        original = self.paths[0].read_bytes()
        self.mutate(0, lambda s: s['optimizers'][1]['state'].clear())
        with self.assertRaisesRegex(ValueError, 'owner state'):
            merge_checkpoints(self.paths, self.output)
        self.paths[0].write_bytes(original)
        self.mutate(7, lambda s: s['optimizers'][1]['state'].update(
            {0: dict(momentum=torch.zeros_like(self.full_states[1]['state'][0]['momentum']))}))
        with self.assertRaisesRegex(ValueError, 'owner state'):
            merge_checkpoints(self.paths, self.output)

    def test_rejects_replica_config_clock_cursor_or_model_conflicts(self):
        original = self.paths[3].read_bytes()
        changes = [
            lambda s: s['model']['proj.bias'].add_(1),
            lambda s: s['config'].update(seed=42),
            lambda s: s['data_cursor'].update(pos=1664),
            lambda s: s['data_cursor'].update(batches=24),
            lambda s: s['counters'].update(forward=26),
            lambda s: s['optimizers'][0]['state'][0]['exp_avg'].add_(1),
            lambda s: s['optimizers'][1]['param_groups'][0].update(mu=.99),
        ]
        for change in changes:
            with self.subTest(change=change):
                self.paths[3].write_bytes(original)
                self.mutate(3, change)
                with self.assertRaises(ValueError):
                    merge_checkpoints(self.paths, self.output)
                self.assertFalse(self.output.exists())

    def test_rejects_controller_transfer_and_wrong_momentum_shape(self):
        original = self.paths[0].read_bytes()
        self.mutate(0, lambda s: s.update(controller={}))
        with self.assertRaisesRegex(ValueError, 'controller'):
            merge_checkpoints(self.paths, self.output)
        self.paths[0].write_bytes(original)
        self.mutate(0, lambda s: s['optimizers'][1]['state'][0].update(momentum=torch.ones(1)))
        with self.assertRaisesRegex(ValueError, 'shape/dtype/value'):
            merge_checkpoints(self.paths, self.output)

    def test_rejects_wrong_parameter_order_and_never_overwrites(self):
        self.mutate(0, lambda s: s['optimizers'][1]['param_groups'][0]['params'].reverse())
        with self.assertRaisesRegex(ValueError, 'IDs/order'):
            merge_checkpoints(self.paths, self.output)
        self.output.write_text('keep me')
        with self.assertRaises(FileExistsError):
            merge_checkpoints(self.paths, self.output)
        self.assertEqual(self.output.read_text(), 'keep me')


if __name__ == '__main__':
    unittest.main()
