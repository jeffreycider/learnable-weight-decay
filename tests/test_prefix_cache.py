import copy
from pathlib import Path
import tempfile
import unittest

import torch

from rewriter.algebra import Anchors, Measurements
from rewriter.controller import (ControllerSnapshot, MatrixObservation, VectorObservation,
                                 GlobalRewriter, coordinate_features, vector_features)
from rewriter.online import OnlineRewriter
from rewriter.prefix_cache import (pack_record, unpack_record, PrefixCacheWriter,
                                   PrefixCacheReader, replay_identity_prefix)


def toy_record(seed=1):
    generator = torch.Generator().manual_seed(seed)
    matrices, vectors, train_m, train_v = {}, {}, {}, {}
    for name, n, m in [('first', 7, 5), ('last', 9, 6)]:
        anchor = Anchors.random(n, m, 2, generator=generator)
        row_banks = [torch.randn(n, 2, generator=generator) for _ in range(4)]
        col_banks = [torch.randn(m, 2, generator=generator) for _ in range(4)]
        rows = coordinate_features(row_banks, anchor.A, torch.rand(n, generator=generator), list(range(10)))
        cols = coordinate_features(col_banks, anchor.B, torch.rand(m, generator=generator), list(range(10)))
        matrices[name] = MatrixObservation(anchor, rows, cols, (row_banks[0].clone(), col_banks[0].clone()))
        train_m[name] = (torch.randn(4, n, 2, generator=generator) * .001,
                         torch.randn(4, m, 2, generator=generator) * .001)
    states = [torch.randn(7, generator=generator) for _ in range(4)]
    vectors['bias'] = VectorObservation(vector_features(states, list(range(10))), states[0])
    train_v['bias'] = torch.randn(4, 7, generator=generator) * .001
    return (ControllerSnapshot(matrices, vectors, torch.randn(3, 12, generator=generator),
                               torch.tensor(10.), torch.tensor(.1)), Measurements(train_m, train_v))


def assert_same_bits(test, left, right):
    test.assertEqual(left.dtype, right.dtype)
    test.assertEqual(left.shape, right.shape)
    test.assertEqual(left.stride(), right.stride())
    test.assertTrue(torch.equal(left.contiguous().reshape(-1).view(torch.uint8), right.contiguous().reshape(-1).view(torch.uint8)))


def assert_record_equal(test, snapshot, train, restored):
    other = restored.snapshot
    test.assertEqual(list(snapshot.matrices), list(other.matrices))
    test.assertEqual(list(snapshot.vectors), list(other.vectors))
    for name, matrix in snapshot.matrices.items():
        result = other.matrices[name]
        for a, b in [(matrix.rows, result.rows), (matrix.columns, result.columns),
                     (matrix.anchors.A, result.anchors.A), (matrix.anchors.B, result.anchors.B)]:
            assert_same_bits(test, a, b)
        for a, b in zip(matrix.weight_probes, result.weight_probes):
            assert_same_bits(test, a, b)
        for a, b in zip(train.matrices[name], restored.train.matrices[name]):
            assert_same_bits(test, a, b)
    for name, vector in snapshot.vectors.items():
        assert_same_bits(test, vector.coordinates, other.vectors[name].coordinates)
        assert_same_bits(test, vector.weight, other.vectors[name].weight)
        assert_same_bits(test, train.vectors[name], restored.train.vectors[name])
    for field in ('trajectories', 'weight_norm', 'radius'):
        assert_same_bits(test, getattr(snapshot, field), getattr(other, field))


class PrefixCacheTests(unittest.TestCase):
    def test_compaction_preserves_every_actor_and_measurement_bit(self):
        snapshot, train = toy_record()
        packed = pack_record(0, snapshot, train)
        self.assertLess(packed['packed_tensor_bytes'], packed['expanded_tensor_bytes'])
        self.assertEqual(packed['matrices']['first']['rows']['encoding'], 'matrix_v1')
        self.assertEqual(packed['vectors']['bias']['coordinates']['encoding'], 'vector_v1')
        assert_record_equal(self, snapshot, train, unpack_record(packed))
        self.assertNotIn('audit', packed)
        self.assertNotIn('controller', packed)
        self.assertNotIn('optimizer', packed)

    def test_signed_zeros_and_nan_payloads_force_lossless_fallback(self):
        snapshot, train = toy_record()
        rows = snapshot.matrices['first'].rows
        rows[:, -1] = 0.
        rows[1, -1] = -0.
        rows[0, 0] = torch.tensor(0x7FC00123, dtype=torch.int32).view(torch.float32)
        coordinates = snapshot.vectors['bias'].coordinates
        coordinates[:, -1] = 0.
        coordinates[1, -1] = -0.
        packed = pack_record(0, snapshot, train)
        self.assertEqual(packed['matrices']['first']['rows']['encoding'], 'dense')
        self.assertEqual(packed['vectors']['bias']['coordinates']['encoding'], 'dense')
        assert_record_equal(self, snapshot, train, unpack_record(packed))

    def test_qr_and_column_major_feature_layouts_are_preserved(self):
        from dataclasses import replace
        snapshot, train = toy_record()
        first = snapshot.matrices['first']
        self.assertEqual(first.anchors.A.stride(), (1, first.anchors.A.shape[0]))
        column_major_rows = first.rows.T.contiguous().T
        column_major_columns = first.columns.T.contiguous().T
        snapshot.matrices['first'] = replace(first, rows=column_major_rows, columns=column_major_columns)
        vector = snapshot.vectors['bias']
        snapshot.vectors['bias'] = replace(vector, coordinates=vector.coordinates.T.contiguous().T)
        # Also preserve a non-dense vector view and a broadcast measurement.
        storage = torch.zeros(vector.weight.numel() * 2)
        storage[::2] = vector.weight
        snapshot.vectors['bias'] = replace(snapshot.vectors['bias'], weight=storage[::2])
        train.vectors['bias'] = train.vectors['bias'][:1].expand(4, -1)
        packed = pack_record(0, snapshot, train)
        self.assertEqual(packed['matrices']['first']['rows']['encoding'], 'matrix_v1')
        assert_record_equal(self, snapshot, train, unpack_record(packed))

    def test_async_freeze_does_not_alias_reused_buffers(self):
        snapshot, train = toy_record()
        original_snapshot, original_train = copy.deepcopy((snapshot, train))
        with tempfile.TemporaryDirectory() as directory:
            with PrefixCacheWriter(directory, {'gpt_seed': 0}, expected_steps=1) as writer:
                writer.append(0, snapshot, train, identity_action=True)
                snapshot.matrices['first'].rows.add_(100)
                snapshot.matrices['first'].anchors.A.zero_()
                train.matrices['first'][0].fill_(900)
            reader = PrefixCacheReader(directory, expected_metadata={'gpt_seed': 0})
            assert_record_equal(self, original_snapshot, original_train, reader.read(0))
            self.assertGreater(writer.entries[0]['file_bytes'], 0)
            self.assertGreaterEqual(writer.entries[0]['freeze_seconds'], 0)
            self.assertGreaterEqual(writer.entries[0]['write_seconds'], 0)

    def test_background_write_failure_is_raised_on_close(self):
        from unittest.mock import patch
        snapshot, train = toy_record()
        with tempfile.TemporaryDirectory() as directory:
            writer = PrefixCacheWriter(directory, {}, expected_steps=1)
            with patch.object(writer, '_write', side_effect=OSError('simulated full disk')):
                writer.append(0, snapshot, train, identity_action=True)
            with self.assertRaisesRegex(OSError, 'full disk'):
                writer.close()
            self.assertFalse((Path(directory) / 'complete.json').exists())

    def test_incomplete_resume_ordering_and_corruption(self):
        snapshot, train = toy_record()
        metadata = {'gpt_seed': 3, 'data_sha256': 'test-fixture'}
        with tempfile.TemporaryDirectory() as directory:
            with PrefixCacheWriter(directory, metadata, expected_steps=2) as writer:
                with self.assertRaises(ValueError):
                    writer.append(1, snapshot, train, identity_action=True)
                with self.assertRaises(ValueError):
                    writer.append(0, snapshot, train, identity_action=False)
                writer.append(0, snapshot, train, identity_action=True)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                PrefixCacheReader(directory)
            self.assertEqual(len(PrefixCacheReader(directory, require_complete=False)), 1)
            with self.assertRaises(ValueError):
                PrefixCacheWriter(directory, {'gpt_seed': 5}, expected_steps=2, resume=True)
            with PrefixCacheWriter(directory, metadata, expected_steps=2, resume=True) as writer:
                self.assertEqual(writer.next_step, 1)
                writer.append(1, snapshot, train, identity_action=True)
            reader = PrefixCacheReader(directory)
            for prefetch in (0, 1, 2):
                self.assertEqual([r.step for r in reader.iter_records(prefetch=prefetch)], [0, 1])
            path = Path(directory) / 'step_0001.lwdpc'
            blob = bytearray(path.read_bytes())
            blob[-1] ^= 1
            path.write_bytes(blob)
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                reader.read(1)

    def test_fresh_controller_replay_matches_live_auxiliary_updates_exactly(self):
        torch.manual_seed(414)
        initial = GlobalRewriter(rank=2, trajectory_dim=12, coordinate_width=8,
                                 global_width=16, pool_queries=2, latents=4, blocks=1, heads=2, ff_width=32)
        config = dict(auxiliary_steps=2, warmup_steps=3, ramp_steps=1)
        direct = OnlineRewriter(copy.deepcopy(initial), config)
        replayed = OnlineRewriter(copy.deepcopy(initial), config)
        with tempfile.TemporaryDirectory() as directory:
            with PrefixCacheWriter(directory, {'gpt_seed': 1}, expected_steps=3) as writer:
                for step in range(3):
                    snapshot, train = toy_record(seed=step+1)
                    writer.append(step, snapshot, train, identity_action=True)
                    self.assertIsNone(direct.propose(snapshot, train, None, step, 5, shadow=True))
            result = replay_identity_prefix(replayed, PrefixCacheReader(directory), max_steps=5)
            self.assertEqual(result['extra_gpt_forwards'], 0)
            self.assertEqual(result['extra_gpt_backwards'], 0)
            self.assertEqual(result['controller_updates'], 6)
            self.assertIsNone(replayed.pending)
            for a, b in zip(direct.controller.parameters(), replayed.controller.parameters()):
                assert_same_bits(self, a, b)
            for a, b in zip(direct.optimizer.state.values(), replayed.optimizer.state.values()):
                for field in a:
                    assert_same_bits(self, a[field], b[field])
            with self.assertRaisesRegex(ValueError, 'fresh controller'):
                replay_identity_prefix(replayed, PrefixCacheReader(directory), max_steps=5)

    def test_replay_refuses_nonidentity_schedule(self):
        snapshot, train = toy_record()
        initial = GlobalRewriter(rank=2, trajectory_dim=12, coordinate_width=8,
                                 global_width=16, pool_queries=2, latents=4, blocks=1, heads=2, ff_width=32)
        online = OnlineRewriter(initial, dict(warmup_steps=0))
        with tempfile.TemporaryDirectory() as directory:
            with PrefixCacheWriter(directory, {}, expected_steps=1) as writer:
                writer.append(0, snapshot, train, identity_action=True)
            with self.assertRaisesRegex(ValueError, 'identity warmup'):
                replay_identity_prefix(online, PrefixCacheReader(directory), max_steps=2)

    def test_replay_rejects_cross_scope_cached_norm_and_radius(self):
        snapshot,train=toy_record()
        initial=GlobalRewriter(rank=2,trajectory_dim=12,coordinate_width=8,
            global_width=16,pool_queries=2,latents=4,blocks=1,heads=2,ff_width=32)
        online=OnlineRewriter(initial,dict(warmup_steps=1,editable_scope='hidden'))
        with tempfile.TemporaryDirectory() as directory:
            with PrefixCacheWriter(directory,{},expected_steps=1) as writer:
                writer.append(0,snapshot,train,identity_action=True)
            with self.assertRaisesRegex(ValueError,'editable scope'):
                replay_identity_prefix(online,PrefixCacheReader(directory),max_steps=2)
            self.assertEqual(online.updates,0)


if __name__ == '__main__':
    unittest.main()
