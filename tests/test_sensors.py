import copy
import math
import os
import tempfile
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from rewriter.sensors import SensorCollector, sequence_signs


def ddp_sensor_worker(rank, init_path):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method="file://" + init_path, rank=rank, world_size=2)
    try:
        torch.manual_seed(19)
        model = TinyCausal()
        x, y = torch.randint(11, (4, 5)), torch.randint(11, (4, 5))
        anchors = anchors_for(model)
        collector = SensorCollector(model, anchors, 4, 5)
        ids = torch.arange(rank * 2, (rank + 1) * 2)
        collector.set_microbatch(ids, x[ids])
        model(x[ids], y[ids]).backward()
        snapshot = collector.finalize(distributed=True)
        for name, param in model.named_parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            global_mean_grad = param.grad / 20
            if name in anchors:
                if name == "embed.weight":
                    global_mean_grad = global_mean_grad.T
                a, b = anchors[name]
                torch.testing.assert_close(snapshot.matrix[name].ones[0], 2 * global_mean_grad @ b, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(snapshot.matrix[name].ones[1], 2 * global_mean_grad.T @ a, atol=2e-6, rtol=2e-5)
            else:
                torch.testing.assert_close(snapshot.vectors[name].ones, 2 * global_mean_grad, atol=2e-6, rtol=2e-5)
        collector.close()
    finally:
        dist.destroy_process_group()


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.first, self.second = nn.Linear(6, 6), nn.Linear(6, 6)
        self.norm2 = nn.RMSNorm(6)

    def forward(self, x):
        x = x + self.first(x)
        return x + self.second(self.norm2(x))


class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(11, 6)
        self.norm1 = nn.RMSNorm(6)
        self.blocks = nn.ModuleList([ResidualBlock(), ResidualBlock()])
        self.head = nn.Linear(6, 11)

    def forward(self, x, y):
        x = self.norm1(self.embed(x))
        for block in self.blocks:
            x = block(x)
        return F.cross_entropy(self.head(x).flatten(0, 1), y.flatten(), reduction="sum")


class TinyCausal(nn.Module):
    def __init__(self, shared=False):
        super().__init__()
        self.embed = nn.Embedding(11, 6)
        self.q = nn.Linear(6, 6)
        self.k = nn.Linear(6, 6)
        self.v = nn.Linear(6, 6)
        self.norm = nn.RMSNorm(6)
        self.proj = nn.Linear(6, 11)
        self.shared = shared
        if shared:
            self.proj.weight = self.embed.weight

    def forward(self, x, y):
        h = self.embed(x)
        z = F.scaled_dot_product_attention(self.q(h)[:, None], self.k(h)[:, None],
                                           self.v(h)[:, None], is_causal=True)[:, 0]
        # Repeated parameter use must accumulate, not overwrite.
        z = self.norm(z + self.q(h))
        return F.cross_entropy(self.proj(z).flatten(0, 1), y.flatten(), reduction="sum")


class SquareTied(nn.Module):
    def __init__(self, embedding_first):
        super().__init__()
        if embedding_first:
            self.embed = nn.Embedding(6, 6)
            self.head = nn.Linear(6, 6, bias=False)
            self.head.weight = self.embed.weight
        else:
            self.head = nn.Linear(6, 6, bias=False)
            self.embed = nn.Embedding(6, 6)
            self.embed.weight = self.head.weight

    def forward(self, x, y):
        z = self.head(self.embed(x).tanh())
        return F.cross_entropy(z.flatten(0, 1), y.flatten(), reduction="sum")


def anchors_for(model, rank=2):
    gen = torch.Generator().manual_seed(112)
    ans = {}
    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue
        n, m = p.shape[::-1] if name == "embed.weight" else p.shape
        ans[name] = (torch.linalg.qr(torch.randn(n, rank, generator=gen)).Q,
                     torch.linalg.qr(torch.randn(m, rank, generator=gen)).Q)
    return ans


class SensorsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        self.x = torch.randint(11, (4, 5))
        self.y = torch.randint(11, (4, 5))

    def compare_exact(self, shared=False, chunks=(2, 2), scale=1.0):
        model = TinyCausal(shared=shared)
        reference = copy.deepcopy(model)
        anchors = anchors_for(model)
        collector = SensorCollector(model, anchors, 4, 5, signs_seed=41, max_float_elements=66)
        offset = 0
        for count in chunks:
            sl = slice(offset, offset+count)
            collector.set_microbatch(torch.arange(offset, offset+count), self.x[sl],
                                     adjoint_to_sequence_loss=1 / (5 * scale))
            (model(self.x[sl], self.y[sl]) * scale).backward()
            offset += count
        result = collector.finalize()
        collector.close()
        signs = torch.cat([sequence_signs(torch.arange(4), 4, 41),
                           sequence_signs(torch.arange(4), 4, 41 + 1000003), torch.ones(1, 4)])
        grads = {n: [] for n, _ in reference.named_parameters()}
        for i in range(4):
            reference.zero_grad(set_to_none=True)
            (reference(self.x[i:i+1], self.y[i:i+1]) / 5).backward()
            for n, p in reference.named_parameters():
                grads[n].append(p.grad.clone())
        for n, values in grads.items():
            gs = torch.stack(values)
            exact = torch.einsum("kb,b...->k...", signs, gs) / 2
            if n in anchors:
                if n == "embed.weight":
                    exact = exact.transpose(-1, -2)
                a, b = anchors[n]
                measured = result.matrix[n]
                got_gb = torch.cat([measured.train[0], measured.audit[0], measured.ones[0][None]])
                got_gta = torch.cat([measured.train[1], measured.audit[1], measured.ones[1][None]])
                torch.testing.assert_close(got_gb, exact @ b, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(got_gta, exact.transpose(-1, -2) @ a, atol=2e-6, rtol=2e-5)
                raw = dict(model.named_parameters())[n].grad / (20 * scale)
                if n == "embed.weight":
                    raw = raw.T
                torch.testing.assert_close(measured.ones[0], 2 * raw @ b, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(measured.ones[1], 2 * raw.T @ a, atol=2e-6, rtol=2e-5)
            else:
                measured = result.vectors[n]
                got = torch.cat([measured.train, measured.audit, measured.ones[None]])
                torch.testing.assert_close(got, exact, atol=2e-6, rtol=2e-5)
        return result

    def test_causal_sequences_microbatches_and_all_ones(self):
        self.compare_exact()

    def test_sum_mean_and_amp_units(self):
        self.compare_exact(chunks=(1, 3), scale=128 / 20)

    def test_tied_embedding_head_and_repeated_linear(self):
        self.compare_exact(shared=True)

    def test_square_tied_weight_uses_canonical_owner_orientation(self):
        # Same matrix shape in both roles: a shape-based orientation heuristic
        # would be wrong for one use in either module registration order.
        x, y = self.x % 6, self.y % 6
        for embedding_first in (True, False):
            with self.subTest(embedding_first=embedding_first):
                model = SquareTied(embedding_first)
                reference = copy.deepcopy(model)
                name = "embed.weight" if embedding_first else "head.weight"
                self.assertEqual(list(dict(model.named_parameters())), [name])
                anchors = anchors_for(model)
                collector = SensorCollector(model, anchors, 4, 5, signs_seed=41,
                                            max_float_elements=18)
                collector.set_microbatch(torch.arange(4), x)
                model(x, y).backward()
                result = collector.finalize()
                collector.close()
                gradients = []
                for i in range(4):
                    reference.zero_grad(set_to_none=True)
                    (reference(x[i:i+1], y[i:i+1]) / 5).backward()
                    gradients.append(dict(reference.named_parameters())[name].grad.clone())
                signs = torch.cat([sequence_signs(torch.arange(4), 4, 41),
                                   sequence_signs(torch.arange(4), 4, 41 + 1000003),
                                   torch.ones(1, 4)])
                exact = torch.einsum("kb,bij->kij", signs, torch.stack(gradients)) / 2
                self.assertGreater(float((exact - exact.transpose(-1, -2)).norm()), .01)
                if embedding_first:
                    exact = exact.transpose(-1, -2)
                a, b = anchors[name]
                measured = result.matrix[name]
                gb = torch.cat([measured.train[0], measured.audit[0], measured.ones[0][None]])
                gta = torch.cat([measured.train[1], measured.audit[1], measured.ones[1][None]])
                torch.testing.assert_close(gb, exact @ b, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(gta, exact.transpose(-1, -2) @ a, atol=2e-6, rtol=2e-5)

    def test_sketches_independent_of_microbatch_partition(self):
        a = self.compare_exact(chunks=(4,))
        torch.manual_seed(91)
        torch.randint(11, (4, 5)); torch.randint(11, (4, 5))
        b = self.compare_exact(chunks=(1, 1, 2))
        for name in a.matrix:
            torch.testing.assert_close(a.matrix[name].train[0], b.matrix[name].train[0], atol=2e-6, rtol=2e-5)

    def test_signs_do_not_consume_default_rng(self):
        before = torch.random.get_rng_state().clone()
        full = sequence_signs(torch.arange(10), 8, 56, 6)
        torch.testing.assert_close(full[:, 3:7], sequence_signs(torch.arange(3, 7), 8, 56, 6))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertFalse(torch.equal(full[:4], sequence_signs(torch.arange(10), 4, 56 + 1000003, 6)))

    def test_coverage_and_shapes_fail_closed(self):
        model = TinyCausal()
        collector = SensorCollector(model, anchors_for(model), 4, 5)
        collector.set_microbatch(torch.tensor([0, 1]), self.x[:2])
        model(self.x[:2], self.y[:2]).backward()
        with self.assertRaises(ValueError):
            collector.set_microbatch(torch.tensor([1, 2]), self.x[:2])
        with self.assertRaises(ValueError):
            collector.finalize()
        collector.close()

    def test_snapshot_does_not_alias_live_buffers(self):
        model = TinyCausal()
        anc = anchors_for(model)
        collector = SensorCollector(model, anc, 4, 5)
        collector.set_microbatch(torch.arange(4), self.x)
        model(self.x, self.y).backward()
        snapshot = collector.finalize()
        old = snapshot.matrix["q.weight"].train[0].clone()
        collector.banks["q.weight"][0].fill_(999)
        collector.reset(anc, 1)
        torch.testing.assert_close(old, snapshot.matrix["q.weight"].train[0])
        collector.close()

    def test_token_chunking_matches_dense_contractions(self):
        model = TinyCausal()
        anchors = anchors_for(model)
        collector = SensorCollector(model, anchors, 4, 5, max_float_elements=22)
        collector.set_microbatch(torch.arange(4), self.x)
        model(self.x, self.y).backward()
        got = collector.finalize()
        for name, (a, b) in anchors.items():
            grad = dict(model.named_parameters())[name].grad / 20
            if name == "embed.weight":
                grad = grad.T
            torch.testing.assert_close(got.matrix[name].ones[0], 2 * grad @ b, atol=2e-6, rtol=2e-5)
        collector.close()

    def test_aligned_trajectories_are_global_loss_adjoints(self):
        model = ResidualNet()
        reference = copy.deepcopy(model)
        collector = SensorCollector(model, anchors_for(model), 4, 5, trajectory_positions=7)
        values = []
        def save_output(mod, args, out):
            out.retain_grad()
            values.append(out)
        def save_input(mod, args):
            args[0].retain_grad()
            values.append(args[0])
        reference.norm1.register_forward_hook(save_output)
        for block in reference.blocks:
            block.norm2.register_forward_pre_hook(save_input)
            block.register_forward_hook(save_output)
        (reference(self.x, self.y) / 20).backward()
        for start in (0, 2):
            collector.set_microbatch(torch.arange(start, start+2), self.x[start:start+2])
            model(self.x[start:start+2], self.y[start:start+2]).backward()
        got = collector.finalize()
        expected = []
        for value in values:
            h = value.detach().flatten(0, 1)[collector.positions] @ collector.omega
            g = value.grad.flatten(0, 1)[collector.positions] @ collector.omega
            expected.extend([h, g])
        torch.testing.assert_close(got.trajectories, torch.cat(expected, dim=1), atol=2e-6, rtol=2e-5)
        collector.close()

    def test_two_rank_sum_reduction_and_all_ones(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(ddp_sensor_worker, args=(os.path.join(d, "gloo_init"),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
