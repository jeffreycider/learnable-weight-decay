import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from chisel.harness import RunState
from chisel import parallel_diagnostics as parallel
from chisel import overfit_diagnostics as single
from chisel.diagnostic_lbfgs import ClosureLBFGS
from chisel.preservation import group_norm_change, prediction_terms
from chisel.preservation_fit import logits


def edited(config, state):
    return {'weight': state.extras['fit']['base']['weight'] + state.model.weight + state.extras['fit']['probes'].weight}


def worker(rank, path, backend):
    torch.set_num_threads(1)
    device = torch.device('cuda', rank) if backend == 'nccl' else torch.device('cpu')
    if device.type == 'cuda': torch.cuda.set_device(device)
    dist.init_process_group(backend, init_method='file://' + path, rank=rank, world_size=2)
    try:
        torch.manual_seed(61)
        teacher = torch.nn.Linear(4, 3, bias=False).to(device)
        shared = torch.nn.Linear(4, 3, bias=False).to(device)
        private = torch.nn.Linear(4, 3, bias=False).to(device)
        base = {'weight': teacher.weight.detach().clone()}
        inputs = torch.randn(8, 4, device=device)
        snapshot = {'weight': dict(raw=base['weight'], consolidated=torch.zeros_like(base['weight']))}
        fit = dict(kind='chisel', entry={'sha256':'test'}, model=teacher, base=base, snapshot=snapshot,
                   batch={'inputs':inputs}, probes=private, private=torch.optim.Adam(private.parameters()))
        state = RunState(model=shared, extras=dict(fit=fit, device=device, rank=rank, world_size=2, logits_forward=logits))
        config = dict(distributed_diagnostic=True, diagnostic_lbfgs=True, gradient_clip=float('inf'),
                      microbatch_size=2, runtime={'autocast':None}, diagnostic_kl_weight=0.,
                      measure_kl=False, measure_rankings=False, preservation=dict(top_k=3, margin_fraction=.5,
                      logit_mse_weight=10., ranking_weight=0., center_logits=False))
        params = list(shared.parameters()) + list(private.parameters())
        state.optimizer = ClosureLBFGS(params, lr=.1, max_iter=2, line_search_fn='strong_wolfe')
        original = single.edited_weights
        single.edited_weights = edited
        parallel.setup()(config, state)
        assert len(fit['batch']['inputs']) == 4
        parallel.prepare(config, state)
        value = state.optimizer.closure()
        actual = [p.grad.clone() for p in params]
        effective = edited(config, state)
        terms = prediction_terms(logits(teacher, effective, inputs), logits(teacher, base, inputs),
                                 center_logits=False, compute_kl=False, compute_ranking=False)
        expected_loss = group_norm_change(snapshot, effective) + 10 * terms['mse'].mean()
        expected = torch.autograd.grad(expected_loss, params)
        torch.testing.assert_close(value.float(), expected_loss.detach(), rtol=2e-5, atol=2e-6)
        for a,e in zip(actual, expected): torch.testing.assert_close(a,e,rtol=2e-5,atol=2e-6)
        assert state.extras['metrics']['evaluated_positions'] == 8
        for _ in range(3): state.optimizer.step()
        for p in params:
            ref = p.detach().clone()
            dist.broadcast(ref, 0)
            torch.testing.assert_close(p, ref, rtol=0, atol=0)
        assert state.extras['train_loss'] < value.item()
        single.edited_weights = original
    finally:
        dist.destroy_process_group()


def test_parallel_fixed_objective_cpu(tmp_path):
    mp.spawn(worker, args=(str(tmp_path/'join'), 'gloo'), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='two CUDA devices required')
def test_parallel_fixed_objective_cuda(tmp_path):
    mp.spawn(worker, args=(str(tmp_path/'join'), 'nccl'), nprocs=2, join=True)
