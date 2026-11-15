"""Jointly fit shared Chisel and private probes on a fixed checkpoint cohort."""
from pathlib import Path
import json
import torch
from chisel import overfit_diagnostics as single
from chisel.diagnostic_lbfgs import ClosureLBFGS


def setup(max_iter=3, history_size=10):
    def hook(config, state):
        if state.extras['world_size'] != 1 or not config['diagnostic_lbfgs'] or config['gradient_clip'] != float('inf'):
            raise ValueError('joint diagnostic fitting requires one device and unclipped L-BFGS gradients')
        entries = next(state.train_batches)
        if len({entry['sha256'] for entry in entries}) != len(entries):
            raise ValueError('fixed cohort must contain distinct checkpoints')
        fits = []
        parameters = list(state.model.parameters())
        for entry in entries:
            model, base, snapshot, _, _, batch, _ = state.extras['replay'].load(entry)
            probes, private = state.extras['private_probes'].load(entry['sha256'], snapshot, base, actor=state.model)
            fits.append(dict(kind='chisel', entry=entry, model=model, base=base, snapshot=snapshot,
                             batch=batch, probes=probes, private=private))
            parameters += list(probes.parameters())
        state.extras['diagnostic_fits'] = fits
        state.optimizer = ClosureLBFGS(parameters, lr=1., max_iter=max_iter, history_size=history_size,
            line_search_fn='strong_wolfe', tolerance_grad=1e-12, tolerance_change=1e-18)
        state.backward = prepare
    return hook


def backward(config, state):
    fits = state.extras['diagnostic_fits']
    rows, losses = [], []
    for fit in fits:
        state.extras['fit'] = fit
        local = config | {'preservation_cache_key': (fit['entry']['sha256'], 'fixed-fitting-batch')}
        single.backward(local, state)
        rows.append(dict(state.extras['metrics']))
        losses.append(state.extras['train_loss'])
    parameters = list(state.model.parameters()) + [p for fit in fits for p in fit['probes'].parameters()]
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(len(fits))
    metrics = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if key.endswith('positions'):
            metrics[key] = sum(values)
        elif key.startswith('maximum_'):
            metrics[key] = max(values)
        elif not key.endswith('gradient_norm'):
            metrics[key] = sum(values) / len(values)
    metrics['minimum_group_norm_reduction_pct'] = min(row['group_norm_reduction_pct'] for row in rows)
    metrics['maximum_checkpoint_logit_mse'] = max(row['logit_mse'] for row in rows)
    state.extras['train_loss'] = sum(losses) / len(losses)
    state.extras['metrics'] = metrics


def prepare(config, state):
    def closure():
        state.optimizer.zero_grad(set_to_none=True)
        backward(config, state)
        state.extras['closure_evaluations'] = state.extras.get('closure_evaluations', 0) + 1
        state.extras['metrics']['cohort_gradient_evaluations'] = state.extras['closure_evaluations']
        return torch.tensor(state.extras['train_loss'], device=state.extras['device'], dtype=torch.float64)
    state.optimizer.closure = closure


def measure(every=5):
    def hook(config, state):
        if (state.step + 1) % every and state.step + 1 != config['train_steps']:
            return
        for fit in state.extras['diagnostic_fits']:
            state.extras['fit'] = fit
            # Each checkpoint gets its own measurement stream; all weights are shared.
            folder = Path(config['output_dir']) / 'audits' / fit['entry']['sha256']
            folder.mkdir(parents=True, exist_ok=True)
            single.measure(every)(config | {'output_dir': str(folder)}, state)
    return hook
