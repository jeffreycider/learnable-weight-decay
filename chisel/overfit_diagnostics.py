"""Single-checkpoint fitting diagnostics using the ordinary injected harness loop."""
from pathlib import Path
import json
import torch
from torch import nn
from chisel.network import propose
from chisel.transfer import apply_edits, bound_edits
from chisel.preservation import group_norm_change, effective_norm_change
from chisel.preservation_fit import preservation_gradient, sampled_inputs
from chisel.logit_diagnostics import logit_errors, summarize
from chisel.training import optimizers


class DirectEdit(nn.Module):
    """Diagnostic effective edits; no learned optimizer or transfer claim."""
    def __init__(self, snapshot, initial, rank=None):
        super().__init__()
        self.names, self.rank = tuple(snapshot), rank
        self.left, self.right = nn.ParameterList(), nn.ParameterList()
        for name in self.names:
            scale = snapshot[name]['raw'].square().mean().sqrt()
            value = initial[name] / scale
            if rank is None:
                self.left.append(nn.Parameter(value.clone()))
            else:
                u, s, vh = torch.linalg.svd(value, full_matrices=False)
                # A nonzero right factor keeps a zero initial edit trainable.
                self.left.append(nn.Parameter((u[:, :rank] * s[:rank]).contiguous()))
                self.right.append(nn.Parameter(vh[:rank].T.contiguous()))

    def edits(self, snapshot):
        result = {}
        for i, name in enumerate(self.names):
            value = self.left[i] if self.rank is None else self.left[i] @ self.right[i].T
            delta = value * snapshot[name]['raw'].square().mean().sqrt()
            result[name] = dict(raw=delta, consolidated=torch.zeros_like(delta))
        return result


def setup(kind='chisel', rank=16, initial='fitted'):
    def hook(config, state):
        entries = next(state.train_batches)
        if len(entries) != 1 or state.extras['world_size'] != 1:
            raise ValueError('overfit diagnostics require exactly one checkpoint on one device')
        entry = entries[0]
        model, base, snapshot, _, _, batch, _ = state.extras['replay'].load(entry)
        probes, private = state.extras['private_probes'].load(entry['sha256'], snapshot, base, actor=state.model)
        for group in private.param_groups:
            group['lr'] = config['private_probes']['optimizer']['lr']
        with torch.no_grad():
            edits = propose(state.model, probes, snapshot)
            initial_edit = {name: item['raw'] + item['consolidated'] for name, item in edits.items()}
        if initial == 'zero':
            initial_edit = {name: torch.zeros_like(value) for name, value in initial_edit.items()}
        elif initial != 'fitted':
            raise ValueError('initial must be fitted or zero')
        if kind in ('dense', 'rank'):
            state.model = DirectEdit(snapshot, initial_edit, rank if kind == 'rank' else None)
            state.extras['forward'] = state.model
            optimizers()(config, state)
        elif kind != 'chisel':
            raise ValueError(kind)
        state.extras['fit'] = dict(kind=kind, entry=entry, model=model, base=base,
                                  snapshot=snapshot, batch=batch, probes=probes, private=private)
        state.backward = backward
        config['preservation_cache_key'] = (entry['sha256'], 'fixed-fitting-batch')
    return hook


def edited_weights(config, state):
    fit = state.extras['fit']
    edits = (state.extras['proposal_forward'](state.model, fit['probes'], fit['snapshot'])
             if fit['kind'] == 'chisel' else state.model.edits(fit['snapshot']))
    edits = bound_edits(fit['snapshot'], edits, config['radius_floor'])
    return apply_edits(fit['snapshot'], edits)[2]


def backward(config, state):
    fit = state.extras['fit']
    fit['private'].zero_grad(set_to_none=True)
    effective = edited_weights(config, state)
    gradient, observed = preservation_gradient(fit['model'], fit['base'], fit['base'] | effective,
        effective, fit['batch'], config, state, config['diagnostic_kl_weight'])
    reduction = group_norm_change(fit['snapshot'], effective)
    credit = sum((value * gradient[name]).sum() for name, value in effective.items())
    target = config.get('diagnostic_group_shrinkage_target')
    norm_objective = (reduction if target is None else
        config['diagnostic_target_penalty'] * (reduction + target).clamp_min(0).square())
    loss_scale = config.get('diagnostic_loss_scale', 1.)
    (loss_scale * (norm_objective + credit)).backward()
    metrics = dict(group_norm_reduction_pct=-100 * reduction.item(),
        effective_norm_reduction_pct=-100 * effective_norm_change(fit['snapshot'], effective).item(), **observed)
    metrics['shared_gradient_norm'] = torch.nn.utils.clip_grad_norm_(state.model.parameters(), config['gradient_clip']).item()
    if fit['kind'] == 'chisel':
        metrics['private_gradient_norm'] = torch.nn.utils.clip_grad_norm_(fit['probes'].parameters(), config['gradient_clip']).item()
        if not config.get('diagnostic_lbfgs', False):
            fit['private'].step()
    state.extras['train_loss'] = loss_scale * (norm_objective.item() + config['preservation']['logit_mse_weight'] * observed['logit_mse'] + config['diagnostic_kl_weight'] * observed.get('kl', 0.))
    metrics['loss_scale'] = loss_scale
    state.extras['metrics'] = metrics


def measure(every=50):
    @torch.no_grad()
    def hook(config, state):
        if (state.step + 1) % every and state.step + 1 != config['train_steps']:
            return
        fit = state.extras['fit']
        effective = edited_weights(config, state)
        rows = []
        for inputs in sampled_inputs(fit['batch'], config).split(config['microbatch_size']):
            reference = fit['base'] | {name: fit['base'][name].detach().requires_grad_() for name in effective}
            with torch.enable_grad():
                teacher = state.extras['logits_forward'](fit['model'], reference, inputs).detach()
                candidate = {name: value.detach().requires_grad_() for name, value in effective.items()}
                student = state.extras['logits_forward'](fit['model'], fit['base'] | candidate, inputs).detach()
            rows.append(logit_errors(student, teacher))
        metrics = dict(step=state.step + 1, **summarize(rows),
            group_norm_reduction_pct=-100 * group_norm_change(fit['snapshot'], effective).item(),
            effective_norm_reduction_pct=-100 * effective_norm_change(fit['snapshot'], effective).item())
        with (Path(config['output_dir']) / 'logit-errors.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics) + '\n')
        if fit['kind'] == 'chisel':
            state.extras['private_probes'].save(fit['entry']['sha256'], state.step, fit['probes'], fit['private'])
    return hook
