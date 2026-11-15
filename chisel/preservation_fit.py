"""Teacher/student preservation gradients for offline fitting; no extra online passes."""
import torch
from chisel.preservation import prediction_terms, ranking_changes
from chisel.training import autocast
from chisel.teacher_cache import TeacherCache


def logits(model, parameters, inputs):
    return torch.func.functional_call(model, parameters, (inputs,))


def sampled_inputs(batch, config):
    inputs = batch['inputs']
    count = config.get('preservation_examples')
    if count is None or count >= len(inputs):
        return inputs
    if count <= 0:
        raise ValueError('preservation_examples must be positive')
    generator = torch.Generator().manual_seed(config['preservation_sample_seed'])
    indices = torch.randperm(len(inputs), generator=generator)[:count].to(inputs.device)
    return inputs.index_select(0, indices)


def preservation_gradient(model, original, edited, effective, batch, config, state, dual_value):
    inputs = sampled_inputs(batch, config)
    microbatch = config['microbatch_size']
    loss_config = config['preservation']
    forward = state.extras['logits_forward']
    cache = state.extras.setdefault('teacher_cache', TeacherCache(config.get('teacher_cache_bytes', 0)))
    identity = config.get('preservation_cache_key')
    sample = config.get('preservation_sample_seed') if len(inputs) < len(batch['inputs']) else None
    gradients = {name: torch.zeros_like(value) for name, value in effective.items()}
    metrics = dict(kl=0., logit_mse=0., ranking_loss=0., maximum_kl=0., rank_changed_positions=0,
                   greedy_changed_positions=0, evaluated_positions=0)
    for start in range(0, len(inputs), microbatch):
        examples = inputs[start:start + microbatch]
        weight = len(examples) / len(inputs)
        with autocast(config, state):
            # Match the student's autograd signature so compilation uses the same
            # floating-point forward. Detach immediately; never backpropagate teacher.
            key = (identity, sample, start, len(examples)) if identity is not None else None
            teacher = cache.get(key)
            if teacher is None:
                reference = original | {name: original[name].detach().requires_grad_() for name in effective}
                teacher = forward(model, reference, examples).detach()
                cache.put(key, teacher)
                state.extras['training_forwards'] = state.extras.get('training_forwards', 0) + 1
            student = forward(model, edited, examples)
        terms = prediction_terms(student, teacher, loss_config['top_k'], loss_config['margin_fraction'], loss_config.get('ranking_power', 2))
        loss = weight * (loss_config['logit_mse_weight'] * terms['mse'].mean()
                        + dual_value * terms['kl'].mean()
                        + loss_config['ranking_weight'] * terms['ranking'].mean())
        cotangents = torch.autograd.grad(loss, tuple(effective.values()))
        for name, gradient in zip(effective, cotangents):
            gradients[name].add_(gradient.detach())
        metrics['kl'] += weight * terms['kl'].detach().mean().item()
        metrics['logit_mse'] += weight * terms['mse'].detach().mean().item()
        metrics['ranking_loss'] += weight * terms['ranking'].detach().mean().item()
        metrics['maximum_kl'] = max(metrics['maximum_kl'], terms['kl'].detach().max().item())
        changes = ranking_changes(student.detach(), teacher, loss_config['top_k'])
        metrics['rank_changed_positions'] += changes['rank_changed'].item()
        metrics['greedy_changed_positions'] += changes['greedy_changed'].item()
        metrics['evaluated_positions'] += changes['positions']
        state.extras['training_forwards'] = state.extras.get('training_forwards', 0) + 1
    return gradients, metrics


@torch.no_grad()
def audit(model, original, edited, batch, config, state):
    inputs = sampled_inputs(batch, config)
    metrics = dict(kl=0., maximum_kl=0., rank_changed_positions=0, greedy_changed_positions=0, evaluated_positions=0)
    for start in range(0, len(inputs), config['microbatch_size']):
        examples = inputs[start:start + config['microbatch_size']]
        with autocast(config, state):
            teacher = state.extras['logits_forward'](model, original, examples)
            student = state.extras['logits_forward'](model, edited, examples)
        terms = prediction_terms(student, teacher, config['preservation']['top_k'], config['preservation']['margin_fraction'])
        metrics['kl'] += len(examples) / len(inputs) * terms['kl'].mean().item()
        metrics['maximum_kl'] = max(metrics['maximum_kl'], terms['kl'].max().item())
        changes = ranking_changes(student, teacher, config['preservation']['top_k'])
        metrics['rank_changed_positions'] += changes['rank_changed'].item()
        metrics['greedy_changed_positions'] += changes['greedy_changed'].item()
        metrics['evaluated_positions'] += changes['positions']
    return metrics


def evaluate(every):
    @torch.no_grad()
    def hook(config, state):
        from pathlib import Path
        from chisel.harness import build
        from chisel.storage import atomic_json
        from chisel.distributed_offline import synchronize_private_store
        from chisel.transfer import action
        from chisel.preservation import group_norm_change, effective_norm_change
        if (state.step + 1) % every and state.step + 1 != config['train_steps']:
            return
        synchronize_private_store(state)
        if state.extras['rank']:
            return
        training_ids = {entry['sha256'] for rows in state.train_batches.domains.values() for entry in rows}
        state.model.eval()
        try:
            for prefix, key in [('train_audit', 'training_audit_data'), ('validation', 'validation_data')]:
                if config.get(key) is None:
                    continue
                catalog = build(config[key])
                entries = [entry for rows in catalog.domains.values() for entry in rows]
                if prefix == 'validation' and training_ids & {entry['sha256'] for entry in entries}:
                    raise ValueError('offline training and validation checkpoints overlap')
                totals = dict(norm_reduction_pct=0., minimum_norm_reduction_pct=float('inf'),
                              effective_frobenius_reduction_pct=0., minimum_effective_frobenius_reduction_pct=float('inf'), kl=0., maximum_kl=0.,
                              rank_changed_positions=0, greedy_changed_positions=0, evaluated_positions=0)
                checkpoint_results = []
                for entry in entries:
                    model, base, snapshot, directions, rates, batch, _ = state.extras['replay'].load(entry)
                    probes, _ = state.extras['private_probes'].load(entry['sha256'], snapshot, base)
                    _, effective, _ = action(state.extras['forward'], probes, snapshot, directions, rates,
                                             config['radius_floor'], advance_optimizer=False,
                                             proposer=state.extras.get('proposal_forward'))
                    measurement_config = config | config.get('domain_settings', {}).get(entry['domain'], {})
                    result = audit(model, base, base | effective, batch, measurement_config, state)
                    reduction = -100 * group_norm_change(snapshot, effective).item()
                    totals['norm_reduction_pct'] += reduction / len(entries)
                    totals['minimum_norm_reduction_pct'] = min(totals['minimum_norm_reduction_pct'], reduction)
                    actual = -100 * effective_norm_change(snapshot, effective).item()
                    checkpoint_results.append(dict(checkpoint=entry['path'], sha256=entry['sha256'],
                        domain=entry['domain'], norm_reduction_pct=reduction,
                        effective_frobenius_reduction_pct=actual, **result))
                    totals['effective_frobenius_reduction_pct'] += actual / len(entries)
                    totals['minimum_effective_frobenius_reduction_pct'] = min(totals['minimum_effective_frobenius_reduction_pct'], actual)
                    totals['kl'] += result.pop('kl') / len(entries)
                    totals['maximum_kl'] = max(totals['maximum_kl'], result.pop('maximum_kl'))
                    for name, value in result.items():
                        totals[name] += value
                atomic_json(Path(config['output_dir']) / 'audits' / f'{prefix}_{state.step + 1:08d}.json',
                            dict(shared_updates=state.step + 1, checkpoints=checkpoint_results))
                state.extras.setdefault('metrics', {}).update({prefix + '_' + name: value for name, value in totals.items()})
        finally:
            state.model.train()
    return hook
