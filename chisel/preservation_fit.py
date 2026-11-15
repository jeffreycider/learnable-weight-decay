"""Teacher/student preservation gradients for offline fitting; no extra online passes."""
import torch
from chisel.preservation import prediction_terms, ranking_changes
from chisel.training import autocast


def logits(model, parameters, inputs):
    return torch.func.functional_call(model, parameters, (inputs,))


def preservation_gradient(model, original, edited, effective, batch, config, state, dual_value):
    inputs = batch['inputs']
    if config.get('preservation_examples') is not None:
        inputs = inputs[:config['preservation_examples']]
    microbatch = config['microbatch_size']
    loss_config = config['preservation']
    forward = state.extras['logits_forward']
    gradients = {name: torch.zeros_like(value) for name, value in effective.items()}
    metrics = dict(kl=0., logit_mse=0., ranking_loss=0., maximum_kl=0., rank_changed_positions=0,
                   greedy_changed_positions=0, evaluated_positions=0)
    for start in range(0, len(inputs), microbatch):
        examples = inputs[start:start + microbatch]
        weight = len(examples) / len(inputs)
        with autocast(config, state):
            with torch.no_grad():
                teacher = forward(model, original, examples)
            student = forward(model, edited, examples)
        terms = prediction_terms(student, teacher, loss_config['top_k'], loss_config['margin_fraction'])
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
        state.extras['training_forwards'] = state.extras.get('training_forwards', 0) + 2
    return gradients, metrics


@torch.no_grad()
def audit(model, original, edited, batch, config, state):
    inputs = batch['inputs']
    if config.get('preservation_examples') is not None:
        inputs = inputs[:config['preservation_examples']]
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
        from chisel.harness import build
        from chisel.transfer import action
        from chisel.preservation import group_norm_change
        if state.extras['rank'] or ((state.step + 1) % every and state.step + 1 != config['train_steps']):
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
                totals = dict(norm_reduction_pct=0., minimum_norm_reduction_pct=float('inf'), kl=0., maximum_kl=0.,
                              rank_changed_positions=0, greedy_changed_positions=0, evaluated_positions=0)
                for entry in entries:
                    model, base, snapshot, directions, rates, batch, _ = state.extras['replay'].load(entry)
                    probes, _ = state.extras['private_probes'].load(entry['sha256'], snapshot)
                    _, effective, _ = action(state.extras['forward'], probes, snapshot, directions, rates,
                                             config['radius_floor'], advance_optimizer=False,
                                             proposer=state.extras.get('proposal_forward'))
                    result = audit(model, base, base | effective, batch, config, state)
                    reduction = -100 * group_norm_change(snapshot, effective).item()
                    totals['norm_reduction_pct'] += reduction / len(entries)
                    totals['minimum_norm_reduction_pct'] = min(totals['minimum_norm_reduction_pct'], reduction)
                    totals['kl'] += result.pop('kl') / len(entries)
                    totals['maximum_kl'] = max(totals['maximum_kl'], result.pop('maximum_kl'))
                    for name, value in result.items():
                        totals[name] += value
                state.extras.setdefault('metrics', {}).update({prefix + '_' + name: value for name, value in totals.items()})
        finally:
            state.model.train()
    return hook
