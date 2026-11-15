"""Fit only checkpoint-private probes on support examples; audit disjoint queries."""
from copy import deepcopy
import torch
from chisel.harness import RunState, build
from chisel.cifar import seed_at
from chisel.transfer import action, penalty
from chisel.preservation import group_norm_change
from chisel.preservation_fit import preservation_gradient, audit


def split_batch(batch, support_examples, query_examples, seed):
    count = len(batch['inputs'])
    if min(support_examples, query_examples) <= 0 or support_examples + query_examples > count:
        raise ValueError('adaptation requires positive, disjoint support and query batches')
    indices = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    def subset(selected):
        return {key: value.index_select(0, selected.to(value.device))
                for key, value in batch.items()}
    return subset(indices[:support_examples]), subset(indices[support_examples:support_examples + query_examples])


@torch.enable_grad()
def fit(config, state, identity, model, base, snapshot, directions, rates, batch, probes, optimizer):
    """No shared gradients, training dual updates, probe-store writes, or query feedback."""
    settings = config['validation_adaptation']
    if settings['steps'] <= 0:
        raise ValueError('adaptation steps must be positive')
    seed = seed_at(settings['seed'], int(identity[:16], 16))
    if settings.get('query_data'):
        if not 0 < settings['support_examples'] <= len(batch['inputs']):
            raise ValueError('support count exceeds the recorded batch')
        indices = torch.randperm(len(batch['inputs']), generator=torch.Generator().manual_seed(seed))[:settings['support_examples']]
        support = {key: value.index_select(0, indices.to(value.device)) for key, value in batch.items()}
        stream = build(settings['query_data'], device=state.extras['device'], rank=0, world_size=1)
        query = stream.replay(0) if hasattr(stream, 'replay') else next(stream)
        if len(query['inputs']) != settings['query_examples']:
            raise ValueError('query source returned the wrong number of examples')
    else:
        support, query = split_batch(batch, settings['support_examples'], settings['query_examples'], seed)
    measurement = config | dict(preservation_examples=None, teacher_cache_bytes=0,
                                preservation_cache_key=None)
    local = RunState(model=state.model, extras=dict(state.extras))
    local.extras.pop('teacher_cache', None)
    local.extras['training_forwards'] = 0
    dual = deepcopy(state.extras['dual'])
    for group in optimizer.param_groups:
        group['lr'] = settings['private_lr']
    trace = []
    for index in range(settings['steps']):
        optimizer.zero_grad(set_to_none=True)
        _, effective, consolidated = action(state.model, probes, snapshot, directions, rates,
            config['radius_floor'], advance_optimizer=False, proposer=state.extras.get('proposal_forward'))
        cotangents, observed = preservation_gradient(model, base, base | effective, effective,
            support, measurement, local, dual.value)
        objective = group_norm_change(snapshot, effective) + config['consolidated_penalty'] * penalty(consolidated)
        if index == 0 or index & (index - 1) == 0:
            trace.append(dict(private_updates=index, norm_reduction_pct=-100 * group_norm_change(snapshot, effective).item(),
                kl=observed['kl'], rank_changed_positions=observed['rank_changed_positions']))
        objective += sum((value * cotangents[name]).sum() for name, value in effective.items())
        gradients = torch.autograd.grad(objective, tuple(probes.parameters()))
        for parameter, gradient in zip(probes.parameters(), gradients):
            parameter.grad = gradient
        torch.nn.utils.clip_grad_norm_(probes.parameters(), config['gradient_clip'])
        optimizer.step()
        dual.update(observed['kl'])
    with torch.no_grad():
        _, effective, _ = action(state.model, probes, snapshot, directions, rates,
            config['radius_floor'], advance_optimizer=False, proposer=state.extras.get('proposal_forward'))
        final = audit(model, base, base | effective, support, measurement, local)
        trace.append(dict(private_updates=settings['steps'],
            norm_reduction_pct=-100 * group_norm_change(snapshot, effective).item(), **final))
    return query, measurement, dict(adaptation_steps=settings['steps'],
        adaptation_query_source='injected_data' if settings.get('query_data') else 'recorded_batch_partition',
        adaptation_support_examples=len(support['inputs']), adaptation_query_examples=len(query['inputs']),
        adaptation_fitting_model_forwards=local.extras['training_forwards'], adaptation_trace=trace)
