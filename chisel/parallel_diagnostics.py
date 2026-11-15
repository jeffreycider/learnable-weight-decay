"""Same fixed objective, with preservation examples split across replicas."""
import torch
import torch.distributed as dist
from chisel import overfit_diagnostics as single
from chisel.preservation_fit import sampled_inputs


def parameters(state):
    return list(state.model.parameters()) + list(state.extras['fit']['probes'].parameters())


def reduce_result(state):
    """Equal-sized example shards: average gradients/objective, sum counts."""
    world = state.extras['world_size']
    if world == 1:
        return
    params = parameters(state)
    # A single collective avoids one synchronization per small probe tensor.
    grads = [torch.zeros_like(p) if p.grad is None else p.grad for p in params]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat)
    flat.div_(world)
    offset = 0
    for p in params:
        p.grad = flat[offset:offset + p.numel()].view_as(p).clone()
        offset += p.numel()
    metrics = state.extras['metrics']
    keys = [k for k in metrics if not k.startswith('maximum_')]
    values = torch.tensor([state.extras['train_loss']] + [metrics[k] for k in keys],
                          dtype=torch.float64, device=flat.device)
    dist.all_reduce(values)
    values.div_(world)
    state.extras['train_loss'] = values[0].item()
    for key, value in zip(keys, values[1:].tolist()):
        metrics[key] = value * world if key.endswith('positions') else value
    for key in metrics:
        if key.startswith('maximum_'):
            value = torch.tensor(metrics[key], device=flat.device, dtype=torch.float64)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            metrics[key] = value.item()
    for name, group in [('shared', state.model.parameters()),
                        ('private', state.extras['fit']['probes'].parameters())]:
        metrics[name + '_gradient_norm'] = torch.linalg.vector_norm(
            torch.stack([p.grad.norm() for p in group])).item()


def setup():
    def hook(config, state):
        if not config.get('distributed_diagnostic') or not config['diagnostic_lbfgs']:
            raise ValueError('parallel fitting requires the diagnostic L-BFGS path')
        fit = state.extras['fit']
        if fit['kind'] != 'chisel' or config['gradient_clip'] != float('inf'):
            raise ValueError('parallel fitting requires Chisel with unclipped gradients')
        world, rank = state.extras['world_size'], state.extras['rank']
        inputs = sampled_inputs(fit['batch'], config)
        if len(inputs) < world or len(inputs) % world:
            raise ValueError('fixed preservation examples must divide evenly across replicas')
        if world > 1:
            for p in parameters(state):
                dist.broadcast(p.data, 0)
        state.extras['full_fitting_batch'] = fit['batch']
        size = len(inputs) // world
        fit['batch'] = {'inputs': inputs[rank * size:(rank + 1) * size]}
        state.extras['parallel_config'] = config | {
            'preservation_examples': None,
            'preservation_cache_key': (fit['entry']['sha256'], 'shard', rank, world),
        }
        state.backward = prepare
    return hook


def prepare(config, state):
    def closure():
        state.optimizer.zero_grad(set_to_none=True)
        single.backward(state.extras['parallel_config'], state)
        reduce_result(state)
        state.extras['closure_evaluations'] = state.extras.get('closure_evaluations', 0) + 1
        state.extras['metrics']['closure_evaluations'] = state.extras['closure_evaluations']
        return torch.tensor(state.extras['train_loss'], device=state.extras['device'], dtype=torch.float64)
    state.optimizer.closure = closure


def measure(every=10):
    def hook(config, state):
        fit = state.extras['fit']
        if not state.extras['rank']:
            shard = fit['batch']
            try:
                fit['batch'] = state.extras['full_fitting_batch']
                single.measure(every)(config, state)
            finally:
                fit['batch'] = shard
        if state.extras['world_size'] > 1:
            store = state.extras['private_probes']
            pointers = [store.latest if state.extras['rank'] == 0 else None]
            dist.broadcast_object_list(pointers, src=0)
            store.latest = pointers[0]
    return hook
