"""Checkpoint-owned private fitting and globally summed shared Chisel gradients."""
import hashlib
from pathlib import Path
import torch
import torch.distributed as dist
import yaml
from chisel.storage import sha256


def owned(entry, state):
    world = state.extras['world_size']
    owner = getattr(state.train_batches, 'owner', None)
    rank = owner(entry['sha256'], world) if owner else int(entry['sha256'], 16) % world
    return rank == state.extras['rank']


def synchronize_private_store(state):
    """Audits need the trained private probes from every checkpoint owner."""
    state.extras['private_probes'].flush()
    world = state.extras['world_size']
    if world == 1:
        return
    store = state.extras['private_probes']
    local = {identity: path for identity, path in store.latest.items()
             if owned(dict(sha256=identity), state)}
    rows = [None] * world
    dist.all_gather_object(rows, local)
    store.latest = {identity: path for row in rows for identity, path in row.items()}
    local = {identity: value for identity, value in store.measurements.items()
             if owned(dict(sha256=identity), state)}
    dist.all_gather_object(rows, local)
    store.measurements = {identity: value for row in rows for identity, value in row.items()}
    bank = state.extras.get('checkpoint_duals')
    if bank is not None:
        local = {identity: value for identity, value in bank.states.items()
                 if owned(dict(sha256=identity), state)}
        dist.all_gather_object(rows, local)
        bank.load_state_dict({identity: value for row in rows for identity, value in row.items()})


def finish(model, metrics, world_size):
    if world_size == 1:
        return
    parameters = list(model.parameters())
    present = torch.tensor([p.grad is not None for p in parameters], device=parameters[0].device, dtype=torch.int32)
    dist.all_reduce(present, op=dist.ReduceOp.MAX)
    for parameter, used in zip(parameters, present.tolist()):
        if not used:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad)  # Each contribution already divided by global checkpoint count.
    values = torch.tensor(list(metrics.values()), dtype=torch.float64, device=parameters[0].device)
    dist.all_reduce(values)
    metrics.update(zip(metrics, values.tolist()))


def gather_audit(totals, rows, entries, state):
    """Retain every checkpoint result; aggregate the same global means and extrema."""
    for prefix, op in [('minimum_', dist.ReduceOp.MIN), ('maximum_', dist.ReduceOp.MAX), ('', dist.ReduceOp.SUM)]:
        keys = [key for key in totals if key.startswith(prefix)] if prefix else [
            key for key in totals if not key.startswith(('minimum_', 'maximum_'))]
        values = torch.tensor([totals[key] for key in keys], device=state.extras['device'], dtype=torch.float64)
        dist.all_reduce(values, op=op)
        totals.update(zip(keys, values.tolist()))
    gathered = [None] * state.extras['world_size']
    dist.all_gather_object(gathered, rows)
    lookup = {row['sha256']: row for group in gathered for row in group}
    if len(lookup) != len(entries) or sum(map(len, gathered)) != len(entries):
        raise ValueError('distributed audit omitted or duplicated checkpoint identities')
    return [lookup[entry['sha256']] for entry in entries]


def commit(every):
    """Publish a restart generation only after every rank's durable checkpoint exists."""
    def hook(config, state):
        step = state.step + 1
        if step % every and step != min(config['train_steps'], config.get('stop_step', config['train_steps'])):
            return
        root = Path(config['output_dir'])
        rank, world = state.extras['rank'], state.extras['world_size']
        path = root / f'checkpoint_{step:06d}_rank{rank}.pt'
        local = dict(path=path.name, sha256=sha256(path))
        rows = [None] * world
        if world > 1:
            dist.all_gather_object(rows, local)
        else:
            rows = [local]
        if rank == 0:
            destination = root / f'generation_{step:06d}.yaml'
            temporary = destination.with_suffix('.tmp')
            temporary.write_text(yaml.safe_dump(dict(step=step, ranks=rows)))
            temporary.replace(destination)
    return hook


def validate_resume():
    def hook(config, state):
        world = state.extras['world_size']
        if world == 1:
            return
        fingerprint = hashlib.sha256()
        for parameter in state.model.parameters():
            fingerprint.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        local = dict(step=state.step, weights=fingerprint.hexdigest())
        states = [None] * world
        dist.all_gather_object(states, local)
        if any(item != states[0] for item in states):
            raise ValueError('offline rank checkpoints disagree on step or shared parameters')
        if not config.get('resume'):
            return
        path = Path(str(config['resume']).format(rank=state.extras['rank']))
        generation = yaml.safe_load((path.parent / f'generation_{state.step + 1:06d}.yaml').read_text())
        if len(generation['ranks']) != world:
            raise ValueError('changing offline world size requires explicit private-state resharding')
        row = generation['ranks'][state.extras['rank']]
        if sha256(path) != row['sha256']:
            raise ValueError('offline resume file differs from committed generation')
    return hook
