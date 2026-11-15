"""Offline Chisel task on the same loop; private probes and Adam state persist per checkpoint."""
from pathlib import Path
import torch
from chisel.harness import build, locate
from chisel.cifar import seed_at
from chisel.network import PrivateProbes, propose
from chisel.offline_data import Replay
from chisel.recording import save_atomic, cpu_copy
from chisel.transfer import action, penalty
from chisel.preservation import Dual, group_norm_change
from chisel.preservation_data import training_batch_step
from chisel.preservation_fit import logits, preservation_gradient, evaluate
import torch.distributed as dist
from chisel.training import autocast
from chisel.distributed_offline import owned, finish


class ProbeStore:
    def __init__(self, root, config, device):
        self.root, self.config, self.device = Path(root), config, device
        self.latest = {}

    def load(self, identity, snapshot):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed_at(self.config['seed'], int(identity[:16], 16)))
            probes = PrivateProbes({name: item['raw'].shape for name, item in snapshot.items()}, self.config['rank']).to(self.device)
        optimizer = build(self.config['optimizer'], params=probes.parameters())
        if identity in self.latest:
            saved = torch.load(self.latest[identity], map_location=self.device, weights_only=False)
            probes.load_state_dict(saved['probes'])
            optimizer.load_state_dict(saved['optimizer'])
        return probes, optimizer

    def save(self, identity, step, probes, optimizer):
        path = self.root / f'{identity}-{step:08d}.pt'
        save_atomic(cpu_copy(dict(probes=probes.state_dict(), optimizer=optimizer.state_dict())), path)
        self.latest[identity] = str(path.resolve())

    def state_dict(self):
        # The actor checkpoint commits these pointers. Older private versions remain valid.
        return dict(latest=dict(self.latest))

    def load_state_dict(self, saved):
        self.latest = dict(saved['latest'])
        if any(not Path(path).is_file() for path in self.latest.values()):
            raise FileNotFoundError('checkpoint-private probe state is missing')


def setup():
    def hook(config, state):
        state.train_batches = build(config['data'])
        state.backward = backward
        state.extras['replay'] = Replay(state.extras['device'], build(config.get('checkpoint_io', {'_target_': 'chisel.checkpoint_io.CheckpointReader'})), config.get('source_compatibility'), phase='current', parameter_dtype=config['runtime'].get('parameter_dtype'), batch_cache_size=config.get('batch_cache_size', 64))
        state.extras['logits_forward'] = torch.compile(logits, **config.get('compile_gpt_loss_kwargs', dict(fullgraph=True, dynamic=False))) if config['compile_gpt_loss'] else logits
        state.extras['proposal_forward'] = torch.compile(propose, fullgraph=True, dynamic=False) if config.get('compile_proposal') else propose
        state.extras['dual'] = build(config['preservation']['dual'])
        store = ProbeStore(Path(config['output_dir']) / 'private', config['private_probes'], state.extras['device'])
        state.extras['private_probes'] = store
        state.extras.setdefault('checkpointables', {}).update(private_probes=store, dual=state.extras['dual'])
    return hook


def functional_loss(model, parameters, inputs, targets, loss_fn, loss_kwargs):
    logits = torch.func.functional_call(model, parameters, (inputs,))
    target = targets.reshape(-1, logits.shape[-1]) if targets.is_floating_point() else targets.reshape(-1)
    return loss_fn(logits.reshape(-1, logits.shape[-1]), target, reduction='sum', **loss_kwargs)


def next_loss_gradient(model, parameters, effective, batch, loss_spec, microbatch, precision_config, state):
    """Backpropagate each GPT microbatch, then contract its detached cotangent once."""
    inputs, targets = batch['inputs'], batch['targets']
    count = len(targets) if targets.is_floating_point() else targets.numel()
    gradients = {name: torch.zeros_like(value) for name, value in effective.items()}
    task_loss = torch.zeros((), device=inputs.device)
    loss_fn = locate(loss_spec['name'])
    for start in range(0, len(inputs), microbatch):
        with autocast(precision_config, state):
            loss = state.extras.get('loss_forward', functional_loss)(model, parameters,
                inputs[start:start + microbatch], targets[start:start + microbatch], loss_fn, loss_spec['kwargs']) / count
        cotangents = torch.autograd.grad(loss, tuple(effective.values()))
        for name, gradient in zip(effective, cotangents):
            gradients[name].add_(gradient.detach())
        task_loss += loss.detach()
        state.extras['training_forwards'] = state.extras.get('training_forwards', 0) + 1
    return task_loss, gradients


def backward(config, state):
    entries = next(state.train_batches)
    metrics = dict(norm_reduction_pct=0., consolidated_penalty=0., kl=0., logit_mse=0., ranking_loss=0.,
                   rank_changed_positions=0., greedy_changed_positions=0., evaluated_positions=0.)
    maximum_kl = 0.
    metrics['private_gradient_norm'] = 0.
    store, replay, dual = (state.extras[name] for name in ('private_probes', 'replay', 'dual'))
    local_entries = [entry for entry in entries if owned(entry, state)]
    for entry, record in replay.reader.iterate(local_entries):
        batch_step = training_batch_step(config, entry['sha256'], state.step, record['step'])
        model, base, snapshot, directions, rates, batch, source = replay.load(entry, record, batch_step=batch_step)
        probes, private_optimizer = store.load(entry['sha256'], snapshot)
        measurement_config = config | config.get('domain_settings', {}).get(entry['domain'], {})
        measurement_config['preservation_cache_key'] = (entry['sha256'], record['step'] if batch_step is None else batch_step)
        if config.get('resample_preservation_examples', True):
            measurement_config['preservation_sample_seed'] = seed_at(config['seeds']['initialization'], int(entry['sha256'][:16], 16) ^ state.step)
        for inner in range(config.get('private_steps', 1)):
            private_optimizer.zero_grad(set_to_none=True)
            _, effective, consolidated = action(state.model, probes, snapshot, directions, rates, config['radius_floor'],
                                                 advance_optimizer=False, proposer=state.extras['proposal_forward'])
            gradient, observed = preservation_gradient(model, base, base | effective, effective, batch, measurement_config, state, dual.value)
            norm_change = group_norm_change(snapshot, effective)
            storage = penalty(consolidated)
            credit = sum((value * gradient[name].detach()).sum() for name, value in effective.items())
            objective = (norm_change + credit + config['consolidated_penalty'] * storage) / len(entries)
            if inner + 1 == config.get('private_steps', 1):
                objective.backward()
                for parameter in probes.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(len(entries))
            else:
                gradients = torch.autograd.grad(objective * len(entries), tuple(probes.parameters()))
                for parameter, gradient in zip(probes.parameters(), gradients):
                    parameter.grad = gradient
            private_norm = torch.nn.utils.clip_grad_norm_(probes.parameters(), config['gradient_clip'])
            private_optimizer.step()
        store.save(entry['sha256'], state.step, probes, private_optimizer)
        metrics['private_gradient_norm'] += private_norm.item() / len(entries)
        metrics['norm_reduction_pct'] -= 100 * norm_change.detach().item() / len(entries)
        metrics['consolidated_penalty'] += storage.detach().item() / len(entries)
        maximum_kl = max(maximum_kl, observed.pop('maximum_kl'))
        for name, value in observed.items():
            metrics[name] += value if name.endswith('positions') else value / len(entries)
    finish(state.model, metrics, state.extras['world_size'])
    maximum = torch.tensor(maximum_kl, device=state.extras['device'], dtype=torch.float64)
    if state.extras['world_size'] > 1:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    shared_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), config['gradient_clip'])
    state.extras['train_loss'] = (-metrics['norm_reduction_pct'] / 100
        + config['consolidated_penalty'] * metrics['consolidated_penalty']
        + config['preservation']['logit_mse_weight'] * metrics['logit_mse']
        + dual.value * metrics['kl'] + config['preservation']['ranking_weight'] * metrics['ranking_loss'])
    metrics.update(maximum_kl=maximum.item(), preservation_multiplier=dual.value, shared_gradient_norm=shared_norm.item(),
                   private_optimizer_steps=len(entries) * config.get('private_steps', 1))
    cache = state.extras.get('teacher_cache')
    if cache is not None:
        metrics.update(teacher_cache_hits=cache.hits, teacher_cache_misses=cache.misses, teacher_cache_bytes=cache.bytes)
    state.extras['metrics'] = metrics
    dual.update(metrics['kl'])
