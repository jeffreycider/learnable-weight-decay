"""Offline Chisel task on the same loop; private probes and Adam state persist per checkpoint."""
from pathlib import Path
from collections import OrderedDict
import torch
from chisel.harness import build, locate
from chisel.cifar import seed_at
from chisel.network import PrivateProbes, propose
from chisel.offline_data import Replay
from chisel.recording import save_atomic, cpu_copy
from chisel.transfer import action, penalty
from chisel.preservation import CheckpointDuals, group_norm_change
from chisel.preservation_data import training_batch_step
from chisel.preservation_fit import logits, preservation_gradient, evaluate
import torch.distributed as dist
from chisel.training import autocast
from chisel.distributed_offline import owned, finish


def initialize_shared_weights(path, sha256):
    """Start a new fitting stage from explicit shared weights, with fresh private probes."""
    def hook(config, state):
        from chisel.storage import sha256 as digest
        if config.get('resume'):
            raise ValueError('choose either shared initialization or exact resume')
        if digest(path) != sha256:
            raise ValueError('shared initialization checkpoint SHA256 mismatch')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        state.model.load_state_dict(saved['model'], strict=True)
    return hook


def initialize_fitted_probes(path, sha256):
    """Begin an explicit new fitting stage from saved private weights and Adam state."""
    def hook(config, state):
        from chisel.storage import sha256 as digest
        if config.get('resume'):
            raise ValueError('choose either staged initialization or exact resume')
        if digest(path) != sha256:
            raise ValueError('private initialization checkpoint SHA256 mismatch')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        state.extras['private_probes'].load_state_dict(saved['extras']['private_probes'])
    return hook


class ProbeStore:
    def __init__(self, root, config, device):
        self.root, self.config, self.device = Path(root), config, device
        self.latest = {}
        self.cache, self.resident_bytes = OrderedDict(), 0
        self.cache_bytes = config.get('cache_bytes', 0)

    def load(self, identity, snapshot, reference=None, actor=None):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed_at(self.config['seed'], int(identity[:16], 16)))
            probes = PrivateProbes({name: item['raw'].shape for name, item in snapshot.items()}, self.config['rank']).to(self.device)
        if identity not in self.latest and identity not in self.cache and self.config.get('initialize'):
            rule = self.config['initialize']
            locate(rule['name'])(probes, snapshot, reference=reference, actor=actor,
                seed=seed_at(self.config['seed'], int(identity[:16], 16)), **rule['kwargs'])
        optimizer = build(self.config['optimizer'], params=probes.parameters())
        if identity in self.latest or identity in self.cache:
            if identity in self.cache:
                item = self.cache.pop(identity)
                self.cache[identity] = item
                saved = cpu_copy(item[0])  # Optimizer loading must not alias the cached state.
            else:
                saved = torch.load(self.latest[identity], map_location=self.device, weights_only=False)
            probes.load_state_dict(saved['probes'])
            optimizer.load_state_dict(saved['optimizer'])
        return probes, optimizer

    def save(self, identity, step, probes, optimizer):
        payload = cpu_copy(dict(probes=probes.state_dict(), optimizer=optimizer.state_dict()))
        from chisel.checkpoint_io import map_tensors
        sizes = []
        map_tensors(payload, lambda value: sizes.append(value.numel() * value.element_size()))
        size = sum(sizes)
        if identity in self.cache:
            self.resident_bytes -= self.cache.pop(identity)[2]
        if not self.cache_bytes or size > self.cache_bytes:
            self.persist(identity, payload, step)
            return
        while self.cache and self.resident_bytes + size > self.cache_bytes:
            oldest = next(iter(self.cache))
            self.flush_identity(oldest)
            self.resident_bytes -= self.cache.pop(oldest)[2]
        self.cache[identity] = (payload, step, size)
        self.resident_bytes += size

    def persist(self, identity, payload, step):
        path = self.root / f'{identity}-{step:08d}.pt'
        save_atomic(payload, path)
        self.latest[identity] = str(path.resolve())

    def flush_identity(self, identity):
        payload, step, size = self.cache[identity]
        if step is not None:
            self.persist(identity, payload, step)
            self.cache[identity] = (payload, None, size)

    def flush(self):
        for identity in self.cache:
            self.flush_identity(identity)

    def state_dict(self):
        # The actor checkpoint commits these pointers. Older private versions remain valid.
        self.flush()
        return dict(latest=dict(self.latest))

    def load_state_dict(self, saved):
        self.latest = dict(saved['latest'])
        if any(not Path(path).is_file() for path in self.latest.values()):
            raise FileNotFoundError('checkpoint-private probe state is missing')
        self.cache.clear()
        self.resident_bytes = 0


def setup():
    def hook(config, state):
        constraint = config['preservation'].get('constraint_metric', 'kl')
        if constraint not in ('kl', 'logit_mse'):
            raise ValueError('preservation constraint must be kl or logit_mse')
        for settings in config.get('domain_settings', {}).values():
            if settings.get('preservation', {}).get('constraint_metric', constraint) != constraint:
                raise ValueError('all domains must use the same preservation metric')
        state.train_batches = build(config['data'])
        state.backward = backward
        state.extras['replay'] = Replay(state.extras['device'], build(config.get('checkpoint_io', {'_target_': 'chisel.checkpoint_io.CheckpointReader'})), config.get('source_compatibility'), phase='current', parameter_dtype=config['runtime'].get('parameter_dtype'), batch_cache_size=config.get('batch_cache_size', 64), model_options=config.get('replay_model_options'), data_options=config.get('replay_data_options'))
        state.extras['logits_forward'] = torch.compile(logits, **config.get('compile_gpt_loss_kwargs', dict(fullgraph=True, dynamic=False))) if config['compile_gpt_loss'] else logits
        from chisel.compiled_preservation import configure
        configure(config, state)
        state.extras['replay'].cache_normalized_sensors = config.get('cache_normalized_sensors', False)
        state.extras['proposal_forward'] = torch.compile(propose, fullgraph=True, dynamic=False) if config.get('compile_proposal') else propose
        state.extras['dual'] = build(config['preservation']['dual'])
        scope = config['preservation'].get('dual_scope', 'shared')
        if scope not in ('shared', 'checkpoint'):
            raise ValueError('dual_scope must be shared or checkpoint')
        if scope == 'checkpoint':
            bank = CheckpointDuals(state.extras['dual'])
            state.extras['checkpoint_duals'] = bank
            state.extras.setdefault('checkpointables', {})['checkpoint_duals'] = bank
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
    curriculum = state.extras.get('offline_curriculum', {})
    preservation_weight = curriculum.get('preservation_weight', 1.)
    constraint = config['preservation'].get('constraint_metric', 'kl')
    metrics = dict(norm_reduction_pct=0., consolidated_penalty=0., kl=0., logit_mse=0., ranking_loss=0.,
                   rank_changed_positions=0., greedy_changed_positions=0., evaluated_positions=0.)
    maximum_kl, maximum_logit_change, maximum_checkpoint_mse = 0., 0., 0.
    metrics['private_gradient_norm'] = 0.
    metrics.update(preservation_penalty=0., fixed_preservation_penalty=0., preservation_multiplier=0.)
    metrics.update(private_first_objective=0., private_last_objective=0.,
                   private_first_rank_changes=0., private_last_rank_changes=0.)
    store, replay, dual = (state.extras[name] for name in ('private_probes', 'replay', 'dual'))
    dual_bank = state.extras.get('checkpoint_duals')
    local_entries = [entry for entry in entries if owned(entry, state)]
    for entry, record in replay.reader.iterate(local_entries):
        if dual_bank is not None:
            dual = dual_bank.get(entry['sha256'])
        batch_step = training_batch_step(config, entry['sha256'], state.step, record['step'])
        model, base, snapshot, directions, rates, batch, source = replay.load(entry, record, batch_step=batch_step)
        probes, private_optimizer = store.load(entry['sha256'], snapshot, base, actor=state.model)
        if 'private_lr' in curriculum:
            for group in private_optimizer.param_groups:
                group['lr'] = curriculum['private_lr']
        measurement_config = config | config.get('domain_settings', {}).get(entry['domain'], {})
        measurement_config = measurement_config | {'preservation': dict(measurement_config['preservation'])}
        for key in ('logit_mse_weight', 'ranking_weight'):
            measurement_config['preservation'][key] *= preservation_weight
        measurement_config['preservation_cache_key'] = (entry['sha256'], record['step'] if batch_step is None else batch_step)
        if config.get('resample_preservation_examples', True):
            measurement_config['preservation_sample_seed'] = seed_at(config['seeds']['initialization'], int(entry['sha256'][:16], 16) ^ state.step)
        for inner in range(config.get('private_steps', 1)):
            private_optimizer.zero_grad(set_to_none=True)
            _, effective, consolidated = action(state.model, probes, snapshot, directions, rates, config['radius_floor'],
                                                 advance_optimizer=False, proposer=state.extras['proposal_forward'])
            gradient, observed = preservation_gradient(model, base, base | effective, effective, batch, measurement_config, state, preservation_weight * dual.value)
            norm_change = group_norm_change(snapshot, effective)
            storage = penalty(consolidated)
            measured_objective = (norm_change.detach().item() + config['consolidated_penalty'] * storage.detach().item()
                + measurement_config['preservation']['logit_mse_weight'] * observed['logit_mse']
                + preservation_weight * dual.value * observed[constraint]
                + measurement_config['preservation']['ranking_weight'] * observed['ranking_loss'])
            for label, selected in [('first', inner == 0), ('last', inner + 1 == config.get('private_steps', 1))]:
                if selected:
                    metrics[f'private_{label}_objective'] += measured_objective / len(entries)
                    metrics[f'private_{label}_rank_changes'] += observed.get('rank_changed_positions', 0)
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
        metrics['preservation_penalty'] += preservation_weight * dual.value * observed[constraint] / len(entries)
        metrics['fixed_preservation_penalty'] += (measurement_config['preservation']['logit_mse_weight'] * observed['logit_mse']
            + measurement_config['preservation']['ranking_weight'] * observed['ranking_loss']) / len(entries)
        metrics['preservation_multiplier'] += preservation_weight * dual.value / len(entries)
        if dual_bank is not None:
            dual_bank.update(entry['sha256'], dual, observed[constraint])
        metrics['private_gradient_norm'] += private_norm.item() / len(entries)
        metrics['norm_reduction_pct'] -= 100 * norm_change.detach().item() / len(entries)
        metrics['consolidated_penalty'] += storage.detach().item() / len(entries)
        maximum_kl = max(maximum_kl, observed.pop('maximum_kl', 0.))
        maximum_logit_change = max(maximum_logit_change, observed.pop('maximum_logit_change', 0.))
        maximum_checkpoint_mse = max(maximum_checkpoint_mse, observed['logit_mse'])
        for name, value in observed.items():
            metrics[name] += value if name.endswith('positions') else value / len(entries)
    finish(state.model, metrics, state.extras['world_size'])
    maximum = torch.tensor([maximum_kl, maximum_logit_change, maximum_checkpoint_mse],
                           device=state.extras['device'], dtype=torch.float64)
    if state.extras['world_size'] > 1:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    shared_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), config['gradient_clip'])
    state.extras['train_loss'] = (-metrics['norm_reduction_pct'] / 100
        + config['consolidated_penalty'] * metrics['consolidated_penalty']
        + metrics['preservation_penalty'] + metrics['fixed_preservation_penalty'])
    metrics.update(curriculum)
    metrics.update(maximum_kl=maximum[0].item(), maximum_logit_change=maximum[1].item(),
                   maximum_checkpoint_logit_mse=maximum[2].item(),
                   preservation_tolerance=dual.tolerance, shared_gradient_norm=shared_norm.item(),
                   private_optimizer_steps=len(entries) * config.get('private_steps', 1))
    if constraint == 'kl':
        metrics.update(kl_penalty=metrics['preservation_penalty'], kl_tolerance=dual.tolerance)
    if not config.get('measure_kl', True) and constraint != 'kl':
        metrics.pop('kl', None)
        metrics.pop('maximum_kl', None)
    if not config.get('measure_rankings', True):
        for key in ('rank_changed_positions', 'greedy_changed_positions', 'private_first_rank_changes', 'private_last_rank_changes'):
            metrics.pop(key, None)
    cache = state.extras.get('teacher_cache')
    if cache is not None:
        metrics.update(teacher_cache_hits=cache.hits, teacher_cache_misses=cache.misses, teacher_cache_bytes=cache.bytes)
    state.extras['metrics'] = metrics
    if dual_bank is None:
        dual.update(metrics[constraint])
