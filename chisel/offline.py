"""Offline Chisel task on the same loop; private probes and Adam state persist per checkpoint."""
from pathlib import Path
import torch
from chisel.harness import build, locate
from chisel.cifar import seed_at
from chisel.network import PrivateProbes
from chisel.offline_data import Replay
from chisel.recording import save_atomic, cpu_copy
from chisel.transfer import action, delayed_loss, penalty
from chisel.training import autocast


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
        if state.extras['world_size'] != 1:
            raise ValueError('offline private-state writes currently require one process per run')
        state.train_batches = build(config['data'])
        state.backward = backward
        state.extras['replay'] = Replay(state.extras['device'], build(config.get('checkpoint_io', {'_target_': 'chisel.checkpoint_io.CheckpointReader'})))
        state.extras['loss_forward'] = torch.compile(functional_loss, fullgraph=True, dynamic=False) if config['compile_gpt_loss'] else functional_loss
        store = ProbeStore(Path(config['output_dir']) / 'private', config['private_probes'], state.extras['device'])
        state.extras['private_probes'] = store
        state.extras.setdefault('checkpointables', {})['private_probes'] = store
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
    metrics = dict(next_batch_loss=0., consolidated_penalty=0.)
    store = state.extras['private_probes']
    replay = state.extras['replay']
    for entry, record in replay.reader.iterate(entries):
        model, base, snapshot, directions, rates, batch, source = replay.load(entry, record)
        probes, private_optimizer = store.load(entry['sha256'], snapshot)
        _, effective, consolidated = action(state.extras['forward'], probes, snapshot, directions, rates, config['radius_floor'])
        parameters = base | effective
        loss, gradient = next_loss_gradient(model, parameters, effective, batch, source['loss'],
                                            config['microbatch_size'], config, state)
        objective = delayed_loss(effective, consolidated, gradient, config['consolidated_penalty']) / len(entries)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(probes.parameters(), config['gradient_clip'])
        private_optimizer.step()
        store.save(entry['sha256'], state.step, probes, private_optimizer)
        metrics['next_batch_loss'] += loss.item() / len(entries)
        metrics['consolidated_penalty'] += penalty(consolidated).detach().item() / len(entries)
    torch.nn.utils.clip_grad_norm_(state.model.parameters(), config['gradient_clip'])
    state.extras['metrics'] = metrics
    state.extras['train_loss'] = metrics['next_batch_loss'] + config['consolidated_penalty'] * metrics['consolidated_penalty']


def evaluate(every):
    @torch.no_grad()
    def hook(config, state):
        if (state.step + 1) % every and state.step + 1 != config['train_steps']:
            return
        if config.get('validation_data') is None:
            return
        catalog = build(config['validation_data'])
        training_ids = {entry['sha256'] for entries in state.train_batches.domains.values() for entry in entries}
        entries = [entry for domain in catalog.domains.values() for entry in domain]
        if training_ids & {entry['sha256'] for entry in entries}:
            raise ValueError('offline training and validation checkpoints overlap')
        totals = {}
        state.model.eval()
        try:
            for entry in entries:
                model, base, snapshot, directions, rates, batch, source = state.extras['replay'].load(entry)
                probes, _ = state.extras['private_probes'].load(entry['sha256'], snapshot)
                _, effective, _ = action(state.extras['forward'], probes, snapshot, directions, rates, config['radius_floor'])
                inputs, targets = batch['inputs'], batch['targets']
                loss = inputs.new_zeros((), dtype=torch.float32)
                for start in range(0, len(inputs), config['microbatch_size']):
                    with autocast(config, state):
                        loss += state.extras['loss_forward'](model, base | effective,
                            inputs[start:start + config['microbatch_size']], targets[start:start + config['microbatch_size']],
                            locate(source['loss']['name']), source['loss']['kwargs'])
                count = len(targets) if targets.is_floating_point() else targets.numel()
                total, examples = totals.get(entry['domain'], (0., 0))
                totals[entry['domain']] = total + loss.item() / count, examples + 1
        finally:
            state.model.train()
        state.extras.setdefault('metrics', {}).update({f'validation_{domain}_loss': total / count for domain, (total, count) in totals.items()})
    return hook
