"""Chisel lifecycle hooks: consume lag-one credit, transfer, then ordinary MuonH."""
import torch
from torch.utils._pytree import tree_map
from chisel.harness import build
from chisel.network import PrivateProbes
from chisel.transfer import observe, action, delayed_loss


class Consolidation:
    def __init__(self, parameters, config):
        self.parameters, self.config = parameters, config
        self.device = next(iter(parameters.values())).device
        self.raw = {name: p.detach().clone() for name, p in parameters.items()}
        self.consolidated = {name: torch.zeros_like(p) for name, p in parameters.items()}
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config['seed'])
            self.actor = build(config['model']).to(self.device)
            self.probes = PrivateProbes({name: p.shape for name, p in parameters.items()}, config['model']['rank']).to(self.device)
        if config.get('pretrained'):
            self.actor.load_state_dict(torch.load(config['pretrained'], map_location=self.device, weights_only=False)['model'])
        self.optimizer = build(config['optimizer'], params=list(self.actor.parameters()) + list(self.probes.parameters()))
        self.pending_inputs, self.pending_graph = None, None

    def generate(self, inputs):
        return action(self.actor, self.probes, *inputs, self.config['radius_floor'])

    def consume(self, gradient_divisor):
        if self.pending_graph is None:
            return
        _, effective, consolidated = self.pending_graph
        gradients = {name: p.grad.detach() / gradient_divisor for name, p in self.parameters.items()}
        self.optimizer.zero_grad(set_to_none=True)
        delayed_loss(effective, consolidated, gradients, self.config['penalty']).backward()
        torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.probes.parameters()), self.config['gradient_clip'])
        self.optimizer.step()
        self.pending_inputs, self.pending_graph = None, None

    def prepare(self, optimizer, activity, enabled):
        optimizer.prepare()
        if enabled:
            self.pending_inputs = observe(self.parameters, self.raw, self.consolidated, optimizer, activity)
            self.pending_graph = self.generate(self.pending_inputs)
            transfer, _, consolidated = self.pending_graph
        with torch.no_grad():
            for name, parameter in self.parameters.items():
                parameter.copy_(self.raw[name] + transfer[name]['raw'].detach() if enabled else self.raw[name])
                if enabled:
                    self.consolidated[name] = consolidated[name].detach().clone()

    @torch.no_grad()
    def finish(self):
        for name, parameter in self.parameters.items():
            self.raw[name].copy_(parameter)
            parameter.add_(self.consolidated[name])

    def state_dict(self):
        return dict(raw=self.raw, consolidated=self.consolidated, actor=self.actor.state_dict(),
                    probes=self.probes.state_dict(), optimizer=self.optimizer.state_dict(), pending=self.pending_inputs)

    def load_state_dict(self, saved):
        saved = tree_map(lambda value: value.to(self.device) if torch.is_tensor(value) else value, saved)
        self.raw, self.consolidated = saved['raw'], saved['consolidated']
        self.actor.load_state_dict(saved['actor'])
        self.probes.load_state_dict(saved['probes'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.pending_inputs = saved['pending']
        self.pending_graph = self.generate(self.pending_inputs) if self.pending_inputs is not None else None


def setup():
    def hook(config, state):
        hidden = state.optimizer.groups[config['chisel']['optimizer_group']]
        identities = {id(p) for group in hidden.param_groups for p in group['params']}
        parameters = {name: p for name, p in state.model.named_parameters() if id(p) in identities}
        consolidation = Consolidation(parameters, config['chisel'])
        state.extras['consolidation'] = consolidation
        state.extras.setdefault('checkpointables', {})['consolidation'] = consolidation
    return hook


def before_optimizer():
    def hook(config, state):
        consolidation = state.extras['consolidation']
        divisor = 1
        if config['gradient_units'] == 'sum':
            targets = state.extras['batch']['targets']
            divisor = (len(targets) if targets.is_floating_point() else targets.numel()) * state.extras['world_size']
        consolidation.consume(divisor)
        settings = config['chisel']
        enabled = state.step >= settings['start_step'] and state.step % settings['interval'] == 0 and state.step < config['train_steps'] - 1
        consolidation.prepare(state.optimizer.groups[settings['optimizer_group']], state.extras.get('activity', {}), enabled)
    return hook


def after_optimizer():
    def hook(config, state):
        consolidation = state.extras['consolidation']
        consolidation.finish()
        state.extras.setdefault('metrics', {})['consolidated_norm'] = sum(value.square().sum() for value in consolidation.consolidated.values()).sqrt().item()
    return hook
