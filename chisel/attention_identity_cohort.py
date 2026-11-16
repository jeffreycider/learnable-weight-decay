"""Mixed-checkpoint learned attention identity diagnostic on the injected loop."""
import torch
from chisel.offline_data import Replay
from chisel.distributed_offline import owned, finish
from chisel.transfer import action
from chisel.preservation import group_norm_change, effective_norm_change


class AttentionReplay(Replay):
    def load(self, entry, record=None, batch_step=None):
        result = super().load(entry, record, batch_step)
        model, base, snapshot = result[:3]
        for index, block in enumerate(model.blocks):
            name = f'blocks.{index}.attn.v.weight'
            snapshot[name]['attention_head_dim'] = block.attn.head_dim
            snapshot[name]['attention_value_bias'] = base[name.removesuffix('weight') + 'bias'].detach().clone()
        return result


def setup():
    def hook(config, state):
        if (config.get('proposal_function') != 'chisel.attention_identity_fit.propose'
                or not isinstance(state.extras['replay'], AttentionReplay)):
            raise ValueError('norm-only diagnostic requires the exact attention identity decoder and metadata')
        if config.get('private_steps', 1) != 1 or config['consolidated_penalty'] != 0:
            raise ValueError('this diagnostic uses one private update and C=0')
        if config.get('training_audit_inputs', 'recorded') != 'recorded':
            raise ValueError('norm-only training does not fit a token batch; audit recorded inputs explicitly')
        state.backward = backward
    return hook


def backward(config, state):
    entries = next(state.train_batches)
    store, replay = state.extras['private_probes'], state.extras['replay']
    metrics = dict(norm_reduction_pct=0., effective_norm_reduction_pct=0., private_gradient_norm=0.,
                   weight_objective_evaluations=0.)
    for entry, record in replay.reader.iterate([e for e in entries if owned(e, state)]):
        _, base, snapshot, directions, rates, _, _ = replay.load(entry, record)
        probes, private = store.load(entry['sha256'], snapshot, base, actor=state.model)
        private.zero_grad(set_to_none=True)
        _, effective, _ = action(state.model, probes, snapshot, directions, rates, config['radius_floor'],
            advance_optimizer=False, proposer=state.extras['proposal_forward'])
        loss = group_norm_change(snapshot, effective)
        (loss / len(entries)).backward()
        for parameter in probes.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(len(entries))
        private_norm = torch.nn.utils.clip_grad_norm_(probes.parameters(), config['gradient_clip'])
        private.step()
        store.save(entry['sha256'], state.step, probes, private)
        metrics['norm_reduction_pct'] -= 100 * loss.item() / len(entries)
        metrics['effective_norm_reduction_pct'] -= 100 * effective_norm_change(snapshot, effective).item() / len(entries)
        metrics['private_gradient_norm'] += private_norm.item() / len(entries)
        metrics['weight_objective_evaluations'] += 1
    finish(state.model, metrics, state.extras['world_size'])
    metrics['shared_gradient_norm'] = torch.nn.utils.clip_grad_norm_(state.model.parameters(), config['gradient_clip']).item()
    state.extras['metrics'] = metrics
    state.extras['train_loss'] = -metrics['norm_reduction_pct'] / 100
