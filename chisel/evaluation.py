"""Read-only evaluation selected by YAML; never provides Chisel feedback."""
import torch
import torch.distributed as dist
from chisel.harness import build
from chisel.training import autocast


def evaluate(every, microbatch_size):
    @torch.no_grad()
    def hook(config, state):
        final = state.step + 1 == config['train_steps']
        if (state.step + 1) % every and not final:
            return
        stream = build(config['evaluation_data'], device=state.extras['device'],
                       rank=state.extras['rank'], world_size=state.extras['world_size'])
        batch = next(stream)
        inputs, targets = (batch['inputs'], batch['targets']) if isinstance(batch, dict) else batch
        loss, correct, count = (torch.zeros((), device=inputs.device) for _ in range(3))
        state.model.eval()
        try:
            for start in range(0, len(inputs), microbatch_size):
                with autocast(config, state):
                    logits = state.extras['forward'](inputs[start:start + microbatch_size])
                    labels = targets[start:start + microbatch_size].reshape(-1)
                    logits = logits.reshape(-1, logits.shape[-1])
                    loss += torch.nn.functional.cross_entropy(logits, labels, reduction='sum')
                correct += (logits.argmax(-1) == labels).sum()
                count += labels.numel()
        finally:
            state.model.train()
        values = torch.stack((loss, correct, count))
        if state.extras['world_size'] > 1:
            dist.all_reduce(values)
        state.extras.setdefault('metrics', {}).update(validation_loss=(values[0]/values[2]).item(),
                                                    validation_accuracy=(values[1]/values[2]).item())
    return hook
