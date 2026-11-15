"""Unprojected, aligned activations and output adjoints from the ordinary pass."""
import re
import torch
from chisel.recording import cpu_copy


class CaptureAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, destination, indices, valid):
        ctx.save_for_backward(destination, indices, valid)
        return output

    @staticmethod
    def backward(ctx, gradient):
        destination, indices, valid = ctx.saved_tensors
        values = gradient.detach().reshape(-1, gradient.shape[-1])
        destination.copy_(torch.where(valid[:, None], values[indices].float(), destination))
        return gradient, None, None, None


class ActivationRecords:
    def __init__(self, model, pattern, positions, tokens_per_example):
        self.positions = positions
        self.tokens_per_example = tokens_per_example
        self.active, self.banks, self.handles = False, {}, []
        for name, module in model.named_modules():
            if not re.fullmatch(pattern, name):
                continue
            if hasattr(module, 'weight') and module.weight.ndim == 2:
                outputs, inputs = module.weight.shape
            else:
                inputs = outputs = module.norm1.gains.numel()
            device = next(module.parameters()).device
            bank = dict(offset=torch.zeros((), dtype=torch.int64, device=device))
            for key, width in [('input', inputs), ('output', outputs), ('adjoint', outputs)]:
                bank[key] = torch.zeros(len(positions), width, device=device)
            bank['seen'] = torch.zeros(len(positions), dtype=torch.bool, device=device)
            self.banks[name] = bank
            self.handles.append(module.register_forward_hook(self.collect(name)))
        if not self.banks:
            raise ValueError('activation pattern matched no modules')

    def collect(self, name):
        def hook(module, inputs, output):
            if not self.active or not module.training:
                return
            bank = self.banks[name]
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            start = bank['offset'].clone()
            indices = self.positions - start
            valid = (indices >= 0) & (indices < len(values))
            indices = indices.clamp(0, len(values) - 1)
            self.copy(bank['input'], values, indices, valid)
            self.copy(bank['output'], output.detach().reshape(-1, output.shape[-1]), indices, valid)
            bank['seen'].logical_or_(valid)
            bank['offset'].add_(len(values))
            if output.requires_grad:
                return CaptureAdjoint.apply(output, bank['adjoint'], indices, valid)
        return hook

    @staticmethod
    def copy(destination, values, indices, valid):
        destination.copy_(torch.where(valid[:, None], values[indices].float(), destination))

    def begin(self, active):
        self.active = active
        for bank in self.banks.values():
            for value in bank.values():
                value.zero_()

    def snapshot(self, rank, world_size):
        for name, bank in self.banks.items():
            if not bank['seen'].all():
                raise ValueError(f'activation positions were not observed: {name}')
        return cpu_copy(dict(rank=rank, world_size=world_size,
            positions=self.positions, tokens_per_example=self.tokens_per_example,
            sample_ids=torch.stack((self.positions // self.tokens_per_example,
                                   self.positions % self.tokens_per_example), dim=1),
            modules={name: {key: value for key, value in bank.items() if key in ('input', 'output', 'adjoint')}
                     for name, bank in self.banks.items()}))

    def close(self):
        for handle in self.handles:
            handle.remove()


def setup(pattern, samples=64):
    def hook(config, state):
        stream, model = state.train_batches, state.model
        tokens = (32 // model.patch) ** 2 + 1 if hasattr(model, 'patch') else stream.seq_len
        examples = stream.batch_size // state.extras['world_size']
        if hasattr(stream, 'seq_len'):
            examples //= tokens  # FineWeb batch_size counts tokens.
        total = examples * tokens
        rank, world = state.extras['rank'], state.extras['world_size']
        count = total if samples is None else samples // world + (rank < samples % world)
        if not 0 < count <= total:
            raise ValueError('activation sample count must fit each rank; null captures every position')
        positions = torch.linspace(0, total - 1, count, dtype=torch.float64).round().long().to(state.extras['device'])
        state.extras['activation_records'] = ActivationRecords(model, pattern, positions, tokens)
    return hook
