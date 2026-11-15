"""Coordinate RMS from ordinary linear operations; no model replay."""
import math
import re
import torch
import torch.distributed as dist


class Observations:
    def __init__(self, model, pattern):
        self.active, self.sums, self.handles = False, {}, []
        for name, module in model.named_modules():
            if re.fullmatch(pattern, name + '.weight'):
                self.handles.append(module.register_forward_hook(self.collect(name + '.weight')))

    def collect(self, name):
        def hook(module, inputs, output):
            if not self.active or not module.training:
                return
            self.add(name, 'input', inputs[0].detach())
            if output.requires_grad:
                output.register_hook(lambda gradient: self.add(name, 'adjoint', gradient.detach()))
        return hook

    def add(self, name, bank, values):
        key = (name, bank)
        squares = values.float().square().sum(dim=tuple(range(values.ndim - 1)))
        count = math.prod(values.shape[:-1])
        if key not in self.sums:
            self.sums[key] = [squares, count]
        else:
            self.sums[key][0].add_(squares)
            self.sums[key][1] += count
        # Backward hooks must return None, not replace the task gradient.

    def finalize(self, world_size):
        result = {}
        for (name, bank), (squares, count) in self.sums.items():
            count_tensor = squares.new_tensor(float(count))
            if world_size > 1:
                dist.all_reduce(squares)
                dist.all_reduce(count_tensor)
            result.setdefault(name, {})[bank] = (squares / count_tensor).sqrt().detach().clone()
        self.active = False
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()


def setup(pattern):
    def hook(config, state):
        state.extras['observations'] = Observations(state.model, pattern)
    return hook


def begin(every):
    def hook(config, state):
        observations = state.extras['observations']
        observations.sums = {}
        observations.active = state.step % every == 0
    return hook


def finish():
    def hook(config, state):
        state.extras['activity'] = state.extras['observations'].finalize(state.extras['world_size'])
    return hook


def close():
    return lambda config, state: state.extras['observations'].close()
