"""Optional whole-step timing, allocator peaks, and compilation accounting."""
import time
import torch


def start():
    def hook(config, state):
        device = state.extras['device']
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        state.extras['step_started_at'] = time.perf_counter()
    return hook


def finish():
    def hook(config, state):
        device = state.extras['device']
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        metrics = state.extras.setdefault('metrics', {})
        metrics['optimizer_step_seconds'] = time.perf_counter() - state.extras['step_started_at']
        metrics['compiled_graphs'] = torch._dynamo.utils.counters['stats']['unique_graphs']
        if device.type == 'cuda':
            metrics['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            metrics['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
    return hook
