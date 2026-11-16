"""Evaluate a training successor without advancing the ordinary data/RNG stream."""
import copy
import random
import numpy as np
import torch
from chisel.harness import RunState
from chisel.training import backward


def next_gradient(config, state, parameters):
    """One ordinary-sized backward, at the caller's temporary successor weights.

    Parameter gradients are scratch; the caller has already consumed g_t. Data,
    RNG, module buffers, training metrics and mode must survive unchanged.
    """
    stream = state.train_batches
    cursor = copy.deepcopy(stream.state_dict())
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    buffers = {name: value.detach().clone() for name, value in state.model.named_buffers()}
    preview = RunState(step=state.step, model=state.model, extras=dict(state.extras))
    preview.extras['training_forwards'] = 0
    state.model.zero_grad(set_to_none=True)
    try:
        # Streams with a direct replay API avoid tearing down prefetch workers.
        batch = stream.preview() if hasattr(stream, 'preview') else next(stream)
        preview.train_batches = iter([batch])
        backward(config, preview)
        gradients = {name: parameter.grad.detach() for name, parameter in parameters.items()}
        metrics = dict(lookahead_loss=preview.extras['train_loss'],
                       lookahead_microbatches=preview.extras['training_forwards'])
        return gradients, metrics
    finally:
        if not hasattr(stream, 'preview'):
            stream.load_state_dict(cursor)
        with torch.no_grad():
            for name, value in state.model.named_buffers():
                value.copy_(buffers[name])
        state.model.zero_grad(set_to_none=True)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
