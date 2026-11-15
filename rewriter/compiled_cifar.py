"""Isolated compiled CNN forward with explicit tensor-valued model binding.

All weights and buffers are arguments, so another verified model of the same
architecture can reuse compilation. Immutable-state checks remain outside the
compiled function. No model/optimizer/cache mutation, no CUDA private pool, and
no production registration. Numerical and finite-update qualification remains
mandatory: compiler fusion can change floating-point reductions.
"""
from torch.func import functional_call
import torch


def _layout(engine):
    def tensors(values):
        return tuple((name, tuple(value.shape), tuple(value.stride()), value.dtype,
                      value.device, value.requires_grad) for name,value in values.items())
    return (type(engine.model), tensors(engine._all_parameters), tensors(engine.buffers),
            tuple(engine._shapes.items()), tuple(engine.parameters), engine.microbatch_sequences)


class CompiledCifarForward:
    """Single-consumer forward/AOT-backward candidate, exact fixed microbatch."""
    def __init__(self, engine, *, backend='inductor'):
        self.layout = _layout(engine)
        self.input_layout = None
        self.microbatch_images = engine.microbatch_sequences
        template = engine.model
        shapes = dict(engine._shapes)

        def pure(base, buffers, replacements, images):
            state = dict(base)
            state.update({name:value.reshape(shapes[name]) for name,value in replacements.items()})
            return functional_call(template, (state,buffers), (images,), strict=True)

        options = {'triton.cudagraphs':False} if backend=='inductor' else None
        self.compiled = torch.compile(pure, backend=backend, fullgraph=True, dynamic=False, options=options)

    def __call__(self, engine, replacements, images):
        engine._validate_replacements(replacements)
        if _layout(engine) != self.layout:
            raise ValueError('compiled CNN model layout changed')
        if images.ndim != 4 or len(images) != self.microbatch_images:
            raise ValueError('compiled CNN requires the pinned complete image microbatch')
        images = images.to(engine.device, dtype=torch.float32)
        layout = (tuple(images.shape), tuple(images.stride()), images.dtype, images.device)
        if self.input_layout is None:
            self.input_layout = layout
        elif layout != self.input_layout:
            raise ValueError('compiled CNN image layout changed')
        if images.is_cuda and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise ValueError('compiled CNN qualification requires TF32 disabled')
        return self.compiled(engine._all_parameters, engine.buffers, replacements, images)
