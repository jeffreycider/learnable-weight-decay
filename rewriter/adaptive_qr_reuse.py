"""Optional eager query+actor evaluation with shared differentiable frame QRs.

One call owns local shallow module views sharing the original Parameters. No
original module changes and no cached QR is reused across evaluations. This is
an experimental CPU-qualified path, not a CUDA graph / torch.compile API.
"""
import copy
import weakref
import torch


def _signature(raw):
    return (id(raw), raw._version, raw.data_ptr(), raw.shape, raw.stride(),
            raw.dtype, raw.device, raw.requires_grad, torch.is_grad_enabled(),
            torch.is_inference_mode_enabled())


def _frame_view(frame):
    view = copy.copy(frame)
    # The mapping is local, while its Parameter objects retain their identity.
    view._parameters = frame._parameters.copy()
    view_ref = weakref.ref(view)
    entry = None

    def orthonormal_matrix():
        nonlocal entry
        signature = _signature(frame.raw)
        if view_ref().raw is not frame.raw:
            raise RuntimeError('frame or autograd mode changed during QR reuse evaluation')
        if entry is not None:
            if signature != entry[0]:
                raise RuntimeError('frame or autograd mode changed during QR reuse evaluation')
            return entry[1]
        value = frame.orthonormal_matrix()
        entry = (signature, value)
        return value

    view.orthonormal_matrix = orthonormal_matrix
    return view


def evaluate_adaptive_with_qr_reuse(actor, snapshot, private, parameters, *,
                                    embedding_names=('embed.weight',)):
    """Run exactly one fresh-core query and eager actor call with private QR reuse.

    Do backward after this call returns. Nested evaluations own independent views;
    exceptions leave originals untouched. Ordinary in-place frame changes fail
    closed; unsafe .data writes are unsupported, as with PyTorch autograd.
    The actor must not retain or mutate its temporary private module argument.
    """
    from .adaptive_sensing import query_weight_cores
    if (torch.compiler.is_compiling()
            or isinstance(actor, torch._dynamo.eval_frame.OptimizedModule)
            or hasattr(actor, '_torchdynamo_orig_callable')):
        raise RuntimeError('QR reuse supports eager actors only')
    view = copy.copy(private)
    view._modules = private._modules.copy()
    for name in ('matrix_rows', 'matrix_columns', 'vector_frames'):
        setattr(view, name, torch.nn.ModuleList(_frame_view(f) for f in getattr(private, name)))
    cores = query_weight_cores(parameters, view, embedding_names=embedding_names)
    return actor(snapshot, view, weight_cores=cores)
