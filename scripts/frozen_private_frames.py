"""Optional eager phase-local QR constants for entirely frozen private probes.

Original modules and Parameter objects are never modified. Use evaluate/query
methods for one full validation per call; accessing view directly is supported
but validates on every frame access. Unsafe .data writes are outside the contract.
"""
import copy
import weakref
from contextlib import contextmanager

import torch


FRAME_LISTS = ('matrix_rows', 'matrix_columns', 'vector_frames')


def _tensor_key(t):
    return (id(t), t._version, t.data_ptr(), t.shape, t.stride(), t.dtype,
            t.device, t.requires_grad)


def _module_key(module):
    return (tuple((n,id(m)) for n,m in module.named_modules()),
            tuple((n,_tensor_key(p)) for n,p in module.named_parameters()),
            tuple((n,_tensor_key(b)) for n,b in module.named_buffers()))


def _shallow_tree(module):
    view = copy.copy(module)
    view._parameters = module._parameters.copy()
    view._buffers = module._buffers.copy()
    view._modules = {name: None if child is None else _shallow_tree(child)
                     for name,child in module._modules.items()}
    return view


class FrozenPrivateFrames:
    """Own cached QRs until close; shared actor parameters may keep learning.

    Every private parameter, including trajectory_input, must already be frozen.
    Mutation, replacement, or unfreezing invalidates the phase permanently.
    Eager synchronous use only; do not share a phase between concurrent calls.
    """
    def __init__(self, private):
        from rewriter.adaptive_coordinate_chisel import AdaptiveCheckpointProbes
        if not isinstance(private, AdaptiveCheckpointProbes):
            raise TypeError('adaptive private probes required')
        if torch.is_inference_mode_enabled():
            raise ValueError('construct frozen frame phase outside inference_mode')
        if any(p.requires_grad for p in private.parameters()):
            raise ValueError('all private parameters must be frozen before QR caching')
        self._closed = False
        self._active = 0
        self._original = private
        self._view = _shallow_tree(private)
        self._original_key = _module_key(private)
        self._view_key = _module_key(self._view)
        self._original_extra = copy.deepcopy(private.get_extra_state())
        self._view_extra = copy.deepcopy(self._view.get_extra_state())
        self._entries = []
        owner = weakref.ref(self)
        for name in FRAME_LISTS:
            for source,view in zip(getattr(private,name),getattr(self._view,name)):
                with torch.no_grad():
                    q = source.orthonormal_matrix().detach()
                index = len(self._entries)
                self._entries.append((source,view,_tensor_key(source.raw),q,_tensor_key(q)))
                def cached(index=index):
                    phase = owner()
                    if phase is None:
                        raise RuntimeError('frozen frame phase is closed')
                    return phase._get(index)
                view.orthonormal_matrix = cached

    def _fail(self):
        self.close()
        raise RuntimeError('frozen private/frame cache changed; create a new phase')

    def validate(self):
        if self._closed:
            raise RuntimeError('frozen frame phase is closed')
        if (_module_key(self._original) != self._original_key
                or _module_key(self._view) != self._view_key
                or self._original.get_extra_state() != self._original_extra
                or self._view.get_extra_state() != self._view_extra):
            self._fail()
        for source,view,key,q,qkey in self._entries:
            if (_tensor_key(source.raw) != key or view.raw is not source.raw
                    or _tensor_key(q) != qkey):
                self._fail()
        return self

    def _get(self,index):
        if self._closed:
            raise RuntimeError('frozen frame phase is closed')
        if not self._active:
            self.validate()
        source,view,key,q,qkey = self._entries[index]
        if (_tensor_key(source.raw) != key or view.raw is not source.raw
                or _tensor_key(q) != qkey):
            self._fail()
        return q

    @property
    def view(self):
        self.validate()
        return self._view

    @contextmanager
    def _evaluation(self):
        self.validate()
        self._active += 1
        try:
            yield self._view
            self.validate()
        finally:
            self._active -= 1

    def evaluate(self, actor, snapshot, *, weight_cores=None):
        if (torch.compiler.is_compiling() or hasattr(actor,'_torchdynamo_orig_callable')
                or isinstance(actor,torch._dynamo.eval_frame.OptimizedModule)):
            raise RuntimeError('frozen frame phase supports eager actors only')
        with self._evaluation() as view:
            return actor(snapshot,view,weight_cores=weight_cores)

    def query_weight_cores(self, parameters, *, embedding_names=('embed.weight',)):
        from rewriter.adaptive_sensing import query_weight_cores
        with self._evaluation() as view:
            return query_weight_cores(parameters,view,embedding_names=embedding_names)

    def close(self):
        self._closed = True
        self._entries.clear()
        self._view = None
        self._original = None

    def __enter__(self):
        return self.validate()

    def __exit__(self,*exc):
        self.close()
