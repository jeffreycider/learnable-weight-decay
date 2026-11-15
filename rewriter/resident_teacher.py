"""Exact teacher reconstruction from a small resident feature panel.

Only immutable cached head inputs and token IDs are staged. The original
TeacherPanel.teacher_batch performs the FP32 head, softcap and FP64 chunked
log-softmax unchanged. No dense full-panel probabilities or learned encodings
are cached. This is an isolated qualification helper, not a fitting default.

For CUDA graphs, call check_binding(engine) BEFORE EVERY replay, on the host.
Python checks/counters/TeacherBatch metadata do not execute during replay.
Bind/replay must be ordered on one stream and have a single consumer.
"""
import copy

import torch

from .offline_systems import BlockSelectedTeacherPanel
from .teacher_cache import TeacherPanel, validate_precision, _head_signature, tensor_hash


def _validate_panel(panel, engine):
    """Mirror the original binding checks without an extra teacher-head GEMM.

    Kept isolated until the training owner chooses a shared validation API.
    Reconstruction still calls the original TeacherPanel method, including its
    validation. Never replace verified cache reading/loading with this helper.
    """
    validate_precision(panel.metadata, engine)
    size = panel.metadata['microbatch_sequences']
    if size < 1 or size > engine.microbatch_sequences or len(panel.inputs) % size:
        raise ValueError('complete generation microbatches within GPT microbatch required')
    if engine.kl_dtype != torch.float64:
        raise ValueError('persisted targets require FP64 KL arithmetic')
    binding = (engine.teacher_id, panel.metadata['state_sha256'])
    if getattr(engine, '_teacher_cache_binding', None) != binding:
        raise ValueError('cache must be bound to currently loaded verified checkpoint')
    signature = getattr(engine, '_teacher_cache_head_signature', None)
    current = _head_signature(engine)
    if signature is not None and signature != current:
        raise ValueError('bound frozen head tensor identity/version changed')
    if getattr(engine, '_teacher_cache_checked_head', None) != binding:
        for name, key in [('proj.weight', 'head_weight_sha256'), ('proj.bias', 'head_bias_sha256')]:
            if tensor_hash(engine.parameters[name]) != panel.metadata[key]:
                raise ValueError('frozen head differs from cached checkpoint')
        engine._teacher_cache_checked_head = binding
        engine._teacher_cache_head_signature = current
    target = torch.device(engine.teacher_cache_device)
    if target.type == 'cuda' and target.index is None:
        target = torch.device('cuda', torch.cuda.current_device())
    if target != engine.device:
        raise ValueError('resident reconstruction requires teacher targets on the GPT device')


def _feature_blocks(panel):
    size = panel.metadata['microbatch_sequences']
    if isinstance(panel, BlockSelectedTeacherPanel):
        # Do not access panel.features: that property materializes the complete
        # selected CPU panel and would undo the deployed lazy-selection win.
        for block in panel.blocks:
            yield panel.parent.features[block*size:(block+1)*size]
    elif isinstance(panel, TeacherPanel):
        for start in range(0, len(panel.inputs), size):
            yield panel.features[start:start+size]
    else:
        raise TypeError('verified TeacherPanel or lazy block view required')


class ResidentTeacherFeatures:
    """Shape-stable teacher feature storage, refillable between checkpoints.

    A bind changes values and provenance but retains all storage addresses.
    Calls to the original teacher reconstruct only the requested microbatch.
    Graph consumers must retain tensors, not capture-time TeacherBatch IDs, and
    obtain current sequence/state metadata from this object outside replay.
    """
    def __init__(self, panel, engine):
        self._binding = None
        self._panel = None
        self.inputs = None
        self.features = None
        self.bind(panel, engine)

    @torch.no_grad()
    def bind(self, panel, engine):
        self._binding = None
        _validate_panel(panel, engine)
        blocks = list(_feature_blocks(panel))
        if not blocks:
            raise ValueError('nonempty teacher panel required')
        size = panel.metadata['microbatch_sequences']
        shape = (len(panel.inputs), *blocks[0].shape[1:])
        dtype = blocks[0].dtype
        if len(blocks)*size != len(panel.inputs) or len(panel.sequence_ids) != len(panel.inputs):
            raise ValueError('feature/input/sequence block alignment differs')
        if any(b.shape != blocks[0].shape or b.dtype != dtype for b in blocks):
            raise ValueError('feature block shapes/dtypes differ')
        if blocks[0].shape[0] != size or tuple(shape[:2]) != tuple(panel.inputs.shape):
            raise ValueError('feature/input token shapes differ')
        schema = (shape, dtype, tuple(panel.inputs.shape), panel.inputs.dtype,
                  engine.device, size, engine.chunk_tokens)
        if self.features is None:
            self._schema = schema
            self.features = torch.empty(shape, dtype=dtype, device=engine.device)
            self.inputs = torch.empty_like(panel.inputs, device=engine.device)
        elif schema != self._schema:
            raise ValueError('resident panel schema changed; construct a separate graph/provider')
        # Sources are verified immutable mmap values. Ordinary synchronous
        # copies match the reference; pinned async staging is a separate option.
        self.inputs.copy_(panel.inputs)
        for index, block in enumerate(blocks):
            self.features[index*size:(index+1)*size].copy_(block)
        self._panel = TeacherPanel(self.inputs, self.features, tuple(panel.sequence_ids),
                                   copy.deepcopy(panel.metadata))
        self._binding = (id(engine), engine.teacher_id,
                         getattr(engine, '_teacher_cache_binding', None), _head_signature(engine))
        self._storage_signature = self._current_storage_signature()
        return self

    def _current_storage_signature(self):
        return tuple((id(t), t.data_ptr(), t._version) for t in (self.inputs, self.features))

    def check_binding(self, engine):
        current = (id(engine), engine.teacher_id,
                   getattr(engine, '_teacher_cache_binding', None), _head_signature(engine))
        if self._binding is None or current != self._binding:
            raise ValueError('resident teacher must be rebound after a checkpoint/head change')
        if self._current_storage_signature() != self._storage_signature:
            raise ValueError('resident teacher input/feature storage was mutated or replaced')
        _validate_panel(self._panel, engine)

    @property
    def sequence_ids(self):
        return self._panel.sequence_ids

    @property
    def metadata(self):
        return copy.deepcopy(self._panel.metadata)

    @property
    def resident_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.inputs, self.features))

    def teacher_batch(self, engine, start, stop):
        self.check_binding(engine)
        size = self._schema[5]
        if start % size or stop-start != size:
            raise ValueError('aligned complete generation microbatch required')
        # This is deliberately the original implementation, not a fused or
        # algebraically equivalent replacement of its FP32/FP64 operation order.
        return self._panel.teacher_batch(engine, start, stop)
