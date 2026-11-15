"""Explicit finite-output preservation experiments, outside the reference budget.

This module performs extra teacher/student GPT forwards and an extra student
GPT backward. Every invocation is counted. It must not be presented as the
zero-extra-evaluation reference algorithm or as benchmark-approved compute.
Inputs must be past training sequences or explicitly declared synthetic data;
validation/test inputs are rejected by the interface.

The teacher is an immutable parameter/buffer snapshot. The student uses the same
GPT architecture and baseline BF16 parameter casts, with the canonical anchored
edit inserted through ``torch.func.functional_call``. Dense differentiable edited
parameters are deliberately materialized for this experimental path: roughly
650 MB of FP32 matrix values at full PR323 size, in addition to GPT activations.
Configure the sequence microbatch for available memory and backward each
microbatch separately. Throughput and peak memory must be measured together.

Teacher log probabilities are cached in FP32 by default (CPU by default), costing
roughly 196.5 MiB per 1024-token sequence at vocabulary 50304. For small edits,
``kl_dtype=torch.float64`` computes and caches log probabilities and KL arithmetic in
FP64 (393 MiB per sequence), while GPT logits and their cotangents remain FP32.
FP32 log-softmax subtraction can produce signed KL errors around 1e-7 nats/token
and negative estimates near identity; FP64 is preferred for that regime.
KL arithmetic streams token chunks. Its custom first-order backward saves logits and cached teacher log
probabilities, but no extra full student softmax/log-softmax tensor. A full student
logits-gradient tensor is still needed by GPT backward and is not hidden here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import uuid

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.func import functional_call

from .algebra import RawAction


def gpt_logits(model, inputs):
    """PR323 forward through its exact 15-softcap, without a CE reduction.

    This duplicates only the short orchestration in baseline.model.GPT.forward;
    it invokes the existing modules and does not change their implementation,
    architecture, normalization, attention, precision, or parameter casting.
    """
    x = model.norm1(model.embed(inputs))
    for block in model.blocks:
        x = block(x)
    logits = model.proj(model.norm2(x)).float()
    return 15 * logits * (logits.square() + 15**2).rsqrt()


class _LogitsView(nn.Module):
    def __init__(self, model, gpt_precision='baseline'):
        super().__init__()
        self.gpt = model
        self.gpt_precision = gpt_precision

    def forward(self, inputs):
        if self.gpt_precision=='fp32':
            from .gpt_precision import fp32_head_inputs
            logits=self.gpt.proj(fp32_head_inputs(self.gpt,inputs))
            return 15*logits*(logits.square()+225).rsqrt()
        return gpt_logits(self.gpt, inputs)


class _ChunkedTeacherKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student_logits, teacher_log_probs, chunk_tokens):
        if student_logits.shape != teacher_log_probs.shape:
            raise ValueError('teacher and student logits must have identical shapes')
        if student_logits.dtype != torch.float32 or teacher_log_probs.dtype not in (torch.float32, torch.float64):
            raise ValueError('finite preservation KL requires FP32 logits and FP32/FP64 teacher log probabilities')
        if chunk_tokens < 1:
            raise ValueError('positive KL token chunk size required')
        student = student_logits.reshape(-1, student_logits.shape[-1])
        teacher = teacher_log_probs.reshape_as(student)
        if not len(student):
            raise ValueError('empty preservation batch')
        loss = student_logits.new_zeros((), dtype=teacher_log_probs.dtype)
        for start in range(0, len(student), chunk_tokens):
            stop = min(start + chunk_tokens, len(student))
            logq = teacher[start:stop].to(device=student.device)
            logp = F.log_softmax(student[start:stop].to(dtype=teacher.dtype), dim=-1)
            loss.add_((logq.exp() * (logq - logp)).sum())
        ctx.save_for_backward(student_logits, teacher_log_probs)
        ctx.chunk_tokens = chunk_tokens
        return loss / len(student)

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        student_logits, teacher_log_probs = ctx.saved_tensors
        student = student_logits.reshape(-1, student_logits.shape[-1])
        teacher = teacher_log_probs.reshape_as(student)
        gradient = torch.empty_like(student)
        for start in range(0, len(student), ctx.chunk_tokens):
            stop = min(start + ctx.chunk_tokens, len(student))
            q = teacher[start:stop].to(device=student.device).exp()
            p = F.log_softmax(student[start:stop].to(dtype=teacher.dtype), dim=-1).exp()
            # Include the finite-precision teacher mass: exp(log_softmax) may
            # sum to 1 +/- roundoff, as in the full automatic-differentiation KL.
            gradient[start:stop] = (p * q.sum(-1, keepdim=True) - q) * (cotangent / len(student))
        return gradient.reshape_as(student_logits), None, None


def chunked_teacher_kl(student_logits, teacher_log_probs, chunk_tokens=128):
    """KL(teacher || student), mean nats/token; first-order autograd only.

    Arithmetic follows teacher_log_probs.dtype. For accurate FP64 operation,
    teacher log probabilities must themselves be computed from FP64 logits;
    casting previously rounded FP32 log probabilities cannot recover accuracy.
    The student model remains FP32 and receives an FP32 logits cotangent.
    """
    return _ChunkedTeacherKL.apply(student_logits, teacher_log_probs.detach(), chunk_tokens)


class _RealizedParameterNorm(torch.autograd.Function):
    """FP64 values without retaining dense FP64 tensors for autograd.

    Autograd uses the usual floating-point cast straight-through convention;
    it does not differentiate the discontinuous rounding operation literally.
    """
    @staticmethod
    def forward(ctx, replacement, base, chunk_rows):
        norm_change = replacement.new_zeros((), dtype=torch.float64)
        edit_sq = replacement.new_zeros((), dtype=torch.float64)
        for start in range(0, len(replacement), chunk_rows):
            new = replacement[start:start+chunk_rows].double()
            old = base[start:start+chunk_rows].double()
            delta = new - old
            squared = delta.square().sum()
            norm_change.add_(2 * (old * delta).sum() + squared)
            edit_sq.add_(squared)
        ctx.save_for_backward(replacement, base)
        ctx.chunk_rows = chunk_rows
        return norm_change, edit_sq

    @staticmethod
    @once_differentiable
    def backward(ctx, norm_cotangent, edit_cotangent):
        replacement, base = ctx.saved_tensors
        gradient = torch.empty_like(replacement)
        for start in range(0, len(replacement), ctx.chunk_rows):
            stop = start + ctx.chunk_rows
            new = replacement[start:stop].double()
            value = torch.zeros_like(new)
            if norm_cotangent is not None:
                value.add_(2 * norm_cotangent * new)
            if edit_cotangent is not None:
                old = base[start:stop].double()
                value.add_(2 * edit_cotangent * (new - old))
            gradient[start:stop] = value.to(replacement.dtype)
        return gradient, None, None


@dataclass
class PreservationCompute:
    teacher_forwards: int = 0
    teacher_tokens: int = 0
    student_forwards: int = 0
    student_tokens: int = 0
    student_backwards: int = 0
    student_backward_tokens: int = 0

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TeacherBatch:
    inputs: Tensor
    log_probs: Tensor
    teacher_id: str
    data_role: str
    sequence_ids: tuple[int, ...] | None = None

    @property
    def tokens(self):
        return self.inputs.numel()

    @property
    def storage_bytes(self):
        return (self.inputs.numel() * self.inputs.element_size()
                + self.log_probs.numel() * self.log_probs.element_size())


class FunctionalPreservation:
    """A frozen teacher and differentiable anchored student for one checkpoint.

    Use a standalone GPT, or ensure its ordinary sensor hooks are closed/dormant.
    ``functional_call`` temporarily swaps parameter attributes, so concurrent
    execution against the same model object is unsupported. Live parameters are
    never changed, and their ``requires_grad`` flags are left untouched.
    """
    def __init__(self, model, *, parameter_state=None, buffer_state=None,
                 microbatch_sequences=1, chunk_tokens=128, teacher_cache_device='cpu',
                 kl_dtype=torch.float32, gpt_precision='baseline'):
        from .gpt_precision import configure_precision
        configure_precision(gpt_precision)
        self.gpt_precision=gpt_precision
        if gpt_precision=='fp32':model.float()
        if type(microbatch_sequences) is not int or microbatch_sequences < 1:
            raise ValueError('microbatch_sequences must be a positive integer')
        if chunk_tokens < 1:
            raise ValueError('positive KL token chunk size required')
        if kl_dtype not in (torch.float32, torch.float64):
            raise ValueError('KL arithmetic/cache dtype must be torch.float32 or torch.float64')
        for attribute in ('embed', 'norm1', 'blocks', 'norm2', 'proj'):
            if not hasattr(model, attribute):
                raise ValueError('finite preservation requires the PR323 GPT module interface')
        self.view = _LogitsView(model,gpt_precision)
        self.device = next(model.parameters()).device
        original = dict(model.named_parameters())
        state = original if parameter_state is None else parameter_state
        if gpt_precision=='fp32':
            state={n:v.float() if isinstance(v,Tensor) and v.is_floating_point() else v for n,v in state.items()}
        if set(state) != set(original):
            raise ValueError('teacher parameter snapshot must match every unique model parameter')
        self.parameters = {}
        for name, parameter in original.items():
            source = state[name]
            if source.shape != parameter.shape or source.dtype != parameter.dtype:
                raise ValueError(f'teacher parameter shape/dtype differs for {name}')
            if parameter.ndim not in (1, 2):
                raise ValueError('anchored prototype supports only matrix and vector parameters')
            self.parameters[name] = source.detach().to(device=self.device, copy=True)
        original_buffers = dict(model.named_buffers())
        buffers = original_buffers if buffer_state is None else buffer_state
        if set(buffers) != set(original_buffers):
            raise ValueError('teacher buffer snapshot must match model buffers')
        for name, source in buffers.items():
            expected = original_buffers[name]
            if source.shape != expected.shape or source.dtype != expected.dtype:
                raise ValueError(f'teacher buffer shape/dtype differs for {name}')
        self.buffers = {name: tensor.detach().to(device=self.device, copy=True) for name, tensor in buffers.items()}
        self.transpose_storage = {
            name: isinstance(model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model, nn.Embedding)
            for name, p in original.items() if p.ndim == 2
        }
        self.microbatch_sequences = microbatch_sequences
        self.chunk_tokens = chunk_tokens
        self.teacher_cache_device = torch.device(teacher_cache_device)
        self.kl_dtype = kl_dtype
        self.teacher_id = uuid.uuid4().hex  # separate from every torch/model RNG stream
        self.compute = PreservationCompute()
        self._forward_callable = self._functional_forward
        self._refresh_base_norm()

    @torch.no_grad()
    def _refresh_base_norm(self):
        # Compute once from the actual stored teacher weights. Chunking avoids
        # a temporary FP64 copy of either full vocabulary matrix.
        with torch.no_grad():
            self.base_weight_norm_sq = torch.zeros((), dtype=torch.float64, device=self.device)
            for parameter in self.parameters.values():
                for start in range(0, len(parameter), 512):
                    self.base_weight_norm_sq.add_(parameter[start:start+512].double().square().sum())

    @torch.no_grad()
    def load_teacher_state(self, parameter_state, buffer_state):
        """Replace checkpoint values while retaining compiled code and storage.

        Finish all student backwards before reloading. A reload invalidates every
        teacher probability cache and resets per-state compute accounting. Full
        validation precedes mutation, so an invalid state leaves this teacher intact.
        """
        if self.gpt_precision=='fp32':
            parameter_state={n:v.float() if isinstance(v,Tensor) and v.is_floating_point() else v for n,v in parameter_state.items()}
        for label, incoming, current in (
                ('parameter', parameter_state, self.parameters),
                ('buffer', buffer_state, self.buffers)):
            if set(incoming) != set(current):
                raise ValueError(f'teacher {label} snapshot must match every stored tensor')
            for name, value in incoming.items():
                expected = current[name]
                if not isinstance(value, Tensor) or value.shape != expected.shape or value.dtype != expected.dtype:
                    raise ValueError(f'teacher {label} shape/dtype differs for {name}')
        for incoming, current in ((parameter_state, self.parameters), (buffer_state, self.buffers)):
            for name, value in incoming.items():
                current[name].copy_(value.detach())
        self.teacher_id = uuid.uuid4().hex
        self.compute = PreservationCompute()
        self._refresh_base_norm()

    def enable_compiled_forward(self, **compile_options):
        """Compile once; reuse across same-shaped checkpoint value changes.

        Teacher inference and differentiable student calls can specialize into
        separate graphs. Callers must measure actual graph counts and steady-state
        reuse on the target backend; this does not promise a single kernel.
        """
        self._forward_callable = torch.compile(self._functional_forward, **compile_options)

    def _functional_forward(self, parameters, buffers, inputs):
        state = {'gpt.' + name: value for name, value in parameters.items()}
        buffers = {'gpt.' + name: value for name, value in buffers.items()}
        return functional_call(self.view, (state, buffers), (inputs,), strict=True, tie_weights=True)

    def _forward(self, parameters, inputs):
        return self._forward_callable(parameters, self.buffers, inputs)

    @torch.no_grad()
    def cache_teacher(self, inputs, *, data_role, sequence_ids=None):
        if data_role not in ('past_training', 'synthetic', 'fresh_next_training_audit'):
            raise ValueError('only past_training or explicitly synthetic inputs are permitted; no validation/test data')
        if inputs.ndim != 2 or min(inputs.shape) < 1 or inputs.dtype not in (torch.int32, torch.int64):
            raise ValueError('expected nonempty integer (sequence, token) inputs')
        if sequence_ids is not None and len(sequence_ids) != len(inputs):
            raise ValueError('sequence ID count differs from replay input count')
        result = []
        for start in range(0, len(inputs), self.microbatch_sequences):
            stop = min(start + self.microbatch_sequences, len(inputs))
            selected = inputs[start:stop].detach().to(device=self.device, copy=True)
            logits = self._forward(self.parameters, selected)
            self.compute.teacher_forwards += 1
            self.compute.teacher_tokens += selected.numel()
            flat = logits.flatten(0, 1)
            log_probs = torch.empty(logits.shape, dtype=self.kl_dtype, device=self.teacher_cache_device)
            output = log_probs.flatten(0, 1)
            for token in range(0, len(flat), self.chunk_tokens):
                output[token:token+self.chunk_tokens].copy_(F.log_softmax(
                    flat[token:token+self.chunk_tokens].to(dtype=self.kl_dtype), dim=-1))
            ids = None if sequence_ids is None else tuple(int(i) for i in sequence_ids[start:stop])
            result.append(TeacherBatch(selected.detach().to(device='cpu', copy=True), log_probs,
                                       self.teacher_id, data_role, ids))
        return tuple(result)

    def edited_parameters(self, action: RawAction, scale=1.):
        """Materialize finite edits with the same stored-dtype rounding as commit."""
        matrix_names = {name for name, p in self.parameters.items() if p.ndim == 2}
        vector_names = {name for name, p in self.parameters.items() if p.ndim == 1}
        if set(action.matrices) != matrix_names or set(action.vectors) != vector_names:
            raise ValueError('a globally accounted action must cover every matrix and vector exactly once')
        replacements = {}
        for name, base in self.parameters.items():
            if base.ndim == 2:
                factor = action.matrices[name]
                delta = factor.dense()
                if self.transpose_storage[name]:
                    delta = delta.T
            else:
                delta = action.vectors[name]
            if delta.shape != base.shape:
                raise ValueError(f'action shape differs from stored parameter {name}')
            replacements[name] = (base.float() + scale * delta).to(dtype=base.dtype)
        return replacements

    def student_loss(self, action: RawAction, teacher: TeacherBatch, *, scale=1.):
        return self.student_parameter_loss(self.edited_parameters(action, scale), teacher)

    def _validate_replacements(self, replacements):
        if set(replacements) != set(self.parameters):
            raise ValueError('student replacement mapping must cover every unique parameter exactly once')
        for name, base in self.parameters.items():
            value = replacements[name]
            if not isinstance(value, Tensor) or value.shape != base.shape or value.dtype != base.dtype:
                raise ValueError(f'student replacement shape/dtype differs for {name}')
            if value.device != base.device:
                raise ValueError(f'student replacement device differs for {name}')

    def realized_parameter_norm_change(self, replacements, *, chunk_rows=512):
        """Return (actual squared-norm change, actual squared edit norm).

        Both scalar values accumulate in FP64 after stored-dtype rounding.
        Gradients use first-order cast STE, exactly as the functional GPT path;
        even an erased BF16 edit can therefore have a nonzero norm gradient.
        No GPT computation is performed. ``None`` processes each tensor whole.
        """
        self._validate_replacements(replacements)
        if chunk_rows is not None and (not isinstance(chunk_rows, int) or chunk_rows < 1):
            raise ValueError('chunk_rows must be a positive integer or None')
        norm_change = self.base_weight_norm_sq.new_zeros(())
        edit_sq = self.base_weight_norm_sq.new_zeros(())
        for name, base in self.parameters.items():
            value = replacements[name]
            change, squared = _RealizedParameterNorm.apply(value, base, chunk_rows or len(base))
            norm_change = norm_change + change
            edit_sq = edit_sq + squared
        return norm_change, edit_sq

    def student_parameter_loss(self, replacements, teacher: TeacherBatch, *, return_ce=False):
        """Evaluate exactly these stored-dtype parameters; no edit is rebuilt."""
        if teacher.teacher_id != self.teacher_id:
            raise ValueError('cached teacher probabilities belong to a different frozen checkpoint')
        if teacher.data_role not in ('past_training', 'synthetic', 'fresh_next_training_audit'):
            raise ValueError('validation/test data cannot train the preservation objective')
        if teacher.data_role=='fresh_next_training_audit' and torch.is_grad_enabled():
            raise ValueError('fresh next-batch data is audit-only; gradients are forbidden')
        if len(teacher.inputs) > self.microbatch_sequences:
            raise ValueError('teacher microbatch exceeds configured student microbatch size')
        if teacher.log_probs.dtype != self.kl_dtype:
            raise ValueError('teacher cache dtype differs from configured KL arithmetic')
        self._validate_replacements(replacements)
        inputs = teacher.inputs.to(device=self.device)
        logits = self._forward(replacements, inputs)
        self.compute.student_forwards += 1
        self.compute.student_tokens += inputs.numel()
        loss = chunked_teacher_kl(logits, teacher.log_probs, self.chunk_tokens)
        ce=None
        if return_ce:
            # Cache inputs contain the true next token except at the final
            # position of each sequence. Audit these known targets without a
            # second GPT forward and without feeding CE to Chisel training.
            with torch.no_grad():
                length=inputs.shape[1]
                if length<2:raise ValueError('next-token CE requires sequence length >=2')
                target=inputs.roll(-1,1).flatten();flat=logits.flatten(0,1)
                teacher_flat=teacher.log_probs.flatten(0,1)
                student_total=torch.zeros((),device=logits.device,dtype=torch.float64)
                teacher_total=student_total.clone();logprob_squared=student_total.clone();centered_squared=student_total.clone();count=0
                for start in range(0,len(flat),self.chunk_tokens):
                    stop=min(start+self.chunk_tokens,len(flat))
                    mask=(torch.arange(start,stop,device=logits.device)%length)<length-1
                    labels=target[start:stop,None]
                    student_lp=F.log_softmax(flat[start:stop].double(),dim=-1)
                    teacher_lp=teacher_flat[start:stop].to(logits.device)
                    student_nll=-student_lp.gather(1,labels).squeeze(1)
                    teacher_nll=-teacher_lp.gather(1,labels).squeeze(1)
                    gap=student_lp-teacher_lp
                    logprob_squared+=gap.square().sum()
                    centered_squared+=(gap-gap.mean(-1,keepdim=True)).square().sum()
                    student_total+=student_nll[mask].sum();teacher_total+=teacher_nll[mask].sum();count+=int(mask.sum())
                ce=dict(student_ce=float(student_total/count),teacher_ce=float(teacher_total/count),
                        ce_change=float((student_total-teacher_total)/count),ce_tokens=count,
                        logprob_rms_shift=float((logprob_squared/logits.numel()).sqrt()),
                        centered_logit_rms_shift=float((centered_squared/logits.numel()).sqrt()),
                        ce_scope='known next tokens within cached inputs; final token of each sequence excluded')
        if loss.requires_grad:
            def count_backward(cotangent):
                self.compute.student_backwards += 1
                self.compute.student_backward_tokens += inputs.numel()
            loss.register_hook(count_backward)
        return (loss,ce) if return_ce else loss
