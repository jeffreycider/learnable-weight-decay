"""Tiled FP64 teacher-KL without materialized log-probabilities.

Production (rewriter.function_preservation._ChunkedTeacherKL plus
rewriter.teacher_cache.TeacherPanel.teacher_batch) performs, per GPT
microbatch of 16x1024 tokens over the 50304 vocabulary:
  teacher : head GEMM -> softcap -> FP64 log_softmax materialized (6.6 GB written)
  forward : per 128-token chunk, FP32->FP64 copy, FP64 log_softmax, exp, sub,
            mul, sum (~110 bytes of traffic per logit element)
  backward: the same FP64 log_softmax again plus exp and the gradient (~90 B/elem)
The kernel-node profile (runs/megakernel-fp32-v3/b16-v1/thin_ieee-profile.txt)
puts these passes at roughly a fifth of the microbatch's GPU time, and the
earlier whole-row fused kernel (rewriter.accelerated_kl) holds a 65536-wide FP64
row per program, which spills.

This module keeps teacher targets as FP32 softcapped logits and streams every
row in column tiles with FP64 accumulators:
  pass 1  row max (exact, FP32), student and teacher      reads 4+4 B/elem
  pass 2  log-sum-exp given the max (FP64), both           reads 4+4 B/elem
  pass 3  loss and teacher mass (FP64)                     reads 8 B/elem
  bwd     gradient (p*mass - q) * cotangent/N              reads 8 B/elem, writes 4 B/elem
  => ~36 B/logit in total (Astra's count) versus ~200 B/logit for the chunked path.
Kernels launch with enable_fp_fusion=False so no FMA contraction changes the
reference's rounding of p*mass - q or (t - max) - log(sum).
Formulas follow the reference exactly: logq = (t - max_t) - log(sum exp(t - max_t)),
q = exp(logq), loss = sum q (logq - logp) / N, gradient = (p * sum q - q) * c / N
cast to FP32 after the FP64 multiply. Only the FP64 summation order differs
from the chunked reference, so agreement is at the 1e-14 level, not bit-level;
it must be qualified like the fused kernel was (value, gradient, then a
measured update), never assumed.

``tiled_teacher_kl`` dispatches to Triton on CUDA and to an equivalent tiled
torch implementation elsewhere (used by the CPU tests and as the semantic
reference for the kernel). Teacher logits must be the softcapped FP32 head
outputs; pass FP64 log-probabilities to the production path instead.
"""
from __future__ import annotations
import math
import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice
except ImportError:  # pragma: no cover - CPU boxes
    triton = None

TILE = 2048


if triton is not None:
    @triton.jit
    def _cap(z):
        # Same FP32 operation sequence as 15*z*(z.square()+225).rsqrt()
        return (15. * z) * libdevice.rsqrt(z * z + 225.)

    @triton.jit
    def _cap_derivative(z):
        # d/dz [15 z (z^2+225)^(-1/2)] = 3375 (z^2+225)^(-3/2)
        r = libdevice.rsqrt(z * z + 225.)
        return 3375. * r * r * r

    @triton.jit
    def _row_max(X, Out, V: tl.constexpr, BLOCK: tl.constexpr, SOFTCAP: tl.constexpr):
        row = tl.program_id(0)
        running = tl.full((BLOCK,), float('-inf'), tl.float32)
        for start in range(0, V, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            x = tl.load(X + row * V + cols, cols < V, other=float('-inf'))
            if SOFTCAP:
                x = tl.where(cols < V, _cap(x), float('-inf'))
            running = tl.maximum(running, x)
        tl.store(Out + row, tl.max(running, 0))

    @triton.jit
    def _row_logsum(X, Max, Out, V: tl.constexpr, BLOCK: tl.constexpr, SOFTCAP: tl.constexpr):
        row = tl.program_id(0)
        m = tl.load(Max + row).to(tl.float64)
        acc = tl.zeros((BLOCK,), tl.float64)
        for start in range(0, V, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            valid = cols < V
            x = tl.load(X + row * V + cols, valid, other=0.)
            if SOFTCAP:
                x = _cap(x)
            x = x.to(tl.float64)
            acc += tl.where(valid, libdevice.exp(x - m), 0.)
        tl.store(Out + row, libdevice.log(tl.sum(acc, 0)))

    @triton.jit
    def _kl_loss(S, T, SMax, SLog, TMax, TLog, Loss, Mass, V: tl.constexpr, BLOCK: tl.constexpr, SOFTCAP: tl.constexpr, LOG_PROBS: tl.constexpr):
        row = tl.program_id(0)
        sm = tl.load(SMax + row).to(tl.float64)
        sl = tl.load(SLog + row)
        tm = tl.full((), 0., tl.float64) if LOG_PROBS else tl.load(TMax + row).to(tl.float64)
        tlg = tl.full((), 0., tl.float64) if LOG_PROBS else tl.load(TLog + row)
        loss = tl.zeros((BLOCK,), tl.float64)
        mass = tl.zeros((BLOCK,), tl.float64)
        for start in range(0, V, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            valid = cols < V
            s = tl.load(S + row * V + cols, valid, other=0.)
            t = tl.load(T + row * V + cols, valid, other=0.)
            if SOFTCAP:
                s = _cap(s)
                t = _cap(t)
            s = s.to(tl.float64)
            t = t.to(tl.float64)
            logp = (s - sm) - sl
            logq = t if LOG_PROBS else (t - tm) - tlg
            q = tl.where(valid, libdevice.exp(logq), 0.)
            loss += tl.where(valid, q * (logq - logp), 0.)
            mass += q
        tl.store(Loss + row, tl.sum(loss, 0))
        tl.store(Mass + row, tl.sum(mass, 0))

    @triton.jit
    def _kl_grad(S, T, SMax, SLog, TMax, TLog, Mass, Scale, Grad, V: tl.constexpr, BLOCK: tl.constexpr, SOFTCAP: tl.constexpr, ROW_SCALE: tl.constexpr, LOG_PROBS: tl.constexpr):
        row = tl.program_id(0)
        sm = tl.load(SMax + row).to(tl.float64)
        sl = tl.load(SLog + row)
        tm = tl.full((), 0., tl.float64) if LOG_PROBS else tl.load(TMax + row).to(tl.float64)
        tlg = tl.full((), 0., tl.float64) if LOG_PROBS else tl.load(TLog + row)
        mass = tl.load(Mass + row)
        scale = tl.load(Scale + row) if ROW_SCALE else tl.load(Scale)
        for start in range(0, V, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            valid = cols < V
            zs = tl.load(S + row * V + cols, valid, other=0.)
            zt = tl.load(T + row * V + cols, valid, other=0.)
            if SOFTCAP:
                s = _cap(zs).to(tl.float64)
                t = _cap(zt).to(tl.float64)
            else:
                s = zs.to(tl.float64)
                t = zt.to(tl.float64)
            p = libdevice.exp((s - sm) - sl)
            q = libdevice.exp(t if LOG_PROBS else (t - tm) - tlg)
            value = ((p * mass - q) * scale).to(tl.float32)
            if SOFTCAP:
                value = value * _cap_derivative(zs)
            tl.store(Grad + row * V + cols, value, valid)


def _warps(block):
    return 4 if block <= 2048 else 8


def _triton_stats(x, block, softcap):
    rows, vocab = x.shape
    row_max = torch.empty(rows, dtype=torch.float32, device=x.device)
    row_log = torch.empty(rows, dtype=torch.float64, device=x.device)
    _row_max[(rows,)](x, row_max, vocab, block, softcap, num_warps=_warps(block), enable_fp_fusion=False)
    _row_logsum[(rows,)](x, row_max, row_log, vocab, block, softcap, num_warps=_warps(block), enable_fp_fusion=False)
    return row_max, row_log


def _softcap(z):
    return 15 * z * (z.square() + 225).rsqrt()


def _softcap_derivative(z):
    r = (z.square() + 225).rsqrt()
    return 3375. * r * r * r


# ---- tiled torch implementation (CPU reference and fallback) -------------

def _torch_stats(x, tile, softcap):
    cap = _softcap if softcap else (lambda z: z)
    row_max = torch.full((x.shape[0],), float('-inf'), dtype=torch.float32, device=x.device)
    for start in range(0, x.shape[1], tile):
        row_max = torch.maximum(row_max, cap(x[:, start:start + tile]).amax(-1))
    m = row_max.double()[:, None]
    acc = torch.zeros(x.shape[0], dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[1], tile):
        acc += (cap(x[:, start:start + tile]).double() - m).exp().sum(-1)
    return row_max, acc.log()


def _torch_loss(student, teacher, sm, sl, tm, tlg, tile, softcap, log_probs=False):
    cap = _softcap if softcap else (lambda z: z)
    loss = torch.zeros(student.shape[0], dtype=torch.float64, device=student.device)
    mass = torch.zeros_like(loss)
    sm, tm = sm.double()[:, None], tm.double()[:, None]
    sl, tlg = sl[:, None], tlg[:, None]
    for start in range(0, student.shape[1], tile):
        logp = (cap(student[:, start:start + tile]).double() - sm) - sl
        logq = teacher[:, start:start + tile] if log_probs else (cap(teacher[:, start:start + tile]).double() - tm) - tlg
        q = logq.exp()
        loss += (q * (logq - logp)).sum(-1)
        mass += q.sum(-1)
    return loss, mass


def _torch_grad(student, teacher, sm, sl, tm, tlg, mass, scale, tile, softcap, log_probs=False):
    cap = _softcap if softcap else (lambda z: z)
    if scale.numel() != 1:scale = scale.reshape(-1, 1)
    grad = torch.empty_like(student)
    sm, tm = sm.double()[:, None], tm.double()[:, None]
    sl, tlg, mass = sl[:, None], tlg[:, None], mass[:, None]
    for start in range(0, student.shape[1], tile):
        zs = student[:, start:start + tile]
        p = ((cap(zs).double() - sm) - sl).exp()
        q = (teacher[:, start:start + tile] if log_probs else (cap(teacher[:, start:start + tile]).double() - tm) - tlg).exp()
        value = ((p * mass - q) * scale).to(torch.float32)
        if softcap:
            value = value * _softcap_derivative(zs)
        grad[:, start:start + tile] = value
    return grad


class _TiledTeacherKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student_logits, teacher_logits, tile, softcap, reduction, teacher_is_log_probs):
        if reduction not in ("mean", "none"):
            raise ValueError("reduction must be mean or none")
        if student_logits.shape != teacher_logits.shape or student_logits.ndim < 2:
            raise ValueError('teacher and student logits require identical nonempty shapes')
        teacher_dtype = torch.float64 if teacher_is_log_probs else torch.float32
        if student_logits.dtype != torch.float32 or teacher_logits.dtype != teacher_dtype:
            raise ValueError('tiled KL requires FP32 student and FP32 logits or explicitly selected FP64 log-prob teacher')
        if teacher_is_log_probs and softcap:
            raise ValueError('log-prob teacher mode requires already softcapped student logits')
        if teacher_logits.device != student_logits.device:
            raise ValueError('student and teacher logits must share a device')
        if tile < 1 or tile & (tile - 1):
            raise ValueError('tile must be a positive power of two')
        student = student_logits.reshape(-1, student_logits.shape[-1]).contiguous()
        teacher = teacher_logits.detach().reshape_as(student).contiguous()
        if not student.shape[0] or not student.shape[1]:
            raise ValueError('empty preservation batch')
        use_triton = triton is not None and student.is_cuda
        softcap = bool(softcap)
        if use_triton:
            sm, sl = _triton_stats(student, tile, softcap)
            if teacher_is_log_probs:
                tm=torch.empty(student.shape[0],device=student.device,dtype=torch.float32)
                tlg=torch.empty(student.shape[0],device=student.device,dtype=torch.float64)
            else:tm, tlg = _triton_stats(teacher, tile, softcap)
            rows = student.shape[0]
            loss = torch.empty(rows, dtype=torch.float64, device=student.device)
            mass = torch.empty_like(loss)
            _kl_loss[(rows,)](student, teacher, sm, sl, tm, tlg, loss, mass, student.shape[1], tile, softcap, teacher_is_log_probs, num_warps=_warps(tile), enable_fp_fusion=False)
        else:
            sm, sl = _torch_stats(student, tile, softcap)
            if teacher_is_log_probs:
                tm=torch.zeros_like(sm);tlg=torch.zeros_like(sl)
            else:tm, tlg = _torch_stats(teacher, tile, softcap)
            loss, mass = _torch_loss(student, teacher, sm, sl, tm, tlg, tile, softcap, teacher_is_log_probs)
        ctx.save_for_backward(student, teacher, sm, sl, tm, tlg, mass)
        ctx.tile = tile
        ctx.softcap = softcap
        ctx.shape = student_logits.shape
        ctx.use_triton = use_triton
        ctx.reduction = reduction
        ctx.teacher_is_log_probs = bool(teacher_is_log_probs)
        return loss.reshape(student_logits.shape[:-1]) if reduction == "none" else loss.sum() / student.shape[0]

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        student, teacher, sm, sl, tm, tlg, mass = ctx.saved_tensors
        # Same FP64 scalar as the reference: cotangent / token count, applied
        # in FP64 before the single cast to FP32.
        if ctx.reduction == "none":
            scale = cotangent.detach().to(torch.float64).reshape(-1).contiguous()
        else:
            scale = (cotangent.detach().to(torch.float64).reshape(()) / student.shape[0]).reshape(1).contiguous()
        if ctx.use_triton:
            grad = torch.empty_like(student)
            _kl_grad[(student.shape[0],)](student, teacher, sm, sl, tm, tlg, mass, scale, grad,
                                          student.shape[1], ctx.tile, ctx.softcap, ctx.reduction == "none", ctx.teacher_is_log_probs, num_warps=_warps(ctx.tile), enable_fp_fusion=False)
        else:
            grad = _torch_grad(student, teacher, sm, sl, tm, tlg, mass, scale, ctx.tile, ctx.softcap, ctx.teacher_is_log_probs)
        return grad.reshape(ctx.shape), None, None, None, None, None


def tiled_teacher_kl(student_logits, teacher_logits, tile=TILE, *, softcap=False, reduction="mean", teacher_is_log_probs=False):
    """Mean KL(teacher || student) in nats/token.

    softcap=False (phase 1): inputs are FP32 softcapped logits; equivalent to
    ``chunked_teacher_kl(student, log_softmax(teacher.double()))`` up to FP64
    summation order.
    softcap=True (phase 2, APPROXIMATE ARITHMETIC): inputs are the RAW FP32 head
    outputs; the 15 z (z^2+225)^(-1/2) softcap is applied inside the kernels and
    its closed-form derivative 3375 (z^2+225)^(-3/2) multiplies the FP32
    cotangent in backward. Forward values follow the same FP32 operation
    sequence as the module, but the backward rounding differs from autograd's
    mul/rsqrt chain, so this arm needs gradient-tolerance qualification like
    any approximate arm. It removes ~8 full-vocabulary FP32 passes per
    microbatch (softcap forward/backward for student and teacher).
    reduction="none" returns the FP64 per-token loss, with shape logits.shape[:-1].
    Its backward accepts arbitrary per-token cotangents; callers can form tail
    objectives without another GPT forward. No tail objective is selected here.
    reduction="mean" preserves the original scalar loss and normalization.
    teacher_is_log_probs=True accepts the existing exact FP64 teacher log-probs
    without casting or reconstructing targets; student logits remain FP32.
    First-order gradients only.
    """
    return _TiledTeacherKL.apply(student_logits, teacher_logits, tile, softcap, reduction, teacher_is_log_probs)


def raw_teacher_head(engine, features):
    """Head GEMM without the softcap, for the softcap=True arm (same F.linear as head_logits)."""
    import torch.nn.functional as F
    features = features.to(engine.device)
    return F.linear(features, engine.parameters['proj.weight'].to(features.dtype),
                    engine.parameters['proj.bias'].to(features.dtype)).float()


def validate_panel_for_tiled(panel, engine):
    """The checks TeacherPanel.teacher_batch performs in Python, run once per
    state OUTSIDE the hot path: precision metadata, verified-checkpoint binding,
    and frozen-head identity/hash. Raises exactly like the production method."""
    from .teacher_cache import validate_precision, _head_signature, tensor_hash
    validate_precision(panel.metadata, engine)
    if getattr(engine, '_teacher_cache_binding', None) != (engine.teacher_id, panel.metadata['state_sha256']):
        raise ValueError('cache must be bound to currently loaded verified checkpoint')
    signature = getattr(engine, '_teacher_cache_head_signature', None)
    if signature is not None and signature != _head_signature(engine):
        raise ValueError('bound frozen head tensor identity/version changed')
    if getattr(engine, '_teacher_cache_checked_head', None) != (engine.teacher_id, panel.metadata['state_sha256']):
        for name, key in [('proj.weight', 'head_weight_sha256'), ('proj.bias', 'head_bias_sha256')]:
            if tensor_hash(engine.parameters[name]) != panel.metadata[key]:
                raise ValueError('frozen head differs from cached checkpoint')
        engine._teacher_cache_checked_head = (engine.teacher_id, panel.metadata['state_sha256'])
        engine._teacher_cache_head_signature = _head_signature(engine)


def teacher_logits_from_features(engine, features):
    """Exact head reconstruction used by TeacherPanel.teacher_batch, without the FP64 log-softmax."""
    from .teacher_cache import head_logits
    return head_logits(engine, features)
