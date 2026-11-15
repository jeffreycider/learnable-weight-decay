"""CUDA fused finite KL with FP64 reductions and first-order gradients.

One Triton program per token fuses student log-softmax, teacher exponential,
KL, and the logits derivative. A separate scalar reduction and a backward
cotangent multiply remain: this is explicitly not a whole-training megakernel.
No probabilities or logits cross the CPU during this operation.
"""
import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _kl_rows(Student, Teacher, Loss, Gradient, V: tl.constexpr,
                 N: tl.constexpr, DOUBLE: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        column = tl.arange(0, BLOCK)
        valid = column < V
        dtype = tl.float64 if DOUBLE else tl.float32
        s = tl.load(Student + row * V + column, valid, other=-float('inf')).to(dtype)
        qlog = tl.load(Teacher + row * V + column, valid, other=0).to(dtype)
        centered = s - tl.max(s, 0)
        if DOUBLE:
            unnormalized = libdevice.exp(centered)
            normalizer = tl.sum(unnormalized, 0)
            log_normalizer = libdevice.log(normalizer)
            q = tl.where(valid, libdevice.exp(qlog), 0.)
        else:
            unnormalized = tl.exp(centered)
            normalizer = tl.sum(unnormalized, 0)
            log_normalizer = tl.log(normalizer)
            q = tl.where(valid, tl.exp(qlog), 0.)
        logp = centered - log_normalizer
        p = unnormalized / normalizer
        mass = tl.sum(q, 0)
        loss = tl.sum(tl.where(valid, q * (qlog - logp), 0.), 0) / N
        gradient = (p * mass - q) / N
        tl.store(Loss + row, loss)
        tl.store(Gradient + row * V + column, gradient, valid)

    @triton.jit
    def _scale_gradient(Gradient, Cotangent, Output, N: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        gradient = tl.load(Gradient + index, index < N, other=0)
        cotangent = tl.load(Cotangent)
        # Match FP64 cotangent multiplication before the reference's FP32 cast.
        value = gradient.to(cotangent.dtype) * cotangent
        tl.store(Output + index, value, index < N)


class _FusedTeacherKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student_logits, teacher_log_probs):
        if triton is None:
            raise RuntimeError('fused_teacher_kl requires Triton on CUDA')
        if student_logits.device.type != 'cuda' or teacher_log_probs.device != student_logits.device:
            raise ValueError('fused KL requires student and teacher on the same CUDA device')
        if student_logits.shape != teacher_log_probs.shape or student_logits.ndim < 2:
            raise ValueError('teacher and student require identical nonempty token/vocabulary shapes')
        if student_logits.dtype != torch.float32 or teacher_log_probs.dtype not in (torch.float32, torch.float64):
            raise ValueError('fused KL requires FP32 student and FP32/FP64 teacher log probabilities')
        vocab = student_logits.shape[-1]
        if not student_logits.numel() or not vocab:
            raise ValueError('empty preservation batch')
        student, teacher = student_logits.contiguous(), teacher_log_probs.detach().contiguous()
        tokens = student.numel() // vocab
        row_loss = torch.empty(tokens, dtype=teacher.dtype, device=student.device)
        # Keep the derivative in the arithmetic dtype until the backward
        # cotangent is applied, exactly matching the reference cast boundary.
        gradient = torch.empty_like(student, dtype=teacher.dtype)
        block = triton.next_power_of_2(vocab)
        warps = 4 if block <= 2048 else 8 if block <= 8192 else 16 if block <= 32768 else 32
        _kl_rows[(tokens,)](student, teacher, row_loss, gradient, vocab, tokens,
                           teacher.dtype == torch.float64, block, num_warps=warps,
                           enable_fp_fusion=False)
        ctx.save_for_backward(gradient)
        ctx.input_dtype = student.dtype
        return row_loss.sum()

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        gradient, = ctx.saved_tensors
        result = torch.empty_like(gradient, dtype=ctx.input_dtype)
        _scale_gradient[(triton.cdiv(gradient.numel(), 1024),)](
            gradient, cotangent, result, gradient.numel(), 1024)
        return result, None


def fused_teacher_kl(student_logits, teacher_log_probs):
    """Mean KL(teacher || student), with the reference's teacher-mass gradient.

    Teacher log probabilities must already be on the student's GPU. FP64
    teacher values must originate from FP64 log-softmax, not a cast of FP32.
    Higher-order differentiation is unsupported, as in chunked_teacher_kl.
    """
    return _FusedTeacherKL.apply(student_logits, teacher_log_probs)
