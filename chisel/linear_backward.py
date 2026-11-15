"""Optional compensated tensor-core gradients with native FP32 linear forwards."""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


class CompensatedBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias):
        if not inputs.is_cuda or inputs.dtype != torch.float32 or weight.dtype != torch.float32:
            raise ValueError('compensated linear backward requires CUDA FP32 operands')
        ctx.save_for_backward(inputs, weight)
        return F.linear(inputs, weight, bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        from chisel.compensated_gemm import launch
        inputs, weight = ctx.saved_tensors
        flat = gradient.reshape(-1, gradient.shape[-1])
        dx = launch(flat, weight).reshape(inputs.shape) if ctx.needs_input_grad[0] else None
        dw = launch(flat.T, inputs.reshape(-1, inputs.shape[-1])) if ctx.needs_input_grad[1] else None
        db = flat.sum(0) if ctx.needs_input_grad[2] else None
        return dx, dw, db


def linear(inputs, weight, bias):
    return CompensatedBackward.apply(inputs, weight, bias)
