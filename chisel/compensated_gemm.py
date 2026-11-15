"""Three-product TF32 backward GEMM, ported from archive/research-20261115.

Source: rewriter/kernels/compensated_dx_triton.py. Never used for GPT forwards.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(configs=[
    triton.Config({'BM': 64, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 64, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=3),
    triton.Config({'BM': 64, 'BN': 64, 'BK': 64}, num_warps=4, num_stages=3),
], key=['M', 'N', 'K', 'AS0', 'AS1', 'BS0', 'BS1'])
@triton.jit
def _gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          AS0: tl.constexpr, AS1: tl.constexpr, BS0: tl.constexpr, BS1: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Group neighbouring M tiles to share B cache lines.
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    group = pid // (8 * nn)
    first_m = group * 8
    group_m = tl.minimum(nm - first_m, 8)
    pm = first_m + (pid % (8 * nn)) % group_m
    pn = (pid % (8 * nn)) // group_m
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + ks
        a = tl.load(A + rows[:, None] * AS0 + k[None, :] * AS1,
                    (rows[:, None] < M) & (k[None, :] < K), other=0.)
        b = tl.load(B + k[:, None] * BS0 + cols[None, :] * BS1,
                    (k[:, None] < K) & (cols[None, :] < N), other=0.)
        acc = tl.dot(a, b, acc, input_precision='tf32x3')
    tl.store(C + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


def launch(a, b):
    m, k = a.shape
    n = b.shape[1]
    result = torch.empty((m, n), device=a.device, dtype=torch.float32)
    with torch.cuda.device(a.device):
        _gemm[lambda meta: (triton.cdiv(m, meta['BM']) * triton.cdiv(n, meta['BN']),)](
            a, b, result, m, n, k, *a.stride(), *b.stride())
    return result
