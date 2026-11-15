"""Experimental top-k ranking loss with exact compact membership lookup.

One bit per vocabulary entry replaces K comparisons per entry. Dense outsider
cotangents and sparse top-k cotangents are written separately; indices from the
same torch.topk are unique, so their writes do not overlap. Integer atomic OR
is order-independent. Only first derivatives are implemented.
"""
import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


def _cpu_membership_bitset(indices, vocabulary):
    mask = torch.zeros((len(indices), (vocabulary+31)//32), dtype=torch.int32, device=indices.device)
    rows = torch.arange(len(indices), device=indices.device)
    for column in range(indices.shape[1]):
        index = indices[:, column]
        word = index//32
        bit = (torch.ones_like(index) << (index % 32)).to(torch.int32)
        mask[rows, word] = mask[rows, word] | bit
    return mask


def _cpu_membership_lookup(mask, vocabulary):
    columns = torch.arange(vocabulary, device=mask.device)
    words = mask[:, columns//32].long()
    return ((words >> (columns % 32)) & 1).bool()


if triton is not None:
    @triton.jit
    def _mark_membership(Indices, Mask, K: tl.constexpr, WORDS: tl.constexpr, BK: tl.constexpr):
        row = tl.program_id(0)
        j = tl.arange(0, BK)
        index = tl.load(Indices+row*K+j, j<K, other=0)
        bit = tl.full((BK,), 1, tl.uint32) << (index % 32).to(tl.uint32)
        tl.atomic_or(Mask+row*WORDS+index//32, bit.to(tl.int32), mask=j<K, sem='relaxed')

    @triton.jit
    def _forward(T, S, Values, Indices, Mask, Loss, Hinges, Member, Outside, Counts,
                 V: tl.constexpr, K: tl.constexpr, WORDS: tl.constexpr, A: tl.constexpr,
                 B: tl.constexpr, BK: tl.constexpr, FP64: tl.constexpr):
        dtype = tl.float64 if FP64 else tl.float32
        row = tl.program_id(0)
        j = tl.arange(0, BK)
        ids = tl.load(Indices+row*K+j, j<K, other=0)
        values = tl.load(Values+row*K+j, j<K, other=0.).to(dtype)
        next_ids = tl.load(Indices+row*K+j+1, j<K-1, other=0)
        next_values = tl.load(Values+row*K+j+1, j<K-1, other=0.).to(dtype)
        edited = tl.load(S+row*V+ids).to(dtype)
        edited_next = tl.load(S+row*V+next_ids).to(dtype)
        hinges = tl.where(j<K-1, tl.maximum(A*(values-next_values)-(edited-edited_next), 0.), 0.)
        tl.store(Hinges+row*K+j, hinges, j<K)
        boundary_id = tl.load(Indices+row*K+K-1)
        boundary_t = tl.load(Values+row*K+K-1).to(dtype)
        boundary = tl.load(S+row*V+boundary_id).to(dtype)-A*boundary_t
        columns = tl.arange(0, B)
        best = tl.full((B,), float('-inf'), dtype)
        count = tl.zeros((B,), tl.int32)
        for start in range(0, V, B):
            c = start+columns
            valid = c<V
            word = tl.load(Mask+row*WORDS+c//32, valid, other=0).to(tl.uint32)
            inside = ((word >> (c % 32).to(tl.uint32)) & 1) != 0
            t = tl.load(T+row*V+c, valid, other=0.).to(dtype)
            s = tl.load(S+row*V+c, valid, other=0.).to(dtype)
            adjusted = s-A*t
            eligible = valid & (~inside)
            z = tl.where(eligible, adjusted, float('-inf'))
            count = tl.where(z>best, eligible.to(tl.int32), count+(eligible & (z==best)).to(tl.int32))
            best = tl.maximum(best, z)
        maximum = tl.max(best, 0)
        ties = tl.sum(tl.where(best==maximum, count, 0), 0)
        member = tl.maximum(maximum-boundary, 0.)
        tl.store(Outside+row, maximum)
        tl.store(Counts+row, ties)
        tl.store(Member+row, member)
        tl.store(Loss+row, (tl.sum(hinges*hinges, 0)+member*member)/K)

    @triton.jit
    def _outside_backward(T, S, Mask, Member, Outside, Counts, Cotangent, Gradient,
                          V: tl.constexpr, K: tl.constexpr, WORDS: tl.constexpr,
                          A: tl.constexpr, B: tl.constexpr, FP64: tl.constexpr):
        dtype = tl.float64 if FP64 else tl.float32
        row = tl.program_id(0)
        columns = tl.arange(0, B)
        scale = tl.load(Cotangent+row)/K
        value = ((2.*tl.load(Member+row))*scale)/tl.maximum(tl.load(Counts+row), 1)
        maximum = tl.load(Outside+row)
        for start in range(0, V, B):
            c = start+columns
            valid = c<V
            word = tl.load(Mask+row*WORDS+c//32, valid, other=0).to(tl.uint32)
            inside = ((word >> (c % 32).to(tl.uint32)) & 1) != 0
            t = tl.load(T+row*V+c, valid, other=0.).to(dtype)
            s = tl.load(S+row*V+c, valid, other=0.).to(dtype)
            gradient = tl.where((~inside) & ((s-A*t)==maximum), value, 0.)
            tl.store(Gradient+row*V+c, gradient, valid)

    @triton.jit
    def _topk_backward(Indices, Hinges, Member, Cotangent, Gradient,
                       V: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
        row = tl.program_id(0)
        j = tl.arange(0, BK)
        indices = tl.load(Indices+row*K+j, j<K, other=0)
        current = tl.load(Hinges+row*K+j, j<K, other=0.)
        previous = tl.load(Hinges+row*K+j-1, (j>0) & (j<K), other=0.)
        scale = tl.load(Cotangent+row)/K
        gradient = (2.*previous)*scale-(2.*current)*scale
        gradient = tl.where(j==K-1, gradient-(2.*tl.load(Member+row))*scale, gradient)
        tl.store(Gradient+row*V+indices, gradient, j<K)


class _BitsetRank(torch.autograd.Function):
    @staticmethod
    def forward(ctx, teacher, student, k, alpha, fp64, return_topk):
        t = teacher.detach().reshape(-1, teacher.shape[-1]).contiguous()
        s = student.reshape_as(t).contiguous()
        values, indices = t.topk(k, dim=-1, sorted=True)
        n, v = t.shape
        dtype = torch.float64 if fp64 else s.dtype
        if s.is_cuda:
            mask = torch.zeros((n, (v+31)//32), dtype=torch.int32, device=s.device)
            loss = torch.empty(n, dtype=dtype, device=s.device)
            hinges = torch.empty((n, k), dtype=dtype, device=s.device)
            member, outside = torch.empty_like(loss), torch.empty_like(loss)
            counts = torch.empty(n, dtype=torch.int32, device=s.device)
            _mark_membership[(n,)](indices, mask, k, mask.shape[1], triton.next_power_of_2(k), num_warps=4)
            _forward[(n,)](t, s, values, indices, mask, loss, hinges, member, outside, counts,
                v, k, mask.shape[1], alpha, 1024, triton.next_power_of_2(k), fp64,
                num_warps=4, enable_fp_fusion=False)
        else:
            mask = _cpu_membership_bitset(indices, v)
            inside = _cpu_membership_lookup(mask, v)
            teacher_values, edited = values.to(dtype), s.to(dtype).gather(1, indices)
            adjacent = (alpha*(teacher_values[:, :-1]-teacher_values[:, 1:])
                        -(edited[:, :-1]-edited[:, 1:])).relu()
            hinges = torch.cat((adjacent, torch.zeros((n, 1), dtype=dtype, device=s.device)), 1)
            adjusted = s.to(dtype)-alpha*t.to(dtype)
            outside = adjusted.masked_fill(inside, -torch.inf).amax(1)
            counts = ((adjusted==outside[:, None]) & (~inside)).sum(1).to(torch.int32)
            member = (outside-(edited[:, -1]-alpha*teacher_values[:, -1])).relu()
            loss = (hinges.square().sum(1)+member.square())/k
        ctx.save_for_backward(t, s, indices, mask, hinges, member, outside, counts)
        ctx.k, ctx.alpha, ctx.fp64, ctx.shape = k, alpha, fp64, student.shape
        if return_topk:
            ctx.mark_non_differentiable(values, indices)
            return loss.reshape(student.shape[:-1]), values, indices
        return loss.reshape(student.shape[:-1])

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent, *unused):
        t, s, indices, mask, hinges, member, outside, counts = ctx.saved_tensors
        cotangent = cotangent.contiguous().reshape(-1)
        if s.is_cuda:
            gradient = torch.empty_like(s)
            _outside_backward[(len(s),)](t, s, mask, member, outside, counts, cotangent, gradient,
                s.shape[1], ctx.k, mask.shape[1], ctx.alpha, 1024, ctx.fp64,
                num_warps=4, enable_fp_fusion=False)
            _topk_backward[(len(s),)](indices, hinges, member, cotangent, gradient,
                s.shape[1], ctx.k, triton.next_power_of_2(ctx.k), num_warps=4, enable_fp_fusion=False)
        else:
            dtype = torch.float64 if ctx.fp64 else s.dtype
            scale = cotangent.to(dtype)/ctx.k
            membership = (2.*member)*scale
            inside = _cpu_membership_lookup(mask, s.shape[1])
            adjusted = s.to(dtype)-ctx.alpha*t.to(dtype)
            gradient = torch.where((~inside) & (adjusted==outside[:, None]),
                                   (membership/counts.clamp_min(1))[:, None], 0.)
            previous = torch.cat((torch.zeros_like(hinges[:, :1]), hinges[:, :-1]), 1)
            selected = (2.*previous)*scale[:, None]-(2.*hinges)*scale[:, None]
            selected[:, -1] -= membership
            gradient.scatter_(1, indices, selected)
            gradient = gradient.to(s.dtype)
        return None, gradient.reshape(ctx.shape), None, None, None, None


def bitset_topk_margin_loss(teacher, student, *, k=100, margin_fraction=.5,
                            arithmetic='fp64', return_teacher_topk=False):
    """Default-off primitive with the dense reference's exact membership rule."""
    if teacher.shape != student.shape or teacher.ndim < 2 or not student.numel():
        raise ValueError('matching nonempty token/vocabulary shapes required')
    if type(k) is not int or not 1 <= k <= teacher.shape[-1]:
        raise ValueError('integer k within vocabulary required')
    if not 0 < margin_fraction <= 1 or arithmetic not in ('fp32', 'fp64'):
        raise ValueError('valid margin fraction and arithmetic required')
    if teacher.device != student.device:
        raise ValueError('teacher/student device mismatch')
    if student.is_cuda:
        if triton is None: raise RuntimeError('Triton required for CUDA bitset ranking')
        if student.dtype != torch.float32 or teacher.dtype not in (torch.float32, torch.float64):
            raise ValueError('CUDA ranking requires FP32 student and FP32/FP64 teacher')
        if arithmetic == 'fp32' and teacher.dtype != torch.float32:
            raise ValueError('FP64 teacher requires FP64 arithmetic')
    result = _BitsetRank.apply(teacher, student, k, float(margin_fraction), arithmetic=='fp64', return_teacher_topk)
    return (result[0], (result[1], result[2])) if return_teacher_topk else result
