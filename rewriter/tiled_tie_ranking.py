"""Default-off exact-teacher-tie ranking primitive.

Selected score-class extrema use segmented scans, not K-by-K masks. Outsiders
are selected by one teacher-threshold comparison. Boundary cutoff selection
deliberately retains torch.topk on an FP64 masked vocabulary buffer, including
its choice of subgradient at an exactly tied student cutoff. That temporary
cost must be measured before production integration. First derivatives only.
"""
import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _min_pair(ak, av, bk, bv):
        return bk, tl.where(ak == bk, tl.minimum(av, bv), bv)

    @triton.jit
    def _max_pair(ak, av, bk, bv):
        return bk, tl.where(ak == bk, tl.maximum(av, bv), bv)

    @triton.jit
    def _sum_pair(ak, av, bk, bv):
        return bk, tl.where(ak == bk, av+bv, bv)

    @triton.jit
    def _class_min(keys, values, BK: tl.constexpr):
        reverse = BK-1-tl.arange(0, BK)
        _, prefix = tl.associative_scan((keys, values), 0, _min_pair)
        _, suffix = tl.associative_scan((tl.gather(keys, reverse, 0), tl.gather(values, reverse, 0)), 0, _min_pair)
        return tl.minimum(prefix, tl.gather(suffix, reverse, 0))

    @triton.jit
    def _class_max(keys, values, BK: tl.constexpr):
        reverse = BK-1-tl.arange(0, BK)
        _, prefix = tl.associative_scan((keys, values), 0, _max_pair)
        _, suffix = tl.associative_scan((tl.gather(keys, reverse, 0), tl.gather(values, reverse, 0)), 0, _max_pair)
        return tl.maximum(prefix, tl.gather(suffix, reverse, 0))

    @triton.jit
    def _class_sum(keys, values, BK: tl.constexpr):
        reverse = BK-1-tl.arange(0, BK)
        _, prefix = tl.associative_scan((keys, values), 0, _sum_pair)
        _, suffix = tl.associative_scan((tl.gather(keys, reverse, 0), tl.gather(values, reverse, 0)), 0, _sum_pair)
        return prefix+tl.gather(suffix, reverse, 0)-values

    @triton.jit
    def _boundary_scores(T, S, Values, Scores, V: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
        row = tl.program_id(0)
        columns = tl.program_id(1)*B+tl.arange(0, B)
        threshold = tl.load(Values+row*K+K-1)
        teacher = tl.load(T+row*V+columns, columns<V, other=float('-inf'))
        student = tl.load(S+row*V+columns, columns<V, other=0.).to(tl.float64)
        tl.store(Scores+row*V+columns, tl.where(teacher==threshold, student, float('-inf')), columns<V)

    @triton.jit
    def _forward(T, S, Values, Indices, BoundaryMax, Cutoff, Loss, Hinges, Member,
                 Outside, OutsideCount, BoundaryCount, BoundaryHinge,
                 V: tl.constexpr, K: tl.constexpr, A: tl.constexpr, B: tl.constexpr, BK: tl.constexpr):
        row = tl.program_id(0)
        j = tl.arange(0, BK)
        keys = tl.load(Values+row*K+j, j<K, other=float('-inf')).to(tl.float64)
        ids = tl.load(Indices+row*K+j, j<K, other=0)
        edited = tl.load(S+row*V+ids).to(tl.float64)
        smallest = _class_min(keys, tl.where(j<K, edited, float('inf')), BK)
        largest = _class_max(keys, tl.where(j<K, edited, float('-inf')), BK)
        threshold = tl.load(Values+row*K+K-1).to(tl.float64)
        boundary_max = tl.load(BoundaryMax+row)
        largest = tl.where(keys==threshold, boundary_max, largest)
        next_j = tl.minimum(j+1, BK-1)
        next_key = tl.gather(keys, next_j, 0)
        next_max = tl.gather(largest, next_j, 0)
        hinge = tl.where((j<K-1) & (keys>next_key), tl.maximum(A*(keys-next_key)-(smallest-next_max), 0.), 0.)
        tl.store(Hinges+row*K+j, hinge, j<K)
        last_hinge = tl.sum(tl.where((j<K-1) & (next_key==threshold), hinge, 0.), 0)
        columns = tl.arange(0, B)
        best = tl.full((B,), float('-inf'), tl.float64)
        count = tl.zeros((B,), tl.int32)
        boundary_count = tl.zeros((B,), tl.int32)
        for start in range(0, V, B):
            c = start+columns
            t = tl.load(T+row*V+c, c<V, other=float('inf')).to(tl.float64)
            s = tl.load(S+row*V+c, c<V, other=0.).to(tl.float64)
            eligible = (c<V) & (t<threshold)
            z = tl.where(eligible, s-A*t, float('-inf'))
            count = tl.where(z>best, eligible.to(tl.int32), count+(eligible & (z==best)).to(tl.int32))
            best = tl.maximum(best, z)
            boundary_count += ((c<V) & (t==threshold) & (s==boundary_max)).to(tl.int32)
        maximum = tl.max(best, 0)
        ties = tl.sum(tl.where(best==maximum, count, 0), 0)
        member = tl.maximum(maximum-(tl.load(Cutoff+row)-A*threshold), 0.)
        tl.store(Outside+row, maximum)
        tl.store(OutsideCount+row, ties)
        tl.store(BoundaryCount+row, tl.sum(boundary_count, 0))
        tl.store(BoundaryHinge+row, last_hinge)
        tl.store(Member+row, member)
        tl.store(Loss+row, (tl.sum(hinge*hinge, 0)+member*member)/K)

    @triton.jit
    def _dense_backward(T, S, Values, BoundaryMax, CutoffIndex, Member, Outside,
                        OutsideCount, BoundaryCount, BoundaryHinge, Cotangent, Gradient,
                        V: tl.constexpr, K: tl.constexpr, A: tl.constexpr, B: tl.constexpr):
        row = tl.program_id(0)
        columns = tl.arange(0, B)
        threshold = tl.load(Values+row*K+K-1).to(tl.float64)
        boundary_max = tl.load(BoundaryMax+row)
        cutoff_index = tl.load(CutoffIndex+row)
        maximum = tl.load(Outside+row)
        scale = tl.load(Cotangent+row).to(tl.float64)/K
        member = (2.*tl.load(Member+row))*scale
        outside_grad = member/tl.maximum(tl.load(OutsideCount+row), 1)
        boundary_grad = ((2.*tl.load(BoundaryHinge+row))*scale)/tl.maximum(tl.load(BoundaryCount+row), 1)
        for start in range(0, V, B):
            c = start+columns
            t = tl.load(T+row*V+c, c<V, other=float('inf')).to(tl.float64)
            s = tl.load(S+row*V+c, c<V, other=0.).to(tl.float64)
            gradient = tl.where((t<threshold) & ((s-A*t)==maximum), outside_grad, 0.)
            gradient += tl.where((t==threshold) & (s==boundary_max), boundary_grad, 0.)
            gradient -= tl.where(c==cutoff_index, member, 0.)
            tl.store(Gradient+row*V+c, gradient, c<V)

    @triton.jit
    def _selected_backward(S, Values, Indices, Hinges, Cotangent, Gradient,
                           V: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
        row = tl.program_id(0)
        j = tl.arange(0, BK)
        keys = tl.load(Values+row*K+j, j<K, other=float('-inf')).to(tl.float64)
        ids = tl.load(Indices+row*K+j, j<K, other=0)
        edited = tl.load(S+row*V+ids).to(tl.float64)
        smallest = _class_min(keys, tl.where(j<K, edited, float('inf')), BK)
        largest = _class_max(keys, tl.where(j<K, edited, float('-inf')), BK)
        min_mask = (j<K) & (edited==smallest)
        max_mask = (j<K) & (edited==largest)
        min_count = _class_sum(keys, min_mask.to(tl.int32), BK)
        max_count = _class_sum(keys, max_mask.to(tl.int32), BK)
        h = tl.load(Hinges+row*K+j, j<K, other=0.)
        upper = _class_max(keys, h, BK)
        previous_key = tl.gather(keys, tl.maximum(j-1, 0), 0)
        previous_h = tl.load(Hinges+row*K+j-1, (j>0) & (j<K), other=0.)
        lower = _class_max(keys, tl.where((j>0) & (keys<previous_key), previous_h, 0.), BK)
        scale = tl.load(Cotangent+row).to(tl.float64)/K
        gradient = tl.where(max_mask, ((2.*lower)*scale)/tl.maximum(max_count, 1), 0.)
        gradient -= tl.where(min_mask, ((2.*upper)*scale)/tl.maximum(min_count, 1), 0.)
        threshold = tl.load(Values+row*K+K-1).to(tl.float64)
        tl.store(Gradient+row*V+ids, gradient, (j<K) & (keys>threshold))


def _cpu_rows(t, s, values, indices, alpha, cutoff, boundary_max):
    """Independent small CPU implementation of the segmented extrema algebra."""
    n, k = indices.shape
    values = values.double()
    edited = s.double().gather(1, indices)
    hinges = torch.zeros((n, k), dtype=torch.float64)
    for row in range(n):
        for j in range(k-1):
            if values[row, j] > values[row, j+1]:
                upper = edited[row, values[row]==values[row, j]].min()
                lower = (boundary_max[row] if values[row, j+1]==values[row, -1]
                         else edited[row, values[row]==values[row, j+1]].max())
                hinges[row, j] = (alpha*(values[row, j]-values[row, j+1])-(upper-lower)).clamp_min(0.)
    threshold = values[:, -1:]
    adjusted = s.double()-alpha*t.double()
    outside = torch.where(t<threshold, adjusted, -torch.inf).amax(1)
    outside_count = ((t<threshold) & (adjusted==outside[:, None])).sum(1)
    boundary_count = ((t==threshold) & (s==boundary_max[:, None])).sum(1)
    boundary_hinge = (hinges[:, :-1]*(values[:, 1:]==threshold)).sum(1)
    member = (outside-(cutoff-alpha*threshold[:, 0])).clamp_min(0.)
    return (hinges.square().sum(1)+member.square())/k, hinges, member, outside, outside_count, boundary_count, boundary_hinge


class _TieRank(torch.autograd.Function):
    @staticmethod
    def forward(ctx, teacher, student, k, alpha, return_topk):
        t = teacher.detach().reshape(-1, teacher.shape[-1]).contiguous()
        s = student.reshape_as(t).contiguous()
        values, indices = t.topk(k, dim=-1, sorted=True)
        n, v = t.shape
        if s.is_cuda:
            scores = torch.empty((n, v), device=s.device, dtype=torch.float64)
            _boundary_scores[(n, triton.cdiv(v, 1024))](t, s, values, scores, v, k, 1024)
        else:
            scores = torch.where(t==values[:, -1:], s.double(), -torch.inf)
        boundary_values, boundary_indices = scores.topk(k, dim=-1, sorted=True)
        remaining = k-(values>values[:, -1:]).sum(1, keepdim=True)
        cutoff = boundary_values.gather(1, remaining-1).flatten().contiguous()
        cutoff_index = boundary_indices.gather(1, remaining-1).flatten().contiguous()
        boundary_max = boundary_values[:, 0].contiguous()
        del scores, boundary_values, boundary_indices
        if s.is_cuda:
            loss = torch.empty(n, dtype=torch.float64, device=s.device)
            hinges = torch.empty((n, k), dtype=torch.float64, device=s.device)
            member, outside, boundary_hinge = (torch.empty_like(loss) for _ in range(3))
            outside_count = torch.empty(n, dtype=torch.int32, device=s.device)
            boundary_count = torch.empty_like(outside_count)
            _forward[(n,)](t, s, values, indices, boundary_max, cutoff, loss, hinges, member,
                outside, outside_count, boundary_count, boundary_hinge, v, k, alpha, 1024,
                triton.next_power_of_2(k), num_warps=4, enable_fp_fusion=False)
        else:
            loss, hinges, member, outside, outside_count, boundary_count, boundary_hinge = _cpu_rows(
                t, s, values, indices, alpha, cutoff, boundary_max)
        ctx.save_for_backward(t, s, values, indices, boundary_max, cutoff_index, hinges, member,
                              outside, outside_count, boundary_count, boundary_hinge)
        ctx.k, ctx.alpha, ctx.shape = k, alpha, student.shape
        if return_topk:
            ctx.mark_non_differentiable(values, indices)
            return loss.reshape(student.shape[:-1]), values, indices
        return loss.reshape(student.shape[:-1])

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent, *unused):
        t, s, values, indices, boundary_max, cutoff_index, hinges, member, outside, outside_count, boundary_count, boundary_hinge = ctx.saved_tensors
        cotangent = cotangent.contiguous().flatten()
        if s.is_cuda:
            gradient = torch.empty_like(s)
            _dense_backward[(len(s),)](t, s, values, boundary_max, cutoff_index, member, outside,
                outside_count, boundary_count, boundary_hinge, cotangent, gradient,
                s.shape[1], ctx.k, ctx.alpha, 1024, num_warps=4, enable_fp_fusion=False)
            _selected_backward[(len(s),)](s, values, indices, hinges, cotangent, gradient,
                s.shape[1], ctx.k, triton.next_power_of_2(ctx.k), num_warps=4, enable_fp_fusion=False)
        else:
            scale = cotangent.double()/ctx.k
            membership = (2.*member)*scale
            threshold = values[:, -1:].double()
            adjusted = s.double()-ctx.alpha*t.double()
            gradient = torch.where((t<threshold) & (adjusted==outside[:, None]),
                (membership/outside_count.clamp_min(1))[:, None], 0.)
            gradient += torch.where((t==threshold) & (s==boundary_max[:, None]),
                ((2.*boundary_hinge)*scale/boundary_count.clamp_min(1))[:, None], 0.)
            gradient.scatter_add_(1, cutoff_index[:, None], -membership[:, None])
            for row in range(len(s)):
                for j in range(ctx.k-1):
                    if values[row, j] <= values[row, j+1]: continue
                    high = indices[row, values[row]==values[row, j]]
                    high = high[s[row, high]==s[row, high].min()]
                    g = (2.*hinges[row, j])*scale[row]
                    gradient[row, high] -= g/len(high)
                    if values[row, j+1] > values[row, -1]:
                        low = indices[row, values[row]==values[row, j+1]]
                        low = low[s[row, low]==s[row, low].max()]
                        gradient[row, low] += g/len(low)
            gradient = gradient.to(s.dtype)
        return None, gradient.reshape(ctx.shape), None, None, None


def tiled_tie_aware_topk_margin_loss(teacher, student, *, k=100, margin_fraction=.5, return_teacher_topk=False):
    """FP64 ranking arithmetic; exact original teacher ties may exchange rank."""
    if teacher.shape!=student.shape or teacher.ndim<2 or not student.numel():
        raise ValueError('matching nonempty token/vocabulary shapes required')
    if type(k) is not int or not 1<=k<=student.shape[-1] or not 0<margin_fraction<=1:
        raise ValueError('valid ranking size and positive margin fraction required')
    if teacher.device!=student.device or not teacher.is_floating_point() or not student.is_floating_point():
        raise ValueError('floating teacher/student on the same device required')
    if student.is_cuda:
        if triton is None: raise RuntimeError('Triton required for CUDA ranking')
        if student.dtype!=torch.float32 or teacher.dtype not in (torch.float32, torch.float64):
            raise ValueError('CUDA ranking requires FP32 student and FP32/FP64 teacher')
    out = _TieRank.apply(teacher, student, k, float(margin_fraction), return_teacher_topk)
    return (out[0], (out[1], out[2])) if return_teacher_topk else out
