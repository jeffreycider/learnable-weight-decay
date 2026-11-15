import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rewriter.proposal_validation import require_finite_proposal


class ScalarReads(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            self.count += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='owner-coordinated CUDA qualification'))])
def test_same_decision_as_sequential_guard_without_changing_values(bad, device):
    proposal = {
        'empty': torch.empty(0, device=device),
        'scalar': torch.tensor(-0., device=device),
        'matrix': torch.tensor([[1e-39, -1e-39], [3e38, -3e38]], device=device),
        'noncontiguous': torch.arange(12., device=device).reshape(3, 4).T,
        'last': torch.tensor([1. if bad is None else bad], device=device),
    }
    expected_bad = any(not bool(torch.isfinite(v).all()) for v in proposal.values())
    signatures = {n: (v.data_ptr(), v._version) for n, v in proposal.items()}
    with ScalarReads() as reads:
        if expected_bad:
            with pytest.raises(FloatingPointError, match='nonfinite proposal'):
                require_finite_proposal(proposal)
        else:
            require_finite_proposal(proposal)
    assert reads.count == 1
    assert signatures == {n: (v.data_ptr(), v._version) for n, v in proposal.items()}


def test_many_parameters_have_one_host_decision_and_empty_has_none():
    proposal = {str(i): torch.ones(2, requires_grad=True) for i in range(173)}
    with ScalarReads() as reads:
        require_finite_proposal(proposal)
    assert reads.count == 1
    assert all(v.grad is None for v in proposal.values())
    with ScalarReads() as reads:
        require_finite_proposal({})
    assert reads.count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='owner-coordinated CUDA qualification')
def test_mixed_devices_keep_checks_local():
    proposal = {'cpu': torch.ones(2), 'cuda': torch.ones(2, device='cuda')}
    with ScalarReads() as reads:
        require_finite_proposal(proposal)
    assert reads.count == 2
