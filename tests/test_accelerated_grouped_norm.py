import pytest
import torch

from baseline.model import GPT
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.grouped_norm import GroupedNormObjective


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA Triton kernel')
@pytest.mark.parametrize('zero_group', [False, True])
def test_multi_tensor_norm_values_gradients_and_storage_changes(zero_group):
    torch.manual_seed(55)
    model = GPT(128, 2, 128).cuda()
    base = {n: v.detach().clone() for n, v in model.named_parameters()}
    if zero_group:
        base['proj.bias'].zero_()
    reference = GroupedNormObjective(base, zero_reference_policy='freeze')
    fast = AcceleratedGroupedNormObjective(base, zero_reference_policy='freeze')
    for magnitude in (0., 1e-5, .01):
        values = {n: (v.float() + torch.randn_like(v.float())*magnitude).to(v.dtype).requires_grad_()
                  for n, v in base.items()}
        if zero_group:
            values['proj.bias'] = base['proj.bias']
        actual, expected = fast.evaluate(values), reference.evaluate(values)
        for key in ('mean_relative_change', 'squared_norm_change', 'edit_norm_sq'):
            torch.testing.assert_close(getattr(actual, key), getattr(expected, key), atol=2e-10, rtol=2e-10)
        leaves = [values[n] for n in fast.names if values[n].requires_grad]
        ga = torch.autograd.grad(actual.mean_relative_change*37. + actual.edit_norm_sq*.7,
                                 leaves, retain_graph=True)
        ge = torch.autograd.grad(expected.mean_relative_change*37. + expected.edit_norm_sq*.7, leaves)
        for a, e in zip(ga, ge):
            torch.testing.assert_close(a, e, atol=1e-7, rtol=2e-6)
    if zero_group:
        values['proj.bias'] = torch.ones_like(base['proj.bias'])
        with pytest.raises(ValueError, match='zero-reference'):
            fast.evaluate(values)
