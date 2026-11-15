import pytest
import torch
from baseline.model import Block
from chisel.attention_gauge_control import balance
from chisel.preservation import group_norm_change


@pytest.mark.parametrize('causal',[False,True])
def test_head_local_finite_edit_preserves_block_with_bias_and_rope(causal):
    torch.manual_seed(19)
    model=torch.nn.Sequential(Block(16,head_dim=8,causal=causal,learned_qk_scale=True)).double()
    base={n:p.detach().clone() for n,p in model.named_parameters()}
    # Retain nonzero biases; cross-head transforms would violate this identity.
    snapshot={n:dict(raw=p,consolidated=torch.zeros_like(p)) for n,p in base.items() if p.ndim==2}
    effective=balance(snapshot,base,head_dim=8,rank=4)
    x=torch.randn(3,11,16,dtype=torch.float64)
    with torch.no_grad():
        before=torch.func.functional_call(model,base,(x,))
        after=torch.func.functional_call(model,base|effective,(x,))
    torch.testing.assert_close(before,after,atol=1e-12,rtol=1e-12)
    assert group_norm_change(snapshot,effective)<-1e-4
    for n,p in effective.items():
        if '.attn.v.' in n or '.attn.proj.' in n:
            assert torch.linalg.matrix_rank(p-base[n],atol=1e-10)<=4
        else: torch.testing.assert_close(p,base[n],rtol=0,atol=0)
