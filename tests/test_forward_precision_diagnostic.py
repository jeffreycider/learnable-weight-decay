import torch

from baseline.model import GPT
from rewriter.function_preservation import _LogitsView
from scripts.diagnose_chisel_forward_precision import diagnostic_logits, compare_logits, UnroundedProposal
from rewriter.adaptive_action import AdaptiveAction, AdaptiveFactors


def test_fp32_matches_executed_gpt_and_fp64_retains_precision():
    torch.manual_seed(123)
    model=GPT(131,2,128).float().eval()
    inputs=torch.randint(131,(1,17))
    parameters=dict(model.named_parameters());buffers=dict(model.named_buffers())
    with torch.no_grad():expected=_LogitsView(model,'fp32')(inputs)
    actual=diagnostic_logits(parameters,buffers,inputs,layers=2,dtype=torch.float32)
    assert torch.equal(actual,expected)
    high=diagnostic_logits(parameters,buffers,inputs,layers=2,dtype=torch.float64)
    assert high.dtype==torch.float64 and torch.isfinite(high).all()
    assert not torch.equal(high,actual.double())
    same=compare_logits(high,high)
    assert same['mean_kl']==0 and same['counts']['strict_topk_changed']==0


def test_unrounded_view_preserves_embedding_orientation_and_readonly():
    torch.manual_seed(8)
    base={'embed.weight':torch.randn(9,5),'bias':torch.randn(5),'norm.gains':torch.ones(5)}
    factor=AdaptiveFactors(torch.randn(5,2,dtype=torch.float64),torch.randn(2,2,dtype=torch.float64),torch.randn(9,2,dtype=torch.float64))
    bias=torch.randn(5,dtype=torch.float64)
    action=AdaptiveAction({'embed.weight':factor},{'bias':bias,'norm.gains':torch.ones(5,dtype=torch.float64)})
    view=UnroundedProposal(base,action,.25,{'embed.weight','bias'},{'embed.weight':True})
    assert torch.equal(view['embed.weight'],base['embed.weight'].double()+.25*factor.dense().T)
    assert torch.equal(view['bias'],base['bias'].double()+.25*bias)
    assert torch.equal(view['norm.gains'],base['norm.gains'].double())
