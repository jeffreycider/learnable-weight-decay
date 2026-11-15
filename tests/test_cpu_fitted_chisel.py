import pytest
import torch
from rewriter.adaptive_action import AdaptiveAction, AdaptiveFactors
from scripts.evaluate_cpu_fitted_chisel import load_action, logit_metrics, stored_edits


def test_stored_edit_embedding_orientation_and_single_cast():
    base = {'embed.weight': torch.arange(20).reshape(5,4).float()/7,
            'w': torch.randn(4,3), 'b': torch.randn(4)}
    u = torch.randn(4,2,dtype=torch.float64)
    v = torch.randn(5,2,dtype=torch.float64)
    c = torch.randn(2,2,dtype=torch.float64)
    action = AdaptiveAction({'embed.weight':AdaptiveFactors(u,c,v)},
                            {'b':torch.full((4,),1e-8,dtype=torch.float64)})
    before = {n:x.clone() for n,x in base.items()}
    out = stored_edits(base,action)
    assert out.keys() == {'embed.weight','b'}
    torch.testing.assert_close(out['embed.weight'],(base['embed.weight'].double()+(u@c@v.T).T).float(),rtol=0,atol=0)
    torch.testing.assert_close(out['b'],(base['b'].double()+action.vectors['b']).float(),rtol=0,atol=0)
    for n in base:assert torch.equal(base[n],before[n])


def test_identity_and_tie_aware_preservation():
    teacher = torch.tensor([[3.,2.,1.,0.],[3.,3.,1.,0.]])
    same = logit_metrics(teacher,teacher,k=3,chunk_tokens=1)
    assert same['kl_sum']==0 and same['different_logits']==0
    assert same['strict_topk_changed']==0 and same['literal_argmax_changed']==0
    student=teacher.clone();student[1,1]=3.01
    result=logit_metrics(teacher,student,k=3)
    assert result['literal_argmax_changed']==1
    assert result['tie_aware_argmax_changed']==0
    assert result['strict_topk_changed']==0


def test_order_change_without_greedy_change_and_kl_reference():
    teacher=torch.tensor([[3.,2.,1.,0.],[5.,3.,2.,0.]])
    student=teacher.clone();student[0,1:3]=torch.tensor([1.,2.])
    result=logit_metrics(teacher,student,k=3,chunk_tokens=1)
    assert result['strict_topk_changed']==1 and result['literal_argmax_changed']==0
    p=teacher.double().log_softmax(-1);q=student.double().log_softmax(-1)
    exact=(p.exp()*(p-q)).sum(-1)
    assert result['kl_sum']==pytest.approx(float(exact.sum()),abs=1e-14)
    assert result['max_token_kl']==pytest.approx(float(exact.max()),abs=1e-14)


def test_nonfinite_logits_and_wrong_checkpoint_rejected():
    with pytest.raises(ValueError,match='nonfinite'):
        logit_metrics(torch.zeros(1,3),torch.full((1,3),float('nan')),k=2)
    with pytest.raises(ValueError,match='identity'):
        load_action({},dict(metadata=dict(schema='wrong',state_sha256='x')),'x')
