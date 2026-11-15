import torch
from chisel.preservation import group_norm_change,prediction_terms,ranking_changes
from chisel.transfer import apply_edits


def test_moving_to_consolidated_earns_no_shrinkage_reward():
    raw=torch.randn(4,6,dtype=torch.float64)
    snapshot={'blocks.0.attn.q.weight':dict(raw=raw,consolidated=torch.zeros_like(raw))}
    name=next(iter(snapshot));edits={name:dict(raw=-raw,consolidated=raw)}
    _,_,effective=apply_edits(snapshot,edits)
    assert group_norm_change(snapshot,effective)==0
    effective={name:raw*.9}
    torch.testing.assert_close(group_norm_change(snapshot,effective),torch.tensor(-.1,dtype=torch.float64))


def test_grouping_weights_types_equally_not_by_parameter_count():
    snapshot={name:dict(raw=torch.ones(*shape),consolidated=torch.zeros(*shape)) for name,shape in [('blocks.0.attn.q.weight',(2,2)),('blocks.1.attn.q.weight',(2,2)),('blocks.0.mlp.fc.weight',(100,2))]}
    effective={name:item['raw']*(.8 if 'attn' in name else 1.) for name,item in snapshot.items()}
    torch.testing.assert_close(group_norm_change(snapshot,effective),torch.tensor(-.1))


def test_identity_kl_small_errors_and_tie_aware_rankings():
    teacher=torch.tensor([[3.,2.,2.,0.]],dtype=torch.float64)
    assert prediction_terms(teacher,teacher,top_k=3)['kl'].item()==0
    assert ranking_changes(torch.tensor([[3.,1.9,2.1,0.]]),teacher,top_k=3)['rank_changed']==0
    student=(teacher+torch.tensor([[0.,.01,-.01,0.]])).requires_grad_()
    terms=prediction_terms(student,teacher,top_k=3)
    exact=torch.nn.functional.kl_div(student.log_softmax(-1),teacher.softmax(-1),reduction='batchmean')
    torch.testing.assert_close(terms['kl'].mean(),exact,rtol=1e-9,atol=1e-14)
    terms['mse'].mean().backward();assert student.grad.norm()>0
    assert ranking_changes(torch.tensor([[3.,1.,2.,2.5]]),teacher,top_k=3)['rank_changed']==1
