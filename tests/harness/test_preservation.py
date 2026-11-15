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


def test_linear_rank_constraint_retains_gradient_near_a_violation():
    teacher=torch.tensor([[1.,0.]],dtype=torch.float64)
    student=torch.tensor([[.499999,0.]],dtype=torch.float64,requires_grad=True)
    term=prediction_terms(student,teacher,top_k=2,ranking_power=1)['ranking'].sum()
    gradient=torch.autograd.grad(term,student)[0]
    torch.testing.assert_close(term,torch.tensor(1e-6,dtype=torch.float64))
    torch.testing.assert_close(gradient,torch.tensor([[-1.,1.]],dtype=torch.float64))
    assert prediction_terms(teacher,teacher,top_k=2,ranking_power=1)['ranking']==0


def test_equal_group_reward_and_total_norm_audit_are_distinct():
    from chisel.preservation import effective_norm_change
    snapshot={name:dict(raw=torch.ones(*shape,dtype=torch.float64),consolidated=torch.zeros(*shape,dtype=torch.float64)) for name,shape in [('small',(1,1)),('large',(100,100))]}
    edited={'small':snapshot['small']['raw']*.5,'large':snapshot['large']['raw']*1.1}
    assert group_norm_change(snapshot,edited)<0
    assert effective_norm_change(snapshot,edited)>0
    before=torch.cat([v['raw'].flatten() for v in snapshot.values()]).norm()
    after=torch.cat([v.flatten() for v in edited.values()]).norm()
    torch.testing.assert_close(effective_norm_change(snapshot,edited),after/before-1)


def test_ties_cannot_hide_real_ordering_changes():
    from chisel.preservation import prediction_terms,ranking_changes
    teacher=torch.tensor([[3.,2.,2.,1.]])
    for student,k in [(torch.tensor([[3.,0.,2.,1.]]),4),(torch.tensor([[3.,2.,0.,2.5]]),2)]:
        student.requires_grad_()
        assert ranking_changes(student,teacher,k)['rank_changed']==1
        loss=prediction_terms(student,teacher,k)['ranking'].sum()
        assert loss>0
        loss.backward()
        assert torch.isfinite(student.grad).all() and student.grad.abs().sum()>0


def test_boundary_tie_replacement_is_allowed_and_top_one_is_finite():
    from chisel.preservation import prediction_terms,ranking_changes
    teacher=torch.tensor([[3.,2.,2.,1.]])
    for student in [torch.tensor([[3.,1.5,2.5,1.]]),torch.tensor([[3.,2.5,1.5,1.]])]:
        assert ranking_changes(student,teacher,2)['rank_changed']==0
        assert prediction_terms(student,teacher,2)['ranking']==0
    for teacher,k in [(torch.ones(2,4),2),(torch.ones(2,1),1)]:
        student=torch.randn_like(teacher,requires_grad=True)
        loss=prediction_terms(student,teacher,k)['ranking'].sum()
        assert loss==0
        loss.backward()
        assert torch.isfinite(student.grad).all()
