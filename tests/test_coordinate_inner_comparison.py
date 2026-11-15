import copy
from scripts.compare_coordinate_inner_program import visit_gates,numeric_visit


def rows():
    result=[]
    for update in range(2):
        visits=[]
        for domain in ('gpt','cifar10'):
            for uid in (1,2):
                inner=[dict(context_ids=[1+i],objective=1.,mean_kl=.1,private_preclip_gradient_norm=.2) for i in range(2)]
                visits.append(dict(domain=domain,local_id=uid,private_visit=3*(update+1),private_updates_this_visit=3,
                    private_inner_rows=inner,context_ids=[3],objective=1.,mean_kl=.1,private_preclip_gradient_norm=.2,state_weight=.25))
        result.append(dict(rows=visits))
    return result


def test_tick_and_disjointness_gate_rejects_real_accounting_errors():
    value=rows();assert all(visit_gates(value,2).values())
    wrong=copy.deepcopy(value);wrong[1]['rows'][0]['private_visit']=2
    assert not visit_gates(wrong,2)['private_tick_accounting']
    wrong=copy.deepcopy(value);wrong[0]['rows'][0]['private_inner_rows'][0]['context_ids']=[3]
    assert not visit_gates(wrong,2)['disjoint_support_and_joint']
    wrong=copy.deepcopy(value);wrong[0]['rows'][0]['mean_kl']=float('nan')
    assert not visit_gates(wrong,2)['finite_losses_and_gradients']


def test_numeric_rows_exclude_only_execution_measurements():
    assert numeric_visit(dict(mean_kl=.1,strict_topk_changes=3,private_stage_seconds=1.,private_snapshot_bytes=20))==dict(mean_kl=.1,strict_topk_changes=3)
