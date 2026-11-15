import copy
import pytest
from scripts.analyze_chisel_graph_gates import diagnose
from scripts.benchmark_chisel_graph_updates import compare_signatures
from tests.test_benchmark_chisel_graph_updates import signatures


def test_triage_separates_objective_scaling_from_gradient_and_adam_failures():
    a,initial=signatures();b=copy.deepcopy(a)
    # A small KL readout discrepancy can coexist with an excessive objective
    # discrepancy at a tighter tau; do not mislabel it as an Adam failure.
    b['rows'][19]['mean_kl']+=1e-10
    b['rows'][19]['objective']+=2e-4
    m=compare_signatures(a,b,initial,.1,1e-8)
    d=diagnose(m,tolerance=1e-8,dual_lr=.1)
    assert d['failed_gates']==['training_rows.objective_difference']
    assert d['maxima']['training_rows']['objective_difference']['exceeded_state_ids']==[19]
    assert d['maxima']['training_rows']['kl_difference']['exceeded_count']==0
    m['within_declared_global_gates']=True
    with pytest.raises(ValueError,match='contradict'):diagnose(m,tolerance=1e-8,dual_lr=.1)


def test_triage_requires_unique_complete_checkpoint_denominators():
    a,initial=signatures();m=compare_signatures(a,a,initial,.1,1e-8)
    assert diagnose(m,tolerance=1e-8,dual_lr=.1)['failed_gates']==[]
    m['post_update_rows'][-1]=m['post_update_rows'][0]
    with pytest.raises(ValueError,match='unique'):diagnose(m,tolerance=1e-8,dual_lr=.1)


def test_prospective_gate_requires_reference_value_and_retains_legacy_failure():
    a,initial=signatures();b=copy.deepcopy(a)
    b['post'][19]['objective']+=1.1e-4
    m=compare_signatures(a,b,initial,.1,1e-8)
    assert diagnose(m,tolerance=1e-8,dual_lr=.1)['failed_gates']==['post_update_rows.objective_difference']
    for key in ['training_rows','post_update_rows']:
        for row,ref in zip(m[key],a['rows' if key=='training_rows' else 'post']):
            row['reference_objective']=ref['objective']
    # New independent run metadata must select the new gate; never mutate the
    # old record to reinterpret its saved decision in a report.
    new=copy.deepcopy(m);new['within_declared_global_gates']=True
    assert diagnose(new,tolerance=1e-8,dual_lr=.1,objective_relative_limit=1e-3,kl_abs_limit=1e-9)['failed_gates']==[]
    new['post_update_rows'][19].pop('reference_objective')
    with pytest.raises(KeyError):diagnose(new,tolerance=1e-8,dual_lr=.1,objective_relative_limit=1e-3)
