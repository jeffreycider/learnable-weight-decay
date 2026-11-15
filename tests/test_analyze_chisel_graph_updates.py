import copy
import json
import pytest
from scripts.analyze_chisel_graph_updates import analyze, response_readout


def fixture(tmp_path):
    states=list(range(200));trials=[]
    for repeat,order in enumerate((('reference','graph'),('graph','reference'),('reference','graph'))):
        for backend in order:
            rankrows=[]
            for rank in range(8):
                # Alternating slow ranks: median(max(ranktime)) differs from max(median(ranktime)).
                seconds=(100. if rank==repeat else 50.) if backend=='reference' else 40.
                rankrows.append(dict(rank=rank,seconds=seconds,accumulation_seconds=seconds-1,
                    setup_seconds=3.,recaptures=0,within_gates=True))
                (tmp_path/f'qualification_{repeat}_{backend}_rank{rank}.json').write_text(json.dumps(dict(
                    within_declared_global_gates=True,reference_repeat_bit_identical=True if backend=='reference' else None)))
            trials.append(dict(repeat=repeat,backend=backend,warmup=repeat==0,
                rank_rows=rankrows,all_rank_gates=True,seconds=max(r['seconds'] for r in rankrows)))
    summary=dict(world_size=8,checkpoint_batch=200,sequences=64,microbatch=16,states=states,trials=trials,
        global_gradient_response_diagnostics=[dict(state_id=s,lr_scale=scale,sequence_ids=list(range(16)),response=dict(finite=True),
                                                   kl_before=1e-6,kl_after_reference=9e-7,kl_after_candidate=9e-7)
                                             for s in states[:8] for scale in (1.,.1)])
    (tmp_path/'summary.json').write_text(json.dumps(summary));return summary


def test_critical_path_is_max_per_trial_then_median(tmp_path):
    fixture(tmp_path);r=analyze(tmp_path)
    assert r['median_seconds']==dict(reference=100.,graph=40.)
    assert r['throughput_ratio']==2.5
    assert r['training_only_updates_per_hour']==dict(reference=36.,graph=90.)
    assert r['trials'][3]['critical_rank']==1
    assert len(r['rank_evidence_sha256'])==48


def test_report_retains_explicit_benchmark_objective_override(tmp_path):
    summary=fixture(tmp_path)
    objective=dict(saved_shrinkage_coefficient=0.,effective_shrinkage_coefficient=1.,
                   explicit_benchmark_override=True)
    summary.update(benchmark_objective=objective,policy_sha256='same-saved-policy')
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    report=analyze(tmp_path)
    assert report['benchmark_objective']==objective
    assert report['policy_sha256']=='same-saved-policy'


@pytest.mark.parametrize('change', ['rank','workload','incomplete','response','wrong_max'])
def test_rejects_unqualified_or_incomplete_workload(tmp_path,change):
    s=fixture(tmp_path)
    if change=='rank':s['trials'][0]['rank_rows'].pop()
    elif change=='workload':s['checkpoint_batch']=8
    elif change=='incomplete':s['trials'].pop()
    elif change=='response':s['global_gradient_response_diagnostics'].pop()
    else:s['trials'][0]['seconds']=50.
    (tmp_path/'summary.json').write_text(json.dumps(s))
    with pytest.raises(ValueError):analyze(tmp_path)


def test_rank_evidence_overrides_summary_assertion(tmp_path):
    fixture(tmp_path)
    p=tmp_path/'qualification_2_graph_rank7.json'
    p.write_text(json.dumps(dict(within_declared_global_gates=False)))
    with pytest.raises(ValueError,match='contradicts'):analyze(tmp_path)


def test_response_uses_change_denominator_and_preserves_direction():
    # Synthetic small improvement: relative disagreement must use the change,
    # not the much larger pre-update KL, and must retain both actual signs.
    d=dict(state_id=999,lr_scale=.1,sequence_ids=list(range(16)),response=dict(finite=True),
           kl_before=1e-6,kl_after_reference=1e-6-4e-9,kl_after_candidate=1e-6-3.8e-9)
    r=response_readout(d)
    assert r['relative_kl_change_disagreement']==pytest.approx(.05)
    assert r['kl_change_reference']==pytest.approx(-4e-9)
    assert r['kl_change_candidate']==pytest.approx(-3.8e-9)
    assert r['kl_change_difference']==pytest.approx(2e-10)
    d['kl_after_reference']=d['kl_before']
    assert response_readout(d)['relative_kl_change_disagreement'] is None
    d['kl_after_candidate']=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):response_readout(d)


def test_hybrid_abba_requires_both_backend_response_populations(tmp_path):
    summary=fixture(tmp_path)
    template=summary['trials'][1]
    summary['graph_comparison']=True
    summary['trials']=[]
    for repeat,order in enumerate((('reference','graph','hybrid'),('hybrid','graph'),('graph','hybrid'))):
        for backend in order:
            trial=copy.deepcopy(template)
            trial.update(repeat=repeat,backend=backend,warmup=repeat==0,seconds=30. if backend=='hybrid' else 40.)
            for row in trial['rank_rows']:
                row.update(seconds=trial['seconds'],accumulation_seconds=trial['seconds']-1)
                (tmp_path/f"qualification_{repeat}_{backend}_rank{row['rank']}.json").write_text(json.dumps(dict(
                    within_declared_global_gates=True,reference_repeat_bit_identical=backend=='reference')))
            summary['trials'].append(trial)
    summary['global_gradient_response_diagnostics']=[dict(d,backend=b) for b in ('graph','hybrid')
        for d in summary['global_gradient_response_diagnostics']]
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    result=analyze(tmp_path)
    assert result['median_seconds']==dict(graph=40.,hybrid=30.)
    assert result['time_reduction_fraction']==.25
    assert len(result['rank_evidence_sha256'])==56
    assert {d['backend'] for d in result['response_readouts']}=={'graph','hybrid'}
    summary['global_gradient_response_diagnostics'].pop()
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):analyze(tmp_path)


def test_disjoint_confirmation_single_warm_measurement_is_explicit(tmp_path):
    s=fixture(tmp_path);s['trials']=s['trials'][:4]
    confirmation=dict(original_state_ids=list(range(200,400)),selected_state_ids=s['states'],disjoint=True,
        kl_abs_limit=1e-9,objective_abs_floor=1e-4,objective_relative_limit=1e-3)
    s['benchmark_objective']=dict(independent_confirmation=confirmation,effective_kl_tolerance=1e-8)
    (tmp_path/'summary.json').write_text(json.dumps(s));r=analyze(tmp_path)
    assert r['median_seconds']==dict(reference=100.,graph=40.)
    assert 'ONE warm' in r['scope'] and 'no repeat variance' in r['timing_unit']
    assert len(r['rank_evidence_sha256'])==32
    confirmation['original_state_ids'][0]=0
    (tmp_path/'summary.json').write_text(json.dumps(s))
    with pytest.raises(ValueError,match='disjoint'):analyze(tmp_path)
    s.pop('benchmark_objective');(tmp_path/'summary.json').write_text(json.dumps(s))
    with pytest.raises(ValueError,match='alternating'):analyze(tmp_path)
