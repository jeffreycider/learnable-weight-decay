"""Avoid presenting rotating minibatch losses as fixed-panel evaluations."""
import json

from scripts.summarize_cpu_norm_preservation import summarize


def test_summary_keeps_rotating_panel_and_full_pool_denominators(tmp_path):
    metadata = dict(cases=[], args=dict(policy='full'), pool_schedule=dict(cycle_updates=15))
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    base = dict(local_id=6, mean_group_l2_percentage_reduction=.01, mean_kl=1e-12,
                token_kl_quantiles=dict(max=2e-12), strict_topk_changes=1,
                tie_aware_argmax_changes=0, norm_objective=-.01, ranking_penalty=.02,
                tail_quadratic_penalty=0., dual_linear_term=0., objective=.01)
    records = []
    for step, scope, tokens, indices in [(0, 'training_subset', 8192, [0, 1]),
                                        (1, 'training_subset', 8192, [10, 11]),
                                        (2, 'full_training_pool', 122880, [0, 1, 10, 11])]:
        records.append(dict(completed_updates_before_measure=step,
                            updates_after=min(step + 1, 2), learning=step < 2,
                            rows=[dict(base, token_count=tokens, panel_scope=scope,
                                       panel_indices=indices, context_ids=[100+i for i in indices])]))
    data = ''.join(json.dumps(r)+'\n' for r in records).encode()
    (tmp_path / 'metrics.jsonl').write_bytes(data+b'{"incomplete":')
    result = summarize(tmp_path)
    assert result['source_prefix_bytes'] == len(data)
    assert result['rotating_training_subsets']
    assert result['final_endpoint_present']
    assert [r['tokens'] for r in result['rows']] == [8192, 8192, 122880]
    assert result['rows'][1]['sequence_ids'] == [110, 111]
    assert result['rows'][-1]['measurement_scope'] == 'full_training_pool'
    assert 'do not isolate learning' in result['objective_warning']

    metadata.pop('pool_schedule')
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    result = summarize(tmp_path)
    assert not result['rotating_training_subsets']
    assert result['scope'].startswith('Fixed training sequences')
