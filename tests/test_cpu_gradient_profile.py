import copy
import json
import sys

import pytest
import torch

from scripts import profile_cpu_joint_gradients as profile


def endpoint():
    cases = [dict(state_id=6, sequence_ids=[10, 12]), dict(state_id=7, sequence_ids=[20, 22])]
    saved = dict(completed_updates=4, dual=dict(updates=4),
                 metadata=dict(fixed_train_panels=True, cases=cases, state_weights=[.5, .5]))
    rows = [dict(local_id=c['state_id'], context_ids=c['sequence_ids'], strict_topk_changes=5) for c in cases]
    result = dict(completed_updates=4, final=dict(completed_updates_before_measure=4,
        updates_after=4, learning=False, gradient_profile=False, rows=rows))
    return saved, result


def test_profile_accepts_failed_preservation_endpoint_but_not_unmeasured_weights():
    saved, result = endpoint()
    profile.validate_profile_endpoint(saved, result)
    result['final']['learning'] = True
    with pytest.raises(ValueError, match='forward-only'):
        profile.validate_profile_endpoint(saved, result)
    result['final']['learning'] = False
    result['final']['completed_updates_before_measure'] = 3
    with pytest.raises(ValueError, match='does not match'):
        profile.validate_profile_endpoint(saved, result)


@pytest.mark.parametrize('change', ['duplicate_case', 'sequence_ids', 'dual_clock', 'panel_weights'])
def test_profile_rejects_mismatched_training_identity(change):
    saved, result = endpoint()
    if change == 'duplicate_case':
        saved['metadata']['cases'][1] = copy.deepcopy(saved['metadata']['cases'][0])
    elif change == 'sequence_ids':
        result['final']['rows'][0]['context_ids'] = [11, 12]
    elif change == 'dual_clock':
        saved['dual']['updates'] = 3
    else:
        saved['metadata']['state_weights'] = [.3, .7]
    with pytest.raises(ValueError):
        profile.validate_profile_endpoint(saved, result)


def test_fitted_inputs_use_saved_id_order_and_hash_not_first_rows():
    panel = dict(inputs=torch.arange(12).reshape(3, 4), metadata=dict(sequence_ids=[90, 80, 70]))
    expected = panel['inputs'][[2, 0]].clone()
    case = dict(sequence_ids=[70, 90], input_sha256=profile.tensor_sha(expected))
    actual = profile.fitted_inputs(panel, case)
    assert torch.equal(actual, expected)
    assert actual.data_ptr() != panel['inputs'].data_ptr()
    panel['inputs'][2, 0] += 1
    with pytest.raises(ValueError, match='hash'):
        profile.fitted_inputs(panel, case)
    case['sequence_ids'] = [999]
    with pytest.raises(ValueError, match='Missing'):
        profile.fitted_inputs(panel, case)


def test_changed_edited_network_measurement_is_not_silently_profiled():
    row = dict(context_ids=[1, 2], norm_objective=-.01, mean_kl=1e-12,
               token_kl_quantiles={'max': 2e-12}, strict_topk_changes=2, ranking_penalty=.1)
    profile.validate_reproduced_measurement(row, copy.deepcopy(row))
    changed = dict(row, strict_topk_changes=3)
    with pytest.raises(RuntimeError, match='reproduce'):
        profile.validate_reproduced_measurement(changed, row)


def test_incomplete_checkpoint_rejected_before_output_or_model_work(monkeypatch, tmp_path):
    saved, result = endpoint()
    result['final']['learning'] = True
    checkpoint = tmp_path / 'checkpoint_0004.pt'
    torch.save(saved, checkpoint)
    (tmp_path / 'complete.json').write_text(json.dumps(result))
    monkeypatch.setattr(sys, 'argv', ['profile', '--checkpoint', str(checkpoint), '--output', str(tmp_path / 'output')])
    with pytest.raises(ValueError, match='forward-only'):
        profile.main()
    assert not (tmp_path / 'output').exists()
