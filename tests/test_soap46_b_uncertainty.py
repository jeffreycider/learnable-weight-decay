import copy
import json
from pathlib import Path
import pytest
from scripts.run_soap46_b_uncertainty import trials,activation_decision,summarize


def config():return json.loads(Path('configs/soap46_b_uncertainty.json').read_text())


def test_manifest_fixes_two_full_pairs_and_only_zero_action_control_differs():
    c=config();declared=trials(c)
    assert len(declared)==4 and len({t['name'] for t in declared})==4
    assert [t['seed'] for t in declared]==[4,4,5,5]
    for a,b in zip(declared[::2],declared[1::2]):
        assert a['warmup_steps']==500 and b['warmup_steps']==2900
        assert {k:v for k,v in a.items() if k not in ('name','warmup_steps')}=={k:v for k,v in b.items() if k not in ('name','warmup_steps')}
    for key,value in [('base_lr',.03),('meta_betas',[.9,.999]),('residual_decay',.01)]:
        bad=copy.deepcopy(c);bad['fixed'][key]=value
        with pytest.raises(ValueError):trials(bad)


def test_only_failed_extension_activates_separate_study_without_mutating_gate():
    state=dict(phase='review',extension_attempted=False)
    assert activation_decision(state)=='wait'
    state.update(extension_attempted=True,extension_incumbent={'name':'old'},extension_selected={'name':'old'})
    old=copy.deepcopy(state)
    assert activation_decision(state)=='launch_separate_uncertainty_study' and state==old
    state.update(extension_selected={'name':'new'},extension_development_comparison={'difference':.001})
    assert activation_decision(state)=='launch_separate_uncertainty_study'
    state['extension_development_comparison']['difference']=-.001
    with pytest.raises(RuntimeError):activation_decision(state)
    state['phase']='confirmation'
    assert activation_decision(state)=='defer_to_A_confirmation'
    state['phase']='complete'
    assert activation_decision(state)=='defer_to_A_confirmation'


def test_primary_mean_uses_both_prespecified_pairs_regardless_first_sign():
    c=config();names=[t['name'] for t in trials(c)];values=dict(zip(names,[3.3,3.2,3.0,3.2]));seen=[]
    def score(name):seen.append(name);return values[name]
    result=summarize(c,score)
    assert seen==names and len(result['pairs'])==2
    assert result['primary_mean_difference']==pytest.approx(-.05)
    assert result['pairs'][0]['difference']>0 and result['pairs'][1]['difference']<0
    values.pop(names[-1])
    with pytest.raises(KeyError):summarize(c,score)
