import copy
import json
from pathlib import Path
import pytest
from scripts.run_soap46_xlr_decay import initial_state,advance,claim_lane


def config():return json.loads(Path('configs/soap46_xlr_decay.json').read_text())


def test_six_predeclared_runs_two_distinct_prospective_controls_and_fixed_baseline():
    c=config();s=initial_state(c)
    assert len(s['jobs'])==2 and sum(len(j['trials']) for j in s['jobs'])==6
    assert len({j['trials'][0]['name'] for j in s['jobs']})==2
    cells=[]
    for j in s['jobs']:
        control,*learned=j['trials'];assert control['warmup_steps']==2900
        assert {t['meta_lr'] for t in learned}=={1e-5,1e-4}
        assert {t['residual_decay'] for t in learned}=={.01,.03}
        for t in j['trials']:
            assert t['seed']==3 and t['variant']=='outer-only'
            assert t['base_lr']==.0375 and t['cwd']==.025
            assert t['meta_betas']==[.9,.99] and t['meta_clip']==1
            assert t['meta_ramp_steps']==500 and t['warmup_meta']=='moments'
        cells.extend((t['meta_lr'],t['residual_decay']) for t in learned)
    assert set(cells)=={(1e-5,.01),(1e-5,.03),(1e-4,.01),(1e-4,.03)}


def test_no_selection_until_both_full_lanes_complete_and_no_best_so_far_stop():
    c=config();s=initial_state(c);s['jobs'][0]['status']='complete'
    advance(s,c,lambda _:(_ for _ in ()).throw(AssertionError('queried incomplete screen')))
    assert s['phase']=='screen'


def scores(state,controls=(1.,1.)):
    result={}
    for j,v in zip(state['jobs'],controls):
        result[j['trials'][0]['name']]=v
        for t in j['trials'][1:]:result[t['name']]=v-.0001
        j['status']='complete'
    return result


def test_practical_threshold_rejects_tiny_improvements_and_preserves_all_rows():
    c=config();s=initial_state(c);values=scores(s);advance(s,c,values.__getitem__)
    assert s['phase']=='review' and len(s['screen_results'])==4 and len(s['controls'])==2
    assert not s['jobs']


def test_must_beat_both_controls_not_just_high_loss_host_reference():
    c=config();s=initial_state(c);values=scores(s,(1.,1.002))
    values[s['jobs'][1]['trials'][1]['name']]=1.001
    advance(s,c,values.__getitem__)
    assert s['phase']=='review'


def test_frozen_tie_break_single_winner_then_both_fresh_pairs_regardless_first_sign():
    c=config();s=initial_state(c);values=scores(s)
    candidates=[s['jobs'][0]['trials'][1],s['jobs'][1]['trials'][1]]
    for t in candidates:values[t['name']]=.999
    expected=min(t['name'] for t in candidates)
    advance(s,c,values.__getitem__)
    assert s['phase']=='confirmation' and s['selected']['trial']['name']==expected
    assert {j['trials'][0]['seed'] for j in s['jobs']}=={6,7}
    first,second=s['jobs'];first['status']='complete'
    advance(s,c,lambda _:(_ for _ in ()).throw(AssertionError('early readout')))
    assert s['phase']=='confirmation' and second['status']=='pending'
    second['status']='complete'
    vals={first['trials'][0]['name']:1.1,first['trials'][1]['name']:1.,second['trials'][0]['name']:.8,second['trials'][1]['name']:1.}
    advance(s,c,vals.__getitem__)
    assert s['phase']=='complete' and s['primary_confirmation_mean']==pytest.approx(-.05)
    assert [r['seed'] for r in s['confirmation_pairs']]==[6,7]


def test_lane_host_identity_and_ambiguous_claim_fail_closed():
    s=initial_state(config());j=claim_lane(s,1,'gpuA');assert j['lane']==1
    with pytest.raises(RuntimeError,match='share'):claim_lane(s,2,'gpuA')
    with pytest.raises(RuntimeError,match='assignment'):claim_lane(s,1,'gpuB')
    with pytest.raises(RuntimeError,match='Ambiguous'):claim_lane(s,1,'gpuA')
    j['status']='complete';assert claim_lane(s,1,'gpuA') is None
    assert claim_lane(s,2,'gpuB')['lane']==2
