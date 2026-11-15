import json
from pathlib import Path
from scripts.run_soap46_lr_cwd import initial_state,advance,control_for,activate_extension
import pytest


def config():return json.loads(Path('configs/soap46_lr_cwd_round3.json').read_text())


def test_original_methods_center_and_only_lr_cwd_change():
    c=config();s=initial_state(c);assert len(s['jobs'])==10
    for variant in c['variants']:
        trials=[j['trials'][0] for j in s['jobs'] if j['trials'][0]['variant']==variant]
        assert {(t['base_lr'],t['cwd']) for t in trials}=={tuple(p) for p in c['neighbors']}
    for j in s['jobs']:
        t=j['trials'][0]
        assert t['meta_betas']==[.9,.99] and t['meta_clip']==1
        assert t['warmup_steps']==500 and t['meta_ramp_steps']==500
        assert 'residual_frame' not in t and 'reset_meta_at_activation' not in t


def test_select_center_and_keep_baseline_hyperparameters_fixed():
    c=config();s=initial_state(c)
    for j in s['jobs']:j['status']='complete'
    advance(s,c,lambda n:0 if 'cell4' in n else 1)
    assert s['phase']=='development'
    for j in s['jobs']:
        treatment,control=j['trials'];assert 'cell4' in s['selected'][int(j['id'][-1])]['name']
        assert control['base_lr']==.0375 and control['cwd']==.025
        assert control['warmup_steps']==2900 and treatment['warmup_steps']==500
        j['status']='complete'
    advance(s,c,lambda n:1 if 'frozen' in n else 2)
    assert s['phase']=='review' and not s['jobs']


def test_only_promising_method_gets_untouched_confirmation():
    c=config();s=initial_state(c)
    for j in s['jobs']:j['status']='complete'
    advance(s,c,lambda n:0 if 'cell0' in n else 1)
    for j in s['jobs']:j['status']='complete'
    advance(s,c,lambda n:1 if 'frozen' in n else (1.1 if 'dev0' in n else .9))
    assert s['phase']=='confirmation' and s['winner']['variant']=='outer-only'
    assert {j['trials'][0]['seed'] for j in s['jobs']}=={4,5}
    for j in s['jobs']:
        assert j['trials'][0]['base_lr']==.03375
        assert j['trials'][1]['base_lr']==.0375 and j['trials'][1]['cwd']==.025


def test_shared_zero_action_control_is_one_observation_not_a_fake_run():
    c=config();s=initial_state(c)
    for j in s['jobs']:j['status']='complete'
    advance(s,c,lambda n:0 if 'cell4' in n else 1)
    a,b=s['jobs'];real_control=a['trials'][1]
    removed=b['trials'][1]['name'];b['trials']=b['trials'][:1]
    b['shared_control']=real_control
    assert control_for(b) is real_control
    for j in s['jobs']:j['status']='complete'
    queried=[]
    def score(name):
        queried.append(name)
        return 1 if name==real_control['name'] else (1.1 if 'dev0' in name else .9)
    advance(s,c,score)
    assert removed not in queried
    assert queried.count(real_control['name'])==2
    assert s['development_comparisons'][1]['shared_control'] is True
    assert s['development_comparisons'][1]['control']['name']==real_control['name']
    assert s['winner']['variant']=='outer-only'
    assert len(s['phase_history'][-1]['jobs'][1]['trials'])==1


def failed_development():
    c=config();s=initial_state(c)
    for j in s['jobs']:j['status']='complete'
    advance(s,c,lambda n:0 if 'cell0' in n else 1)
    c['conditional_extension']=dict(cells=[[.030,.01875],[.03375,.0125],[.030,.0125]],shared_control=s['jobs'][0]['trials'][1])
    for j in s['jobs']:j['status']='complete'
    advance(s,c,lambda n:1 if 'frozen' in n else 2)
    return c,s


def test_extension_gate_and_exact_cells_preserve_residual_settings():
    c,s=failed_development();activate_extension(s,c)
    trials=s['jobs'][0]['trials'];assert len(trials)==3
    assert [(t['base_lr'],t['cwd']) for t in trials]==[(.030,.01875),(.03375,.0125),(.030,.0125)]
    for t in trials:
        for key,value in s['extension_incumbent'].items():
            if key not in ('name','base_lr','cwd'):assert t[key]==value
    with pytest.raises(RuntimeError,match='gate'):activate_extension(s,c)
    c,s=failed_development();s['development_comparisons'][1]['difference']=-.0001
    with pytest.raises(RuntimeError,match='gate'):activate_extension(s,c)


def test_extension_never_repeats_failed_incumbent_or_forces_confirmation():
    c,s=failed_development();activate_extension(s,c)
    for j in s['jobs']:j['status']='complete'
    advance(s,c,lambda n:0 if 'cell0' in n else 1)
    assert s['phase']=='review' and not s['jobs']
    c,s=failed_development();activate_extension(s,c)
    for j in s['jobs']:j['status']='complete'
    advance(s,c,lambda n:0 if 'extension1' in n else 1)
    assert s['phase']=='extension-development' and len(s['jobs'][0]['trials'])==1
    assert control_for(s['jobs'][0])['name']==c['conditional_extension']['shared_control']['name']
    s['jobs'][0]['status']='complete'
    advance(s,c,lambda n:1 if 'frozen' in n else 1.1)
    assert s['phase']=='review' and not s['jobs']


def test_extension_only_negative_development_gets_fresh_pairs():
    c,s=failed_development();activate_extension(s,c)
    s['jobs'][0]['status']='complete';advance(s,c,lambda n:0 if 'extension2' in n else 1)
    s['jobs'][0]['status']='complete';advance(s,c,lambda n:1 if 'frozen' in n else .9)
    assert s['phase']=='confirmation'
    assert {j['trials'][0]['seed'] for j in s['jobs']}=={4,5}
    assert all(j['trials'][1]['base_lr']==.0375 and j['trials'][1]['cwd']==.025 for j in s['jobs'])
