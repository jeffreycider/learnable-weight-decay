import copy
import pytest
import json
from scripts.run_soap46_round2 import advance,claim,paired_trials,select_candidates,validate_adoption


def candidate(name,lr=1e-5,seed=0):
    return dict(name=name,variant='outer-only',meta_lr=lr,meta_clip=1.,meta_betas=[.9,.99],
        warmup_steps=500,warmup_meta='moments',meta_ramp_steps=500,residual_decay=.1,
        seed=seed,base_lr=.0375,cwd=.025,checkpoint_every=0)


def test_bounded_stages_use_development3_then_unseen_paired4_5():
    reference=candidate('old');new=candidate('new',3e-5)
    state=dict(phase='screen',phase_history=[],jobs=[dict(id='s',status='complete',trials=[new])])
    score={'old':3.27,'new':3.269}
    advance(state,reference,score.__getitem__)
    assert state['phase']=='development' and len(state['jobs'])==2
    for i,j in enumerate(state['jobs']):
        treatment,control=j['trials'];assert treatment['seed']==control['seed']==3
        assert treatment['warmup_steps']==500 and control['warmup_steps']==2900
        assert treatment['meta_betas']==control['meta_betas']
        score[treatment['name']]=3.27+i*.001;score[control['name']]=3.272
        j['status']='complete'
    advance(state,reference,score.__getitem__)
    assert state['phase']=='confirmation' and [j['trials'][0]['seed'] for j in state['jobs']]==[4,5]
    for j in state['jobs']:
        t,c=j['trials'];score[t['name']]=3.27;score[c['name']]=3.271;j['status']='complete'
    advance(state,reference,score.__getitem__)
    assert state['phase']=='complete' and state['confirmation_differences']==pytest.approx([-.001,-.001])
    assert 'not project completion' in state['next_action']


def test_no_selection_before_all_archives_and_no_duplicate_worker_claim():
    state=dict(phase='screen',phase_history=[],jobs=[dict(id='p',status='pending',trials=[candidate('new')])])
    original=copy.deepcopy(state);advance(state,candidate('old'),lambda _:pytest.fail('no premature selection'))
    assert state==original
    assert claim(state,'node')['id']=='p'
    assert claim(state,'other') is None
    with pytest.raises(RuntimeError,match='manual review'):claim(state,'node')


def test_candidates_deduplicate_settings_and_exclude_A_frozen():
    ref=candidate('old');same=candidate('same');new=candidate('new',3e-5)
    frozen={**candidate('frozen'),'warmup_steps':2900};a={**candidate('A'),'variant':'inner-outer'}
    state=dict(jobs=[dict(trials=[same,new,frozen,a])])
    scores={'old':3.27,'same':3.269,'new':3.268}
    assert [t['name'] for t in select_candidates(state,ref,scores.__getitem__)]==['new','same']


def test_restart_adopts_only_exact_surviving_panel_without_relaunch(tmp_path):
    path=tmp_path/'job.json';trial=candidate('t')
    config={'numeric_source':{'source_archive':'pin','source_sha256':'abc'}}
    job={'gpu':'node','trials':[trial]}
    path.write_text(json.dumps({**config['numeric_source'],'gpu':'node','trials':[trial]}))
    command=['python','scripts/run_soap46_panel.py','--plan',str(path),'--execute']
    validate_adoption(job,config,'node',path,{},command)
    validate_adoption(job,config,'node',path,{'finished':True},[])
    for bad_command in ([],['python','other.py','--plan',str(path)],command[:-2]+[str(tmp_path/'other.json'),'--execute']):
        with pytest.raises(RuntimeError):validate_adoption(job,config,'node',path,{},bad_command)
    with pytest.raises(RuntimeError,match='failed'):
        validate_adoption(job,config,'node',path,{'error':'bad'},command)
    with pytest.raises(RuntimeError,match='mismatch'):
        validate_adoption(job,config,'wrong-node',path,{},command)
    with pytest.raises(RuntimeError,match='mismatch'):
        validate_adoption(job,{'numeric_source':{'source_sha256':'changed'}},'node',path,{},command)
