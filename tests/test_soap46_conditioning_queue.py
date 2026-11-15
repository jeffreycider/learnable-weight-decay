import copy
import json
from pathlib import Path

from scripts.run_soap46_conditioning import initial_state,advance


def config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/soap46_conditioning_round3.json').read_text())


def test_factorial_has_only_frame_memory_differences_and_fixed_base():
    cfg=config();state=initial_state(cfg)
    trials=[t for j in state['jobs'] for t in j['trials']]
    assert len(trials)==8
    assert {(t['residual_frame'],t['meta_betas'][1]) for t in trials}=={('basis',.99),('parameter',.99),('basis',.999),('parameter',.999)}
    for t in trials:
        assert {k:t[k] for k in cfg['fixed'] if k!='seed'}=={k:v for k,v in cfg['fixed'].items() if k!='seed'}
    for j in state['jobs']:assert [t['seed'] for t in j['trials']]==[0,3]
    before=copy.deepcopy(state)
    advance(state,cfg,lambda _: (_ for _ in ()).throw(AssertionError('no incomplete selection')))
    assert state==before


def test_only_complete_archived_screen_selects_then_fresh_paired6_7():
    cfg=config();state=initial_state(cfg)
    scores={}
    for i,j in enumerate(state['jobs']):
        j['status']='complete'
        for t in j['trials']:scores[t['name']]=3.27-i*.0001
    # A lucky single seed must not select a worse two-seed mean.
    scores[state['jobs'][0]['trials'][0]['name']]=3.0
    scores[state['jobs'][0]['trials'][1]['name']]=3.6
    advance(state,cfg,scores.__getitem__)
    assert state['phase']=='confirmation'
    assert [j['trials'][0]['seed'] for j in state['jobs']]==[6,7]
    for j in state['jobs']:
        learned,frozen=j['trials']
        assert learned['residual_frame']==frozen['residual_frame']=='parameter'
        assert learned['meta_betas']==frozen['meta_betas']==[.9,.999]
        assert learned['warmup_steps']==500 and frozen['warmup_steps']==2900
        assert learned['base_lr']==frozen['base_lr']==.0375
        assert learned['cwd']==frozen['cwd']==.025
        scores[learned['name']]=3.27;scores[frozen['name']]=3.271;j['status']='complete'
    advance(state,cfg,scores.__getitem__)
    assert state['phase']=='complete' and len(state['confirmation_differences'])==2
    assert sum(len(j['trials']) for j in state['phase_history'][0]['jobs'])+sum(len(j['trials']) for j in state['jobs'])==12
