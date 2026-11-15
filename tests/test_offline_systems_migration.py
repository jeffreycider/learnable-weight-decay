import json
from pathlib import Path
import pytest
import torch
from scripts import migrate_offline_systems_checkpoint as m

@pytest.fixture
def fixture(tmp_path,monkeypatch):
    source=tmp_path/'new';(source/'scripts').mkdir(parents=True);(source/'rewriter').mkdir()
    (source/m.TRAINER).write_text('new trainer');(source/m.HELPER).write_text('qualified helper');(source/'rewriter/unchanged.py').write_text('same');(source/m.PLOT_HELPER).write_text('incremental plot')
    monkeypatch.setattr(m,'OLD_TRAINER','old-trainer');monkeypatch.setattr(m,'NEW_TRAINER',m.sha(source/m.TRAINER));monkeypatch.setattr(m,'HELPER_SHA',m.sha(source/m.HELPER));monkeypatch.setattr(m,'PLOT_HELPER_SHA',m.sha(source/m.PLOT_HELPER))
    state=dict(metadata=dict(options={'lr':2e-6,'direction_init':'warmup.pt'},implementation_sha256={m.TRAINER:'old-trainer','rewriter/unchanged.py':m.sha(source/'rewriter/unchanged.py')},gpt_precision={'rms_epsilon':2**-23,'tf32':False},state_bank_identity={'teacher_cache_manifest_sha256':m.CACHE_SHA},world_size=8),controller={'w':torch.arange(6,dtype=torch.float32).reshape(2,3)},optimizer={'state':{0:{'step':torch.tensor(9.),'exp_avg':torch.randn(2,3)}},'param_groups':[{'lr':2e-6,'params':[0]}]},primal_dual={'values':torch.tensor([.1,.2])},fit_schedule={'best':0.,'stale_checks':2},per_rank_rng=[{'torch':torch.get_rng_state(),'cuda':torch.arange(4,dtype=torch.uint8)}]*8,next_epoch=5,next_batch=0,updates=20,completed=False)
    input=tmp_path/'input.pt';torch.save(state,input)
    q=tmp_path/'qualification';q.mkdir();launch=dict(source_archive_sha256=m.ARCHIVE_SHA,teacher_cache_manifest_sha256=m.CACHE_SHA,rms_epsilon=2**-23,source_policy_sha256='policy')
    summary=dict(world_size=8,checkpoint_batch=200,state_ids=list(range(200)),source_policy_sha256='policy',rows=[dict(variant=v,repeat=i,warmup=i==0,seconds=72. if v=='baseline' else 67.,bit_exact_to_baseline=True) for i in range(3) for v in ['baseline','lazy_untimed_allcache']],median_seconds={'baseline':72.,'lazy_untimed_allcache':67.})
    for name,data in [('launch',launch),('summary',summary),('finished',{'exit_code':0})]:(q/(name+'.json')).write_text(json.dumps(data))
    return input,tmp_path/'output.pt',source,q,state

def test_preserves_every_numerical_field(fixture):
    old,new,source,q,state=fixture;oldsha=m.sha(old)
    record=m.migrate(old,new,source,q);got=torch.load(new,weights_only=False)
    assert m.sha(old)==oldsha
    assert m.numerical_digest(got)==m.numerical_digest(state)==record['numerical_state_sha256']
    assert got['metadata']['options']==dict(state['metadata']['options'],offline_systems_cache=True,no_component_timing=True)
    assert got['metadata']['implementation_sha256'][m.HELPER]==m.HELPER_SHA
    assert record['numerical_state_changed'] is False
    assert new.with_suffix('.pt.migration.json').exists()

@pytest.mark.parametrize('failure',['old_source','other_source','helper','plot_helper','wrong_epsilon','wrong_cache','completed','existing_destination','failed_gpu','inexact_trial','no_speedup','duplicate_state','bad_warmup','bad_median','rolling_input'])
def test_rejects_unqualified_transition(fixture,failure):
    old,new,source,q,state=fixture
    if failure=='old_source':state['metadata']['implementation_sha256'][m.TRAINER]='unknown'
    elif failure=='other_source':(source/'rewriter/unchanged.py').write_text('changed')
    elif failure=='helper':(source/m.HELPER).write_text('different')
    elif failure=='plot_helper':(source/m.PLOT_HELPER).write_text('different')
    elif failure=='wrong_epsilon':state['metadata']['gpt_precision']['rms_epsilon']=2**-7
    elif failure=='wrong_cache':state['metadata']['state_bank_identity']['teacher_cache_manifest_sha256']='old-cache'
    elif failure=='completed':state['completed']=True
    elif failure=='existing_destination':new.write_bytes(b'keep')
    elif failure=='rolling_input':old=old.with_name('controller_latest.pt')
    elif failure=='failed_gpu':(q/'finished.json').write_text('{"exit_code":1}')
    else:
        x=json.loads((q/'summary.json').read_text())
        if failure=='inexact_trial':x['rows'][3]['bit_exact_to_baseline']=False
        elif failure=='no_speedup':
            x['median_seconds']['lazy_untimed_allcache']=73.
            for row in x['rows']:
                if row['variant']=='lazy_untimed_allcache':row['seconds']=73.
        elif failure=='duplicate_state':x['state_ids'][0]=x['state_ids'][1]
        elif failure=='bad_warmup':x['rows'][2]['warmup']=True
        elif failure=='bad_median':x['median_seconds']['baseline']=99.
        (q/'summary.json').write_text(json.dumps(x))
    torch.save(state,old)
    with pytest.raises(ValueError):m.migrate(old,new,source,q)
    if failure=='existing_destination':assert new.read_bytes()==b'keep'
    else:assert not new.exists()
