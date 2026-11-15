"""Supervisor lifecycle checks use no GPU or external services."""
import json
import sys
import pytest
from pathlib import Path
from scripts import run_soap46_panel as panel


def test_only_safe_progress_operations_retry_transient_transport(monkeypatch):
    from types import SimpleNamespace
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1 if len(calls)==1 else 0,
            stderr='HTTP 502 temporary gateway error',stdout='ok')
    monkeypatch.setattr(panel.subprocess,'run',run)
    monkeypatch.setattr(panel.time,'sleep',lambda _:None)
    assert panel.command(['python','scripts/snapshot_soap46_metrics.py'])=='ok'
    assert len(calls)==2
    calls.clear()
    with pytest.raises(RuntimeError,match='HTTP 502'):
        panel.command(['python','scripts/launch_soap46.py'])
    assert len(calls)==1


def test_explicit_boundary_pause_preserves_before_handoff_and_never_autoresumes(tmp_path,monkeypatch):
    path=tmp_path/'panel.json';marker=path.with_suffix('.pause.json')
    marker.write_text(json.dumps({'reason':'authorized independent qualification'}))
    calls=[];pending=['finished-run']
    monkeypatch.setattr(panel,'script',lambda *args:calls.append(args) or '{}')
    def hand_back(_):
        ack=json.loads(path.with_suffix('.paused.json').read_text())
        assert ack['preserved'] and ack['completed']==['finished-run']
        assert calls[0][0]=='archive_soap46.py' and not pending
        marker.unlink()  # Explicit owner handback, not a timeout.
    monkeypatch.setattr(panel.time,'sleep',hand_back)
    panel.pause_at_boundary(path,'owned',{'completed':['finished-run']},pending)
    assert not path.with_suffix('.paused.json').exists()


@pytest.mark.parametrize("release",[True,False])
@pytest.mark.parametrize("checkpoint_every",[0,1000])
def test_checkpoint_archives_during_trial_and_completed_run_after_next_launch(tmp_path,monkeypatch,release,checkpoint_every):
    monkeypatch.setattr(panel,'ROOT',tmp_path);monkeypatch.setattr(panel.time,'sleep',lambda _:None)
    trials=[dict(name=f'trial{i}',variant='outer-only',meta_lr=1e-5,warmup_steps=500,meta_ramp_steps=500,residual_decay=.1,seed=2,base_lr=.03,cwd=.0125,meta_clip=.003,meta_betas=[.5,.999],checkpoint_every=checkpoint_every) for i in range(2)]
    plan=dict(gpu='owned-gpu',archive='/scratch/pinned.tar.gz',sha256='pinned',source_commit='fixed',already_mounted=True,trials=trials,release_on_completion=release)
    path=tmp_path/'panel.json';path.write_text(json.dumps(plan));events=[];polls={}
    def cli(*args):
        events.append(('cli',args))
        if args[:2]==('gpu','status'):return json.dumps(dict(status='ready',expires_in_seconds=10000,artifact_dir='/persist/owned'))
        if args[:2]==('gpu','exec'):return ''
        return '{}'
    def script(name,*args):
        args=list(map(str,args));run=args[args.index('--name')+1] if '--name' in args else None
        if name=='launch_soap46.py':
            assert args[args.index('--seed')+1]=='2'
            assert args[args.index('--base-lr')+1]=='0.03'
            assert args[args.index('--cwd')+1]=='0.0125'
            assert args[args.index('--meta-clip')+1]=='0.003'
            assert args[args.index('--meta-betas')+1:args.index('--meta-betas')+3]==['0.5','0.999']
            assert args[args.index('--checkpoint-every')+1]==str(checkpoint_every)
            events.append(('launch',run));return '{}'
        if name=='poll_soap46.py':
            run=Path(args[args.index('--output')+1]).name;i=polls.get(run,0);polls[run]=i+1;step=(1200,2200,2900)[i]
            events.append(('poll',run,step))
            return json.dumps(dict(latest_train=dict(step=step),latest_validation=None,wandb=[],launcher_alive=True,
                                   finished=dict(completed_steps=2900) if step==2900 else None))
        if name=='archive_soap46.py':
            cp=int(args[args.index('--checkpoint-step')+1]) if '--checkpoint-step' in args else 0
            events.append(('archive',run,cp));return '{}'
        if name=='snapshot_soap46_metrics.py':return '{}'
        raise AssertionError(name)
    monkeypatch.setattr(panel,'cli',cli);monkeypatch.setattr(panel,'script',script)
    monkeypatch.setattr(sys,'argv',['panel','--plan',str(path),'--execute']);panel.main()
    for run in ('trial0','trial1'):
        if checkpoint_every:
            assert events.index(('poll',run,1200))<events.index(('archive',run,1000))<events.index(('poll',run,2200))
            assert events.index(('poll',run,2200))<events.index(('archive',run,2000))<events.index(('poll',run,2900))
        else:
            assert not any(e[0]=='archive' and e[1]==run and e[2] for e in events)
    assert events.index(('launch','trial1'))<events.index(('archive','trial0',0))<events.index(('poll','trial1',1200))
    assert (('cli',('gpu','down','owned-gpu')) in events)==release
    result=json.loads(path.with_suffix('.progress.json').read_text())
    assert result['finished'] and result['completed']==['trial0','trial1']
    assert result['checkpoint_archives']=={'trial0':[1000,2000] if checkpoint_every else [],'trial1':[1000,2000] if checkpoint_every else []}


def test_restart_adopts_active_trial_and_does_not_repeat_checkpoint_archives(tmp_path,monkeypatch):
    monkeypatch.setattr(panel,'ROOT',tmp_path);monkeypatch.setattr(panel.time,'sleep',lambda _:None)
    trial=dict(name='adopted',variant='outer-only',meta_lr=1e-5,warmup_steps=500,meta_ramp_steps=500,residual_decay=.1)
    plan=dict(gpu='owned-gpu',archive='/scratch/pinned',sha256='pinned',source_commit='fixed',trials=[trial]);path=tmp_path/'panel.json';path.write_text(json.dumps(plan))
    path.with_suffix('.progress.json').write_text(json.dumps(dict(completed=[],pid=-1,active='adopted',checkpoint_archives={'adopted':[1000,2000]})))
    record=tmp_path/'runs/soap46-v1/adopted-launch.json';record.parent.mkdir(parents=True);record.write_text(json.dumps(dict(**trial,gpu='owned-gpu',sha256='pinned',source_commit='fixed')))
    calls=[]
    def cli(*args):
        if args[:2]==('gpu','status'):return json.dumps(dict(status='ready',expires_in_seconds=10000,artifact_dir='/persist/owned'))
        return ''
    def script(name,*args):
        calls.append((name,args))
        assert name!='launch_soap46.py'
        if name=='poll_soap46.py':return json.dumps(dict(latest_train=dict(step=2900),latest_validation=None,wandb=[],finished=dict(completed_steps=2900)))
        if name=='archive_soap46.py':assert '--checkpoint-step' not in args
        return '{}'
    monkeypatch.setattr(panel,'cli',cli);monkeypatch.setattr(panel,'script',script);monkeypatch.setattr(sys,'argv',['panel','--plan',str(path),'--execute'])
    panel.main();result=json.loads(path.with_suffix('.progress.json').read_text());assert result['pid']!=-1 and result['finished']


def test_followup_selection_neighbors_and_heldout_controls_are_bounded():
    from scripts.run_soap46_followup import best_trial,neighbors,confirmations
    a=dict(name='a',variant='outer-only',meta_lr=1e-5,warmup_steps=500,meta_ramp_steps=500,residual_decay=.1)
    b={**a,'name':'b','meta_lr':1e-4}
    frozen={**a,'name':'frozen','warmup_steps':2900}
    winner=best_trial([a,b,frozen],lambda name:{'a':3.2,'b':3.3,'frozen':3.1}[name])
    assert winner==a
    near=neighbors(winner,'B');assert len(near)==4
    assert {(t['base_lr'],t['cwd']) for t in near}=={(.03,.025),(.045,.025),(.0375,.0125),(.0375,.0375)}
    runs=confirmations(near[0],'B');assert len(runs)==4
    for seed in (1,2):
        treatment,control=[t for t in runs if t['seed']==seed]
        assert treatment['base_lr']==.03 and treatment['warmup_steps']==500
        assert control['base_lr']==.0375 and control['cwd']==.025 and control['warmup_steps']==2900
        assert control['meta_lr']>0


def test_followup_recognizes_only_explicit_replacement_supervisor(tmp_path):
    from scripts.run_soap46_followup import replacement_supervisor_alive
    plan=tmp_path/'plan.json';proc=tmp_path/'proc';(proc/'123').mkdir(parents=True)
    assert not replacement_supervisor_alive(plan,proc)
    plan.with_suffix('.pid').write_text('123\n');(proc/'123/cmdline').write_bytes(b'python\x00unrelated.py')
    assert not replacement_supervisor_alive(plan,proc)
    (proc/'123/cmdline').write_bytes(('python\0run_soap46_panel.py\0--plan\0'+str(plan)).encode())
    assert replacement_supervisor_alive(plan,proc)


def test_followup_rejects_executed_edits_in_nominal_frozen_control(tmp_path,monkeypatch):
    from scripts import run_soap46_followup as followup
    monkeypatch.setattr(followup,'ROOT',tmp_path)
    d=tmp_path/'runs/soap46-v1/control';d.mkdir(parents=True)
    (d/'archive_result.json').write_text('{}')
    (d/'config.json').write_text(json.dumps(dict(warmup_steps=2900,base_lr=.0375,cwd=.025,warmup_meta='moments',meta_lr=1e-5)))
    rows=[dict(event='train',step=i,basis_residual_norm=0.,residual_updates=0,zero_lr_moment_updates=max(0,i-2)) for i in range(1,2901)]
    rows.append(dict(event='validation',step=2900,val_loss=3.27))
    path=d/'metrics.jsonl';path.write_text('\n'.join(map(json.dumps,rows)))
    assert followup.endpoint('control')==3.27
    rows[900]['basis_residual_norm']=.1;path.write_text('\n'.join(map(json.dumps,rows)))
    with pytest.raises(RuntimeError,match='nonzero residual'):followup.endpoint('control')


def test_attribution_is_only_two_same_base_controls_after_two_promising_seeds():
    from scripts.run_soap46_followup import needs_attribution,attribution_controls
    winner=dict(base_lr=.03,cwd=.025,meta_lr=1e-5,warmup_steps=500,variant='outer-only')
    assert needs_attribution(winner,[-.001,-.002])
    assert not needs_attribution(winner,[-.001,.002])
    assert not needs_attribution({**winner,'base_lr':.0375},[-.001,-.002])
    controls=attribution_controls(winner,'B');assert len(controls)==2
    assert [c['seed'] for c in controls]==[1,2]
    assert all(c['base_lr']==.03 and c['cwd']==.025 and c['warmup_steps']==2900 for c in controls)


def test_outer_supervisor_restart_adopts_existing_stage_without_new_launch(tmp_path,monkeypatch):
    from scripts import run_soap46_followup as followup
    import subprocess
    plan=tmp_path/'stage.json';base=dict(gpu='owned');trials=[dict(name='trial')]
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)',str(plan)])
    try:
        progress=plan.with_suffix('.progress.json');progress.write_text(json.dumps(dict(pid=child.pid,completed=[])))
        monkeypatch.setattr(followup.time,'sleep',lambda _:progress.write_text(json.dumps(dict(pid=child.pid,finished=True))))
        monkeypatch.setattr(followup.subprocess,'run',lambda *a,**k:pytest.fail('must not launch a duplicate stage'))
        followup.run_stage(plan,base,trials)
    finally:
        child.terminate();child.wait(timeout=5)


def test_clean_handoff_requires_ready_pinned_successor_otherwise_releases(tmp_path,monkeypatch):
    from scripts import run_soap46_followup as followup
    import hashlib
    calls=[];monkeypatch.setattr(followup.panel,'cli',lambda *args:calls.append(args))
    path=tmp_path/'followup-A-v1.json';state=dict(lane='A',finished=True)
    followup.release_or_handoff('owned',state,path);assert calls==[('gpu','down','owned')]
    calls.clear();state=dict(lane='A',finished=True)
    launcher=tmp_path/'launch.py';launcher.write_text('pass');archive=tmp_path/'source.tar';archive.write_bytes(b'pinned')
    marker=tmp_path/'handoff-A-v1.json';marker.write_text(json.dumps(dict(gpu='owned',ready=True,next_owner='next',launcher=str(launcher),source_archive=str(archive),source_sha256=hashlib.sha256(b'pinned').hexdigest())))
    followup.release_or_handoff('owned',state,path);assert not calls and state['handoff_ready']['next_owner']=='next'
    archive.write_bytes(b'changed');state=dict(lane='A',finished=True)
    followup.release_or_handoff('owned',state,path);assert calls==[('gpu','down','owned')] and 'handoff_skipped' in state
