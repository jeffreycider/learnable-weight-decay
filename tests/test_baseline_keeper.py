import ast
import sys
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_keeper_dry_run_never_contacts_service(monkeypatch,capsys):
    m=load('keep_baselines')
    monkeypatch.setattr(m,'cli',lambda *a,**k:pytest.fail('dry-run contacted service'))
    monkeypatch.setattr(m.sys,'argv',['keep_baselines.py'])
    m.main()
    text=capsys.readouterr().out
    assert '"max_hours": 6' in text and '"new_training_runs": 0' in text
    assert set(m.JOBS)=={'baseline_seed0','baseline_seed1'}


def test_keeper_mutation_failure_is_not_replayed(tmp_path):
    m=load('keep_baselines');keeper=m.Keeper(tmp_path)
    calls=[]
    def fail():
        calls.append(1);raise RuntimeError('ambiguous timeout')
    assert keeper.mutate('capture_rolling',fail) is None
    assert keeper.mutate('capture_rolling',fail) is None
    assert len(calls)==1
    assert 'capture_rolling' in keeper.state['attempts']
    assert 'capture_rolling' in keeper.state['errors']


def test_release_requires_final_archive_and_closed_trainer(tmp_path,monkeypatch):
    m=load('keep_baselines');keeper=m.Keeper(tmp_path)
    monkeypatch.setattr(m,'cli',lambda *a,**k:pytest.fail('unsafe release attempted'))
    result=dict(active=[],finished={'steps':3250})
    with pytest.raises(RuntimeError,match='verified final'):keeper.release('baseline_seed0',result)
    keeper.state['preserved']['baseline_seed0']={}
    with pytest.raises(RuntimeError,match='active'):keeper.release('baseline_seed0',dict(result,active=['123']))


def test_remote_capture_program_preserves_single_inode_and_checks_step():
    m=load('keep_baselines');compile(ast.parse(m.CAPTURE),'<capture>','exec')
    assert "with src.open('rb') as inp,tmp.open('xb') as out:" in m.CAPTURE
    assert "saved['data_cursor']['batches']==actual" in m.CAPTURE
    assert "subprocess.run(['sync','/scratch'],check=True)" in m.CAPTURE
    assert "source_commit='102ab99'" not in m.PRESERVE


@pytest.mark.parametrize('name',['supervise_six_runs','launch_seed1_pair'])
def test_obsolete_execute_entrypoints_fail_before_service_access(name,monkeypatch):
    m=load(name)
    monkeypatch.setattr(m,'cli',lambda *a,**k:pytest.fail('obsolete launcher contacted service'))
    monkeypatch.setattr(sys,'argv',[name+'.py','--execute'])
    with pytest.raises(SystemExit,match='SUPERSEDED'):m.main()


def test_keeper_log_preserves_checkpoint_timestamp_and_label(tmp_path,monkeypatch,capsys):
    import json
    m=load('keep_baselines')
    monkeypatch.setattr(m.time,'time',lambda:2000.)
    keeper=m.Keeper(tmp_path)
    original_state=(tmp_path/'state.json').read_bytes()
    metadata=dict(time=1500.,label='baseline_seed1',step=250,seed=1,
                  source='/persist/run/latest_rank0.pt',destination='/scratch/seed1_step250.pt')
    keeper.log('rolling_preserved',**metadata)
    record=json.loads((tmp_path/'events.jsonl').read_text())
    assert record['time']==2000. and record['source_time']==1500.
    assert record['event']=='rolling_preserved'
    assert record['label']=='baseline_seed1' and record['step']==250
    assert metadata['time']==1500. and 'source_time' not in metadata
    assert json.loads(capsys.readouterr().out)==record
    assert (tmp_path/'state.json').read_bytes()==original_state


def test_capture_rolling_logs_real_manifest_shape_and_retains_deadline(tmp_path,monkeypatch):
    import json
    m=load('keep_baselines');keeper=m.Keeper(tmp_path)
    deadline=keeper.state['deadline']
    row=dict(step=250,seed=1,source='/persist/latest_rank0.pt',
             destination='/scratch/seed1_step250.pt',sha256='a'*64,bytes=1376913691,time=1794182175.)
    monkeypatch.setattr(keeper,'remote',lambda *a,**k:dict(preserved=[row],latest_step=250))
    monkeypatch.setattr(m,'cli',lambda *a,**k:pytest.fail('fixture contacted service'))
    keeper.capture_rolling()
    saved=json.loads((tmp_path/'state.json').read_text())
    assert saved['rolling']['250']==row
    assert saved['deadline']==deadline
    assert not saved['errors'] and not saved['attempts']
    assert json.loads((tmp_path/'rolling/seed1_step250.json').read_text())==row
    events=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert events[-1]['source_time']==row['time']
    assert events[-1]['event']=='rolling_preserved'
    # Reopening the same keeper state must not extend the six-hour budget.
    restored=m.Keeper(tmp_path)
    assert restored.state['deadline']==deadline
    assert restored.state['rolling']['250']==row
