"""Supervisor/launcher dry-run and transition checks; never call Strange Loop."""
import ast
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest


ROOT=Path(__file__).resolve().parents[1]

def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_remote_programs_compile_without_escape_corruption():
    supervisor=module('supervise_six_runs')
    launcher=module('launch_seed1_pair')
    for source in [supervisor.OBSERVE,supervisor.PRESERVE,launcher.QUALIFY]:
        tree=ast.parse(source)
        compile(tree,'<remote>','exec')
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='split':
                if node.args and isinstance(node.args[0],ast.Constant):
                    assert node.args[0].value=='\0'


def test_default_supervisor_dryrun_never_calls_cli(monkeypatch,capsys):
    m=module('supervise_six_runs')
    monkeypatch.setattr(m,'cli',lambda *a,**k:pytest.fail('dryrun touched service'))
    monkeypatch.setattr(m.sys,'argv',['supervise_six_runs.py'])
    m.main()
    assert '"new_gpu_allocations": 0' in capsys.readouterr().out


def minimal_supervisor(m,tmp_path):
    s=m.Supervisor.__new__(m.Supervisor)
    s.allowed={'lease'};s.directory=tmp_path
    s.args=SimpleNamespace(predecessors='runs/panel_leases.json')
    s.state=dict(jobs={},released=[],errors={},launch_attempts={},deadline=10**12)
    s.save=lambda:None;s.log=lambda *a,**k:None
    return s


def test_predecessor_not_released_until_successor_is_preserved(tmp_path,monkeypatch):
    m=module('supervise_six_runs');s=minimal_supervisor(m,tmp_path)
    old=dict(label='baseline_seed0',gpu='lease',phase='preserved',seed=0)
    new=dict(label='reference_seed1',gpu='lease',phase='running',seed=1)
    s.state['jobs']={old['label']:old,new['label']:new}
    calls=[];monkeypatch.setattr(m,'cli',lambda *a,**k:calls.append(a))
    s.release_if_unused(old);assert calls==[]
    new['phase']='preserved'
    s.release_if_unused(old);assert len(calls)==1 and calls[0][:3]==('gpu','down','lease')
    s.release_if_unused(new);assert len(calls)==1


def test_ambiguous_launch_is_never_automatically_replayed(tmp_path,monkeypatch):
    m=module('supervise_six_runs');s=minimal_supervisor(m,tmp_path)
    old=dict(label='baseline_seed0',gpu='lease',phase='preserved',seed=0)
    calls=[]
    def timeout(*a,**k):
        calls.append(a)
        raise subprocess.TimeoutExpired('launcher',600)
    monkeypatch.setattr(m.subprocess,'run',timeout)
    with pytest.raises(subprocess.TimeoutExpired):s.launch_follow_on(old)
    assert 'reference_seed1' in s.state['launch_attempts']
    with pytest.raises(RuntimeError,match='ambiguous'):s.launch_follow_on(old)
    assert len(calls)==1


def test_unknown_lease_rejected_without_service_call(tmp_path):
    m=module('supervise_six_runs');s=minimal_supervisor(m,tmp_path)
    with pytest.raises(ValueError,match='outside'):s.assert_lease('someone-elses-gpu')
