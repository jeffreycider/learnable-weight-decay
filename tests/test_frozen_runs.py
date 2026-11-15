import ast
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]

def helper():
    spec=importlib.util.spec_from_file_location('frozen_runs',ROOT/'scripts/frozen_runs.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def spec():
    return dict(name='approved_lr_test',gpu='existing-owned-lease',
                source_archive='/scratch/staging/frozen.tar.gz',source_sha256='a'*64,
                checkpoint='/scratch/checkpoints/model.pt',checkpoint_sha256='b'*64,
                parameters=dict(steps=1000,lr=.0002,optimizer_mode='adam',action_units='relative',
                                preservation_penalty='deadband',kl_budget=.001))


def test_structured_arguments_preserve_new_runner_flags_and_no_extra_wallcap():
    m=helper();s=m.validate_spec(spec());args=m.trainer_arguments(s,'/workspace/run','/persist/output')
    assert '--steps' in args and args[args.index('--steps')+1]=='1000'
    assert args[args.index('--optimizer-mode')+1]=='adam'
    assert args[args.index('--action-units')+1]=='relative'
    assert args[args.index('--preservation-penalty')+1]=='deadband'
    assert '--wandb' in args and 'timeout' not in args
    assert 'wall_time_seconds' not in s


def test_launch_dryrun_does_not_contact_gpu(tmp_path,monkeypatch,capsys):
    m=helper();path=tmp_path/'spec.json';path.write_text(json.dumps(spec()))
    monkeypatch.setattr(m,'cli',lambda *a,**k:pytest.fail('dryrun contacted GPU'))
    monkeypatch.setattr(sys,'argv',['frozen_runs.py','launch','--spec',str(path)])
    m.main()
    assert json.loads(capsys.readouterr().out)['dry_run'] is True


@pytest.mark.parametrize('change',[dict(name='../overwrite'),dict(source_sha256='bad'),
                                  dict(parameters=dict(output='/other/run')),dict(parameters=dict(lr=float('nan')))])
def test_spec_rejects_unreviewable_or_reserved_fields(change):
    m=helper()
    with pytest.raises(ValueError):m.validate_spec(dict(spec(),**change))


def test_remote_templates_compile_and_launch_refuses_occupied_gpu():
    m=helper()
    for text in (m.LAUNCH,m.POLL,m.ARCHIVE):compile(ast.parse(text),'<remote>','exec')
    assert "if active:raise RuntimeError('GPU has active compute processes" in m.LAUNCH
    assert "os.O_EXCL" in m.LAUNCH
    assert "source archive SHA256 mismatch" in m.LAUNCH
    assert "input checkpoint SHA256 mismatch" in m.LAUNCH
    assert "if 'wall_time_seconds' in s:" in m.LAUNCH
