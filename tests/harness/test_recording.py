import torch
from chisel.recording import cpu_copy, save_atomic

def test_snapshot_is_independent_and_atomic(tmp_path):
    x=torch.ones(2,requires_grad=True);saved=cpu_copy({'x':[x]})
    with torch.no_grad():x.add_(3)
    assert saved['x'][0].tolist()==[1,1]
    assert not saved['x'][0].requires_grad
    path=tmp_path/'state.pt';save_atomic(saved,path)
    assert torch.load(path,weights_only=True)['x'][0].tolist()==[1,1]
    assert not path.with_suffix('.pt.tmp').exists()


def test_rejected_resume_cannot_overwrite_original_provenance(tmp_path, monkeypatch):
    import pytest
    from chisel.harness import RunState
    from chisel.recording import open_run, restore
    monkeypatch.chdir(tmp_path)
    for name in ['chisel/component.py', 'baseline/model.py', 'baseline/data.py']:
        path=tmp_path/name;path.parent.mkdir(exist_ok=True);path.write_text('# original\n')
    monkeypatch.setattr('chisel.recording.importlib.metadata.distributions', lambda: [])
    config=dict(output_dir=str(tmp_path/'run'),train_steps=10)
    state=RunState(extras=dict(rank=0))
    open_run()(config,state)
    root=tmp_path/'run'
    before={p.name:p.read_bytes() for p in root.iterdir()}
    saved=tmp_path/'saved.pt'
    torch.save(dict(config=config,source_hashes=state.extras['source_hashes']),saved)
    (tmp_path/'chisel/component.py').write_text('# incompatible edit\n')
    resumed=dict(config,resume=str(saved))
    attempt=RunState(extras=dict(rank=0))
    open_run()(resumed,attempt)
    with pytest.raises(ValueError,match='source hashes differ'):
        restore()(resumed,attempt)
    assert all((root/name).read_bytes()==value for name,value in before.items())
    records=list((root/'resume_attempts').iterdir())
    assert len(records)==1 and (records[0]/'source.tar.gz').exists()
    import yaml
    assert yaml.safe_load((records[0]/'hyperparameters.yaml').read_text())==resumed
    assert (records[0]/'hyperparameters.yaml').read_bytes()==(records[0]/'recipe.yaml').read_bytes()
