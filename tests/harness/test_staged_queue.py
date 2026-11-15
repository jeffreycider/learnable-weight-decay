import yaml
from chisel.staged_queue import staged_recipe


def test_staging_preserves_training_contract_and_changes_only_storage_and_microbatch(tmp_path):
    original = dict(output_dir='/shared/seed7', microbatch_size=32,
        pre_forward=[], post_optimizer=[dict(name='chisel.evaluation.evaluate', hyperparams=dict(every=125, microbatch_size=32))],
        data=dict(batch_size=524288, seq_len=1024), optimizers=dict(hidden=dict(lr=.018)),
        seeds=dict(initialization=7), train_steps=3325, runtime=dict(world_size=8))
    path=tmp_path/'source.yaml';path.write_text(yaml.safe_dump(original))
    staged=staged_recipe(path, tmp_path/'local', 8)
    for key in ('data','optimizers','seeds','train_steps'):
        assert staged[key]==original[key]
    assert staged['microbatch_size']==8
    assert staged['output_dir']==str(tmp_path/'local/seed7-local-stage')
    assert yaml.safe_load(path.read_text())==original
    grouped=staged_recipe(path,tmp_path/'grouped',8,world_size=2,space_floor=100)
    assert grouped['runtime']['world_size']==2 and grouped['data']==original['data']
    assert grouped['pre_forward'][1]['name']=='chisel.capture.wait_for_space'


def test_space_guard_waits_before_model_work(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import torch
    from chisel.harness import RunState
    from chisel.capture import wait_for_space
    free=iter([0,200]);sleeps=[]
    monkeypatch.setattr('chisel.capture.shutil.disk_usage',lambda _:SimpleNamespace(free=next(free)))
    monkeypatch.setattr('chisel.capture.time.sleep',sleeps.append)
    state=RunState(extras=dict(rank=0,world_size=1,device=torch.device('cpu')))
    wait_for_space(100,poll_seconds=3)(dict(output_dir=str(tmp_path)),state)
    assert sleeps==[3]
