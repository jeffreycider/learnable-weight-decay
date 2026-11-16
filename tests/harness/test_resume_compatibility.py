from copy import deepcopy
import pytest
import torch
import yaml
from chisel.harness import run
from chisel.resume_compatibility import validate
from chisel.storage import sha256
from tests.harness.test_checkpoint_duals import configuration


def test_resume_does_not_rerun_stage_initialization(tmp_path):
    from chisel.offline import initialize_shared_weights, initialize_fitted_probes
    from chisel.harness import RunState
    for constructor in [initialize_shared_weights, initialize_fitted_probes]:
        constructor(str(tmp_path/'unused-initialization.pt'),'unused')(
            dict(resume='committed-checkpoint.pt'),RunState())


def declaration(tmp_path, fields, changes):
    path=tmp_path/'compatibility.yaml'
    path.write_text(yaml.safe_dump(dict(configuration_fields=fields,source_changes=changes)))
    return dict(path=str(path),sha256=sha256(path))


def test_cache_migration_preserves_full_training_trajectory(tmp_path):
    config=configuration(tmp_path)
    reference=run(config)
    config.update(output_dir=str(tmp_path/'prefix'),stop_step=2)
    run(config)
    config.update(output_dir=str(tmp_path/'migrated'),stop_step=4,
        resume=str(tmp_path/'prefix/latest_rank0.pt'),
        resume_compatibility=declaration(tmp_path,['private_probes.cache_bytes'],{}))
    config['private_probes']['cache_bytes']=1<<20
    actual=run(config)
    torch.testing.assert_close(actual.model.state_dict(),reference.model.state_dict(),rtol=0,atol=0)
    torch.testing.assert_close(actual.optimizer.state_dict(),reference.optimizer.state_dict(),rtol=0,atol=0)
    assert actual.extras['checkpoint_duals'].state_dict()==reference.extras['checkpoint_duals'].state_dict()
    a,b=actual.extras['private_probes'].state_dict(),reference.extras['private_probes'].state_dict()
    for identity,path in a['latest'].items():
        left=torch.load(path,weights_only=False);right=torch.load(b['latest'][identity],weights_only=False)
        torch.testing.assert_close(left,right,rtol=0,atol=0)


def test_migration_rejects_unlisted_training_or_source_changes(tmp_path):
    saved=dict(config=dict(optimizers={'lr':.01},private_probes={}),source_hashes={'a.py':'old'})
    config=deepcopy(saved['config'])
    config['resume_compatibility']=declaration(tmp_path,['private_probes.cache_bytes'],{'a.py':dict(captured='old',current='new')})
    config['private_probes']['cache_bytes']=100
    validate(saved,config,{'a.py':'new'})
    with pytest.raises(ValueError,match='unqualified'):
        validate(saved,config,{'a.py':'other'})
    config['optimizers']['lr']=.02
    with pytest.raises(ValueError,match='training configuration'):
        validate(saved,config,{'a.py':'new'})
    config['resume_compatibility']=declaration(tmp_path,['optimizers.lr'],{})
    with pytest.raises(ValueError,match='performance fields'):
        validate(saved,config,{'a.py':'new'})
    config['resume_compatibility']['sha256']='wrong'
    with pytest.raises(ValueError,match='SHA256'):
        validate(saved,config,{'a.py':'new'})


def test_device_cache_migration_preserves_full_offline_updates(tmp_path):
    config=configuration(tmp_path)
    reference=run(config)
    config.update(output_dir=str(tmp_path/'prefix_device'),stop_step=2)
    run(config)
    config.update(output_dir=str(tmp_path/'migrated_device'),stop_step=4,
        resume=str(tmp_path/'prefix_device/latest_rank0.pt'),
        resume_compatibility=declaration(tmp_path,['private_store'],{}),
        private_store=dict(_target_='chisel.device_probe_store.DeviceProbeStore',cache_bytes=1<<20))
    actual=run(config)
    torch.testing.assert_close(actual.model.state_dict(),reference.model.state_dict(),rtol=0,atol=0)
    torch.testing.assert_close(actual.optimizer.state_dict(),reference.optimizer.state_dict(),rtol=0,atol=0)
    assert actual.extras['checkpoint_duals'].state_dict()==reference.extras['checkpoint_duals'].state_dict()
    a,b=actual.extras['private_probes'].state_dict(),reference.extras['private_probes'].state_dict()
    for identity,path in a['latest'].items():
        torch.testing.assert_close(torch.load(path,weights_only=False),torch.load(b['latest'][identity],weights_only=False),rtol=0,atol=0)
    assert actual.extras['private_probes'].device_hits>0


def test_private_store_migration_rejects_unqualified_implementations(tmp_path):
    saved=dict(config={},source_hashes={})
    config=dict(private_store={'_target_':'unqualified.OtherStore'},
        resume_compatibility=declaration(tmp_path,['private_store'],{}))
    with pytest.raises(ValueError,match='unqualified private store'):
        validate(saved,config,{})
