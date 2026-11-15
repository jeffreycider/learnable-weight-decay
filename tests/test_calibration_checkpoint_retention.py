"""Retention regression on a new tiny CPU trajectory, not recovery of lost actors."""
import hashlib
import json
import struct

import numpy as np
import pytest
import torch

import scripts.train_multistate_chisel as trainer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_exact(left,right):
    if isinstance(left,torch.Tensor):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:assert_exact(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):assert_exact(a,b)
    else:assert left==right


def test_immutable_save_refuses_existing_and_racing_collisions(tmp_path,monkeypatch):
    path=tmp_path/'controller_update0064.pt'
    state={'updates':64,'controller':{'w':torch.tensor([1.,2.])}}
    trainer.save_calibration_checkpoint(path,state);before=digest(path)
    with pytest.raises(FileExistsError):trainer.save_calibration_checkpoint(path,state)
    assert digest(path)==before
    real_link=trainer.os.link
    racing=tmp_path/'controller_update0128.pt'
    def competing_writer(source,destination):
        destination.write_bytes(b'existing concurrent writer')
        real_link(source,destination)
    monkeypatch.setattr(trainer.os,'link',competing_writer)
    with pytest.raises(FileExistsError):trainer.save_calibration_checkpoint(racing,{'updates':128})
    assert racing.read_bytes()==b'existing concurrent writer'
    assert not list(tmp_path.glob('.calibration-checkpoint-*'))


def test_failed_staging_never_publishes_a_partial_checkpoint(tmp_path,monkeypatch):
    def fail(path,state):
        path.write_bytes(b'incomplete')
        raise OSError('simulated incomplete serialization')
    monkeypatch.setattr(trainer,'save_atomic',fail)
    path=tmp_path/'controller_update0064.pt'
    with pytest.raises(OSError):trainer.save_calibration_checkpoint(path,{'updates':64})
    assert not path.exists() and not list(tmp_path.glob('.calibration-checkpoint-*'))


def test_default_off_trainer_does_not_publish_retained_calibration_files(tmp_path,monkeypatch):
    from test_multistate_chisel import test_two_epoch_state_bank_training_audits_and_canonical_checkpoint as legacy_run
    def forbidden(*args,**kwargs):raise AssertionError('calibration retention called by default-off trainer')
    monkeypatch.setattr(trainer,'save_calibration_checkpoint',forbidden)
    legacy_run(tmp_path,[])
    assert not list((tmp_path/'run').glob('controller_update*.pt'))


def test_actual_64_to_128_resume_preserves_bytes_and_all_training_state(tmp_path):
    from test_multistate_chisel import _tiny_state_bank
    from baseline.model import GPT
    from rewriter.function_preservation import FunctionalPreservation
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import write_panel
    torch.set_num_threads(1)
    bank=tmp_path/'bank';_tiny_state_bank(bank)
    data=tmp_path/'fineweb_train_000.bin';tokens=(np.arange(1000)%16).astype(np.uint16)
    data.write_bytes(struct.pack('<iii',20240520,1,len(tokens))+bytes(1012)+tokens.tobytes())
    reader=StateBankReader(bank);cache=tmp_path/'cache';cache.mkdir();panels=[]
    engine=FunctionalPreservation(GPT(16,1,128).float().requires_grad_(False),
        microbatch_sequences=1,kl_dtype=torch.float64,gpt_precision='fp32')
    for state_id in range(1,6):
        record=reader.read(state_id)
        engine.load_teacher_state({n:record.model[n] for n in engine.parameters},{n:record.model[n] for n in engine.buffers})
        pool=trainer.state_replay([str(data)],state_id,reader.manifest['provenance']['training_config'],
            pool_sequences=32,seed=400001,epoch=0,split_seed=500001)
        for role in ('train','audit'):
            inputs,ids=trainer.select_sequences(pool[role],2,seed=trainer.stable_seed('fixed_teacher_panel',400001,state_id,role))
            panels.append(write_panel(cache,state_id,role,engine,inputs,ids,
                state_sha256=record.assert_verified(),execution={'gpt_precision':'fp32'}))
    (cache/'manifest.json').write_text(json.dumps(dict(complete=True,args={'states':5},execution={'gpt_precision':'fp32'},panels=panels)))
    output=tmp_path/'calibration'
    args=['--state-bank',str(bank),'--teacher-cache',str(cache),'--train-data',str(data),'--output',str(output),
        '--checkpoint-batch','1','--sequences-per-checkpoint','1','--gpt-microbatch','1','--replay-pool-sequences','32',
        '--lr','2e-5','--expected-gpt-seed','7','--audit-states','1','--audit-sequences','1','--eager-chisel','--device','cpu',
        '--no-plots','--chisel-model-dim','128','--chisel-layers','1','--gpt-precision','fp32','--objective','tokenwise-v1',
        '--kl-tolerance','.001','--token-tail-tolerance','.002','--token-normalization-scale','.01','--augmented-rho','.03',
        '--checkpoint-every','17','--calibration-state-ids','1','--calibration-readout-updates','0','1','64','128']
    trainer.main([*args,'--epochs','64'])
    retained=output/'controller_update0064.pt';hash64=digest(retained)
    old=torch.load(retained,weights_only=False);latest=torch.load(output/'controller_latest.pt',weights_only=False)
    fields=('controller','optimizer','primal_dual','fit_schedule','updates','next_epoch','next_batch',
            'per_rank_rng','per_rank_compute','clipped_updates','completed','metadata')
    for key in fields:assert_exact(old[key],latest[key])
    assert old['updates']==old['next_epoch']==old['primal_dual']['updates']==64
    assert old['next_batch']==0 and old['fit_schedule'] is None and not old['completed']
    assert old['optimizer']['state'] and all(int(s['step'])==64 for s in old['optimizer']['state'].values())
    assert old['per_rank_compute'][0]['student_backwards']==64
    names_before={p.name:digest(p) for p in output.glob('controller_update*.pt')}
    assert set(names_before)=={f'controller_update{i:04d}.pt' for i in (0,1,17,34,51,64)}
    trainer.main([*args,'--epochs','128','--resume',str(retained)])
    assert digest(retained)==hash64
    for name,value in names_before.items():assert digest(output/name)==value
    assert_exact(old,torch.load(retained,weights_only=False))
    new=torch.load(output/'controller_update0128.pt',weights_only=False)
    latest=torch.load(output/'controller_latest.pt',weights_only=False)
    for key in fields:assert_exact(new[key],latest[key])
    assert new['updates']==new['next_epoch']==new['primal_dual']['updates']==128
    assert all(int(s['step'])==128 for s in new['optimizer']['state'].values())
    assert new['per_rank_compute'][0]['student_backwards']==128
    assert set(p.name for p in output.glob('controller_update*.pt'))=={
        f'controller_update{i:04d}.pt' for i in (0,1,17,34,51,64,68,85,102,119,128)}
    assert old['metadata']['options']['epochs']==64 and new['metadata']['options']['epochs']==128
