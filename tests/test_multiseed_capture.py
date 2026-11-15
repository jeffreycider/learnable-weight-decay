import json
from pathlib import Path
import pytest
from scripts.launch_multiseed_capture import configuration, ROOT


def test_seed_expansion_changes_only_seed_and_paths():
    baseline=json.loads((ROOT/'configs/state_bank1000_seed0.json').read_text())
    for seed in range(1,10):
        actual=configuration(seed)
        assert actual['seed']==seed and f'seed{seed}-v1' in actual['state_bank']['directory']
        actual['seed']=0
        for key in ('train_data','val_data'):actual[key]=baseline[key]
        actual['state_bank']['directory']=baseline['state_bank']['directory']
        assert actual==baseline


def test_only_authorized_new_seeds():
    for seed in (0,10,-1):
        with pytest.raises(ValueError):configuration(seed)


def test_frozen_plan_full_state_and_panel_coverage():
    plan=json.loads((ROOT/'runs/multiseed-capture/plan.json').read_text())
    seeds=[s for lane in plan['lanes'] for s in lane['seeds']]
    assert sorted(seeds)==list(range(1,10)) and len(seeds)==len(set(seeds))
    assert plan['teacher_cache']['train_sequences']==128
    assert plan['teacher_cache']['audit_sequences']==64
    for seed,config in plan['capture_configs'].items():
        assert config==configuration(int(seed))
        assert config['state_bank']['steps']==config['max_steps']==1000
        assert config['batch_tokens']==524288 and config['schedule_steps']==3250


def test_final_archiver_preserves_all_rank_rng_and_cursors(tmp_path,monkeypatch):
    import hashlib,sys,torch
    from scripts import archive_multiseed_capture as archive
    out=tmp_path/'run';bank=tmp_path/'bank';out.mkdir();bank.mkdir()
    entries=[]
    for step in range(1,1001):
        p=bank/f'state_{step:04d}.pt';p.write_bytes(b'fixture')
        entries.append(dict(completed_steps=step,file=p.name,file_bytes=p.stat().st_size))
    index=bank/'index.jsonl';index.write_text(''.join(json.dumps(r)+'\n' for r in entries))
    (bank/'complete.json').write_text(json.dumps(dict(steps=1000,index_sha256=archive.digest(index))))
    (bank/'manifest.json').write_text('{}')
    train=[dict(event='train',step=s,train_loss=4.,capture_allones_relative_error=.0005,training_forwards_per_rank=s,training_backwards_per_rank=s,captured_states=s,capture_extra_gpt_forwards=0,capture_extra_gpt_backwards=0) for s in range(1,1001)]
    (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in [*train,dict(event='finished',steps=1000)]))
    for rank in range(8):
        torch.save(dict(step=1000,config=dict(seed=1,world_size=8),data_cursor=dict(batches=1000,rank=rank),rng=dict(cpu=torch.tensor([rank])),optimizers=[{},{}]),out/f'final_rank{rank}.pt')
    launch=out/'launch.json';launch.write_text(json.dumps(dict(output=str(out),seed=1,source_sha256='a'*64,config=dict(state_bank=dict(directory=str(bank))))))
    monkeypatch.setattr(sys,'argv',['archive','--launch',str(launch)])
    monkeypatch.setattr(archive.subprocess,'check_output',lambda *a,**k:'')
    monkeypatch.setattr(archive.subprocess,'run',lambda *a,**k:None)
    archive.main()
    result=json.loads((bank/'resume1000/manifest.json').read_text())
    assert len(result['files'])==8
    for rank in range(8):
        p=bank/'resume1000'/f'rank{rank}.pt'
        state=torch.load(p,weights_only=False)
        assert state['rng']['cpu'].item()==rank and state['data_cursor']['rank']==rank
        assert result['files'][p.name]['sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    assert (bank/'archive_manifest.json').is_file()
    with pytest.raises(FileExistsError):archive.main()


def test_teacher_recipe_is_qualified_full_panel_fp32_source():
    import hashlib
    recipe=ROOT/'runs/multiseed-capture/teacher_recipe.json'
    assert hashlib.sha256(recipe.read_bytes()).hexdigest()=='28e06feaf6aa8a694ad4a36cd2122fcdd9f116bccfae539280af7263f82a8e7e'
    r=json.loads(recipe.read_text());a=r['argv_template']
    assert r['source_commit']=='60b04228ede48bd2fa8e2b0f46ce7492f9367004'
    for flag,value in [('--states','1000'),('--train-sequences','128'),('--audit-sequences','64'),('--microbatch','16'),('--gpt-precision','fp32'),('--qualification-states','1,2,3,999,1000')]:
        assert a[a.index(flag)+1]==value
    assert '--resume' not in a and len(r['source_files_sha256'])==5
