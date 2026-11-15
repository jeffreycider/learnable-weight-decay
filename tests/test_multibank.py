from dataclasses import replace
import json

import pytest
import torch

from rewriter import multibank as m
from rewriter.state_bank import StateBankWriter, StateBankReader
from rewriter.teacher_cache import TeacherPanel
from tests.test_prefix_cache import toy_record
from tests.test_state_bank import provenance, model_state


@pytest.fixture
def banks(tmp_path, monkeypatch):
    # Tests filesystem/record identity, not the system-wide durability command.
    monkeypatch.setattr('rewriter.state_bank.subprocess.run', lambda *a, **k: None)
    items = []
    for seed in (0, 1):
        path = tmp_path / f'seed{seed}'; snapshot, train = toy_record()
        weights = model_state(snapshot)
        with StateBankWriter(path, dict(provenance(), seed=seed), expected_steps=5) as writer:
            for step in range(1, 6):
                writer.append(step, weights, snapshot, train, data_cursor_batches=step, identity_action=True)
        cache = tmp_path / f'cache{seed}'; cache.mkdir()
        child = StateBankReader(path)
        panels = [dict(state_id=local, role=role, state_sha256=child.entries[local-1]['sha256'])
                  for local in range(1,6) for role in ('train','audit')]
        (cache/'manifest.json').write_text(json.dumps(dict(args={'states':5}, panels=panels,
            execution={'gpt_precision':'fp32','precision':{'rms_epsilon':2**-23},'microbatch':16,'softcap':'same'})))
        items.append(dict(seed=seed, state_bank=str(path), state_manifest_sha256=m.file_sha(path/'manifest.json'),
            state_index_sha256=m.file_sha(path/'index.jsonl'), teacher_cache=str(cache),
            teacher_manifest_sha256=m.file_sha(cache/'manifest.json')))
    manifest = tmp_path/'multibank.json'
    manifest.write_text(json.dumps(dict(schema=m.SCHEMA,state_id_stride=5,banks=items,heldout_online_seeds=[2])))
    return manifest


def test_global_identity_preserves_local_gpt_and_data_clock(banks):
    reader=m.MultiStateBankReader(banks,verify_once=True,mmap_model=True)
    a=reader.read(1);b=reader.read(6)
    assert a.state_id==1 and b.state_id==6
    assert a.completed_steps==b.completed_steps==a.data_cursor_batches==b.data_cursor_batches==1
    assert a.bank_seed==0 and b.bank_seed==1
    assert reader.describe_state(6)==dict(state_id=6,bank_seed=1,gpt_completed_steps=1)
    assert a.assert_verified()!=b.assert_verified()
    assert reader.manifest['split']==dict(train=[1,2,3,4,6,7,8,9],audit=[5,10])
    assert len(reader.entries)==10 and reader.entries[5]['local_completed_steps']==1
    original=reader.readers[1].read(1)
    assert original.state_id==1 and original.training_state_id is None
    for name in original.model:assert torch.equal(original.model[name],b.model[name])


def test_composite_teacher_routes_local_metadata_and_rejects_wrong_namespace(banks,monkeypatch):
    class FakeCache:
        def __init__(self,path,**kwargs):
            self.manifest=json.loads((path/'manifest.json').read_text())
            self.expected={(r['state_id'],r['role']):r for r in self.manifest['panels']}
        def read(self,local,role):
            return TeacherPanel(torch.tensor([[local]]),torch.tensor([float(local)]),(10,),dict(self.expected[(local,role)]))
        def load_engine(self,engine,record):
            assert record.assert_verified()==self.expected[(record.completed_steps,'train')]['state_sha256']
            engine.append(record.completed_steps)
    monkeypatch.setattr(m,'TeacherCacheReader',FakeCache)
    states=m.MultiStateBankReader(banks);cache=m.MultiTeacherCacheReader(states)
    panel=cache.read(6,'train')
    assert panel.metadata['state_id']==1 and panel.metadata['training_state_id']==6 and panel.metadata['bank_seed']==1
    assert panel.sequence_ids==(10,) and panel.inputs.item()==1
    calls=[];record=states.read(6);cache.load_engine(calls,record);assert calls==[1]
    with pytest.raises(ValueError,match='namespace'):cache.load_engine(calls,replace(record,bank_seed=0))
    assert len(cache.expected)==20


@pytest.mark.parametrize('case',['seed_order','holdout_overlap','manifest_hash','index_hash','local_count','source_seed','layout'])
def test_rejects_mixed_or_unpinned_banks(banks,case):
    index=json.loads(banks.read_text())
    if case=='seed_order':index['banks'].reverse()
    elif case=='holdout_overlap':index['heldout_online_seeds']=[1]
    elif case=='manifest_hash':index['banks'][0]['state_manifest_sha256']='wrong'
    elif case=='index_hash':index['banks'][0]['state_index_sha256']='wrong'
    elif case=='local_count':index['state_id_stride']=6
    else:
        item=index['banks'][1];path=m.Path(item['state_bank'])/'manifest.json';value=json.loads(path.read_text())
        if case=='source_seed':value['provenance']['seed']=7
        else:value['provenance']['model_config']={'different':True}
        path.write_text(json.dumps(value));item['state_manifest_sha256']=m.file_sha(path)
    banks.write_text(json.dumps(index))
    with pytest.raises(ValueError):m.MultiStateBankReader(banks)


def test_rejects_cross_seed_teacher_even_if_manifest_is_repinned(banks,monkeypatch):
    class FakeCache:
        def __init__(self,path,**kwargs):
            self.manifest=json.loads((path/'manifest.json').read_text());self.expected={(r['state_id'],r['role']):r for r in self.manifest['panels']}
    monkeypatch.setattr(m,'TeacherCacheReader',FakeCache)
    index=json.loads(banks.read_text());a,b=index['banks'];b['teacher_cache']=a['teacher_cache'];b['teacher_manifest_sha256']=a['teacher_manifest_sha256'];banks.write_text(json.dumps(index))
    with pytest.raises(ValueError,match='another state or seed'):m.MultiTeacherCacheReader(m.MultiStateBankReader(banks))
