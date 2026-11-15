import json
import pytest
import torch
from rewriter.state_bank import StateBankWriter,StateBankReader,state_split
from tests.test_prefix_cache import toy_record


def provenance():
    return dict(baseline_commit='fixed-commit',model_config={'tiny':True},dataset={'sha256':'fixed-data'},
                seed=1,schedule_steps=3250,capture_world_size=8,sensor_config={'rank':2,'banks':['W','G','M','U']})


def model_state(snapshot):
    return dict(first=torch.randn(7,5),last=torch.randn(9,6),bias=torch.randn(7),
                fixed_buffer=torch.arange(3))


def test_exact_immutable_roundtrip_and_disjoint_states(tmp_path):
    snapshot,train=toy_record();weights=model_state(snapshot)
    expected={n:v.clone() for n,v in weights.items()};rows=snapshot.matrices['first'].rows.clone()
    writer=StateBankWriter(tmp_path,provenance(),expected_steps=5)
    writer.append(1,weights,snapshot,train,data_cursor_batches=1,identity_action=True)
    weights['first'].zero_();snapshot.matrices['first'].rows.zero_()
    for step in range(2,6):writer.append(step,weights,snapshot,train,data_cursor_batches=step,identity_action=True)
    entries=writer.close()
    reader=StateBankReader(tmp_path,expected_provenance=provenance())
    restored=reader.read(1)
    assert all(torch.equal(restored.model[n],v) for n,v in expected.items())
    assert torch.equal(restored.snapshot.matrices['first'].rows,rows)
    assert restored.completed_steps==restored.data_cursor_batches==1
    assert reader.manifest['provenance']['schedule_steps']==3250
    assert [r.completed_steps for r in reader.iter_split('train')]==[1,2,3,4]
    assert [r.completed_steps for r in reader.iter_split('audit')]==[5]
    assert entries[0]['weight_tensor_bytes']==sum(v.numel()*v.element_size() for v in expected.values())
    saved=torch.load(tmp_path/'state_0001.pt',weights_only=False)
    assert set(saved)=={'schema','completed_steps','data_cursor_batches','model','observation'}
    assert 'optimizer' not in saved and 'controller' not in saved


def test_reject_bad_identity_clock_provenance_and_integrity(tmp_path):
    snapshot,train=toy_record();weights=model_state(snapshot)
    with StateBankWriter(tmp_path,provenance(),expected_steps=1) as writer:
        with pytest.raises(ValueError,match='unedited'):writer.append(1,weights,snapshot,train,data_cursor_batches=1,identity_action=False)
        with pytest.raises(ValueError,match='clocks'):writer.append(1,weights,snapshot,train,data_cursor_batches=0,identity_action=True)
        with pytest.raises(ValueError,match='contiguous'):writer.append(2,weights,snapshot,train,data_cursor_batches=2,identity_action=True)
        writer.append(1,weights,snapshot,train,data_cursor_batches=1,identity_action=True)
    with pytest.raises(ValueError,match='provenance'):StateBankReader(tmp_path,expected_provenance=dict(provenance(),seed=2))
    reader=StateBankReader(tmp_path)
    with (tmp_path/'state_0001.pt').open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='SHA256'):reader.read(1)


def test_model_loading_validates_every_tensor_before_mutation(tmp_path):
    snapshot,train=toy_record();weights=model_state(snapshot)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name,value in weights.items():
                if value.is_floating_point():self.register_parameter(name,torch.nn.Parameter(torch.zeros_like(value)))
                else:self.register_buffer(name,torch.zeros_like(value))
    model=Tiny()
    with StateBankWriter(tmp_path,provenance(),expected_steps=1) as writer:
        writer.append(1,weights,snapshot,train,data_cursor_batches=1,identity_action=True)
    record=StateBankReader(tmp_path).read(1);record.load_model(model)
    assert all(torch.equal(v,weights[n]) for n,v in model.state_dict().items())
    model.first.data.zero_();model.bias.data=model.bias.data.double()
    with pytest.raises(ValueError,match='dtype'):record.load_model(model)
    assert not model.first.any()


def test_writer_failure_propagates_and_no_completion(tmp_path,monkeypatch):
    snapshot,train=toy_record();weights=model_state(snapshot)
    writer=StateBankWriter(tmp_path,provenance(),expected_steps=1)
    def fail(*args):raise OSError('disk full')
    monkeypatch.setattr(writer,'_write',fail)
    writer.append(1,weights,snapshot,train,data_cursor_batches=1,identity_action=True)
    with pytest.raises(OSError,match='disk full'):writer.close()
    assert not (tmp_path/'complete.json').exists()


def test_split_rejects_duplicates_and_overlap(tmp_path):
    with pytest.raises(ValueError):state_split([1,1,2])
    snapshot,train=toy_record();weights=model_state(snapshot)
    with StateBankWriter(tmp_path,provenance(),expected_steps=1) as writer:
        writer.append(1,weights,snapshot,train,data_cursor_batches=1,identity_action=True)
    path=tmp_path/'manifest.json';manifest=json.loads(path.read_text());manifest['split']['audit']=[1]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='overlaps'):StateBankReader(tmp_path)


def test_optional_verify_once_rechecks_changed_file_and_mmap_is_private(tmp_path,monkeypatch):
    import rewriter.state_bank as module
    snapshot,train=toy_record();weights=model_state(snapshot)
    with StateBankWriter(tmp_path,provenance(),expected_steps=1) as writer:
        writer.append(1,weights,snapshot,train,data_cursor_batches=1,identity_action=True)
    reader=StateBankReader(tmp_path,verify_once=True,mmap_model=True)
    original=module._digest;calls=[]
    def digest(path):calls.append(path);return original(path)
    monkeypatch.setattr(module,'_digest',digest)
    first=reader.read(1);first.model['first'].zero_()
    second=reader.read(1)
    assert torch.equal(second.model['first'],weights['first'])
    assert len(calls)==1
    path=tmp_path/'state_0001.pt'
    with path.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='SHA256'):reader.read(1)
    assert len(calls)==2


def test_default_reader_verifies_each_read(tmp_path,monkeypatch):
    import rewriter.state_bank as module
    snapshot,train=toy_record();weights=model_state(snapshot)
    with StateBankWriter(tmp_path,provenance(),expected_steps=1) as writer:
        writer.append(1,weights,snapshot,train,data_cursor_batches=1,identity_action=True)
    reader=StateBankReader(tmp_path);original=module._digest;calls=[]
    def digest(path):calls.append(path);return original(path)
    monkeypatch.setattr(module,'_digest',digest)
    reader.read(1);reader.read(1)
    assert len(calls)==2
