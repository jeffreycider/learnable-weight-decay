import pytest
import torch
from chisel.checkpoint_io import CheckpointReader
from chisel.storage import sha256


@pytest.mark.parametrize('buffered',[False,True])
def test_prefetch_matches_sync_and_rejects_replaced_files(tmp_path,buffered):
    entries=[]
    for index in range(3):
        path=tmp_path/f'{index}.pt';torch.save(dict(value=torch.ones(3)*index),path)
        entries.append(dict(path=str(path),sha256=sha256(path)))
    reader=CheckpointReader(prefetch=True,buffered=buffered)
    actual=list(reader.iterate(entries))
    assert [entry for entry,_ in actual]==entries
    for index,(_,record) in enumerate(actual):torch.testing.assert_close(record['value'],torch.ones(3)*index)
    torch.save(dict(value=torch.ones(3)*9),entries[1]['path'])
    with pytest.raises(ValueError,match='SHA256'):list(reader.iterate(entries))


def test_prefetch_errors_propagate_before_consumption(tmp_path):
    reader=CheckpointReader()
    with pytest.raises(FileNotFoundError):list(reader.iterate([dict(path=str(tmp_path/'missing'),sha256='x')]))
    assert list(reader.iterate([]))==[]


@pytest.mark.parametrize('buffered',[False,True])
def test_resident_cache_owns_file_bytes_and_checks_changed_files(tmp_path,buffered):
    import pytest
    from chisel.checkpoint_io import CheckpointReader
    from chisel.storage import sha256
    path=tmp_path/'state.pt';torch.save(dict(value=torch.arange(8.)),path)
    entry=dict(path=str(path),sha256=sha256(path));reader=CheckpointReader(cache_bytes=1024,buffered=buffered)
    first=reader.read(entry);second=reader.read(entry)
    assert first is second and reader.resident_bytes==32
    torch.save(dict(value=torch.ones(8)),path)
    assert torch.equal(first['value'],torch.arange(8.))
    with pytest.raises(ValueError,match='SHA256'):reader.read(entry)


def test_buffered_reader_owns_tensors_without_a_resident_cache(tmp_path):
    path=tmp_path/'state.pt';torch.save(dict(value=torch.arange(20.)),path)
    entry=dict(path=str(path),sha256=sha256(path))
    reader=CheckpointReader(buffered=True)
    first=reader.read(entry)
    torch.save(dict(value=torch.zeros(20)),path)
    torch.testing.assert_close(first['value'],torch.arange(20.),rtol=0,atol=0)
    with pytest.raises(ValueError,match='SHA256'):reader.read(entry)


def test_buffered_reader_detects_mutation_during_deserialization(tmp_path,monkeypatch):
    path=tmp_path/'state.pt';torch.save(dict(value=torch.arange(20.)),path)
    entry=dict(path=str(path),sha256=sha256(path));original=torch.load
    def changed(*args,**kwargs):
        result=original(*args,**kwargs)
        torch.save(dict(value=torch.zeros(20)),path)
        return result
    monkeypatch.setattr(torch,'load',changed)
    with pytest.raises(ValueError,match='changed while reading'):
        CheckpointReader(buffered=True,cache_bytes=1024).read(entry)


def test_buffered_and_mmap_projection_are_identical(tmp_path):
    from chisel.current_replay import CurrentCheckpointReader
    source=dict(config=dict(optimizers=dict(hidden=dict(parameters='weight'))),
                weights=dict(weight=torch.randn(9,7),bias=torch.randn(9)),
                gradients=dict(weight=torch.randn(9,7),bias=torch.randn(9)),
                optimizer=dict(hidden=dict(momentum=torch.randn(9,7)),scalars=dict(unused=torch.randn(9))))
    path=tmp_path/'state.pt';torch.save(source,path);entry=dict(path=str(path),sha256=sha256(path))
    a=CurrentCheckpointReader(cache_bytes=100000).read(entry)
    b=CurrentCheckpointReader(cache_bytes=100000,buffered=True).read(entry)
    from chisel.checkpoint_io import map_tensors
    left=[];right=[]
    map_tensors(a,lambda x:left.append(x));map_tensors(b,lambda x:right.append(x))
    assert len(left)==len(right)==4
    for x,y in zip(left,right):torch.testing.assert_close(x,y,rtol=0,atol=0)
