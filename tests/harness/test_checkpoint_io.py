import pytest
import torch
from chisel.checkpoint_io import CheckpointReader
from chisel.storage import sha256


def test_prefetch_matches_sync_and_rejects_replaced_files(tmp_path):
    entries=[]
    for index in range(3):
        path=tmp_path/f'{index}.pt';torch.save(dict(value=torch.ones(3)*index),path)
        entries.append(dict(path=str(path),sha256=sha256(path)))
    reader=CheckpointReader(prefetch=True)
    actual=list(reader.iterate(entries))
    assert [entry for entry,_ in actual]==entries
    for index,(_,record) in enumerate(actual):torch.testing.assert_close(record['value'],torch.ones(3)*index)
    torch.save(dict(value=torch.ones(3)*9),entries[1]['path'])
    with pytest.raises(ValueError,match='SHA256'):list(reader.iterate(entries))


def test_prefetch_errors_propagate_before_consumption(tmp_path):
    reader=CheckpointReader()
    with pytest.raises(FileNotFoundError):list(reader.iterate([dict(path=str(tmp_path/'missing'),sha256='x')]))
    assert list(reader.iterate([]))==[]


def test_resident_cache_owns_file_bytes_and_checks_changed_files(tmp_path):
    import pytest
    from chisel.checkpoint_io import CheckpointReader
    from chisel.storage import sha256
    path=tmp_path/'state.pt';torch.save(dict(value=torch.arange(8.)),path)
    entry=dict(path=str(path),sha256=sha256(path));reader=CheckpointReader(cache_bytes=1024)
    first=reader.read(entry);second=reader.read(entry)
    assert first is second and reader.resident_bytes==32
    torch.save(dict(value=torch.ones(8)),path)
    assert torch.equal(first['value'],torch.arange(8.))
    with pytest.raises(ValueError,match='SHA256'):reader.read(entry)
