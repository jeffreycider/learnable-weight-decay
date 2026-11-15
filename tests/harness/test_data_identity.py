import hashlib
import json
import numpy as np
import pytest
from baseline.data import FinewebStream


def test_shard_identity_and_cursor_are_pinned(tmp_path):
    path=tmp_path/'shard.bin'
    header=np.zeros(256,dtype=np.int32);header[:3]=[20240520,1,40]
    path.write_bytes(header.tobytes()+np.arange(40,dtype=np.uint16).tobytes())
    manifest=tmp_path/'identity.json'
    manifest.write_text(json.dumps([dict(type='file',path=path.name,lfs=dict(oid=hashlib.sha256(path.read_bytes()).hexdigest()))]))
    kwargs=dict(filename_pattern=str(path),batch_size=8,seq_len=4,rank=0,world_size=1,device='cpu',identity_manifest=str(manifest))
    stream=FinewebStream(**kwargs);next(stream);saved=stream.state_dict()
    restored=FinewebStream(**kwargs);restored.load_state_dict(saved)
    assert restored.state_dict()==saved
    path.write_bytes(path.read_bytes()[:-1]+b'x')
    with pytest.raises(ValueError,match='SHA256'):next(FinewebStream(**kwargs))
