import hashlib
import io
import json
from types import SimpleNamespace
import pytest
from chisel.storage import upload_verified, upload_receipt

class Missing(Exception):
    response={'Error':{'Code':'404'}}

class FakeR2:
    exceptions=SimpleNamespace(ClientError=Missing)
    def __init__(self):self.objects={};self.corrupt=False
    def head_object(self,Bucket,Key):
        if Key not in self.objects:raise Missing()
        body,metadata=self.objects[Key]
        return dict(ContentLength=len(body),Metadata=metadata)
    def upload_file(self,path,bucket,key,ExtraArgs,Config):
        from pathlib import Path
        self.objects[key]=(Path(path).read_bytes(),ExtraArgs['Metadata'])
    def get_object(self,Bucket,Key):
        return dict(Body=io.BytesIO(b'bad' if self.corrupt else self.objects[Key][0]))


def test_upload_checks_local_and_remote_bytes(tmp_path):
    path=tmp_path/'state.pt';path.write_bytes(b'checkpoint')
    client=FakeR2();digest=hashlib.sha256(path.read_bytes()).hexdigest()
    for _ in range(2):
        assert upload_verified(client,path,'bucket','key',digest)['verification']=='full_get_sha256'
    with pytest.raises(ValueError,match='local checkpoint checksum'):
        upload_verified(client,path,'bucket','different','0'*64)
    client.corrupt=True
    with pytest.raises(ValueError,match='full-read'):
        upload_verified(client,path,'bucket','key',digest)
    client.corrupt=False;path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='collision'):
        upload_verified(client,path,'bucket','key')


def test_marker_exists_only_after_verified_data_and_receipt(tmp_path):
    path=tmp_path/'state.pt';path.write_bytes(b'checkpoint')
    receipt=tmp_path/'state.ready.json'
    receipt.write_text(json.dumps(dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    client=FakeR2();client.corrupt=True
    with pytest.raises(ValueError):upload_receipt(client,tmp_path,receipt,'bucket','prefix')
    marker=tmp_path/'state.uploaded.json';assert not marker.exists()
    client.corrupt=False;upload_receipt(client,tmp_path,receipt,'bucket','prefix')
    assert marker.exists() and path.exists()
    assert set(client.objects)=={'prefix/state.pt','prefix/state.ready.json'}
