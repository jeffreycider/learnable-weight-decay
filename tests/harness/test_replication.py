import json
import pytest
from chisel.replicate import copy_record
from chisel.storage import sha256


def test_record_is_published_after_copy_and_source_is_retained(tmp_path):
    source=tmp_path/'local';source.mkdir()
    value=source/'state.pt';value.write_bytes(b'weights and gradients')
    receipt=source/'state.ready.json'
    receipt.write_text(json.dumps(dict(file=value.name,bytes=value.stat().st_size,sha256=sha256(value))))
    destination=tmp_path/'shared'
    copy_record(source,destination,receipt)
    copied=json.loads((destination/receipt.name).read_text())
    assert copied['path']==str(destination/value.name)
    assert (destination/value.name).read_bytes()==value.read_bytes()
    copy_record(source,destination,receipt)
    assert value.exists()
    value.write_bytes(b'corrupt')
    receipt.write_text(json.dumps(dict(file=value.name,bytes=7,sha256=sha256(value))))
    with pytest.raises(ValueError,match='collision'):copy_record(source,destination,receipt)


def test_bad_source_cannot_publish_ready_marker(tmp_path):
    source=tmp_path/'local';source.mkdir()
    (source/'state.pt').write_bytes(b'bad')
    receipt=source/'state.ready.json'
    receipt.write_text(json.dumps(dict(file='state.pt',bytes=3,sha256='wrong')))
    destination=tmp_path/'shared'
    with pytest.raises(ValueError,match='differs'):copy_record(source,destination,receipt)
    assert not (destination/receipt.name).exists()
