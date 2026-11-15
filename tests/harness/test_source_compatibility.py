import hashlib
import pytest
import yaml
from chisel.source_compatibility import verify_sources


def test_compatibility_requires_both_exact_hashes(tmp_path):
    source=tmp_path/'source.py';source.write_text('new')
    current=hashlib.sha256(source.read_bytes()).hexdigest()
    mapping=tmp_path/'allowed.yaml'
    mapping.write_text(yaml.safe_dump({str(source):[dict(captured='old',current=current)]}))
    with pytest.raises(ValueError):verify_sources({str(source):'old'})
    verify_sources({str(source):'old'},mapping)
    with pytest.raises(ValueError):verify_sources({str(source):'other'},mapping)
    source.write_text('changed again')
    with pytest.raises(ValueError):verify_sources({str(source):'old'},mapping)
