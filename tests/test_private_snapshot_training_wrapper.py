import hashlib
import pytest
from rewriter.private_probe_store import PrivateProbeStore
from scripts.qualify_private_snapshot_training import snapshot_backend,validate_qualification


def test_wrapper_is_scoped_and_preserves_default(tmp_path):
    original=PrivateProbeStore.__init__
    with pytest.raises(RuntimeError,match='injected'):
        with snapshot_backend('pinned_d2h_v1'):
            with PrivateProbeStore(tmp_path/'candidate') as store:
                assert store.snapshot_backend=='pinned_d2h_v1'
            raise RuntimeError('injected')
    assert PrivateProbeStore.__init__ is original
    with PrivateProbeStore(tmp_path/'reference') as store:
        assert store.snapshot_backend=='reference'


def test_explicit_backend_and_source_bind_resume_identity(tmp_path):
    source=tmp_path/'driver.py';source.write_text('frozen qualifier')
    recipe=dict(qualification_only=True,qualification_snapshot_backend='pinned_d2h_v1',
                snapshot_qualification_driver_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    validate_qualification(recipe,'pinned_d2h_v1',source)
    with pytest.raises(ValueError,match='backend'):validate_qualification(recipe,'reference',source)
    with pytest.raises(ValueError,match='qualification-only'):
        validate_qualification(dict(recipe,qualification_only=False),'pinned_d2h_v1',source)
    source.write_text('mutated')
    with pytest.raises(ValueError,match='source'):validate_qualification(recipe,'pinned_d2h_v1',source)
