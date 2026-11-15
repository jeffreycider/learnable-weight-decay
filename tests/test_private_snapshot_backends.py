import copy
import pytest
import torch
from rewriter.private_probe_store import PrivateProbeStore
from tests.test_private_probe_store import context, step
from scripts.qualify_coordinate_staging import differences


@pytest.mark.parametrize('backend', ['owned_cpu_v1', 'pinned_d2h_v1'])
def test_snapshot_backend_preserves_exact_payload_and_restored_adam(tmp_path, backend):
    torch.set_num_threads(1)
    ctx = context(); step(ctx)
    actor, private, snapshot, slot, optimizer, shared_optimizer, identity = ctx
    before = copy.deepcopy(optimizer.state_dict())
    with PrivateProbeStore(tmp_path / 'reference') as reference, PrivateProbeStore(tmp_path / backend, snapshot_backend=backend) as candidate:
        first = reference.snapshot_slot(identity, slot, optimizer, visits=1)
        second = candidate.snapshot_slot(identity, slot, optimizer, visits=1)
        a = reference.read_private(identity, version=first)
        b = candidate.read_private(identity, version=second)
        assert not differences(a, b)
        for name, value in a['private'].items():
            if isinstance(value, torch.Tensor): assert value.stride() == b['private'][name].stride()
        step(ctx)
        assert candidate.stage(identity, slot, optimizer, version=second) == 1
        assert not differences(optimizer.state_dict(), before)
        assert not differences(private.state_dict(), a['private'])


def test_owned_writer_rejects_non_cpu_payload_before_file_creation(tmp_path):
    with PrivateProbeStore(tmp_path) as store:
        with pytest.raises(ValueError, match='CPU resident'):
            store._write({'invalid':torch.empty(1,device='meta')}, 'private', owned_cpu=True)
    assert not list(tmp_path.glob('private-*'))


def test_unknown_backend_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='backend'):
        PrivateProbeStore(tmp_path, snapshot_backend='silently_changed')
