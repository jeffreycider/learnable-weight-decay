import unittest
import pytest
import torch
from rewriter.prefix_cache import pack_record, unpack_record
from scripts.benchmark_observation_transfer import transfer_packed_observation
from tests.test_prefix_cache import toy_record, assert_record_equal


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_coalesced_observation_preserves_bits_layout_and_independent_storage(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    snapshot, measurements=toy_record()
    packed=pack_record(0,snapshot,measurements)
    # Include signed zeros and a NaN payload in an uncompressed trajectory bank.
    packed['trajectories'][0,0]=-0.
    packed['trajectories'][0,1]=torch.tensor(0x7FC00123,dtype=torch.int32).view(torch.float32)
    direct=unpack_record(packed,device=device)
    coalesced=unpack_record(transfer_packed_observation(packed,device),device=device)
    assert_record_equal(unittest.TestCase(),direct.snapshot,direct.train,coalesced)
    packed['matrices']['first']['anchors'][0].zero_()
    assert coalesced.snapshot.matrices['first'].anchors.A.abs().sum()>0
    coalesced.snapshot.vectors['bias'].weight.zero_()
    assert direct.snapshot.vectors['bias'].weight.abs().sum()>0


def test_transfer_handles_mixed_dtypes_empty_scalars_and_nested_metadata():
    value={'float':torch.tensor([-0.,1.]),'integer':torch.tensor([2,3]),
           'nested':(torch.empty(0,2),[torch.tensor(4.,dtype=torch.float64),'metadata'])}
    moved=transfer_packed_observation(value,'cpu')
    assert moved['float'].data_ptr()!=value['float'].data_ptr()
    assert torch.equal(moved['float'].view(torch.int32),value['float'].view(torch.int32))
    assert moved['integer'].dtype==torch.int64
    assert moved['nested'][0].shape==(0,2)
    assert moved['nested'][1][0].shape==()
    assert moved['nested'][1][1]=='metadata'
