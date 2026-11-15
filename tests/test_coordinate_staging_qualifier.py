import pytest
import torch
from scripts.qualify_coordinate_staging import differences, transport
from rewriter.multibank import MultiStateBankReader
from scripts.qualify_coordinate_chisel import FixedContexts


def test_numeric_comparison_preserves_none_clocks_and_all_tensor_bits():
    value = dict(grad=None, state={'step':torch.tensor(3.), 'moment':torch.tensor([0., 1.])})
    assert differences(value, value) == []
    assert differences(value, dict(value, grad=torch.zeros(1))) == ['/grad']
    changed = dict(grad=None, state={'step':torch.tensor(4.), 'moment':torch.tensor([0., 1.])})
    assert differences(value, changed) == ['/state/step']


def test_experimental_transport_is_scoped_even_after_failure():
    read = MultiStateBankReader.read; initialize = FixedContexts.__init__
    with pytest.raises(RuntimeError, match='injected'):
        with transport('coalesced', {}):
            assert MultiStateBankReader.read is not read
            assert FixedContexts.__init__ is not initialize
            raise RuntimeError('injected')
    assert MultiStateBankReader.read is read
    assert FixedContexts.__init__ is initialize
