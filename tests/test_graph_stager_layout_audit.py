"""Independent CPU oracle for pinned-stager layout reconstruction.

The allocator is deliberately dirty. Check only each leaf's valid storage
footprint, never alignment padding belonging to the surrounding arena.
"""
import itertools

import pytest
import torch

from rewriter.algebra import Measurements
from rewriter.controller import ControllerSnapshot
from rewriter.graph_stager import PinnedLeafBuilder


def cases():
    base = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    return [
        base,
        base.t(),
        base[1::2, 1::2],  # nonzero source offset and holes
        base[2:3, :].expand(5, 8),
        base[2:3, 3:4].expand(4, 7),
        base[:0, :],
        torch.tensor(-0.),
        torch.tensor([0x7FC00001, -2147483648, 0], dtype=torch.int32).view(torch.float32),
        torch.arange(5, dtype=torch.int64)[1::2],
    ]


@pytest.mark.parametrize('source', cases())
def test_layout_and_bits_with_dirty_allocator_and_valid_gap_bounds(source, monkeypatch):
    original_empty = torch.empty

    def dirty_empty(*args, **kwargs):
        return original_empty(*args, **kwargs).fill_(17)

    monkeypatch.setattr(torch, 'empty', dirty_empty)
    snapshot = ControllerSnapshot({}, {}, source, torch.tensor(2.), torch.tensor(.02))
    built, _ = PinnedLeafBuilder(pin=False).build(snapshot, Measurements({}, {}))
    result = built['trajectories']
    assert result.shape == source.shape and result.stride() == source.stride()
    assert result.dtype == source.dtype
    assert torch.equal(result.contiguous().reshape(-1).view(torch.uint8),
                       source.contiguous().reshape(-1).view(torch.uint8))
    size = 0 if source.numel() == 0 else 1 + sum((d - 1) * s for d, s in zip(source.shape, source.stride()))
    addresses = {sum(i * s for i, s in zip(index, source.stride()))
                 for index in itertools.product(*(range(d) for d in source.shape))}
    storage = result.as_strided((size,), (1,))
    gaps = sorted(set(range(size)) - addresses)
    if gaps:
        assert torch.count_nonzero(storage[gaps]) == 0
    # A neighbouring leaf must not overwrite the reconstructed logical values.
    built['weight_norm'].fill_(9.)
    assert torch.equal(result.contiguous().reshape(-1).view(torch.uint8),
                       source.contiguous().reshape(-1).view(torch.uint8))
