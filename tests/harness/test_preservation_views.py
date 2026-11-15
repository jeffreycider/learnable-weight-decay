import json
import pytest
import torch
from chisel.harness import run
from chisel.offline_data import Replay
from tests.harness.test_integration import source_config, ToyStream


class ViewStream(ToyStream):
    def __init__(self, device, rank=0, world_size=1, mix='original'):
        super().__init__(device, rank, world_size)
        self.mix = mix

    def __next__(self):
        inputs, targets = super().__next__()
        return (torch.zeros_like(inputs) if self.mix is None else inputs), targets


def test_alternative_views_preserve_source_state_and_dataset_identity(tmp_path):
    config = source_config(tmp_path)
    config['data'] = dict(_target_=__name__+'.ViewStream')
    config['stop_step'] = 2
    run(config)
    entry = json.loads((tmp_path/'source/trajectory/000001/state.ready.json').read_text()) | dict(domain='cifar')
    normal = Replay('cpu', phase='current').load(entry)
    alternative = Replay('cpu', phase='current', data_options=dict(cifar=dict(mix=None)))
    changed = alternative.load(entry)
    assert normal[5]['inputs'].any() and not changed[5]['inputs'].any()
    torch.testing.assert_close(normal[5]['targets'], changed[5]['targets'], rtol=0, atol=0)
    assert normal[-1]['data'] == changed[-1]['data'] == config['data']
    for name in normal[2]:
        for bank in ['raw', 'consolidated', 'gradient', 'momentum', 'proposal']:
            torch.testing.assert_close(normal[2][name][bank], changed[2][name][bank], rtol=0, atol=0)
    again = alternative.load(entry)
    torch.testing.assert_close(changed[5]['inputs'], again[5]['inputs'], rtol=0, atol=0)
    # An override for another domain cannot affect this entry.
    untouched = Replay('cpu', phase='current', data_options=dict(other=dict(mix=None))).load(entry)
    torch.testing.assert_close(normal[5]['inputs'], untouched[5]['inputs'], rtol=0, atol=0)


def test_view_overrides_cannot_switch_to_heldout_data():
    for key in ['split', 'path', 'sha256', 'split_seed', 'heldout_per_class']:
        with pytest.raises(ValueError, match='dataset identity or splits'):
            Replay('cpu', data_options=dict(cifar={key: 'changed'}))


def test_unmixed_cifar_view_uses_the_same_images_and_augmentations(tmp_path):
    from tests.harness.test_cifar_replay import stream
    mixed, unmixed = stream(tmp_path), stream(tmp_path)
    unmixed.mix = None
    before = torch.get_rng_state().clone()
    for step in range(3):
        assert mixed.recipe(step) == unmixed.recipe(step)
        raw, actual = unmixed.replay(step), mixed.replay(step)
        images, targets = mixed.mix(raw['inputs'], raw['targets'], step)
        torch.testing.assert_close(images, actual['inputs'], rtol=0, atol=0)
        torch.testing.assert_close(targets, actual['targets'], rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())
