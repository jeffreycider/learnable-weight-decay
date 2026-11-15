from types import SimpleNamespace
import pytest
import torch
from chisel.fitting_history import setup


class Replay:
    def __init__(self):
        self.reader = SimpleNamespace(read=lambda entry: {'step': 9})
        self.calls = []

    def load(self, entry, record, batch_step):
        self.calls.append(batch_step)
        batch = {'inputs': torch.full((4,2), batch_step)}
        return None,None,None,None,None,batch,None


def test_fixed_past_examples_are_reproducible_and_not_subsampled_again():
    replay=Replay();fit={'entry': {'sha256': 'abc'}}
    state=SimpleNamespace(extras={'fit':fit,'replay':replay})
    config={'preservation_examples':1,'preservation_sample_seed':13}
    setup(fractions=(0,.5,1,1),examples_per_batch=2)(config,state)
    assert replay.calls==[0,4,9]
    assert fit['fitting_batch_steps']==[0,4,9]
    assert config['preservation_examples'] is None
    assert fit['batch']['inputs'][:,0].tolist()==[0,0,4,4,9,9]


@pytest.mark.parametrize('fractions',[[],[-.1],[1.1]])
def test_future_or_invalid_history_rejected(fractions):
    with pytest.raises(ValueError): setup(fractions=fractions)
