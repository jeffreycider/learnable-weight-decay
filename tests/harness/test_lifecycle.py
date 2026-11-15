from pathlib import Path
import pytest
from chisel.harness import RunState, run, resolve, build

EVENTS = []
class Optimizer:
    def step(self): EVENTS.append('step')
    def zero_grad(self, **kwargs): EVENTS.append('zero')
def setup():
    def hook(config, state):
        EVENTS.append('setup')
        state.optimizer = Optimizer()
        state.backward = lambda config, state: EVENTS.append('backward')
    return hook

def mark(value):
    return lambda config, state: EVENTS.append(value)

def test_order_and_schedule():
    EVENTS.clear()
    prefix = __name__
    cfg = {'train_steps': 100, 'stop_step': 2, 'setup': [{'name': f'{prefix}.setup'}]}
    for site in ['pre_forward','pre_optimizer','post_optimizer','teardown']:
        cfg[site] = [{'name': f'{prefix}.mark', 'hyperparams': {'value': site}}]
    state = run(cfg)
    assert state.step == 1
    assert cfg['train_steps'] == 100
    assert EVENTS == ['setup'] + ['pre_forward','backward','pre_optimizer','step','zero','post_optimizer']*2 + ['teardown']

def test_yaml_merge_and_cycle(tmp_path):
    (tmp_path/'base.yaml').write_text('train_steps: 10\nmodel: {width: 8, depth: 2}\nsetup: [old]\n')
    (tmp_path/'run.yaml').write_text('extends: base.yaml\nmodel: {width: 16}\nsetup: [new]\n')
    assert resolve(tmp_path/'run.yaml') == dict(train_steps=10,model=dict(width=16,depth=2),setup=['new'])
    (tmp_path/'base.yaml').write_text('extends: run.yaml\n')
    with pytest.raises(ValueError,match='cyclic'):resolve(tmp_path/'run.yaml')

def test_dependency_injection():
    assert build({'_target_':'builtins.dict','a':1}, b=2) == {'a':1,'b':2}
    with pytest.raises(ValueError):build({'_target_':'builtins.dict','a':1}, a=2)
