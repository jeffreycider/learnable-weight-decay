import torch
from chisel.harness import RunState
from chisel.performance import start,finish


def test_step_measurement_preserves_task_metrics_and_rng():
    state=RunState(extras=dict(device=torch.device('cpu'),metrics=dict(kl=.1)))
    rng=torch.get_rng_state().clone()
    start()({},state)
    finish()({},state)
    assert state.extras['metrics']['optimizer_step_seconds']>=0
    assert state.extras['metrics']['kl']==.1
    assert 'compiled_graphs' in state.extras['metrics']
    assert torch.equal(rng,torch.get_rng_state())
