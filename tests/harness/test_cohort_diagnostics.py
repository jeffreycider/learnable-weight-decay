import torch
from chisel.harness import RunState
from chisel.cohort_diagnostics import backward


def test_joint_loss_and_gradients_average_shared_and_private_once(monkeypatch):
    from chisel import cohort_diagnostics as module
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad(): model.weight.fill_(2)
    probes = [torch.nn.Linear(1, 1, bias=False) for _ in range(2)]
    for i,p in enumerate(probes):
        with torch.no_grad(): p.weight.fill_(i + 1)
    fits = [dict(entry={'sha256':str(i)}, probes=p) for i,p in enumerate(probes)]
    state = RunState(model=model, extras={'diagnostic_fits':fits})
    def one(config, state):
        p = state.extras['fit']['probes'].weight
        loss = state.model.weight.square().sum() + p.square().sum()
        loss.backward()
        state.extras.update(train_loss=loss.item(), metrics=dict(group_norm_reduction_pct=p.item(),logit_mse=loss.item(),evaluated_positions=3))
    monkeypatch.setattr(module.single, 'backward', one)
    backward({}, state)
    assert state.extras['train_loss'] == 6.5
    torch.testing.assert_close(model.weight.grad, torch.tensor([[4.]]))
    for i,p in enumerate(probes): torch.testing.assert_close(p.weight.grad, torch.tensor([[float(i+1)]]))
    assert state.extras['metrics']['evaluated_positions'] == 6
