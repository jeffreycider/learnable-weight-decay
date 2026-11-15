import torch
from chisel.overfit_diagnostics import DirectEdit


def test_dense_initialization_reproduces_effective_edit():
    w = torch.randn(7, 9)
    snapshot = {'w': {'raw': w}}
    delta = torch.randn_like(w) * .01
    model = DirectEdit(snapshot, {'w': delta})
    torch.testing.assert_close(model.edits(snapshot)['w']['raw'], delta)


def test_rank_initialization_preserves_rank_edit_and_zero_is_trainable():
    w = torch.randn(7, 9)
    snapshot = {'w': {'raw': w}}
    delta = torch.randn(7, 2) @ torch.randn(2, 9) * .01
    model = DirectEdit(snapshot, {'w': delta}, 2)
    torch.testing.assert_close(model.edits(snapshot)['w']['raw'], delta, atol=1e-7, rtol=1e-5)
    model = DirectEdit(snapshot, {'w': torch.zeros_like(w)}, 2)
    model.edits(snapshot)['w']['raw'].sum().backward()
    assert model.left[0].grad.norm() > 0


def test_rank_edit_supports_lbfgs_flattened_gradients():
    w = torch.randn(7, 9)
    snapshot = {'w': {'raw': w}}
    model = DirectEdit(snapshot, {'w': torch.zeros_like(w)}, 2)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=2)
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = (model.edits(snapshot)['w']['raw'] - .01).square().sum()
        loss.backward()
        return loss
    optimizer.step(closure)


def test_diagnostic_loss_scaling_scales_value_and_true_gradient_equally():
    from chisel.harness import RunState
    from chisel.overfit_diagnostics import backward
    from chisel.preservation_fit import logits
    teacher = torch.nn.Linear(4, 3, bias=False)
    base = {'weight': teacher.weight.detach().clone()}
    snapshot = {'weight': {'raw': base['weight'], 'consolidated': torch.zeros_like(base['weight'])}}
    delta = torch.randn_like(base['weight']) * .001
    batch = {'inputs': torch.randn(5, 4)}
    results = []
    for scale in [1., 1e6]:
        model = DirectEdit(snapshot, {'weight': delta})
        unused = torch.nn.Parameter(torch.zeros(()))
        state = RunState(model=model, extras={'logits_forward': logits, 'fit': dict(kind='dense',
            model=teacher, base=base, snapshot=snapshot, batch=batch,
            private=torch.optim.Adam([unused]))})
        config = dict(radius_floor=1e-6, microbatch_size=2, runtime={'autocast':None},
            gradient_clip=float('inf'), diagnostic_kl_weight=0., diagnostic_loss_scale=scale,
            preservation=dict(top_k=3, margin_fraction=.5, logit_mse_weight=1., ranking_weight=0.))
        backward(config, state)
        results.append((state.extras['train_loss'], model.left[0].grad.clone()))
    assert abs(results[1][0] / results[0][0] - 1e6) < .01
    torch.testing.assert_close(results[1][1], results[0][1] * 1e6)
