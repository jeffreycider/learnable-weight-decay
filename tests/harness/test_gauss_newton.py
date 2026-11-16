import copy
import pytest
import torch
from chisel.gauss_newton import GaussNewton, conjugate_gradient


def test_cg_matches_dense_solve_and_counts_products():
    matrix = torch.tensor([[4., 1.], [1., 2.]], dtype=torch.float64)
    rhs = torch.tensor([2., -3.], dtype=torch.float64)
    calls = []
    def product(v):
        calls.append(1)
        return matrix @ v
    actual, iterations, residual = conjugate_gradient(product, rhs, 20, 1e-12)
    torch.testing.assert_close(actual, torch.linalg.solve(matrix, rhs))
    assert iterations == len(calls) == 2
    assert residual < 1e-12
    zero, iterations, residual = conjugate_gradient(product, torch.zeros_like(rhs), 20, 1e-12)
    assert iterations == 0 and zero.count_nonzero() == 0 and residual == 0


def test_underdetermined_step_is_the_damped_minimum_motion_solution():
    matrix = torch.tensor([[1., 2., 0.], [0., 1., 1.]], dtype=torch.float64)
    value = torch.nn.Parameter(torch.tensor([2., -1., 4.], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.tensor(9., dtype=torch.float64))
    target = torch.tensor([1., -2.], dtype=torch.float64)
    initial = value.detach().clone()
    opt = GaussNewton([value, unused], damping=.01, cg_steps=20, cg_tolerance=1e-12)
    opt.closure = lambda: matrix @ value - target
    expected = initial - matrix.T @ torch.linalg.solve(
        matrix @ matrix.T + .01 * torch.eye(2, dtype=torch.float64), matrix @ initial - target)
    opt.step()
    torch.testing.assert_close(value, expected)
    assert unused == 9 and opt.metrics['refinement_accepted']
    assert opt.evaluations == 2 and opt.products == 3


def test_nonlinear_backtracking_and_exact_resume():
    value = torch.nn.Parameter(torch.tensor([.1, 2.], dtype=torch.float64))
    opt = GaussNewton([value], damping=1e-8, cg_steps=20, cg_tolerance=1e-12)
    opt.closure = lambda: value.square() - 1
    initial = opt.closure().square().sum().item()
    opt.step()
    assert opt.metrics['refinement_scale'] < 1
    assert opt.closure().square().sum() < initial
    saved_value, saved_optimizer = value.detach().clone(), copy.deepcopy(opt.state_dict())
    for _ in range(4): opt.step()
    expected = value.detach().clone()
    restored = torch.nn.Parameter(saved_value)
    other = GaussNewton([restored], damping=1e-8, cg_steps=20, cg_tolerance=1e-12)
    other.closure = lambda: restored.square() - 1
    other.load_state_dict(saved_optimizer)
    for _ in range(4): other.step()
    torch.testing.assert_close(restored, expected, atol=0, rtol=0)
    assert other.state_dict() == opt.state_dict()


def test_failed_finite_trial_restores_parameters():
    value = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    opt = GaussNewton([value], damping=1e-6)
    def residual():
        # The finite trial is deliberately invalid; no partially applied update survives.
        return value if torch.is_grad_enabled() else value * float('nan')
    opt.closure = residual
    with pytest.raises(FloatingPointError, match='nonfinite refinement residual'):
        opt.step()
    assert value.item() == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_real_chisel_refinement_uses_full_network_jacobian():
    from chisel.network import Chisel, PrivateProbes, BANKS, propose
    from chisel.models import CifarGPT
    from chisel.transfer import apply_edits
    from chisel.preservation import group_norm_change
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.manual_seed(531)
    torch.set_float32_matmul_precision('highest')
    model = CifarGPT(16, 1, 16, 'mean', head_dim=8, causal=False, learned_qk_scale=True).cuda()
    with torch.no_grad(): model.proj.weight.normal_(std=.1)
    base = {n:p.detach().clone() for n,p in model.named_parameters()}
    snapshot = {n:{bank:(p.clone() if bank=='raw' else torch.zeros_like(p)) for bank in BANKS}
                for n,p in base.items() if n.startswith('blocks.') and p.ndim==2}
    actor = Chisel(4,16,1,head_dim=8,causal=False).cuda()
    probes = PrivateProbes({n:s['raw'].shape for n,s in snapshot.items()},4).cuda()
    with torch.no_grad(): actor.proj.weight.normal_(std=.04)
    inputs=torch.randn(4,3,32,32,device='cuda')
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        teacher=torch.func.functional_call(model,base,(inputs,))
        target=group_norm_change(snapshot,apply_edits(snapshot,propose(actor,probes,snapshot))[2])
    def residual():
        effective=apply_edits(snapshot,propose(actor,probes,snapshot))[2]
        logits=torch.func.functional_call(model,base|effective,(inputs,))
        norm=group_norm_change(snapshot,effective)
        return torch.cat(((logits-teacher).flatten()/teacher.numel()**.5,(norm-target).reshape(1)))
    opt=GaussNewton(list(actor.parameters())+list(probes.parameters()),cg_steps=80,damping=1e-7)
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):before=residual().square().sum().item()
    original_private=[p.detach().clone() for p in probes.parameters()]
    for _ in range(4):opt.closure=residual;opt.step()
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):after=residual().square().sum().item()
    assert after < before*.1
    assert any(not torch.equal(a,b) for a,b in zip(original_private,probes.parameters()))


def perturb_heads():
    def hook(config, state):
        with torch.no_grad(): state.model.proj.weight.normal_(std=.03)
    return hook


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='CUDA required'))])
def test_harness_refinement_targets_private_weights_and_resume(tmp_path, device):
    from pathlib import Path
    import yaml
    from chisel.harness import run
    from tests.harness.test_checkpoint_duals import configuration
    config=configuration(tmp_path)
    config['runtime'].update(device=device,autocast=None)
    manifest=Path(config['data']['manifest'])
    manifest.write_text(yaml.safe_dump({'states':yaml.safe_load(manifest.read_text())['states'][:1]}))
    config['data']['batch_counts']={'text':1}
    config.update(diagnostic_lbfgs=False,compile_proposal=False)
    model_index=next(i for i,h in enumerate(config['setup']) if h['name']=='chisel.training.model')+1
    config['setup'].insert(model_index,dict(name='tests.harness.test_gauss_newton.perturb_heads'))
    insert=next(i for i,h in enumerate(config['setup']) if h['name']=='chisel.offline.setup')+1
    config['setup'][insert:insert]=[
        dict(name='chisel.overfit_diagnostics.setup',hyperparams=dict(kind='chisel')),
        dict(name='chisel.gauss_newton.setup',hyperparams=dict(cg_steps=20,damping=1e-5))]
    config['pre_forward']=[]
    config['post_optimizer']=[dict(name='chisel.gauss_newton.progress'),
        dict(name='chisel.parallel_diagnostics.checkpoint',hyperparams=dict(every=2))]
    expected=run(config)
    config.update(output_dir=str(tmp_path/'prefix'),stop_step=2)
    prefix=run(config)
    config.update(output_dir=str(tmp_path/'resumed'),stop_step=4,resume=str(tmp_path/'prefix/latest_rank0.pt'))
    actual=run(config)
    for left,right in [(actual.model,expected.model),
                       (actual.extras['fit']['probes'],expected.extras['fit']['probes']),
                       (actual.extras['checkpointables']['refinement_targets'],expected.extras['checkpointables']['refinement_targets'])]:
        torch.testing.assert_close(left.state_dict(),right.state_dict(),rtol=0,atol=0)
    assert actual.optimizer.state_dict()==expected.optimizer.state_dict()
    assert actual.optimizer.evaluations>4
    assert any(not torch.equal(a,b) for a,b in zip(prefix.model.parameters(),actual.model.parameters()))


def test_relative_damping_matches_dense_scaled_system():
    matrix=torch.tensor([[1000.,20.,0.],[0.,10.,1.]],dtype=torch.float64)
    value=torch.nn.Parameter(torch.tensor([.1,2.,-3.],dtype=torch.float64))
    initial=value.detach().clone();residual=matrix@initial
    rayleigh=(residual@(matrix@matrix.T)@residual)/(residual@residual)
    damping=.01+.03*rayleigh
    expected=initial-matrix.T@torch.linalg.solve(matrix@matrix.T+damping*torch.eye(2,dtype=torch.float64),residual)
    opt=GaussNewton([value],damping=.01,relative_damping=.03,cg_steps=20,cg_tolerance=1e-12)
    opt.closure=lambda:matrix@value
    opt.step()
    torch.testing.assert_close(value,expected)
    assert abs(opt.metrics['damping']-damping.item())<1e-9
