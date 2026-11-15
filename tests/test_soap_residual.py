import ast
import math
from pathlib import Path
import torch
from rewriter.soap_residual import (FrozenMap,committed_proposal,finite_polar,
    inverse_fourth_factor,residual_scale,displacement_ema_forward)
from rewriter.lag_one import LagOneHypergradient

torch.set_num_threads(2)


def upstream_helpers():
    tree=ast.parse(Path('baseline/pr395/upstream.py').read_text())
    names={'gram_frobenius_norm_estimate','_ns_inner','zeropower_via_newtonschulz5',
           'shampoo_ns_r4_coeffs','apply_inverse_fourth_root_left','scale_radial_update',
           'target_radius_after_update','rescale_to_radius','spectral_anneal'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope=dict(torch=torch,Tensor=torch.Tensor,SHAMPOO_TF32_MIN_SIDE=1024,
               RADIAL_OUTWARD_SCALE=.5,RADIAL_INWARD_SCALE=1.,math=math,SPA_RMIN=.3,SPA_BETA=.99,SPA_CLIP=2.,SPA_ALPHA=.25)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned upstream>','exec'),scope)
    return scope


def test_finite_polar_matches_upstream_including_transpose():
    torch.manual_seed(73)
    up=upstream_helpers()
    for shape in [(4,7),(7,4),(4,4)]:
        g=torch.randn(shape)
        torch.testing.assert_close(finite_polar(g),up['zeropower_via_newtonschulz5'](g),rtol=0,atol=0)


def test_inverse_factor_matches_pinned_polynomial_on_identity():
    up=upstream_helpers()
    a=torch.randn(5,5); gram=a@a.T
    actual=inverse_fourth_factor(gram)
    expected=up['apply_inverse_fourth_root_left'](torch.eye(5),gram,8,.01)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_polar_derivative_finite_polynomial_not_svd():
    x=torch.randn(4,6,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda v:finite_polar(v,compute_dtype=torch.float64),
                                   (x,),eps=1e-6,atol=2e-5,rtol=2e-4)


def record(dtype=torch.float64):
    torch.manual_seed(51)
    w=torch.randn(3,4,dtype=dtype)
    return FrozenMap(w,torch.randn_like(w),torch.eye(3,dtype=dtype),torch.eye(4,dtype=dtype),
                     torch.tensor(.7,dtype=dtype),torch.tensor(.4,dtype=dtype),.013,
                     polar_dtype=dtype,cwd=.025)


def test_qzero_matches_common_fixed_control_and_trainable():
    r=record(); h=torch.zeros_like(r.left,requires_grad=True);k=torch.zeros_like(r.right,requires_grad=True)
    a=committed_proposal(r,h,k); b=committed_proposal(r,torch.zeros_like(h),torch.zeros_like(k))
    assert torch.equal(a,b)
    grads=torch.autograd.grad((a*torch.randn_like(a)).sum(),(h,k))
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in grads)


def test_complete_map_matches_explicit_upstream_postprocessing():
    r=record(torch.float32);up=upstream_helpers()
    u=finite_polar(r.left@r.direction@r.right,compute_dtype=r.polar_dtype).float()
    u=u*(math.sqrt(r.direction.shape[0])/u.norm().clamp_min(1e-20))
    u=up['scale_radial_update'](u,r.weight)
    u=u*(r.target_uw*r.weight.norm(dim=1,keepdim=True).clamp_min(1e-8)/u.norm(dim=1,keepdim=True).clamp_min(1e-8)).clamp_min(1)
    radius=up['target_radius_after_update'](r.weight,u,r.lr)
    mask=(u*r.weight>0).to(r.weight.dtype)
    p=r.weight.clone();p.add_(u,alpha=-r.lr);up['rescale_to_radius'](p,radius);p.mul_(1-r.lr*r.cwd*mask)
    actual=committed_proposal(r,torch.zeros_like(r.left),torch.zeros_like(r.right))
    torch.testing.assert_close(actual,p,atol=3e-7,rtol=3e-7)


def test_lagged_full_map_and_actual_wrapper_equal_explicit_next_loss():
    r=record();hl=torch.randn_like(r.left,requires_grad=True);hr=torch.randn_like(r.right,requires_grad=True)
    previous=r.weight.clone();buf=torch.randn_like(previous)
    inputs=torch.randn(4,5,dtype=torch.float64);targets=torch.randn(3,5,dtype=torch.float64)
    def forward():
        return displacement_ema_forward(committed_proposal(r,hl,hr),previous,buf,gamma=.99,beta=.3)
    q=forward(); full=((q@inputs-targets)**2).mean()
    expected=torch.autograd.grad(full,(hl,hr))
    actual_q=forward().detach().requires_grad_();loss=((actual_q@inputs-targets)**2).mean()
    next_grad=torch.autograd.grad(loss,actual_q)[0]
    lag=LagOneHypergradient()
    lag.from_contraction(action_step=0,hyperparameters={'hl':hl,'hr':hr},
                         contraction=lambda g:(forward()*g['weight']).sum())
    result=lag.consume({'weight':next_grad},gradient_step=1)
    for name,g in zip(['hl','hr'],expected):torch.testing.assert_close(result.gradients[name],g)


def test_dimensionless_scale_homogeneity_and_zero():
    gram=torch.diag(torch.tensor([1e-24,3e-24]))
    torch.testing.assert_close(residual_scale(gram*81),residual_scale(gram)/3)
    assert torch.isfinite(residual_scale(torch.zeros(3,3)))


def test_zero_direction_finite_derivative():
    x=torch.zeros(3,4,requires_grad=True)
    finite_polar(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_pinned_source_patch_preserves_training_and_contains_credit_order():
    from scripts.train_soap_hyper import patched_source
    source=Path('baseline/pr395/upstream.py').read_text()
    patched=patched_source(source,meta_lr=1e-5,output='/tmp/test',steps=256,microbatch=64)
    compile(patched,'<generated>','exec')
    assert patched.index('hyper.consume(step)')<patched.index('    set_hparams(step)\n')
    assert 'FINAL_TRAIN_STEPS = 2900' in patched
    assert 'batch_size = 8 * 64 * 1024' in patched
    assert 'soap_precondition_pair(direction, residual, state)' in patched
    assert 'hyper.propose(p, direction, residual, state, group, self.step_count, use_soap)' in patched
    assert patched.count('loss.backward()')==source.count('loss.backward()')==1


def test_bfloat16_radius_statistics_are_float32():
    import dataclasses
    r=record(torch.float32)
    # Actual PR395 hidden params are FP32; this separately covers a BF16 master.
    r=dataclasses.replace(r,weight=r.weight.bfloat16(),polar_dtype=torch.bfloat16)
    up=upstream_helpers()
    u=finite_polar(r.left@r.direction@r.right).float()
    u=u*(math.sqrt(r.direction.shape[0])/u.norm().clamp_min(1e-20))
    u=up['scale_radial_update'](u,r.weight)
    u=u*(r.target_uw*r.weight.float().norm(dim=1,keepdim=True).clamp_min(1e-8)/u.norm(dim=1,keepdim=True).clamp_min(1e-8)).clamp_min(1)
    radius=up['target_radius_after_update'](r.weight,u,r.lr)
    p=(r.weight-r.lr*u).to(r.weight.dtype)
    up['rescale_to_radius'](p,radius)
    p.mul_(1-r.lr*r.cwd*(u*r.weight.float()>0).to(r.weight.dtype))
    actual=committed_proposal(r,torch.zeros_like(r.left),torch.zeros_like(r.right))
    assert torch.equal(actual,p)


def test_cooldown_spectral_annealing_matches_pinned_source():
    import dataclasses
    r=record(torch.float32);up=upstream_helpers()
    qr=torch.linalg.qr(torch.randn(3,3))[0];qc=torch.linalg.qr(torch.randn(4,4))[0]
    gradient=torch.randn_like(r.weight)
    state={'q_row':qr,'q_col':qc};group={'lr':.01,'initial_lr':.0325}
    u=finite_polar(r.left@r.direction@r.right,compute_dtype=r.polar_dtype)
    u=u*(math.sqrt(r.direction.shape[0])/u.norm().clamp_min(1e-20))
    u=up['spectral_anneal'](u,gradient,state,group)
    row=torch.diag(qr.T@state['spa_rr']@qr).clamp_min(1e-30).log()
    col=torch.diag(qc.T@state['spa_cc']@qc).clamp_min(1e-30).log()
    lv=row[:,None]+col[None,:]
    sc=((lv-lv.mean())/lv.std().clamp_min(1e-6)).clamp(-2,2)
    r=dataclasses.replace(r,spectral_row=qr,spectral_col=qc,spectral_scale=torch.exp(.25*math.log(.01/.0325)*sc))
    u=up['scale_radial_update'](u,r.weight)
    u=u*(r.target_uw*r.weight.norm(dim=1,keepdim=True).clamp_min(1e-8)/u.norm(dim=1,keepdim=True).clamp_min(1e-8)).clamp_min(1)
    radius=up['target_radius_after_update'](r.weight,u,r.lr)
    p=(r.weight-r.lr*u).clone();up['rescale_to_radius'](p,radius)
    p.mul_(1-r.lr*r.cwd*(u*r.weight>0))
    result=committed_proposal(r,torch.zeros_like(r.left),torch.zeros_like(r.right))
    torch.testing.assert_close(result,p,rtol=1e-6,atol=2e-7)
