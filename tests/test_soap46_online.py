import copy
import dataclasses
import io
import types
from pathlib import Path
import pytest
import torch
from rewriter.soap46_online import Soap46Online
from rewriter.soap46_map import committed_weight,soap_direction
from test_soap46_map import reference_scope


def bare(model,lr=0,variant='inner-outer'):
    h=Soap46Online.__new__(Soap46Online);h.scope=reference_scope()
    h.scope.update(model=model,batch_size=512,CWD=.025,MUON_LR=.0375,optimizers=[types.SimpleNamespace(it=0,prefill_steps=300,rest_steps=1950,lookahead_stepsize=.3,lookahead_ema=.99,get_lr_lambda=lambda:1.)])
    h.lr=lr;h.clip=1.;h.betas=(.9,.99);h.compile_map=False;h.stop=10;h.variant=variant
    h.warmup_steps=0;h.warmup_meta='moments';h.warmup_meta_steps=0;h.meta_ramp_steps=0;h.residual_decay=0.;h.residual_updates=0
    h.entries={};h.optimizer=None;h.meta_steps=0;h.metrics={};h.map=committed_weight
    return h


def initial_state(p):
    return dict(momentum=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p),lam_row=torch.ones(p.shape[0]),lam_col=torch.ones(p.shape[1]),
        row_gg=torch.zeros(p.shape[0],p.shape[0]),col_gg=torch.zeros(p.shape[1],p.shape[1]),q_row=None,q_col=None,soap_step=0)


def source_step(s,p,state,group,step):
    g=p.grad;state['momentum'].lerp_(g,1-group['mu']);d=g.lerp(state['momentum'],group['mu'])
    soap=s['soap_precondition_momentum'](d,state,step=step)
    gate=s['bounded_trust_gate'](s['trust_gate'](d,soap,g),step)
    u=s['muon_update'](s['norm_preserving_blend'](d,soap,gate));u=s['scale_radial_update'](u,p)
    factor=(.3825*p.float().norm(dim=1,keepdim=True).clamp_min(1e-8)/u.float().norm(dim=1,keepdim=True).clamp_min(1e-8)).clamp(min=1.).pow(1.)
    u=(u.float()*factor).to(u.dtype);radius=s['target_radius_after_update'](p,u,group['lr']);mask=(u.float()*p.float()>0).to(p.dtype)
    p.add_(u,alpha=-group['lr']);s['rescale_to_radius'](p,radius);p.mul_(1-group['lr']*.025*mask)
    s['soap_update_preconditioner'](g,state,shampoo_beta=.9)


@pytest.mark.parametrize('variant',['inner-outer','outer-only'])
def test_frozen_x_identity_trajectory_and_all_baseline_states(variant):
    torch.manual_seed(634);model=torch.nn.Linear(7,4,bias=False);p=model.weight
    h=bare(model,variant=variant);state=initial_state(p);control=torch.nn.Parameter(p.detach().clone());control_state=copy.deepcopy(state)
    group={'lr':.0375,'mu':.85}
    for step in range(5):
        p.grad=torch.randn_like(p);control.grad=p.grad.clone()
        with torch.no_grad():source_step(h.scope,control,control_state,group,step)
        h.propose(p,state,group,step,True,True)
        assert torch.equal(p,control)
        for k in state:
            if isinstance(state[k],torch.Tensor):assert torch.equal(state[k],control_state[k]),k
            else:assert state[k]==control_state[k]
        h.consume(step+1);assert h.meta_steps==0


def test_variance_update_uses_unperturbed_q_with_nonzero_x_and_old_snapshot_stays_fixed(monkeypatch):
    torch.manual_seed(635);model=torch.nn.Linear(7,4,bias=False);p=model.weight;h=bare(model,lr=1e-4);st=initial_state(p);group={'lr':.0375,'mu':.85}
    p.grad=torch.randn_like(p);h.propose(p,st,group,0,True,True)
    assert not h.entries[p]['lag'].pending
    # No fictitious task update on initial basis-free pass.
    h.consume(1);assert h.optimizer is None and h.meta_steps==0
    with torch.no_grad():h.entries[p]['xl'].normal_(std=.02);h.entries[p]['xr'].normal_(std=.02)
    p.grad=torch.randn_like(p);q=st['q_row'].clone();qr=st['q_col'].clone();v=st['exp_avg_sq'].clone()
    m=st['momentum'].lerp(p.grad,.15);d=p.grad.lerp(m,.85);expected=v.mul(.9).add((q.T@d@qr).square(),alpha=.1)
    h.propose(p,st,group,1,True,True);r=h.entries[p]['record']
    # Current bank is already permuted by refresh; pending snapshot is pre-refresh.
    torch.testing.assert_close(r.variance,expected,rtol=1e-6,atol=1e-7)
    assert torch.equal(r.q_left,q);assert r.variance.data_ptr()!=st['exp_avg_sq'].data_ptr()
    clone=r.variance.clone();st['exp_avg_sq'].zero_();assert torch.equal(r.variance,clone)


def test_pending_resume_and_adam_update_exact(monkeypatch):
    monkeypatch.setattr(torch.distributed,'all_reduce',lambda x:None);monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    model=torch.nn.Linear(7,4,bias=False);p=model.weight;h=bare(model,lr=1e-4);st=initial_state(p);group={'lr':.0375,'mu':.85}
    for step in range(2):p.grad=torch.randn_like(p);h.propose(p,st,group,step,True,True)
    b=io.BytesIO();torch.save(h.state_dict(),b);b.seek(0);resumed=bare(model,lr=1e-4);resumed.load_state_dict(torch.load(b,weights_only=False))
    p.grad=torch.randn_like(p);h.consume(2);resumed.consume(2)
    for k in ('xl','xr'):
        assert torch.equal(h.entries[p][k],resumed.entries[p][k]);assert h.entries[p][k].norm()>0
    assert h.meta_steps==resumed.meta_steps==1


def test_driver_pins_active_env_and_has_one_gpt_backward(monkeypatch,tmp_path):
    from scripts.train_soap46 import configure_environment,patched_source
    source=Path('baseline/record46/upstream.py').read_text()
    monkeypatch.setenv('SOAP_OFF','1');monkeypatch.setenv('OPT_BASE','psgd')
    env=configure_environment(source,steps=10,output=tmp_path,base_lr=.075,cwd=.05)
    assert env['ROWFLOOR']=='1' and env['ROWFLOOR_FROB']=='0' and env['CWD']=='0.05' and 'SOAP_OFF' not in env
    patched=patched_source(source,variant='outer-only',meta_lr=1e-4,output=tmp_path,steps=10,microbatch=8)
    compile(patched,'<record46 patched>','exec')
    assert patched.count('loss.backward()')==source.count('loss.backward()')==1
    assert patched.index('hyper.consume(step)')<patched.index('    set_hparams(step)\n')
    assert 'MUON_POWER_C *= MUON_LR / 0.0375' in patched
    assert 'FINAL_TRAIN_STEPS = 2900' in patched
    assert 'shampoo_update' not in patched and 'spectral_anneal' not in patched


def test_compiled_map_reuses_graph_for_changed_state_lr_and_gate_phase():
    from test_soap46_map import make_record
    from torch._dynamo.testing import CompileCounter
    backend=CompileCounter();compiled=torch.compile(committed_weight,backend=backend,fullgraph=True,dynamic=False)
    for step,lr in [(1,.02),(1500,.01),(1700,.005)]:
        r=make_record();r=dataclasses.replace(r,step=torch.tensor(float(step)),lr=torch.tensor(lr),cwd=torch.tensor(.025),direction=r.direction+step*.00001)
        xl=torch.zeros_like(r.q_left,requires_grad=True);xr=torch.zeros_like(r.q_right,requires_grad=True)
        expected=committed_weight(r,xl,xr);actual=compiled(r,xl,xr)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        actual.sum().backward();assert torch.isfinite(xl.grad).all()
    assert backend.frame_count==1


def test_complete_checkpoint_roundtrip_and_policy_mismatch(monkeypatch):
    from rewriter.soap46_checkpoint import payload,restore
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda:1)
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda:torch.zeros(1,dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda _:None)
    def setup():
        model=torch.nn.Linear(7,4,bias=False);h=bare(model)
        opt=torch.optim.Adam(model.parameters(),lr=.03)
        class Wrapper(torch.optim.Optimizer):
            def __init__(self):
                super().__init__(model.parameters(),{});self.inner_optimizer=[opt];self.it=2401
                self.lookahead_status=False;self.current_lookahead_stepsize=0.
        h.scope.update(optimizers=[Wrapper()],optimizer2=types.SimpleNamespace(step_count=2401),SEED=3,mbs=64,
                       _tailema=[torch.randn_like(model.weight)],training_time=1.,t0=0.)
        return h
    a=setup();b=io.BytesIO();torch.save(payload(a.scope,a,2401),b);b.seek(0);saved=torch.load(b,weights_only=False)
    c=setup();assert restore(c.scope,c,saved)==2401
    assert torch.equal(a.scope['model'].weight,c.scope['model'].weight)
    assert torch.equal(a.scope['_tailema'][0],c.scope['_tailema'][0])
    assert c.scope['optimizers'][0].it==2401
    with pytest.raises(ValueError,match='cwd'):restore(c.scope,c,dict(saved,cwd=.1))


def test_compiled_scalar_path_does_not_round_lr_times_bfloat16_update():
    from test_soap46_map import make_record
    r=make_record();tensor=dataclasses.replace(r,lr=torch.tensor(r.lr),step=torch.tensor(float(r.step)),cwd=torch.tensor(r.cwd))
    eager=committed_weight(r,torch.zeros_like(r.q_left),torch.zeros_like(r.q_right))
    candidate=committed_weight(tensor,torch.zeros_like(r.q_left),torch.zeros_like(r.q_right))
    # Float FMA vs separate FP32 add/multiply may differ by an ulp; BF16 product
    # rounding would produce much larger errors and is not permitted here.
    torch.testing.assert_close(candidate,eager,atol=3e-7,rtol=3e-7)

@pytest.mark.parametrize('variant',['inner-outer','outer-only'])
@pytest.mark.parametrize('name',['0.mlp.fc.weight','0.attn.proj.weight'])
def test_verbatim_muon_class_zero_residual_multistep(variant,name,tmp_path,monkeypatch):
    """The oracle is the complete pinned class, including active branch routing."""
    import os
    from scripts.train_soap46 import configure_environment
    source=Path('baseline/record46/upstream.py').read_text()
    old=dict(os.environ)
    configure_environment(source,steps=125,output=tmp_path)
    class LocalDist:
        def get_world_size(self):return 1
        def get_rank(self):return 0
        def all_gather(self,*args):pass
    scope=dict(torch=torch,nn=torch.nn,Tensor=torch.Tensor,F=torch.nn.functional,
               os=os,args=types.SimpleNamespace(seed=0),Path=Path,dist=LocalDist())
    try:exec(source[source.index('SEED = args.seed'):source.index('\nclass Adam(')],scope)
    finally:os.environ.clear();os.environ.update(old)
    torch.manual_seed(912);model=torch.nn.Linear(7,4,bias=False)
    p=model.weight
    with torch.no_grad():
        if '.attn.proj.' in name:p.zero_()
    control=torch.nn.Parameter(p.detach().clone());opt=scope['Muon']([(name,control)],lr=.0375,mu=.85)
    h=bare(model,variant=variant);h.scope.update(scope);h.scope['CWD']=.025
    state=initial_state(p)
    for step in range(4):
        p.grad=torch.randn_like(p);control.grad=p.grad.clone()
        opt.step();h.propose(p,state,opt.param_groups[0],step,scope['is_attn_param'](name),scope['is_attn_proj_param'](name))
        assert torch.equal(p,control)
        assert set(state)==set(opt.state[control])
        for key,value in state.items():
            expected=opt.state[control][key]
            assert torch.equal(value,expected) if isinstance(value,torch.Tensor) else value==expected

@pytest.mark.parametrize('mode',['moments','off'])
@pytest.mark.parametrize('variant',['inner-outer','outer-only'])
def test_warmup_zero_action_moments_activation_and_pending_resume(mode,variant,monkeypatch):
    monkeypatch.setattr(torch.distributed,'all_reduce',lambda x:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    torch.manual_seed(1912);model=torch.nn.Linear(7,4,bias=False);p=model.weight
    h=bare(model,lr=1e-4,variant=variant);h.warmup_steps=4;h.warmup_meta=mode
    state=initial_state(p);control=torch.nn.Parameter(p.detach().clone());control_state=copy.deepcopy(state)
    group={'lr':.0375,'mu':.85}
    for step in range(4):
        p.grad=torch.randn_like(p);control.grad=p.grad.clone()
        h.consume(step)
        with torch.no_grad():source_step(h.scope,control,control_state,group,step)
        h.propose(p,state,group,step,True,True)
        assert torch.equal(p,control)
        assert all(torch.count_nonzero(h.entries[p][k])==0 for k in ('xl','xr'))
    assert h.meta_steps==(2 if mode=='moments' else 0)
    assert h.warmup_meta_steps==h.meta_steps
    if mode=='moments':
        assert h.optimizer.param_groups[0]['lr']==0
        assert any(s['exp_avg'].norm()>0 for s in h.optimizer.state.values())
        assert all(s['step']==2 for s in h.optimizer.state.values())
    else:assert h.optimizer is None
    assert h.entries[p]['lag'].pending  # action3 zero-X supplies activation4 credit
    b=io.BytesIO();torch.save(h.state_dict(),b);b.seek(0)
    resumed=bare(model,lr=1e-4,variant=variant);resumed.warmup_steps=4;resumed.warmup_meta=mode
    resumed.load_state_dict(torch.load(b,weights_only=False))
    p.grad=torch.randn_like(p);h.consume(4);resumed.consume(4)
    assert h.optimizer.param_groups[0]['lr']==1e-4
    assert not h.metrics['moment_only_update']
    for key in ('xl','xr'):
        assert torch.equal(h.entries[p][key],resumed.entries[p][key])
        assert h.entries[p][key].norm()>0
    assert h.meta_steps==(3 if mode=='moments' else 1)


def test_warmup_endpoint_preserves_pending_zero_action_for_prefix_fork():
    model=torch.nn.Linear(7,4,bias=False);p=model.weight;h=bare(model,lr=1e-4)
    h.warmup_steps=h.stop=2;state=initial_state(p);group={'lr':.0375,'mu':.85}
    for step in range(2):p.grad=torch.randn_like(p);h.propose(p,state,group,step,True,True)
    assert h.entries[p]['lag'].pending and h.entries[p]['action_step']==1


def test_delayed_linear_meta_lr_schedule_and_legacy_hold():
    h=bare(torch.nn.Linear(7,4,bias=False),lr=1e-5);h.warmup_steps=500;h.meta_ramp_steps=500
    assert [h.effective_lr(t) for t in (0,499,500,750,1000,2000)]==[0.,0.,0.,5e-6,1e-5,1e-5]
    h.meta_ramp_steps=0
    assert h.effective_lr(499)==0 and h.effective_lr(500)==1e-5


@pytest.mark.parametrize('zero_lr_boundary',[False,True])
def test_direct_residual_decay_after_credit_is_independent_of_lr(zero_lr_boundary,monkeypatch):
    monkeypatch.setattr(torch.distributed,'all_reduce',lambda x:None);monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    torch.manual_seed(1923);model=torch.nn.Linear(7,4,bias=False);p=model.weight;h=bare(model,lr=1e-4)
    state=initial_state(p);group={'lr':.0375,'mu':.85}
    for step in range(3):
        p.grad=torch.randn_like(p);h.consume(step);h.propose(p,state,group,step,True,True)
    b=io.BytesIO();torch.save(h.state_dict(),b);b.seek(0);saved=torch.load(b,weights_only=False)
    control=bare(model,lr=1e-4);control.load_state_dict(copy.deepcopy(saved))
    decayed=bare(model,lr=1e-4);decayed.load_state_dict(copy.deepcopy(saved));decayed.residual_decay=.1
    if zero_lr_boundary:
        for a in (control,decayed):a.warmup_steps=3;a.meta_ramp_steps=3
    before={k:control.entries[p][k].detach().clone() for k in ('xl','xr')};model_before=p.detach().clone()
    p.grad=torch.randn_like(p);control.consume(3);decayed.consume(3)
    assert torch.equal(p,model_before)  # consuming credit never edits GPT weights
    for key in before:
        expected=before[key]*.9+(control.entries[p][key].detach()-before[key])
        torch.testing.assert_close(decayed.entries[p][key],expected,atol=1e-10,rtol=1e-6)
    assert decayed.metrics['applied_residual_decay']==.1
    assert decayed.metrics['meta_effective_lr']==(0 if zero_lr_boundary else 1e-4)
    for a,b in zip(control.optimizer.state.values(),decayed.optimizer.state.values()):
        for key in a:assert torch.equal(a[key],b[key])


@pytest.mark.parametrize('field,value',[('meta_ramp_steps',500),('residual_decay',.1)])
def test_resume_refuses_changed_meta_ramp_or_residual_decay(field,value):
    model=torch.nn.Linear(7,4,bias=False);h=bare(model);saved=h.state_dict();other=bare(model);setattr(other,field,value)
    with pytest.raises(ValueError,match='ramp/decay'):other.load_state_dict(saved)


def test_ramp_decay_pending_resume_exact(monkeypatch):
    monkeypatch.setattr(torch.distributed,'all_reduce',lambda x:None);monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    torch.manual_seed(1931);model=torch.nn.Linear(7,4,bias=False);p=model.weight;h=bare(model,lr=1e-4)
    h.warmup_steps=2;h.meta_ramp_steps=3;h.residual_decay=.1;state=initial_state(p);group={'lr':.0375,'mu':.85}
    for step in range(6):
        p.grad=torch.randn_like(p);h.consume(step);h.propose(p,state,group,step,True,True)
    assert h.meta_steps==4 and h.residual_updates==3
    b=io.BytesIO();torch.save(h.state_dict(),b);b.seek(0);resumed=bare(model,lr=1e-4)
    resumed.warmup_steps=2;resumed.meta_ramp_steps=3;resumed.residual_decay=.1
    resumed.load_state_dict(torch.load(b,weights_only=False));p.grad=torch.randn_like(p)
    h.consume(6);resumed.consume(6)
    for key in ('xl','xr'):assert torch.equal(h.entries[p][key],resumed.entries[p][key])
    assert h.residual_updates==resumed.residual_updates==4
    for a,b in zip(h.optimizer.state.values(),resumed.optimizer.state.values()):
        for key in a:assert torch.equal(a[key],b[key])


def test_parameter_frame_custom_epsilon_pending_resume_and_policy_guards(monkeypatch):
    from rewriter.soap46_conditioning import parameter_frame_weight
    monkeypatch.setattr(torch.distributed,'all_reduce',lambda _:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    torch.manual_seed(943);model=torch.nn.Linear(7,4,bias=False);p=model.weight
    h=bare(model,lr=1e-5,variant='outer-only');h.residual_frame='parameter';h.meta_epsilon=1e-9;h.map=parameter_frame_weight
    st=initial_state(p);group={'lr':.0375,'mu':.85}
    for step in range(2):p.grad=torch.randn_like(p);h.propose(p,st,group,step,True,True)
    saved=copy.deepcopy(h.state_dict())
    resumed=bare(model,lr=1e-5,variant='outer-only');resumed.residual_frame='parameter';resumed.meta_epsilon=1e-9;resumed.map=parameter_frame_weight
    resumed.load_state_dict(saved);p.grad=torch.randn_like(p);h.consume(2);resumed.consume(2)
    assert h.optimizer.param_groups[0]['eps']==1e-9
    for key in ('xl','xr'):assert torch.equal(h.entries[p][key],resumed.entries[p][key])
    bad=bare(model,lr=1e-5,variant='outer-only')
    with pytest.raises(ValueError,match='frame or meta epsilon'):bad.load_state_dict(saved)
    bad.residual_frame='parameter'
    with pytest.raises(ValueError,match='frame or meta epsilon'):bad.load_state_dict(saved)


def test_driver_records_explicit_conditioning_without_extra_gpt_backward(tmp_path):
    from scripts.train_soap46 import patched_source
    source=Path('baseline/record46/upstream.py').read_text()
    actual=patched_source(source,variant='outer-only',meta_lr=1e-5,output=tmp_path,steps=20,
        microbatch=64,residual_frame='parameter',meta_epsilon=1e-9)
    compile(actual,'<conditioned record46>','exec')
    assert "residual_frame='parameter'" in actual and 'meta_epsilon=1e-09' in actual
    assert actual.count('loss.backward()')==source.count('loss.backward()')==1
