import dataclasses
import torch
import pytest
from scripts.fit_cpu_edit_teacher import physical_action,make_target,target_loss,stored_readout
from rewriter.adaptive_action import AdaptiveAction,AdaptiveFactors
from rewriter.adaptive_preservation import build_adaptive_proposal
from rewriter.edit_scope import editable_names,editable_weight_norm_sq,scoped_snapshot
from tests.test_adaptive_preservation_actor import actual_actor_case


@pytest.mark.parametrize('radius_fraction',[1.,1e-7])
def test_physical_conversion_value_and_gradient_matches_real_adapter(radius_fraction):
    torch.set_num_threads(2);torch.manual_seed(14)
    engine,_,_,snapshot,grouped,_=actual_actor_case()
    matrices={}
    for n,o in snapshot.matrices.items():
        u=torch.linalg.qr(torch.randn(o.rows.shape[0],2,dtype=torch.float64)).Q.requires_grad_()
        v=torch.linalg.qr(torch.randn(o.columns.shape[0],2,dtype=torch.float64)).Q.requires_grad_()
        c=torch.randn(2,2,dtype=torch.float64,requires_grad=True)
        matrices[n]=AdaptiveFactors(u,c,v)
    vectors={n:torch.randn_like(engine.parameters[n],dtype=torch.float64,requires_grad=True) for n,v in snapshot.vectors.items()}
    raw=AdaptiveAction(matrices,vectors);captured=[]
    build_adaptive_proposal(lambda _:raw,snapshot,engine,grouped,radius_fraction=radius_fraction,
        detach_credit=False,execution_capture=lambda a,s:captured.append(a.scaled(s)))
    editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
    scoped=scoped_snapshot(snapshot,editable_weight_norm_sq(engine.parameters,editable),radius_fraction,scope='grouped')
    candidate,_,guard=physical_action(raw,grouped,scoped)
    assert bool(guard<1)==(radius_fraction==1e-7)
    params=[t for f in raw.matrices.values() for t in (f.U,f.C,f.V)]+list(vectors.values())
    cot={n:torch.randn_like(v) for n,v in candidate.dense().items()}
    loss=sum((v*cot[n]).sum() for n,v in candidate.dense().items())
    ref=sum((v*cot[n]).sum() for n,v in captured[0].dense().items() if n in editable)
    torch.testing.assert_close(loss,ref,rtol=0,atol=0)
    a=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(ref,params,allow_unused=True)
    for x,y in zip(a,b):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-10)


def test_target_orientation_all_zero_outputs_count_and_storage():
    # Includes an embedding in transposed computational orientation.
    p={'embed.weight':torch.randn(11,8),'first.weight':torch.randn(8,8),
       'second.weight':torch.randn(8,8),'first.bias':torch.randn(8),'other.bias':torch.randn(8)}
    snap=type('Snapshot',(),{'matrices':{n:None for n in p if p[n].ndim==2},
                            'vectors':{n:None for n in p if p[n].ndim==1}})()
    rows=[{'rank_limit':8,'inverse_power':2,'selected':[{'first_weight':'first.weight','second_weight':'second.weight','first_bias':'first.bias','channel':2,'exponent':1}]}]
    target,used=make_target(p,snap,rows,set(p))
    assert target.matrices['embed.weight'].dense().shape==(8,11)
    expected=p['first.weight'].double().clone();expected[2]*=2
    torch.testing.assert_close((p['first.weight'].double()+target.matrices['first.weight'].dense()).float(),expected.float(),rtol=0,atol=0)
    zero=target.scaled(0.);loss,terms=target_loss(zero,target,target.norm_sq())
    torch.testing.assert_close(loss,torch.tensor(1.,dtype=torch.float64))
    assert terms.keys()==p.keys() and 'other.bias' in terms
    changed=AdaptiveAction(zero.matrices,{**zero.vectors,'other.bias':torch.ones(8,dtype=torch.float64)})
    assert target_loss(changed,target,target.norm_sq())[0]>loss
    readout=stored_readout(target,target,p)
    assert readout['stored_different_elements']==0 and readout['gpt_forwards']==0


def test_stored_group_and_frobenius_readout_matches_dense_norms():
    torch.set_num_threads(2)
    engine,_,_,snapshot,grouped,_=actual_actor_case()
    matrices={}
    for n,o in snapshot.matrices.items():
        u=torch.ones(o.rows.shape[0],1,dtype=torch.float64)
        v=torch.ones(o.columns.shape[0],1,dtype=torch.float64)
        matrices[n]=AdaptiveFactors(u,torch.tensor([[.002]],dtype=torch.float64),v)
    editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
    action=AdaptiveAction({n:f for n,f in matrices.items() if n in editable},
        {n:torch.full_like(engine.parameters[n],.001,dtype=torch.float64) for n in snapshot.vectors if n in editable})
    result=stored_readout(action,action,engine.parameters,grouped)
    actual=dict(engine.parameters)
    for n,d in action.dense().items():
        if n=='embed.weight':d=d.T
        actual[n]=(actual[n].double()+d).float()
    direct=grouped.evaluate(actual).metrics()
    assert result['stored_norm_metrics']['predicted']['mean_group_l2_percentage_reduction']==pytest.approx(direct['mean_group_l2_percentage_reduction'],abs=1e-10)
    for role,ndim in [('whole_parameter',None),('matrix',2)]:
        old=sum(p.double().square().sum() for p in engine.parameters.values() if ndim is None or p.ndim==ndim).sqrt()
        new=sum(p.double().square().sum() for p in actual.values() if ndim is None or p.ndim==ndim).sqrt()
        assert result['stored_norm_metrics']['predicted'][role+'_frobenius_percentage_reduction']==pytest.approx(float(100*(1-new/old)),abs=1e-10)


def test_span_capture_equals_dense_projection_with_tight_and_thin_frames():
    from types import SimpleNamespace
    from scripts.fit_cpu_edit_teacher import write_span_capture
    def frame(q):return SimpleNamespace(orthonormal_matrix=lambda:q)
    torch.manual_seed(53)
    u=torch.linalg.qr(torch.randn(5,2,dtype=torch.float64)).Q
    v=torch.linalg.qr(torch.randn(3,2,dtype=torch.float64)).Q
    q=torch.linalg.qr(torch.randn(4,2,dtype=torch.float64)).Q.T # 2x4 row-orthonormal
    target=AdaptiveAction({'w':AdaptiveFactors(torch.eye(5,dtype=torch.float64),torch.randn(5,3,dtype=torch.float64),torch.eye(3,dtype=torch.float64))},{'b':torch.randn(2,dtype=torch.float64)})
    private=SimpleNamespace(matrix_names=('w',),matrix_rows=[frame(u)],matrix_columns=[frame(v)],vector_names=('b',),vector_frames=[frame(q)])
    result=write_span_capture(private,target)
    dense=target.matrices['w'].dense();projected=u@(u.T@dense@v)@v.T
    energy=float(projected.square().sum()+target.vectors['b'].square().sum())
    assert result['captured_target_energy_fraction']==pytest.approx(energy/float(target.norm_sq()),abs=1e-12)


def test_mixed_precision_lbfgs_and_hard_budget_restore():
    from scripts.fit_cpu_edit_teacher import bounded_lbfgs
    x=torch.nn.Parameter(torch.tensor([0.],dtype=torch.float32))
    y=torch.nn.Parameter(torch.tensor([0.],dtype=torch.float64))
    def loss(_):return (x.double()-2).square().sum()+(y+3).square().sum()
    opt,result=bounded_lbfgs([x,y],loss,max_iter=10,closure_budget=20)
    assert float(loss(0).detach())<1e-10 and x.dtype==torch.float32 and y.dtype==torch.float64
    assert result['gradient_dtypes']==['torch.float32','torch.float64'] and result['flat_gradient_dtype']=='torch.float64'
    with torch.no_grad():x.zero_();y.zero_()
    opt,result=bounded_lbfgs([x,y],loss,max_iter=10,closure_budget=1)
    assert result['budget_aborted_restored_initial'] and result['closure_evaluations']==1
    assert torch.equal(x,torch.zeros_like(x)) and torch.equal(y,torch.zeros_like(y)) and not opt.state


def test_explicit_wolfe_tolerance_overcomes_tiny_scale_and_restores_internal_function():
    import importlib
    from scripts.fit_cpu_edit_teacher import bounded_lbfgs
    module=importlib.import_module('torch.optim.lbfgs');original=module._strong_wolfe
    x=torch.nn.Parameter(torch.tensor([0.],dtype=torch.float64))
    def loss(_):return (x*1e10-1).square().sum()
    _,result=bounded_lbfgs([x],loss,max_iter=5,closure_budget=40,wolfe_bracket_tolerance=1e-15)
    assert module._strong_wolfe is original
    assert not result['budget_aborted_restored_initial']
    assert float(loss(0).detach())<1e-10


def test_relative_physical_error_dense_gradient_scaling_and_zero_targets():
    from types import SimpleNamespace
    from scripts.fit_cpu_edit_teacher import relative_target_error
    torch.manual_seed(19)
    x=torch.randn(3,2,dtype=torch.float64,requires_grad=True)
    y=torch.randn(2,dtype=torch.float64,requires_grad=True)
    target=AdaptiveAction({'w':AdaptiveFactors(torch.eye(3,dtype=torch.float64),torch.ones(3,2,dtype=torch.float64),torch.eye(2,dtype=torch.float64))}, {'b':torch.zeros(2,dtype=torch.float64)})
    predicted=AdaptiveAction({'w':AdaptiveFactors(torch.eye(3,dtype=torch.float64),x,torch.eye(2,dtype=torch.float64))},{'b':y})
    grouped=SimpleNamespace(groups={'weight':('w',),'bias':('b',),'frozen':('z',)},reference_squared={'weight':torch.tensor(100.),'bias':torch.tensor(2.),'frozen':torch.tensor(0.)})
    from rewriter.adaptive_edit_target import edit_target_terms
    den=relative_target_error(edit_target_terms(target.scaled(0.),target),grouped)
    actual=relative_target_error(edit_target_terms(predicted,target),grouped)/den
    expected=((x-1).square().sum()/100+y.square().sum()/2)/(6/100)
    torch.testing.assert_close(actual,expected)
    ga=torch.autograd.grad(actual,(x,y),retain_graph=True)
    ge=torch.autograd.grad(expected,(x,y))
    for a,e in zip(ga,ge):torch.testing.assert_close(a,e)
    # Every zero-target output remains penalized, and the fresh zero head is identity.
    assert torch.count_nonzero(ga[1])==2
    assert float(relative_target_error(edit_target_terms(target.scaled(0.),target),grouped)/den)==1.
    # Scale each group physically along with its reference: relative error unchanged.
    scaled=AdaptiveAction({'w':predicted.matrices['w'].scaled(7.)},{'b':y*3})
    scaled_target=AdaptiveAction({'w':target.matrices['w'].scaled(7.)},{'b':target.vectors['b']*3})
    scaled_grouped=SimpleNamespace(groups=grouped.groups,reference_squared={'weight':torch.tensor(4900.),'bias':torch.tensor(18.),'frozen':torch.tensor(0.)})
    scaled_den=relative_target_error(edit_target_terms(scaled_target.scaled(0.),scaled_target),scaled_grouped)
    torch.testing.assert_close(relative_target_error(edit_target_terms(scaled,scaled_target),scaled_grouped)/scaled_den,actual)


def test_adam_resume_matches_uninterrupted_and_rejects_changed_identity(tmp_path):
    import copy
    from scripts.fit_cpu_edit_teacher import resume_adam
    torch.manual_seed(61)
    actor=torch.nn.Linear(2,2).double();private=torch.nn.Linear(2,1).double()
    opt=torch.optim.Adam(actor.parameters(),lr=.01);popt=torch.optim.Adam(private.parameters(),lr=.02)
    metadata={k:k for k in ('state_sha256','selection_sha256','finite_evidence_sha256','architecture','private_identity','target_norm_squared','relative_target_squared','group_reference_squared','radius')}
    metadata.update(args=dict(optimizer='adam',objective='group-relative',head_lr=.01,representation_lr=.01,private_lr=.02,seed=61,private_seed=62,steps=8),source_sha256={'model.py':'fixed','scripts/fit_cpu_edit_teacher.py':'before-resume'})
    def update():
        opt.zero_grad();popt.zero_grad()
        loss=private(actor(torch.randn(3,2,dtype=torch.float64))).square().sum()
        loss.backward();opt.step();popt.step()
    for _ in range(3):update()
    path=tmp_path/'checkpoint.pt'
    torch.save(dict(metadata=metadata,step=3,actor=actor.state_dict(),private=private.state_dict(),optimizer=opt.state_dict(),private_optimizer=popt.state_dict(),rng=torch.get_rng_state(),lbfgs_optimizer=None),path)
    for _ in range(5):update()
    expected=[p.detach().clone() for p in (*actor.parameters(),*private.parameters())]
    expected_rng=torch.get_rng_state().clone()
    for field in ('objective','head_lr','private_seed'):
        bad=copy.deepcopy(metadata);bad['args'][field]='wrong'
        with pytest.raises(ValueError,match='configuration mismatch'):resume_adam(path,bad,actor,private,opt,popt)
    bad=copy.deepcopy(metadata);bad['state_sha256']='wrong'
    with pytest.raises(ValueError,match='identity mismatch'):resume_adam(path,bad,actor,private,opt,popt)
    bad=copy.deepcopy(metadata);bad['args']['steps']=3
    with pytest.raises(ValueError,match='endpoint'):resume_adam(path,bad,actor,private,opt,popt)
    over=copy.deepcopy(metadata);over['args']['steps']=1004
    with pytest.raises(ValueError,match='endpoint'):resume_adam(path,over,actor,private,opt,popt)
    equal=copy.deepcopy(metadata);equal['args'].update(steps=3,evaluate_only=True)
    assert resume_adam(path,equal,actor,private,opt,popt)==3
    assert resume_adam(path,metadata,actor,private,opt,popt)==3
    for _ in range(5):update()
    for actual,wanted in zip((*actor.parameters(),*private.parameters()),expected):torch.testing.assert_close(actual,wanted,rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),expected_rng)


def test_explicit_private_rate_fork_preserves_adam_moments_and_counters():
    import copy
    from scripts.fit_cpu_edit_teacher import fork_private_rate
    x=torch.nn.Parameter(torch.tensor([2.],dtype=torch.float64))
    optimizer=torch.optim.Adam([x],lr=.001)
    x.square().sum().backward();optimizer.step()
    before=copy.deepcopy(optimizer.state_dict())
    assert fork_private_rate(optimizer,.01)==[.001]
    after=optimizer.state_dict()
    assert after['param_groups'][0]['lr']==.01
    for key,value in before['state'][0].items():torch.testing.assert_close(after['state'][0][key],value,rtol=0,atol=0)
    for key,value in before['param_groups'][0].items():
        if key!='lr':assert after['param_groups'][0][key]==value
    with pytest.raises(ValueError):fork_private_rate(optimizer,float('nan'))


def test_objective_fork_resets_both_histories_without_changing_parameters():
    from scripts.fit_cpu_edit_teacher import reset_objective_fork
    head=torch.nn.Parameter(torch.tensor([2.]));body=torch.nn.Parameter(torch.tensor([3.]));private=torch.nn.Parameter(torch.tensor([4.]))
    opt=torch.optim.Adam([dict(params=[head],lr=1e-6,role='heads'),dict(params=[body],lr=1e-4,role='representation')])
    popt=torch.optim.Adam([private],lr=1e-3)
    (head.square()+body.square()+private.square()).sum().backward();opt.step();popt.step()
    before=[p.detach().clone() for p in (head,body,private)]
    assert opt.state and popt.state
    assert reset_objective_fork(opt,popt,1e-8)==1e-6
    assert not opt.state and not popt.state
    assert [g['lr'] for g in opt.param_groups]==[1e-8,1e-4] and popt.param_groups[0]['lr']==1e-3
    for p,b in zip((head,body,private),before):assert torch.equal(p,b)


@pytest.mark.parametrize('control_kind',['conditional_shared_head_global_lstsq','conditional_shared_head_balanced_lstsq'])
def test_explicit_analytic_seed_loads_only_parameters_rng_and_freezes_private(tmp_path,control_kind):
    from scripts.fit_cpu_edit_teacher import seed_analytic_control,exact_state_equal
    actor=torch.nn.Linear(2,2);private=torch.nn.Linear(2,2)
    meta={k:k for k in ('state_sha256','selection_sha256','finite_evidence_sha256','architecture','private_identity','target_norm_squared','relative_target_squared','group_reference_squared','radius')}
    path=tmp_path/'control.pt';torch.save(dict(diagnostic_control=True,metadata={**meta,'diagnostic_control':control_kind},step=1000,actor=actor.state_dict(),private=private.state_dict(),rng=torch.get_rng_state()),path)
    other=torch.nn.Linear(2,2);probe=torch.nn.Linear(2,2);opt=torch.optim.Adam(other.parameters())
    result=seed_analytic_control(path,meta,other,probe)
    assert exact_state_equal(other.state_dict(),actor.state_dict()) and exact_state_equal(probe.state_dict(),private.state_dict())
    assert all(not p.requires_grad for p in probe.parameters()) and not opt.state and result['parent_step']==1000
    before={k:v.clone() for k,v in probe.state_dict().items()}
    other(probe(torch.ones(1,2))).sum().backward();torch.nn.utils.clip_grad_norm_(other.parameters(),1.);opt.step()
    assert exact_state_equal(before,probe.state_dict())
    with pytest.raises(ValueError,match='identity'):seed_analytic_control(path,{**meta,'radius':'wrong'},other,probe)


def test_balanced_objective_dense_value_gradients_zero_targets_and_global_default():
    from types import SimpleNamespace
    from scripts.fit_cpu_edit_teacher import normalized_target_objective,error_metrics
    x=torch.tensor([.3,-.4],dtype=torch.float64,requires_grad=True);z=torch.tensor([.2,.5],dtype=torch.float64,requires_grad=True)
    t=torch.tensor([1.,2.],dtype=torch.float64);terms={'w':(x-t).square().sum(),'zero':z.square().sum()}
    grouped=SimpleNamespace(groups={'target':('w',),'zero':('zero',)},reference_squared={'target':torch.tensor(100.),'zero':torch.tensor(4.)})
    den=t.square().sum();rel_den=den/100
    global_expected=((x-t).square().sum()+z.square().sum())/den
    relative_expected=((x-t).square().sum()/100+z.square().sum()/4)/rel_den
    actual=normalized_target_objective(terms,den,rel_den,grouped,'balanced');expected=.5*(global_expected+relative_expected)
    torch.testing.assert_close(actual,expected)
    a=torch.autograd.grad(actual,(x,z),retain_graph=True);b=torch.autograd.grad(expected,(x,z),retain_graph=True)
    for u,v in zip(a,b):torch.testing.assert_close(u,v)
    assert torch.count_nonzero(a[1])==2
    torch.testing.assert_close(normalized_target_objective(terms,den,rel_den,grouped),global_expected,rtol=0,atol=0)
    torch.testing.assert_close(normalized_target_objective(terms,den,rel_den,grouped,'group-relative'),relative_expected)
    identity={'w':den,'zero':den.new_zeros(())}
    assert normalized_target_objective(identity,den,rel_den,grouped,'balanced')==1
    metrics=error_metrics(terms,den,rel_den,grouped)
    assert metrics['balanced_normalized_edit_error']==pytest.approx(.5*(metrics['normalized_edit_error']+metrics['relative_normalized_edit_error']))


def test_span_only_control_cannot_silently_seed_shared_fit(tmp_path):
    from scripts.fit_cpu_edit_teacher import seed_analytic_control
    path=tmp_path/'span.pt';torch.save(dict(diagnostic_control=True,metadata={'diagnostic_control':'private_span_projection_only'}),path)
    with pytest.raises(ValueError,match='fitted conditional-head'):seed_analytic_control(path,{},None,None)


@pytest.mark.parametrize('optimizer,clip,valid',[('adam',True,True),('adam',False,False),('lbfgs',False,True),('lbfgs',True,False)])
def test_analytic_seed_optimizer_clipping_validation(optimizer,clip,valid):
    from types import SimpleNamespace
    from scripts.fit_cpu_edit_teacher import validate_seed_flags
    args=SimpleNamespace(seed_analytic_control='seed.pt',resume=None,evaluate_only=False,fork_objective=None,fork_private_lr=None,optimizer=optimizer,freeze_private=True,global_grad_clip=clip)
    if valid:validate_seed_flags(args)
    else:
        with pytest.raises(ValueError,match='clipping'):validate_seed_flags(args)
    args.freeze_private=False
    with pytest.raises(ValueError,match='frozen-private'):validate_seed_flags(args)
