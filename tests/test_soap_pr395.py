"""Published-baseline parity; legacy extra-polar tests remain separate."""
import ast
import dataclasses
import math
import textwrap
from pathlib import Path
import pytest
import torch
from rewriter.soap_pr395_residual import PublishedMap, committed_pr395_proposal
from rewriter.soap_residual import inverse_fourth_factor, residual_scale
from rewriter.lag_one import LagOneHypergradient
from test_soap_residual import upstream_helpers


def upstream():
    s=upstream_helpers()
    tree=ast.parse(Path('baseline/pr395/upstream.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='shampoo_update')
    s.update(SHAMPOO_BETA2=.8,SHAMPOO_DAMPING=.01,SHAMPOO_NS_STEPS=8,
             SHAMPOO_NS_FINAL_STEPS=10,SHAMPOO_NS_TRANSITION_STEP=1450,
             SOAP_TARGET_UW=.3825,NONSOAP_TARGET_UW=.2,ROWFLOOR=True,ROWFLOOR_RHO=1.,CWD=.025)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<upstream shampoo>','exec'),s)
    return s


def make_record(shape=(3,5),dtype=torch.float32,spa=False):
    torch.manual_seed(64)
    s=upstream();n,m=shape
    weight=torch.randn(n,m,dtype=dtype);direction=torch.randn(n,m)
    a=torch.randn(n,n);b=torch.randn(m,m)
    state={'shp_left':a@a.T,'shp_right':b@b.T,'q_row':None}
    residual=torch.randn_like(direction)
    # Update the exact same Gram buffers once, as upstream shampoo_update does.
    state['shp_left'].lerp_(residual@residual.T,.2)
    state['shp_right'].lerp_(residual.T@residual,.2)
    app=s['apply_inverse_fourth_root_left']
    raw=app(direction,state['shp_left'],8,.01)
    raw=app(raw.T.contiguous(),state['shp_right'],8,.01).T
    factors=[inverse_fourth_factor(state['shp_left']),inverse_fourth_factor(state['shp_right']).T.contiguous()]
    spectral={};gradient=torch.randn_like(direction);group={'lr':.01,'initial_lr':.0325}
    if spa:
        state['q_row']=torch.linalg.qr(torch.randn(n,n))[0];state['q_col']=torch.linalg.qr(torch.randn(m,m))[0]
        state['spa_rr']=gradient@gradient.T;state['spa_cc']=gradient.T@gradient
        row=torch.diag(state['q_row'].T@state['spa_rr']@state['q_row']).clamp_min(1e-30).log()
        col=torch.diag(state['q_col'].T@state['spa_cc']@state['q_col']).clamp_min(1e-30).log()
        lv=row[:,None]+col[None,:];sc=((lv-lv.mean())/lv.std().clamp_min(1e-6)).clamp(-2,2)
        spectral=dict(spectral_row=state['q_row'],spectral_col=state['q_col'],spectral_scale=torch.exp(.25*math.log(.01/.0325)*sc))
        del state['spa_rr'];del state['spa_cc']
    record=PublishedMap(weight,direction,*factors,residual_scale(state['shp_left']),residual_scale(state['shp_right']),.01,base_raw=raw,target_uw=.3825 if spa else .2,**spectral)
    return record,s,state,gradient,group


def exact_postprocessing(record,s,state,gradient,group,spa):
    # Executed verbatim from the pinned Muon.step, starting immediately after
    # its shampoo_update call. Includes all geometry and storage operations.
    source=Path('baseline/pr395/upstream.py').read_text()
    start=source.index('                    if use_soap and state["q_row"] is not None:',source.index('                    ns_steps = SHAMPOO_NS_FINAL_STEPS'))
    end=source.index('                    if use_soap:\n                        soap_update_preconditioner',start)
    body=textwrap.dedent(source[start:end])
    update=record.base_raw*(math.sqrt(record.direction.shape[0])/record.base_raw.norm().clamp_min(1e-20))
    p=record.weight.clone()
    s.update(p=p,update=update,grad=gradient,state=state,group=group,use_soap=spa,lr=record.lr,ROWFLOOR=record.rowfloor)
    exec(compile(body,'<verbatim upstream postprocessing>','exec'),s)
    return p


@pytest.mark.parametrize('shape',[(3,5),(5,3),(4,4),(768,64)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('spa',[False,True])
def test_zero_residual_matches_exact_upstream_postprocessing(shape,dtype,spa):
    r,s,state,g,group=make_record(shape,dtype,spa)
    expected=exact_postprocessing(r,s,state,g,group,spa)
    actual=committed_pr395_proposal(r,torch.zeros_like(r.left),torch.zeros_like(r.right))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_direct_shampoo_and_grams_match_upstream():
    s=upstream();torch.manual_seed(65)
    direction=torch.randn(7,4);residual=torch.randn_like(direction)
    states=[{'shp_left':torch.eye(7),'shp_right':torch.eye(4)} for _ in range(2)]
    expected=s['shampoo_update'](direction,residual,states[0],8)
    state=states[1]
    state['shp_left'].lerp_(residual@residual.T,1-s['SHAMPOO_BETA2'])
    state['shp_right'].lerp_(residual.T@residual,1-s['SHAMPOO_BETA2'])
    raw=s['apply_inverse_fourth_root_left'](direction,state['shp_left'],8,.01)
    raw=s['apply_inverse_fourth_root_left'](raw.T.contiguous(),state['shp_right'],8,.01).T
    actual=raw*(direction.size(0)**.5/raw.norm().clamp_min(1e-20))
    assert torch.equal(expected,actual)
    assert all(torch.equal(states[0][key],states[1][key]) for key in state)


def double_record():
    r,*_=make_record()
    return dataclasses.replace(r,**{f.name:getattr(r,f.name).double() for f in dataclasses.fields(r)
        if isinstance(getattr(r,f.name),torch.Tensor)})


def test_nonzero_residual_finite_difference_and_zero_trainability():
    r=double_record();hl=(torch.randn_like(r.left)*.01).requires_grad_();hr=(torch.randn_like(r.right)*.01).requires_grad_()
    assert torch.autograd.gradcheck(lambda h,k:committed_pr395_proposal(r,h,k),(hl,hr),eps=1e-6,atol=2e-5,rtol=2e-4)
    hl=torch.zeros_like(hl,requires_grad=True);hr=torch.zeros_like(hr,requires_grad=True)
    gradients=torch.autograd.grad((committed_pr395_proposal(r,hl,hr)*torch.randn_like(r.weight)).sum(),(hl,hr))
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in gradients)


def test_lag_credit_includes_full_tail_and_displacement_envelope():
    r=double_record();hl=torch.zeros_like(r.left,requires_grad=True);hr=torch.zeros_like(r.right,requires_grad=True)
    gamma,beta,anchor,c=.99,.23,.45,.01
    previous=torch.randn_like(r.weight);buffer=torch.randn_like(previous);tail=torch.randn_like(previous)
    def next_forward():
        p=committed_pr395_proposal(r,hl,hr)
        next_tail=(1-c)*tail+c*p
        next_buffer=gamma*buffer+(1-gamma)*(p-previous)
        return (1+anchor)*(p+beta*next_buffer)-anchor*next_tail
    q=next_forward();inputs=torch.randn(5,3,dtype=q.dtype);target=torch.randn(3,3,dtype=q.dtype)
    loss=((q@inputs-target)**2).mean();expected=torch.autograd.grad(loss,(hl,hr))
    leaf=q.detach().requires_grad_();g=torch.autograd.grad(((leaf@inputs-target)**2).mean(),leaf)[0]
    coefficient=(1+anchor)*(1+beta*(1-gamma))-anchor*c
    lag=LagOneHypergradient();lag.from_contraction(action_step=2100,hyperparameters={'hl':hl,'hr':hr},
        contraction=lambda cot:(committed_pr395_proposal(r,hl,hr)*cot['weight']).sum()*coefficient)
    actual=lag.consume({'weight':g},gradient_step=2101)
    for key,want in zip(('hl','hr'),expected):torch.testing.assert_close(actual.gradients[key],want)


def test_default_source_is_published_and_legacy_explicit():
    from scripts.train_soap_hyper import patched_source
    source=Path('baseline/pr395/upstream.py').read_text();kw=dict(meta_lr=0,output='/tmp/test',steps=256,microbatch=64)
    assert "variant='pr395'" in patched_source(source,**kw)
    assert "variant='legacy-extra-polar'" in patched_source(source,variant='legacy-extra-polar',**kw)
    with pytest.raises(ValueError):patched_source(source,variant='oops',**kw)


@pytest.mark.parametrize('rowfloor',[False,True])
def test_production_square_shape_and_zero_direction(rowfloor):
    r,s,state,g,group=make_record((768,768),torch.bfloat16,False)
    r=dataclasses.replace(r,rowfloor=rowfloor)
    expected=exact_postprocessing(r,s,state,g,group,False)
    actual=committed_pr395_proposal(r,torch.zeros_like(r.left),torch.zeros_like(r.right))
    assert torch.equal(actual,expected)
    zero=dataclasses.replace(r,direction=torch.zeros_like(r.direction),base_raw=torch.zeros_like(r.base_raw))
    h=torch.zeros_like(r.left,requires_grad=True);k=torch.zeros_like(r.right,requires_grad=True)
    result=committed_pr395_proposal(zero,h,k)
    grads=torch.autograd.grad(result.float().sum(),(h,k))
    assert torch.isfinite(result).all() and all(torch.isfinite(v).all() for v in grads)


def test_online_published_proposal_matches_upstream_and_updates_statistics_once(tmp_path):
    from rewriter.soap_online import SoapResidualOnline
    s=upstream();torch.manual_seed(87)
    direction=torch.randn(6,4);residual=torch.randn_like(direction);p=torch.nn.Parameter(torch.randn_like(direction));p.grad=torch.randn_like(p)
    states=[{'shp_left':torch.eye(6),'shp_right':torch.eye(4),'q_row':None} for _ in range(2)]
    expected_raw=s['shampoo_update'](direction,residual,states[0],8)
    r=PublishedMap(p.detach().clone(),direction,torch.eye(6),torch.eye(4),torch.tensor(1.),torch.tensor(1.),.01,
                   base_raw=expected_raw,target_uw=.2)
    # expected_raw is already normalized, so remove only the reference helper's
    # second normalization by executing the verbatim body directly here.
    source=Path('baseline/pr395/upstream.py').read_text()
    start=source.index('                    if use_soap and state["q_row"] is not None:',source.index('                    ns_steps = SHAMPOO_NS_FINAL_STEPS'))
    end=source.index('                    if use_soap:\n                        soap_update_preconditioner',start)
    expected=p.detach().clone();group={'lr':.01,'initial_lr':.01}
    s.update(p=expected,update=expected_raw,grad=p.grad,state=states[0],group=group,use_soap=False,lr=.01)
    exec(textwrap.dedent(source[start:end]),s)
    online=SoapResidualOnline.__new__(SoapResidualOnline)
    online.scope=s;online.variant='pr395';online.learned=False;online.entries={};online.map=committed_pr395_proposal
    online.propose(p,direction,residual,states[1],group,1,False)
    assert torch.equal(p,expected)
    for key in ('shp_left','shp_right'):assert torch.equal(states[0][key],states[1][key])


def test_corrected_record_resume_cannot_cross_variants():
    from rewriter.soap_online import SoapResidualOnline
    model=torch.nn.Linear(5,3,bias=False).double();r=double_record()
    def bare(variant):
        h=SoapResidualOnline.__new__(SoapResidualOnline);h.variant=variant;h.scope={'model':model};h.learned=True;h.lr=1e-5
        h.entries={};h.optimizer=None;h.meta_steps=0;h.map=committed_pr395_proposal
        return h
    h=bare('pr395');entry={'hl':torch.nn.Parameter(torch.zeros_like(r.left)),
                          'hr':torch.nn.Parameter(torch.zeros_like(r.right)),'lag':LagOneHypergradient()}
    h.entries[model.weight]=entry;h.install_pending(entry,r,1.45,2100)
    saved=h.state_dict();resumed=bare('pr395');resumed.load_state_dict(saved)
    assert isinstance(resumed.entries[model.weight]['pending_record'],PublishedMap)
    cotangent={'weight':torch.randn_like(model.weight)}
    a=entry['lag'].consume(cotangent,gradient_step=2101)
    b=resumed.entries[model.weight]['lag'].consume(cotangent,gradient_step=2101)
    for key in a.gradients:torch.testing.assert_close(a.gradients[key],b.gradients[key],rtol=0,atol=0)
    with pytest.raises(ValueError,match='variant'):bare('legacy-extra-polar').load_state_dict(saved)
