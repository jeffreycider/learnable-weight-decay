import dataclasses
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from baseline.model import GPT
from rewriter.function_preservation import FunctionalPreservation
from rewriter.ranking_loss import topk_margin_loss
from rewriter.token_preservation import token_preservation_terms
from rewriter.tokenwise_reference import (TokenwiseConfig,TokenwiseDual,
    logit_terms_and_cotangent,tokenwise_panel,tokenwise_config_from_options,validate_objective_resume)


def config(**kwargs):
    return dataclasses.replace(TokenwiseConfig(.001,.002,.01,.03,.4,.5),**kwargs)


@pytest.mark.parametrize('chunk',[1,3,128])
def test_rowwise_cotangent_matches_independent_dense_autograd(chunk):
    torch.manual_seed(17);student=torch.randn(2,4,16,requires_grad=True)
    teacher=torch.randn_like(student).double().log_softmax(-1)
    c=config();g,sums,raw=logit_terms_and_cotangent(student,teacher,config=c,multiplier=.7,panel_tokens=24,chunk_tokens=chunk)
    kl=(teacher.exp()*(teacher-student.double().log_softmax(-1))).sum(-1)
    rank=topk_margin_loss(teacher,student.double(),k=10,margin_fraction=.5)
    terms=token_preservation_terms(kl,multiplier=.7,mean_tolerance=c.mean_tolerance,tail_tolerance=c.tail_tolerance,
        scale=c.scale,rho=c.rho,ranking_loss=rank,ranking_weight=c.ranking_weight)
    direct=torch.autograd.grad(terms['total'].sum()/24,student)[0]
    torch.testing.assert_close(g,direct,rtol=2e-6,atol=2e-7)
    torch.testing.assert_close(raw,kl.detach().flatten(),rtol=0,atol=0)
    torch.testing.assert_close(sums[:3],torch.stack([terms[k].sum() for k in ('dual_linear','tail_quadratic','ranking')]),rtol=1e-14,atol=1e-14)


@pytest.mark.parametrize('microbatch',[1,2,3])
def test_real_gpt_uneven_microbatch_one_backward_and_actor_credit(microbatch):
    from scripts.train_multistate_chisel import detached_action_credit
    from rewriter.algebra import RawAction
    torch.set_num_threads(1);torch.manual_seed(19)
    model=GPT(16,1,128).float().requires_grad_(False)
    engine=FunctionalPreservation(model,microbatch_sequences=microbatch,kl_dtype=torch.float64,gpt_precision='fp32')
    engine.chunk_tokens=3;inputs=torch.randint(16,(3,4));direction=torch.linspace(-1,1,16)
    c=config();dual=TokenwiseDual([1,2],config=c,dual_lr=.0001);dual.multipliers[1]=.7
    theta=torch.tensor(.08,requires_grad=True)
    action,credit=detached_action_credit(RawAction({},dict(bias=theta*direction)))
    parameters=dict(engine.parameters);parameters['proj.bias']=parameters['proj.bias']+action.vectors['bias']
    norm=.3*action.vectors['bias'].square().sum()
    teacher_count=student_count=actual_backwards=0;original=engine._forward
    def counted(p,x):
        nonlocal teacher_count,student_count,actual_backwards
        v=original(p,x)
        if v.requires_grad:
            student_count+=1
            def hook(g):
                nonlocal actual_backwards
                actual_backwards+=1
            v.register_hook(hook)
        else:teacher_count+=1
        return v
    engine._forward=counted
    measured=tokenwise_panel(engine,parameters,norm,inputs,config=c,dual=dual,state_id=1,state_weight=.25,actor_credit=credit)
    actual=theta.grad.clone();assert dual.updates==0 and dual.multipliers[1]==.7
    calls=(len(inputs)+microbatch-1)//microbatch
    assert (teacher_count,student_count,actual_backwards)==(calls,calls,calls)
    assert engine.compute.student_backwards==engine.compute.student_forwards==calls
    direct=torch.tensor(.08,requires_grad=True);parts=[]
    for start in range(0,len(inputs),microbatch):
        x=inputs[start:start+microbatch]
        t=original(engine.parameters,x).double().log_softmax(-1).detach()
        p=dict(engine.parameters);p['proj.bias']=p['proj.bias']+direct*direction
        s=original(p,x);kl=(t.exp()*(t-s.double().log_softmax(-1))).sum(-1)
        rank=topk_margin_loss(t,s.double(),k=10,margin_fraction=c.margin_fraction)
        parts.append(token_preservation_terms(kl,multiplier=.7,mean_tolerance=c.mean_tolerance,tail_tolerance=c.tail_tolerance,
            scale=c.scale,rho=c.rho,ranking_loss=rank,ranking_weight=c.ranking_weight)['total'].sum()/inputs.numel())
    expected=(sum(parts)+.3*(direct*direction).square().sum())*.25
    expected.backward();torch.testing.assert_close(actual,direct.grad,rtol=3e-5,atol=1e-7)
    assert measured['token_count']==12 and measured['sequence_count']==3


def test_dual_fixed_scale_panel_update_resume_and_no_heldout_feedback():
    c=config(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-8)
    dual=TokenwiseDual([1,2],config=c,dual_lr=1e-4)
    dual.step([(1,2e-8),(2,0.)],expected_ids=[1,2])
    assert dual.multipliers[1]==pytest.approx(1e-4*(2e-8-1e-10)/1e-8)
    assert dual.multipliers[2]==0 and dual.updates==1
    other=TokenwiseDual([1,2],config=c,dual_lr=1e-4);other.load_state_dict(dual.state_dict())
    assert other.state_dict()==dual.state_dict()
    frozen=dual.state_dict()
    with pytest.raises(ValueError,match='held-out'):dual.step([(5,1.)],expected_ids=[5])
    assert dual.state_dict()==frozen
    for changed in (config(scale=1e-7),dataclasses.replace(c,ranking_weight=.2),dataclasses.replace(c,tail_tolerance=2e-10)):
        with pytest.raises(ValueError,match='resume'):TokenwiseDual([1,2],config=changed,dual_lr=1e-4).load_state_dict(frozen)
    # Tightening tolerances does not change the dual's KL slope.
    tight=TokenwiseDual([1,2],config=dataclasses.replace(c,mean_tolerance=1e-12),dual_lr=1e-4)
    assert dual.terms(2e-8)['constraint_violation']-dual.terms(1e-8)['constraint_violation']==pytest.approx(tight.terms(2e-8)['constraint_violation']-tight.terms(1e-8)['constraint_violation'])


def test_default_off_and_objective_identity_fail_closed():
    args=SimpleNamespace(objective='primal-dual',token_tail_tolerance=None,token_normalization_scale=None,
        ranking_weight=None,ranking_margin_fraction=None,offline_backend='reference',gpt_precision='fp32',gradient_diagnostics=False,
        kl_tolerance=1e-10,augmented_rho=.01)
    assert tokenwise_config_from_options(args) is None
    validate_objective_resume({},None)
    with pytest.raises(ValueError,match='actor-only'):validate_objective_resume({},config())
    with pytest.raises(ValueError,match='actor-only'):validate_objective_resume({'tokenwise_objective':dataclasses.asdict(config())},None)
    args.objective='tokenwise-v1';args.token_tail_tolerance=1e-10;args.token_normalization_scale=1e-8
    c=tokenwise_config_from_options(args);validate_objective_resume({'tokenwise_objective':dataclasses.asdict(c)},c)
    args.offline_backend='strict_graph'
    with pytest.raises(ValueError,match='reference backend'):tokenwise_config_from_options(args)


def test_clipped_dual_is_explicit_and_primal_unchanged():
    c=config(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-8)
    clipped=dataclasses.replace(c,dual_update_mode='clipped_relative')
    a=TokenwiseDual([1,2,3],config=c,dual_lr=.01);b=TokenwiseDual([1,2,3],config=clipped,dual_lr=.01)
    for d in (a,b):d.multipliers.update({1:.1,2:.1,3:.1})
    samples=[(1,1e-7),(2,0.),(3,1.5e-10)]
    a.step(samples,expected_ids=[1,2,3]);b.step(samples,expected_ids=[1,2,3])
    assert b.multipliers==pytest.approx({1:.11,2:.09,3:.105})
    assert a.multipliers!=b.multipliers
    with pytest.raises(ValueError,match='resume'):a.load_state_dict(b.state_dict())
    with pytest.raises(ValueError,match='positive mean'):dataclasses.replace(clipped,mean_tolerance=0.)
    torch.manual_seed(8);s=torch.randn(4,16);t=torch.randn(4,16).double().log_softmax(-1)
    ga,sa,_=logit_terms_and_cotangent(s,t,config=c,multiplier=.5,panel_tokens=4)
    gb,sb,_=logit_terms_and_cotangent(s,t,config=clipped,multiplier=.5,panel_tokens=4)
    assert torch.equal(ga,gb) and torch.equal(sa,sb)


def test_balance_readout_row_vjps_and_stationary_score():
    from rewriter.tokenwise_reference import tokenwise_feasibility_score
    torch.manual_seed(22);s=torch.randn(3,4,16);t=torch.randn_like(s).double().log_softmax(-1);c=config()
    g,sums,_=logit_terms_and_cotangent(s,t,config=c,multiplier=0.,panel_tokens=12,chunk_tokens=3,backward=False,balance=True)
    assert g is None and sums[11]>0 and sums[12]>0
    direct=s.clone().requires_grad_();kl=(t.exp()*(t-direct.double().log_softmax(-1))).sum(-1)
    tail=.5*c.rho*((kl.clamp_min(0)-c.tail_tolerance)/c.scale).relu().square().mean()
    ranking=topk_margin_loss(t,direct.double(),k=10,margin_fraction=c.margin_fraction).mean()
    gt=torch.autograd.grad(tail,direct,retain_graph=True)[0];gr=torch.autograd.grad(ranking,direct)[0]
    torch.testing.assert_close(sums[11],gt.double().square().sum(),rtol=1e-7,atol=1e-7)
    torch.testing.assert_close(sums[12],gr.double().square().sum(),rtol=1e-7,atol=1e-7)
    feasible=dict(mean_kl=1e-11,norm_objective=-.1,tail_quadratic_penalty=1e9,ranking_penalty=2.,dual_linear_term=1e12)
    assert tokenwise_feasibility_score([feasible],1e-10)<1
    assert tokenwise_feasibility_score([dict(feasible,mean_kl=1.01e-10)],1e-10)>1
    assert tokenwise_feasibility_score([feasible],1e-10)==tokenwise_feasibility_score([dict(feasible,dual_linear_term=-1e20)],1e-10)


def test_actual_trainer_resume_and_actor_only_objective_transition(tmp_path):
    from test_multistate_chisel import _tiny_state_bank
    from scripts.train_multistate_chisel import main
    torch.set_num_threads(1);bank=tmp_path/'bank';_tiny_state_bank(bank)
    data=tmp_path/'fineweb_train_000.bin';tokens=(np.arange(1000)%16).astype(np.uint16)
    data.write_bytes(struct.pack('<iii',20240520,1,len(tokens))+bytes(1012)+tokens.tobytes())
    common=['--state-bank',str(bank),'--train-data',str(data),'--checkpoint-batch','4','--sequences-per-checkpoint','3',
        '--gpt-microbatch','2','--replay-pool-sequences','32','--lr','2e-5','--expected-gpt-seed','7','--audit-states','0',
        '--audit-sequences','1','--audit-every-epochs','1','--convergence','--minimum-epochs','1','--eager-chisel','--device','cpu',
        '--no-plots','--chisel-model-dim','128','--chisel-layers','1','--gpt-precision','fp32']
    token=['--objective','tokenwise-v1','--kl-tolerance','.001','--token-tail-tolerance','.002','--token-normalization-scale','.01',
        '--augmented-rho','.03','--ranking-weight','.4','--ranking-margin-fraction','.5']
    old=tmp_path/'legacy';main([*common,'--output',str(old),'--epochs','1'])
    init=old/'controller_latest.pt'
    with pytest.raises(ValueError,match='actor-only'):main([*common,*token,'--output',str(old),'--epochs','2','--resume',str(init)])
    resumed=tmp_path/'resumed';direct=tmp_path/'direct'
    main([*common,*token,'--output',str(resumed),'--epochs','1','--primal-init',str(init)])
    first=torch.load(resumed/'controller_latest.pt',weights_only=False)
    assert first['updates']==1 and first['primal_dual']['updates']==1
    assert all(int(s['step'])==1 for s in first['optimizer']['state'].values())
    assert all(first['metadata']['objective_transition'][k] for k in ('actor_preserved','adam_reset','duals_reset','schedule_reset','data_cursor_reset'))
    main([*common,*token,'--output',str(resumed),'--epochs','2','--resume',str(resumed/'controller_latest.pt')])
    main([*common,*token,'--output',str(direct),'--epochs','2','--primal-init',str(init)])
    a=torch.load(resumed/'controller_latest.pt',weights_only=False);b=torch.load(direct/'controller_latest.pt',weights_only=False)
    assert a['primal_dual']==b['primal_dual'] and a['fit_schedule']==b['fit_schedule'] and a['updates']==b['updates']==2
    for n in a['controller']:torch.testing.assert_close(a['controller'][n],b['controller'][n],rtol=0,atol=0)
    for i,state in a['optimizer']['state'].items():
        for k,v in state.items():torch.testing.assert_close(v,b['optimizer']['state'][i][k],rtol=0,atol=0)
    rows=[json.loads(line) for line in (direct/'metrics.jsonl').read_text().splitlines()]
    assert rows[-1]['compute']['student_backwards']==16 # 4 states x 2 microbatches x 2 updates
    assert {r['state_id'] for r in rows if r['event']=='train_state'}=={1,2,3,4}
    assert all(r['dual_multiplier']==0 for r in rows if r['event']=='audit_state' and r['state_id']==5)
    assert all('ranking_penalty' in r and 'token_kl_quantiles' in r for r in rows if r['event']=='train_state')


def test_calibration_uses_only_fixed_fitting_panels_and_can_extend(tmp_path):
    from test_multistate_chisel import _tiny_state_bank
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import write_panel
    from scripts.train_multistate_chisel import main,state_replay,select_sequences,stable_seed
    torch.set_num_threads(1);bank=tmp_path/'bank';_tiny_state_bank(bank)
    data=tmp_path/'fineweb_train_000.bin';tokens=(np.arange(1000)%16).astype(np.uint16)
    data.write_bytes(struct.pack('<iii',20240520,1,len(tokens))+bytes(1012)+tokens.tobytes())
    reader=StateBankReader(bank);cache=tmp_path/'cache';cache.mkdir();panel_rows=[]
    model=GPT(16,1,128).float().requires_grad_(False)
    engine=FunctionalPreservation(model,microbatch_sequences=1,kl_dtype=torch.float64,gpt_precision='fp32')
    for state_id in range(1,6):
        record=reader.read(state_id)
        engine.load_teacher_state({n:record.model[n] for n in engine.parameters},{n:record.model[n] for n in engine.buffers})
        pool=state_replay([str(data)],state_id,reader.manifest['provenance']['training_config'],pool_sequences=32,seed=400001,epoch=0,split_seed=500001)
        for role in ('train','audit'):
            x,ids=select_sequences(pool[role],4,seed=stable_seed('fixed_teacher_panel',400001,state_id,role))
            panel_rows.append(write_panel(cache,state_id,role,engine,x,ids,state_sha256=record.assert_verified(),execution={'gpt_precision':'fp32'}))
    (cache/'manifest.json').write_text(json.dumps(dict(complete=True,args={'states':5},execution={'gpt_precision':'fp32'},panels=panel_rows)))
    common=['--state-bank',str(bank),'--teacher-cache',str(cache),'--train-data',str(data),'--checkpoint-batch','2',
        '--sequences-per-checkpoint','2','--gpt-microbatch','1','--replay-pool-sequences','32','--lr','2e-5','--expected-gpt-seed','7',
        '--audit-states','1','--audit-sequences','1','--eager-chisel','--device','cpu','--no-plots',
        '--chisel-model-dim','128','--chisel-layers','1','--gpt-precision','fp32','--objective','tokenwise-v1',
        '--kl-tolerance','.001','--token-tail-tolerance','.002','--token-normalization-scale','.01','--augmented-rho','.03']
    output=tmp_path/'calibration';subset=['--calibration-state-ids','1','4','--calibration-readout-updates','0','1','2']
    with pytest.raises(SystemExit):main([*common,'--output',str(tmp_path/'bad'),'--epochs','1','--calibration-state-ids','1','5'])
    main([*common,*subset,'--output',str(output),'--epochs','1'])
    first=torch.load(output/'controller_latest.pt',weights_only=False)
    assert not first['completed'] and first['updates']==1 and set(first['primal_dual']['multipliers'])=={1,4}
    main([*common,*subset,'--output',str(output),'--epochs','2','--resume',str(output/'controller_latest.pt')])
    rows=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    expected={s:next(r['sequence_ids'][:2] for r in panel_rows if r['state_id']==s and r['role']=='train') for s in (1,4)}
    for row in rows:
        if row['event'] in ('train_state','audit_state','calibration_panel'):
            assert row['state_id'] in expected and row['sequence_ids']==expected[row['state_id']]
    assert {r['role'] for r in rows if r['event']=='audit_state'}=={'fixed_training'}
    assert [r['update'] for r in rows if r['event']=='audit_summary']==[0,1,2]
    assert rows[-1]['compute']['teacher_forwards']==0 and rows[-1]['compute']['student_backwards']==8
    assert rows[-1]['compute']['student_forwards']==20 # 8 fitting + 12 fixed readout microbatches
    with pytest.raises(ValueError,match='calibration actors'):
        main([*common,'--output',str(tmp_path/'forbidden-transfer'),'--epochs','1','--primal-init',str(output/'controller_latest.pt')])
