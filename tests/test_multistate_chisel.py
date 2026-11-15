import json,struct
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scripts.train_multistate_chisel import (epoch_batches,heldout_sequence,state_replay,
    mean_kl_and_backward,choose_audit_states,select_sequences,deadband)
from rewriter.grouped_norm import GroupedNormObjective
from baseline.model import GPT
from rewriter.function_preservation import FunctionalPreservation


def test_shuffled_epochs_cover_only_train_states_and_are_reproducible():
    training=[1,2,3,4,6,7,8,9];audit={5,10}
    first=epoch_batches(training,0,3,3);second=epoch_batches(training,1,3,3)
    assert first==epoch_batches(training,0,3,3) and first!=second
    assert sorted(x for b in first for x in b)==training
    assert all(not set(b)&audit for b in first+second)
    assert len(first[-1])==2
    assert choose_audit_states([5,10,15,20],2)==[5,20]


def test_full_validation_keeps_all_200_states_with_small_fixed_training_probe():
    validation=list(range(5,1001,5));training=[s for s in range(1,1001) if s%5]
    assert choose_audit_states(validation,0)==validation
    assert choose_audit_states(validation,200)==validation
    probe=choose_audit_states(training,32)
    assert len(probe)==32 and not set(probe)&set(validation)
    assert probe[0]==1 and probe[-1]==999


def test_global_sequence_holdout_does_not_leak_between_states_or_epochs(tmp_path):
    path=tmp_path/'fineweb_train_000001.bin';tokens=np.arange(1000,dtype=np.uint16)
    path.write_bytes(struct.pack('<iii',20240520,1,len(tokens))+bytes(1012)+tokens.tobytes())
    config=dict(batch_tokens=128,seq_len=4);train_ids=set();audit_ids=set()
    for state,epoch in [(1,0),(2,0),(3,1),(4,2)]:
        split=state_replay([path],state,config,pool_sequences=128,seed=9,epoch=epoch,split_seed=8)
        for role,values in split.items():
            ids=[x['id'] for x in values.provenance]
            assert all(x<state*32 for x in ids)
            assert all(heldout_sequence(x,8)==(role=='audit') for x in ids)
            (train_ids if role=='train' else audit_ids).update(ids)
    assert train_ids and audit_ids and not train_ids&audit_ids


def test_zero_reference_freeze_contributes_zero_without_changing_group_average():
    base={'weight':torch.ones(3,2,dtype=torch.float64),'bias':torch.zeros(2,dtype=torch.float64)}
    groups={'matrix':('weight',),'bias':('bias',)}
    obj=GroupedNormObjective(base,groups,zero_reference_policy='freeze')
    assert obj.zero_reference_groups==('bias',) and obj.frozen_parameter_names=={'bias'}
    result=obj.evaluate({'weight':base['weight']*.8,'bias':base['bias']})
    assert float(result.mean_relative_change)==pytest.approx(-.1)
    assert torch.isfinite(result.mean_relative_change)
    with pytest.raises(ValueError,match='must remain frozen'):
        obj.evaluate({'weight':base['weight'],'bias':torch.ones(2,dtype=torch.float64)})
    positive=dict(base,bias=torch.ones(2,dtype=torch.float64))
    assert not GroupedNormObjective(positive,groups,zero_reference_policy='freeze').frozen_parameter_names


class LinearEngine:
    def __init__(self,microbatch):self.microbatch_sequences=microbatch;self.forward_count=0
    def cache_teacher(self,inputs,**kwargs):
        return [SimpleNamespace(inputs=inputs,logq=(inputs@torch.tensor([[.2,-.4],[.3,.7]],dtype=torch.float64)).log_softmax(-1))]
    def student_parameter_loss(self,parameters,teacher):
        self.forward_count+=1
        logp=(teacher.inputs@parameters['weight']+parameters['bias']).log_softmax(-1)
        return (teacher.logq.exp()*(teacher.logq-logp)).sum(-1).mean()


def test_microbatch_accumulated_kl_vjp_matches_direct_mean_objective():
    torch.set_num_threads(1);torch.manual_seed(5)
    inputs=torch.randn(7,2,dtype=torch.float64)
    for microbatch in (1,2,4,7):
        theta=torch.tensor(.07,dtype=torch.float64,requires_grad=True)
        params={'weight':torch.tensor([[.4,-.2],[.6,.1]],dtype=torch.float64)*theta.sin(),
                'bias':torch.tensor([.03,-.02],dtype=torch.float64)}
        regularizer=.2*theta.square()-.3*theta
        engine=LinearEngine(microbatch)
        measured=mean_kl_and_backward(engine,params,regularizer,inputs,kl_budget=.001,state_weight=.25)
        actual=theta.grad.clone()
        assert engine.forward_count==(len(inputs)+microbatch-1)//microbatch
        direct=torch.tensor(.07,dtype=torch.float64,requires_grad=True)
        dp=dict(params,weight=torch.tensor([[.4,-.2],[.6,.1]],dtype=torch.float64)*direct.sin())
        teacher=engine.cache_teacher(inputs)[0]
        kl=engine.student_parameter_loss(dp,teacher)
        loss=.2*direct.square()-.3*direct+(kl/.001-1).clamp_min(0).square()
        (loss*.25).backward()
        torch.testing.assert_close(actual,direct.grad,rtol=1e-12,atol=1e-12)
        assert measured['mean_kl']==pytest.approx(float(kl.detach()),abs=1e-14)


def test_multiple_checkpoint_contributions_average_in_one_shared_gradient():
    torch.set_num_threads(1)
    theta=torch.tensor(.09,dtype=torch.float64,requires_grad=True);expected=0.
    for offset in (.2,.5):
        inputs=torch.tensor([[offset,.3],[.7,-.5]],dtype=torch.float64)
        params={'weight':theta*torch.ones(2,2,dtype=torch.float64),'bias':torch.zeros(2,dtype=torch.float64)}
        mean_kl_and_backward(LinearEngine(1),params,theta.square(),inputs,kl_budget=.1,state_weight=.5)
        expected+=float(theta.detach())*1.  # KL has identical logits and zero directional derivative here.
    assert float(theta.grad)==pytest.approx(expected)


def test_real_gpt_microbatch_vjp_matches_direct_backward():
    torch.set_num_threads(1);torch.manual_seed(3)
    model=GPT(16,1,128).float().requires_grad_(False)
    engine=FunctionalPreservation(model,microbatch_sequences=1,kl_dtype=torch.float64)
    inputs=torch.randint(16,(3,4));direction=torch.linspace(-1.,1.,16)
    theta=torch.tensor(.02,requires_grad=True)
    params=dict(engine.parameters);params['proj.bias']=params['proj.bias']+theta*direction
    measured=mean_kl_and_backward(engine,params,theta.square(),inputs,kl_budget=.00001)
    actual=theta.grad.clone();direct=torch.tensor(.02,requires_grad=True)
    params=dict(engine.parameters);params['proj.bias']=params['proj.bias']+direct*direction
    losses=[engine.student_parameter_loss(params,engine.cache_teacher(inputs[i:i+1],data_role='past_training')[0]) for i in range(3)]
    kl=torch.stack(losses).mean();(direct.square()+(kl/.00001-1).clamp_min(0).square()).backward()
    torch.testing.assert_close(actual,direct.grad,rtol=2e-5,atol=2e-5)
    assert measured['mean_kl']==pytest.approx(float(kl.detach()),rel=1e-10,abs=1e-14)


def _tiny_state_bank(directory):
    from rewriter.state_bank import StateBankWriter
    from rewriter.controller import ControllerSnapshot,MatrixObservation,VectorObservation,coordinate_features,vector_features
    from rewriter.algebra import Anchors,Measurements
    torch.manual_seed(81)
    model=GPT(16,1,128)
    cfg=dict(model=dict(vocab_size=16,num_layers=1,model_dim=128),batch_tokens=128,seq_len=4,schedule_steps=20)
    prov=dict(baseline_commit='unit-test-only',model_config=cfg['model'],training_config=cfg,dataset={'name':'synthetic-test-fixture'},
              seed=7,schedule_steps=20,capture_world_size=1,sensor_config={'probe_rank':2})
    gen=torch.Generator().manual_seed(19)
    with StateBankWriter(directory,prov,expected_steps=5) as writer:
        for step in range(1,6):
            with torch.no_grad():model.proj.bias.add_(.001)
            matrices={};vectors={};tm={};tv={};norm=torch.zeros(())
            for name,p in model.named_parameters():
                w=(p.T if name=='embed.weight' else p).detach().float();norm+=w.square().sum()
                if p.ndim==2:
                    a=Anchors.random(*w.shape,2,generator=gen);wb=w@a.B;wta=w.T@a.A
                    rows=coordinate_features([wb]+[torch.zeros_like(wb)]*3,a.A,torch.ones(len(w)),[0.]*10)
                    columns=coordinate_features([wta]+[torch.zeros_like(wta)]*3,a.B,torch.ones(w.shape[1]),[0.]*10)
                    matrices[name]=MatrixObservation(a,rows,columns,(wb,wta));tm[name]=(torch.zeros(4,*wb.shape),torch.zeros(4,*wta.shape))
                else:
                    vectors[name]=VectorObservation(vector_features([w]+[torch.zeros_like(w)]*3,[0.]*10),w.clone());tv[name]=torch.zeros(4,len(w))
            snapshot=ControllerSnapshot(matrices,vectors,torch.zeros(4,12),norm.sqrt(),norm.sqrt()*.01)
            writer.append(step,model.state_dict(),snapshot,Measurements(tm,tv),data_cursor_batches=step,identity_action=True)
    return cfg


@pytest.mark.parametrize('reader_flags',[[],['--verify-state-once','--mmap-state']])
def test_two_epoch_state_bank_training_audits_and_canonical_checkpoint(tmp_path,reader_flags):
    from scripts.train_multistate_chisel import main
    bank=tmp_path/'bank';_tiny_state_bank(bank)
    data=tmp_path/'fineweb_train_000001.bin';tokens=(np.arange(1000)%16).astype(np.uint16)
    data.write_bytes(struct.pack('<iii',20240520,1,len(tokens))+bytes(1012)+tokens.tobytes())
    output=tmp_path/'run'
    main(['--state-bank',str(bank),'--output',str(output),'--train-data',str(data),
          '--epochs','2','--checkpoint-batch','2','--sequences-per-checkpoint','2','--gpt-microbatch','1',
          '--replay-pool-sequences','32','--lr','.00002','--kl-budget','.1','--expected-gpt-seed','7',
          '--audit-states','1','--audit-sequences','1','--audit-every','999','--checkpoint-every','1',
          '--eager-chisel','--device','cpu','--no-plots','--chisel-model-dim','128','--chisel-layers','1',*reader_flags])
    rows=[json.loads(x) for x in (output/'metrics.jsonl').read_text().splitlines()]
    updates=[x for x in rows if x['event']=='update'];training=[x for x in rows if x['event']=='train_state']
    assert len(updates)==4 and len(training)==8
    assert {x['state_id'] for x in training}=={1,2,3,4}
    assert all(not heldout_sequence(i,500001) for x in training for i in x['sequence_ids'])
    audits=[x for x in rows if x['event']=='audit_state']
    assert {x['role'] for x in audits}=={'heldout_sequences','heldout_states'}
    assert all(heldout_sequence(i,500001) for x in audits for i in x['sequence_ids'])
    assert all(x['read_only_unchanged'] for x in audits)
    assert all(x['state_id']==5 for x in audits if x['role']=='heldout_states')
    saved=torch.load(output/'controller_latest.pt',weights_only=False)
    assert saved['updates']==4 and saved['next_epoch']==2
    assert saved['metadata']['options']['verify_state_once']==bool(reader_flags)
    assert saved['metadata']['options']['mmap_state']==bool(reader_flags)
    assert saved['metadata']['architecture']['name']=='nanogpt_blocks_patch_io_v1'
    assert not any(k.startswith('_orig_mod.') for k in saved['controller'])
    assert rows[-1]['event']=='finished' and rows[-1]['compute']['student_backwards']==16
    assert any(x['mean_group_l2_percentage_reduction']!=0 for x in training[2:])


@pytest.mark.parametrize('progress_flags',[[],['--track-trained-policy-progress']])
@pytest.mark.parametrize('direction_initialized',[False,True])
def test_convergence_epoch_cap_can_resume_with_exact_schedule_and_optimizer(tmp_path,progress_flags,direction_initialized):
    from scripts.train_multistate_chisel import main
    bank=tmp_path/'bank';_tiny_state_bank(bank)
    data=tmp_path/'fineweb_train_000.bin';tokens=(np.arange(1000)%16).astype(np.uint16)
    data.write_bytes(struct.pack('<iii',20240520,1,len(tokens))+bytes(1012)+tokens.tobytes())
    common=['--state-bank',str(bank),'--train-data',str(data),'--checkpoint-batch','4',
        '--sequences-per-checkpoint','2','--gpt-microbatch','1','--replay-pool-sequences','32',
        '--lr','2e-5','--kl-budget','.1','--expected-gpt-seed','7','--audit-states','0',
        '--audit-sequences','1','--audit-every-epochs','1','--convergence','--minimum-epochs','1',
        '--eager-chisel','--device','cpu','--no-plots','--chisel-model-dim','128','--chisel-layers','1',*progress_flags]
    resumed=tmp_path/'resumed';direct=tmp_path/'direct'
    initialization=[]
    if direction_initialized:
        import hashlib
        from rewriter.chisel_factory import chisel_architecture,build_chisel
        architecture=chisel_architecture(SimpleNamespace(chisel_model_dim=128,chisel_layers=1,chisel_patch_size=32),rank=2,trajectory_dim=12)
        torch.manual_seed(171);actor=build_chisel(architecture)
        warmup=tmp_path/'direction.pt';manifest=json.loads((bank/'manifest.json').read_text())
        torch.save(dict(controller=actor.state_dict(),updates=13,metadata=dict(architecture=architecture,
            bank_manifest_sha256=hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest(),
            bank_provenance=manifest['provenance'],options={'prefix_states':0})),warmup)
        initialization=['--direction-init',str(warmup)]
        common+=['--direction-head-scale','0']
    main([*common,*initialization,'--output',str(resumed),'--epochs','1'])
    saved=torch.load(resumed/'controller_latest.pt',weights_only=False)
    assert not saved['completed'] and saved['stop_reason']=='maximum_epochs'
    assert saved['fit_schedule'] is not None and (resumed/'controller_best.pt').exists()
    main([*common,'--output',str(resumed),'--epochs','2','--resume',str(resumed/'controller_latest.pt')])
    main([*common,*initialization,'--output',str(direct),'--epochs','2'])
    a=torch.load(resumed/'controller_latest.pt',weights_only=False)
    b=torch.load(direct/'controller_latest.pt',weights_only=False)
    assert a['fit_schedule']==b['fit_schedule'] and a['updates']==b['updates']==2
    assert a['primal_dual']==b['primal_dual']
    if direction_initialized:
        assert a['metadata']['same_run_direction_warmup']==b['metadata']['same_run_direction_warmup']
        assert a['metadata']['same_run_direction_warmup']['updates']==13
    assert set(a['primal_dual']['multipliers'])=={1,2,3,4}
    for name in a['controller']:torch.testing.assert_close(a['controller'][name],b['controller'][name],rtol=0,atol=0)
    for index,state in a['optimizer']['state'].items():
        for key,value in state.items():torch.testing.assert_close(value,b['optimizer']['state'][index][key],rtol=0,atol=0)

    from scripts.evaluate_chisel_fresh import main as evaluate
    test_output=tmp_path/'fresh'
    evaluate(['--checkpoint',str(resumed/'controller_best.pt'),'--state-bank',str(bank),
        '--train-data',str(data),'--output',str(test_output),'--sequences','1','--gpt-microbatch','1',
        '--state-ids','5','--device','cpu','--eager-chisel'])
    summary=json.loads((test_output/'summary.json').read_text())
    assert summary['state_count']==1 and summary['compute']['student_backwards']==0
    assert summary['compute']['teacher_forwards']==summary['compute']['student_forwards']==1
    samples=[json.loads(line) for line in (test_output/'states.jsonl').read_text().splitlines()]
    assert samples[0]['state_id']==5 and all(heldout_sequence(i,500001) for i in samples[0]['sequence_ids'])
    assert len(samples[0]['group_relative_edit_norm'])==15
    assert all(v>=0 for v in samples[0]['group_relative_edit_norm'].values())


def test_proposal_masks_zero_groups_until_a_later_positive_state():
    from scripts.train_multistate_chisel import build_proposal
    from rewriter.algebra import Anchors,Factors,RawAction
    from rewriter.controller import ControllerSnapshot,MatrixObservation,VectorObservation
    torch.set_num_threads(1);torch.manual_seed(4)
    model=GPT(16,1,128).float().requires_grad_(False);model.proj.bias.zero_()
    gen=torch.Generator().manual_seed(9);matrices={};vectors={};matrix_obs={};vector_obs={}
    for n,p in model.named_parameters():
        if p.ndim==2:
            w=p.T if n=='embed.weight' else p;a=Anchors.random(*w.shape,2,generator=gen)
            matrices[n]=Factors(torch.zeros_like(a.A),torch.zeros_like(a.B),a)
            matrix_obs[n]=MatrixObservation(a,torch.empty(w.shape[0],25),torch.empty(w.shape[1],25),(w@a.B,w.T@a.A))
        else:
            vectors[n]=torch.ones_like(p)
            vector_obs[n]=VectorObservation(torch.empty(p.numel(),18),p)
    action=RawAction(matrices,vectors);snapshot=ControllerSnapshot(matrix_obs,vector_obs,torch.zeros(1,12),torch.tensor(1.),torch.tensor(.01))
    engine=FunctionalPreservation(model);groups=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
    parameters,regularizer,stats,editable,_=build_proposal(lambda _:action,snapshot,engine,groups,radius_fraction=.01)
    assert 'proj.bias' not in editable and torch.equal(parameters['proj.bias'],engine.parameters['proj.bias'])
    assert stats['zero_reference_groups']==['head.bias'] and torch.isfinite(regularizer)
    model.proj.bias.fill_(.01)
    engine=FunctionalPreservation(model);groups=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
    parameters,_,stats,editable,_=build_proposal(lambda _:action,snapshot,engine,groups,radius_fraction=.01)
    assert 'proj.bias' in editable and not torch.equal(parameters['proj.bias'],engine.parameters['proj.bias'])
    assert stats['zero_reference_groups']==[]


def test_fixed_state_shards_visit_every_state_once_and_keep_rank_ownership():
    from scripts.train_multistate_chisel import sharded_epoch_batches
    ids=list(range(1,81))
    previous=None
    for epoch in range(3):
        batches=sharded_epoch_batches(ids,epoch,17,16,4)
        assert sorted(s for batch in batches for s in batch)==ids
        for rank in range(4):
            assert sorted(s for batch in batches for s in batch[rank::4])==ids[rank::4]
        assert batches==sharded_epoch_batches(ids,epoch,17,16,4)
        assert batches!=previous;previous=batches


def test_two_rank_checkpoint_minibatch_training(tmp_path):
    import os,subprocess,sys
    from pathlib import Path
    bank=tmp_path/'bank';_tiny_state_bank(bank)
    data=tmp_path/'fineweb_train_000001.bin';tokens=(np.arange(1000)%16).astype(np.uint16)
    data.write_bytes(struct.pack('<iii',20240520,1,len(tokens))+bytes(1012)+tokens.tobytes())
    output=tmp_path/'distributed';root=Path(__file__).resolve().parents[1]
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
             str(root/'scripts/train_multistate_chisel.py'),'--state-bank',str(bank),'--output',str(output),
             '--train-data',str(data),'--epochs','1','--checkpoint-batch','2','--sequences-per-checkpoint','2',
             '--gpt-microbatch','1','--replay-pool-sequences','32','--kl-budget','.1','--expected-gpt-seed','7',
             '--audit-states','1','--audit-sequences','1','--audit-every','999','--checkpoint-every','1',
             '--eager-chisel','--device','cpu','--no-plots','--chisel-model-dim','128','--chisel-layers','1']
    result=subprocess.run(command,capture_output=True,text=True,timeout=90,env=dict(os.environ,OMP_NUM_THREADS='1'))
    assert result.returncode==0,result.stdout[-4000:]+result.stderr[-4000:]
    rows=[json.loads(x) for x in (output/'metrics.jsonl').read_text().splitlines()]
    trained=[r for r in rows if r['event']=='train_state']
    assert len(trained)==4 and {r['state_id'] for r in trained}=={1,2,3,4}
    finished=rows[-1];assert finished['event']=='finished'
    assert finished['compute']['student_backwards']==8
    saved=torch.load(output/'controller_latest.pt',weights_only=False)
    assert saved['metadata']['world_size']==2 and len(saved['per_rank_compute'])==2
    assert saved['updates']==2 and saved['completed']
    assert [v['student_backwards'] for v in saved['per_rank_compute']]==[4,4]
    from rewriter.primal_dual import PrimalDual
    expected=PrimalDual([1,2,3,4],tolerance=1e-8)
    for update in sorted({r['update'] for r in trained}):
        batch=[r for r in trained if r['update']==update]
        expected.step([(r['state_id'],r['mean_kl']) for r in batch],expected_ids=[r['state_id'] for r in batch])
    assert saved['primal_dual']==expected.state_dict()
    assert all(r['trust_term']==0 and not r['edit_size_penalty_enabled'] for r in trained)
    assert len([r for r in rows if r['event']=='dual_update'])==2


def test_action_units_are_homogeneous_group_rms_without_absolute_floor():
    from scripts.train_multistate_chisel import group_action_scales
    base={'w.weight':torch.ones(4,4),'a.bias':torch.tensor([1e-7,2e-7]),'b.bias':torch.zeros(2)}
    groups={'matrix':('w.weight',),'bias':('a.bias','b.bias')}
    first=group_action_scales(GroupedNormObjective(base,groups),2)
    scaled={n:value*7 if n.endswith('bias') else value for n,value in base.items()}
    second=group_action_scales(GroupedNormObjective(scaled,groups),2)
    assert 0<float(first['a.bias'])<1e-6
    assert first['a.bias']==first['b.bias']  # a zero tensor in a positive group is editable
    torch.testing.assert_close(second['a.bias'],7*first['a.bias'])
    torch.testing.assert_close(second['w.weight'],first['w.weight'])
    zero={n:torch.zeros_like(value) if n.endswith('bias') else value for n,value in base.items()}
    frozen=group_action_scales(GroupedNormObjective(zero,groups,zero_reference_policy='freeze'),2)
    assert frozen['a.bias']==frozen['b.bias']==0


def test_anchored_ceiling_matches_dense_orthogonal_projection():
    from scripts.train_multistate_chisel import anchored_norm_ceiling
    from rewriter.algebra import Anchors
    torch.manual_seed(33);w=torch.randn(7,5,dtype=torch.float64);v=torch.randn(3,dtype=torch.float64)
    anchors=Anchors.random(7,5,2,generator=torch.Generator().manual_seed(12),dtype=torch.float64)
    snapshot=SimpleNamespace(matrices={'weight':SimpleNamespace(anchors=anchors,weight_probes=(w@anchors.B,w.T@anchors.A))})
    grouped=GroupedNormObjective({'weight':w,'bias':v},{'matrix':('weight',),'vector':('bias',)})
    actual=anchored_norm_ceiling(snapshot,grouped)
    p=w@anchors.B-anchors.A@(anchors.A.T@w@anchors.B);q=w.T@anchors.A
    edited=w-p@anchors.B.T-anchors.A@q.T
    expected=100*(1-edited.norm()/w.norm())
    assert actual['unconstrained_anchored_ceiling_group_percentage']['matrix']==pytest.approx(float(expected),abs=1e-12)
    assert actual['unconstrained_anchored_ceiling_group_percentage']['vector']==100.


def test_split_actor_vjp_matches_uninterrupted_chain_rule():
    torch.set_num_threads(1);inputs=torch.tensor([[.2,.3],[-.4,.1]],dtype=torch.float64)
    theta=torch.tensor([[.2,-.4],[.1,.7]],dtype=torch.float64,requires_grad=True)
    output=theta.sin();leaf=output.detach().requires_grad_()
    params={'weight':leaf*.7,'bias':torch.zeros(2,dtype=torch.float64)}
    timings={}
    mean_kl_and_backward(LinearEngine(1),params,.1*leaf.square().sum(),inputs,kl_budget=.001,
                        actor_credit=([output],[leaf]),timings=timings)
    actual=theta.grad.clone();reference=theta.detach().clone().requires_grad_();value=reference.sin()
    engine=LinearEngine(2);kl=engine.student_parameter_loss({'weight':value*.7,'bias':torch.zeros(2,dtype=torch.float64)},engine.cache_teacher(inputs)[0])
    (.1*value.square().sum()+(kl/.001-1).clamp_min(0).square()).backward()
    torch.testing.assert_close(actual,reference.grad,rtol=1e-12,atol=1e-12)
    assert {'actor_backward_seconds','norm_edit_backward_seconds','gpt_kl_backward_seconds'}<=timings.keys()


def test_launcher_preserves_explicit_batch_and_false_backend_flags():
    from scripts.launch_multistate import command_arguments
    spec=dict(world_size=8,parameters=dict(diagnostic_prefix_states=25,epochs=100,checkpoint_batch=8,
              sequences_per_checkpoint=32,gpt_microbatch=16,compile_gpt=False,wandb=True))
    command=command_arguments(spec,'/work','/bank','/data','/output')
    assert '--nproc_per_node=8' in command and '--compile-gpt' not in command
    assert command[command.index('--gpt-microbatch')+1]=='16'
    assert command[command.index('--epochs')+1]=='100' and '--wandb' in command


def test_cached_targets_skip_teacher_forward_and_component_diagnostics_preserve_gradient():
    inputs=torch.tensor([[.2,.3],[-.4,.1]],dtype=torch.float64)
    results=[]
    for diagnostic in (False,True):
        theta=torch.tensor([[.2,-.4],[.1,.7]],dtype=torch.float64,requires_grad=True)
        output=theta.sin();leaf=output.detach().requires_grad_();engine=LinearEngine(1)
        targets={start:engine.cache_teacher(inputs[start:start+1])[0] for start in range(2)}
        def forbidden(*args,**kwargs):raise AssertionError('cached target path re-executed teacher')
        engine.cache_teacher=forbidden
        measured=mean_kl_and_backward(engine,{'weight':leaf*.7,'bias':torch.zeros(2,dtype=torch.float64)},
            .1*leaf.square().sum(),inputs,kl_budget=1e-8,actor_credit=([output],[leaf]),
            teacher_provider=lambda start,stop:targets[start],diagnostic_parameters=[theta] if diagnostic else None)
        results.append(theta.grad.clone())
        if diagnostic:
            assert measured['norm_component_gradient_norm']>0
            assert measured['kl_component_gradient_norm']>0
    torch.testing.assert_close(*results,rtol=1e-12,atol=1e-12)


def test_fixed_factorized_panels_train_without_teacher_transformer_replay(tmp_path):
    from scripts.train_multistate_chisel import main,select_sequences,stable_seed
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import write_panel
    from rewriter.function_preservation import FunctionalPreservation
    bank=tmp_path/'bank';cfg=_tiny_state_bank(bank)
    data=tmp_path/'fineweb_train_000001.bin';tokens=(np.arange(1000)%16).astype(np.uint16)
    data.write_bytes(struct.pack('<iii',20240520,1,len(tokens))+bytes(1012)+tokens.tobytes())
    reader=StateBankReader(bank);cache=tmp_path/'targets';engine=FunctionalPreservation(GPT(**cfg['model']),microbatch_sequences=1,kl_dtype=torch.float64)
    names=set(engine.parameters);panel_rows=[]
    for state_id in range(1,6):
        record=reader.read(state_id)
        engine.load_teacher_state({n:v for n,v in record.model.items() if n in names},{n:v for n,v in record.model.items() if n not in names})
        pools=state_replay([str(data)],state_id,cfg,pool_sequences=32,seed=400001,epoch=0,split_seed=500001)
        for role in ('train','audit'):
            inputs,ids=select_sequences(pools[role],2,seed=stable_seed('fixed-panel-test',state_id,role))
            panel_rows.append(write_panel(cache,state_id,role,engine,inputs,ids,state_sha256=reader.entries[state_id-1]['sha256'],execution={'test':True}))
    (cache/'manifest.json').write_text(json.dumps({'complete':True,'args':{'states':5},'panels':panel_rows,'purpose':'tiny exact cache integration test'}))
    output=tmp_path/'run'
    main(['--state-bank',str(bank),'--teacher-cache',str(cache),'--output',str(output),'--train-data',str(data),
          '--epochs','2','--checkpoint-batch','2','--sequences-per-checkpoint','2','--gpt-microbatch','1',
          '--replay-pool-sequences','32','--lr','.00002','--kl-budget','.1','--expected-gpt-seed','7',
          '--audit-states','1','--audit-sequences','2','--audit-every-epochs','1','--checkpoint-every','1',
          '--full-training-audit','--gradient-diagnostics','--verify-state-once','--mmap-state',
          '--eager-chisel','--device','cpu','--no-plots'])
    rows=[json.loads(x) for x in (output/'metrics.jsonl').read_text().splitlines()]
    assert rows[-1]['compute']['teacher_forwards']==0
    assert rows[-1]['compute']['cached_teacher_head_forwards']>0
    fixed=[x for x in rows if x['event']=='audit_summary' and x['role']=='fixed_training']
    assert len(fixed)==3 and fixed[0]['objective']==pytest.approx(0,abs=1e-12)
    assert fixed[-1]['state_ids']==[1,2,3,4]
    assert [x['completed_epochs'] for x in fixed]==[0,1,2]
    assert all('checkpoint_gradient_coherence' in x for x in rows if x['event']=='update')
    assert all('kl_component_gradient_norm' in x for x in rows if x['event']=='train_state')


def test_fixed_cached_panel_declares_four_disjoint_gpu_pairs_and_equal_exposure():
    from scripts.launch_cached_panel import arm_specs,training_command
    arms=arm_specs()
    assert sorted(g for arm in arms for g in arm['gpus'])==list(range(8))
    assert {(a['lr'],a['batch']) for a in arms}=={(lr,batch) for lr in (2e-5,2e-4) for batch in (8,20)}
    for arm in arms:
        command=training_command('/w','/b','/c','/d','/o',arm)
        assert command[command.index('--epochs')+1]=='30'
        assert command[command.index('--checkpoint-batch')+1]==str(arm['batch'])
        assert '--compile-gpt' not in command
        assert '--gradient-diagnostics' in command and '--teacher-cache' in command
