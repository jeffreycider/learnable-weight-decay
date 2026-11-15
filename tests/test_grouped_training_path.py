"""Exercise opted-in grouped heads through finite measurement and full driver state.

Toy driver runs establish staging/Adam/resume correctness, not science or GPU
qualification. Separate tiny real GPT/CNN checks exercise FixedContexts.measure.
"""
from dataclasses import replace
import pytest
import torch
import torch.distributed as dist
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
from scripts.train_coordinate_chisel import run_training,build_actor,build_shared_adam,measure_context
from scripts.qualify_coordinate_chisel import FixedContexts,proposal_builder_for_family
from scripts.qualify_coordinate_fullbank import read_joint
from tests.test_grouped_actor_recipe import grouped_recipe
from tests.test_coordinate_training_driver import ToyContexts
from tests.test_distributed_probe_store import assert_tree_equal


class GroupedToyContexts(ToyContexts):
    action_family='adaptive_ucv_v1'

    def load(self,*args,**kwargs):
        c=super().load(*args,**kwargs);s=c['snapshot'];gpt=c['domain']=='gpt'
        names={'first':'blocks.0.mlp.fc.weight' if gpt else 'layers.1.conv1.weight',
               'second':'blocks.0.mlp.proj.weight','gain':'proj.bias' if gpt else 'whiten.bias'}
        s=replace(s,matrices={names[n]:o for n,o in s.matrices.items()},vectors={names[n]:o for n,o in s.vectors.items()})
        c['snapshot']=s;c['primary']=names['first']
        c['dense_weights']={n:torch.full((o.rows.shape[0],o.columns.shape[0]),.05) for n,o in s.matrices.items()}
        c['dense_weights'][c['primary']]=c['weight']
        return c

    def measure(self,c,actor,private,config,dual,*,learning,weight=1.):
        cores=query_weight_cores(c['dense_weights'],private,embedding_names=())
        action=actor(c['snapshot'],private,weight_cores=cores)
        logits=c['inputs']@(c['weight']+action.matrices[c['primary']].dense().float()).T
        teacher=(c['inputs']@c['weight'].T).log_softmax(-1).detach()
        kl=(teacher.exp()*(teacher-logits.log_softmax(-1))).sum(-1).mean()
        dtype=next(iter(action.matrices.values())).U.dtype
        norm=.01*action.norm_change({n:t.to(dtype) for n,t in c['dense_weights'].items()},
                                   {n:t.to(dtype) for n,t in c['snapshot'].vector_weights.items()})
        loss=kl+norm
        if learning:(loss*weight).backward()
        self.compute['toy_forwards']=self.compute.get('toy_forwards',0)+1
        self.compute['toy_backwards']=self.compute.get('toy_backwards',0)+int(learning)
        return dict(domain=c['domain'],local_id=c['local_id'],context_ids=c['ids'],mean_kl=float(kl.detach()),
            mean_group_l2_percentage_reduction=-float(norm.detach()),strict_topk_changes=int((logits.argmax(-1)!=teacher.argmax(-1)).sum()),objective=float(loss.detach()))


@pytest.mark.parametrize('policy',['full','heads_only'])
def test_grouped_driver_inner_steps_real_adam_and_exact_resume(tmp_path,policy):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(tmp_path/'rendezvous'),rank=0,world_size=1)
    try:
        cfg=grouped_recipe();cfg['architecture']['fresh_weight_core']=True
        cfg.update(shared_update_policy=policy,private_inner_steps=1,qualification_only=True,
                   fitting_subset={'gpt':[1,2],'cifar10':[1,2]})
        cfg['readout']['heldout']['enabled']=policy=='full'
        cfg['stopping'].update(max_updates=3,full_sweep_every=2)
        meta={'recipe':cfg,'scope':'typed-layout toy finite driver test'}
        torch.manual_seed(cfg['actor_seed']);initial=build_actor(cfg).state_dict()
        full=run_training(cfg,tmp_path/'full',GroupedToyContexts(),torch.device('cpu'),metadata=meta)
        def interrupt(step):
            if step==1:raise RuntimeError('intentional committed interruption')
        with pytest.raises(RuntimeError,match='intentional'):
            run_training(cfg,tmp_path/'split',GroupedToyContexts(),torch.device('cpu'),metadata=meta,after_commit=interrupt)
        resumed=run_training(cfg,tmp_path/'split',GroupedToyContexts(),torch.device('cpu'),metadata=meta,resume=True)
        assert_tree_equal(full,resumed)
        complete=read_joint(tmp_path/'full');assert_tree_equal(complete,read_joint(tmp_path/'split'))
        model=complete[0]['model'];optimizer=complete[0]['optimizer']
        for role in ('mlp__in__weight','conv__entry__weight'):
            key='matrix_output.'+role+'.weight';assert not torch.equal(initial[key],model[key])
        body=[n for n in initial if n!='_extra_state' and not n.startswith(('matrix_output.','vector_output.'))]
        if policy=='heads_only':
            assert all(torch.equal(initial[n],model[n]) for n in body)
            assert not any(i in optimizer['state'] for i in optimizer['param_groups'][1]['params'])
        else:assert any(not torch.equal(initial[n],model[n]) for n in body)
        assert all(int(v['step'])==3 for v in optimizer['state'].values())
        assert all(payload['optimizer']['state'] for payload in complete[2].values())
        assert any(t.get('step',torch.tensor(0))>1 for payload in complete[2].values() for t in payload['optimizer']['state'].values())
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('domain',['gpt','cifar10'])
def test_grouped_real_finite_measurement_backward_and_optimizer(domain):
    from tests.test_adaptive_preservation_actor import actual_actor_case
    from tests.test_cifar_adapter import fixture
    from rewriter.cifar_adapter import observe_cifar_terminal
    from rewriter.grouped_norm import GroupedNormObjective
    from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
    torch.set_num_threads(1)
    if domain=='gpt':
        engine,_,_,snapshot,grouped,inputs=actual_actor_case()
        # FixedContexts expects inputs with a sequence batch dimension.
        cached=[engine.cache_teacher(x[None],data_role='synthetic')[0] for x in inputs]
        teacher=lambda start,end:cached[start]
        trajectory_dim=12;target=None
    else:
        _,engine,inputs,target=fixture();engine.microbatch_sequences=2
        snapshot,_=observe_cifar_terminal(engine,inputs,target,momentum={},anchor_seed=87,rank=2,trajectory_images=4)
        grouped=GroupedNormObjective(engine.parameters,groups=engine.groups,zero_reference_policy='freeze')
        logits=engine._forward(engine.parameters,inputs).detach()
        teacher=lambda start,end:engine.teacher_batch(inputs[start:end],logits[start:end],range(start,end))
        trajectory_dim=100
    actor=GroupedCoordinateChisel(rank=2,trajectory_dim=trajectory_dim,model_dim=128,num_layers=1,latent_tokens=2,fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
    with torch.no_grad():
        for bank in actor.output_heads:
            for head in bank.values():head.weight.normal_(std=.001)
        actor.blocks[0].attn.proj.weight.normal_(std=.001)
    private=actor.make_private(snapshot,state_key=(domain,1),seed=19)
    contexts=FixedContexts.__new__(FixedContexts);contexts.action_family='adaptive_ucv_v1';contexts.compute={}
    contexts.proposal_builder=proposal_builder_for_family(contexts.action_family)
    c=dict(domain=domain,local_id=1,engine=engine,snapshot=snapshot,grouped=grouped,inputs=inputs,target=target,ids=list(range(len(inputs))),norm_squared=None,teacher=teacher)
    config=TokenwiseConfig(mean_tolerance=1e-3,tail_tolerance=1e-3,scale=1e-3,rho=.01,ranking_weight=0.,ranking_k=3,ranking_tie_policy='exact_teacher_ties')
    dual=TokenwiseDual([1],config=config,dual_lr=.01)
    before={n:p.clone() for n,p in engine.parameters.items()}
    row=measure_context(contexts,c,actor,private,config,dual,learning=True)
    assert row['action_family']=='adaptive_ucv_v1'
    assert actor.weight_core_input.weight.grad.norm()>0
    assert any(p.grad is not None and p.grad.norm()>0 for p in private.parameters())
    shared=build_shared_adam(actor,grouped_recipe()['shared_optimizer'])
    shared.step();torch.optim.Adam(private.parameters(),lr=1e-4).step()
    assert shared.state
    assert all(torch.equal(p,before[n]) and p.grad is None for n,p in engine.parameters.items())
