"""CPU ownership checks, not a GPU/partial-mask graph qualification."""
import copy
import json

import pytest
import torch
import torch.distributed as dist

from scripts.train_coordinate_chisel import (
    adapt_private, build_actor, build_shared_adam, configure_shared_trainability,
    heldout_readout, run_training, shared_update_policy, validate_recipe,
)
from scripts.qualify_coordinate_fullbank import read_joint
from rewriter.distributed_probe_fork import fork_distributed_checkpoint
from tests.test_coordinate_shared_freeze import identity
from tests.test_coordinate_training_driver import ToyContexts, recipe
from tests.test_distributed_probe_store import assert_tree_equal


def test_policy_defaults_conflicts_and_both_capture_guards():
    cfg=recipe()
    assert shared_update_policy(cfg)=='full'
    cfg['shared_optimizer_step']=False
    assert shared_update_policy(cfg)=='frozen'
    cfg['shared_update_policy']='heads_only'
    with pytest.raises(ValueError,match='conflicts'):validate_recipe(cfg)
    del cfg['shared_optimizer_step']
    validate_recipe(cfg)
    for policy in ('full','frozen','heads_only'):
        cfg['shared_update_policy']=policy;validate_recipe(cfg)
    for policy in ('head_only',None,True,[]):
        cfg['shared_update_policy']=policy
        with pytest.raises(ValueError,match='shared_update_policy'):validate_recipe(cfg)
    cfg['shared_update_policy']='heads_only';cfg['private_inner_steps']=2
    for key in ('actor_backend','private_actor_backend'):
        cfg[key]='eager_actor_graph'
        with pytest.raises(ValueError,match='heads_only requires eager'):validate_recipe(cfg)
        cfg[key]='eager'
    cfg.update(action_family='adaptive_ucv_v1',student_backend='dense')
    cfg['architecture']['fresh_weight_core']=True
    for key in ('actor_backend','private_actor_backend'):
        cfg[key]='adaptive_actor_graph'
        with pytest.raises(ValueError,match='heads_only requires eager'):validate_recipe(cfg)
        cfg[key]='eager'


def test_same_parent_heads_only_keeps_nonzero_body_moments_and_exact_resume(tmp_path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(tmp_path/'rendezvous'),rank=0,world_size=1)
    try:
        cfg=recipe();cfg['private_inner_steps']=2
        cfg['qualification_only']=True;cfg['fitting_subset']={'gpt':[1,2],'cifar10':[1,2]}
        cfg['readout']['heldout']['enabled']=False
        cfg['stopping']['review_after_full_sweeps']=[1]
        meta={'recipe':cfg,'fixture':'heads-only same-parent continuation'}
        parent=tmp_path/'parent'
        run_training(cfg,parent,ToyContexts(),torch.device('cpu'),metadata=meta)
        before=read_joint(parent)
        ref=json.loads((parent/'joint_store/global/CURRENT.json').read_text())
        actor=build_actor(cfg)
        head_ids={id(p) for h in actor.output_heads for p in h.parameters()}
        head_names={n for n,p in actor.named_parameters() if id(p) in head_ids}
        body_names={n for n,p in actor.named_parameters() if id(p) not in head_ids}
        body_ids=before[0]['optimizer']['param_groups'][1]['params']
        assert any(before[0]['optimizer']['state'][i]['exp_avg'].count_nonzero()>0
                   for i in body_ids if i in before[0]['optimizer']['state'])
        def fork(name):
            child=tmp_path/name;child.mkdir()
            c=copy.deepcopy(cfg);c['shared_update_policy']='heads_only'
            c['stopping'].update(max_updates=6,review_after_full_sweeps=[2])
            m=dict(meta,recipe=c);(child/'metadata.json').write_text(json.dumps(m))
            fork_distributed_checkpoint(parent/'joint_store',child/'joint_store',parent_manifest_ref=ref,
                source_run_identity=identity(meta),destination_run_identity=identity(m),
                provenance={'reason':'explicit heads-only test','source_code_id':'test','destination_code_id':'test'})
            assert_tree_equal(read_joint(child),before)
            return child,c,m
        full,c,m=fork('full')
        run_training(c,full,ToyContexts(),torch.device('cpu'),metadata=m,resume=True)
        split,c2,m2=fork('split')
        def interrupt(batch):
            if batch==3:raise RuntimeError('intentional postcommit interruption')
        with pytest.raises(RuntimeError,match='intentional'):
            run_training(c2,split,ToyContexts(),torch.device('cpu'),metadata=m2,resume=True,after_commit=interrupt)
        run_training(c2,split,ToyContexts(),torch.device('cpu'),metadata=m2,resume=True)
        after=read_joint(full);assert_tree_equal(after,read_joint(split))
        for n in body_names:assert torch.equal(before[0]['model'][n],after[0]['model'][n]),n
        assert any(not torch.equal(before[0]['model'][n],after[0]['model'][n]) for n in head_names)
        assert_tree_equal(before[0]['optimizer']['param_groups'],after[0]['optimizer']['param_groups'])
        for i in body_ids:
            assert (i in before[0]['optimizer']['state'])==(i in after[0]['optimizer']['state'])
            if i in before[0]['optimizer']['state']:
                assert_tree_equal(before[0]['optimizer']['state'][i],after[0]['optimizer']['state'][i])
        head_opt_ids=before[0]['optimizer']['param_groups'][0]['params']
        assert any(after[0]['optimizer']['state'][i]['step']>before[0]['optimizer']['state'][i]['step'] for i in head_opt_ids)
        assert any(not torch.equal(p['private'][k],before[2][token]['private'][k]) for token,p in after[2].items()
                   for k in p['private'] if torch.is_tensor(p['private'][k]))
        assert after[1][0]['shared_optimizer_steps']==4
        events=[json.loads(x) for x in (full/'metrics.jsonl').read_text().splitlines()]
        assert all(x['shared_update_policy']=='heads_only' for x in events)
        assert [x['global_batch'] for x in events if x['event']=='joint_update']==[3,4]
        bad=copy.deepcopy(m);bad['recipe']['shared_update_policy']='full'
        with pytest.raises(ValueError,match='metadata'):
            run_training(bad['recipe'],full,ToyContexts(),torch.device('cpu'),metadata=bad,resume=True)
    finally:dist.destroy_process_group()


@pytest.mark.parametrize("fresh_core",[False,True])
def test_adaptive_heads_private_uv_update_and_inner_mask_restoration(fresh_core):
    from tests.test_adaptive_coordinate_chisel import setup
    from types import SimpleNamespace
    snapshot,actor,private=setup()
    if fresh_core:
        from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
        actor=AdaptiveCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=2,latent_tokens=3,fresh_weight_core=True)
        private=actor.make_private(snapshot,state_key=('gpt',0,6),seed=73)
    with torch.no_grad():
        for head in actor.output_heads:
            for p in head.parameters():p.normal_(std=.01)
    config=recipe()['shared_optimizer'];shared=build_shared_adam(actor,config)
    opt=torch.optim.Adam(private.parameters(),lr=1e-3)
    matrices={n:torch.randn(o.rows.shape[0],o.columns.shape[0]) for n,o in snapshot.matrices.items()}
    vectors={n:torch.randn_like(o.weight) for n,o in snapshot.vectors.items()}
    class Contexts:
        def measure(self,context,a,p,config,dual,*,learning,weight):
            from rewriter.adaptive_sensing import query_weight_cores
            kwargs={'weight_cores':query_weight_cores(matrices,p)} if fresh_core else {}
            loss=a(snapshot,p,**kwargs).contraction(matrices,vectors)
            if learning:(loss*weight).backward()
            return dict(local_id=1,mean_kl=0.)
    contexts=Contexts();dual=SimpleNamespace(step=lambda *args,**kwargs:None)
    # Populate moments in every active shared owner before the policy switch.
    contexts.measure({},actor,private,None,dual,learning=True,weight=1.)
    shared.step();shared.zero_grad(set_to_none=True);opt.zero_grad(set_to_none=True)
    configure_shared_trainability(actor,'heads_only')
    if fresh_core:assert not actor.weight_core_input.weight.requires_grad
    mask=[p.requires_grad for p in actor.parameters()]
    before=copy.deepcopy(actor.state_dict());optimizer_before=copy.deepcopy(shared.state_dict())
    u=private.matrix_rows[0].raw.detach().clone();v=private.matrix_columns[0].raw.detach().clone()
    contexts.measure({},actor,private,None,dual,learning=True,weight=1.)
    head_grads=[p.grad.clone() if p.grad is not None else None for p in actor.parameters()]
    adapt_private(contexts,{},actor,private,opt,None,dual,steps=2,clip=1.)
    assert mask==[p.requires_grad for p in actor.parameters()]
    for p,g in zip(actor.parameters(),head_grads):
        if g is None:assert p.grad is None
        else:assert torch.equal(g,p.grad)
    shared.step()
    assert not torch.equal(u,private.matrix_rows[0].raw)
    assert not torch.equal(v,private.matrix_columns[0].raw)
    for n,p in actor.named_parameters():
        if not p.requires_grad:assert torch.equal(before[n],p)
    assert any(not torch.equal(before[n],p) for n,p in actor.named_parameters() if p.requires_grad)
    after=shared.state_dict()
    for i in optimizer_before['param_groups'][1]['params']:
        if i in optimizer_before['state']:assert_tree_equal(optimizer_before['state'][i],after['state'][i])


def test_readout_restores_partial_mask_on_failure(tmp_path):
    # Fail before any distributed call; the finally block must still restore ownership.
    from unittest.mock import patch
    actor=build_actor(recipe());configure_shared_trainability(actor,'heads_only')
    flags=[p.requires_grad for p in actor.parameters()]
    cfg=recipe();cfg['readout']['heldout']['ids']['gpt']=[999]
    with patch('torch.distributed.get_rank',return_value=0),patch('torch.distributed.get_world_size',return_value=1):
        with pytest.raises(ValueError,match='wholly held-out'):
            heldout_readout(cfg,None,actor,{'audit':{'gpt':[],'cifar10':[]}},None,torch.device('cpu'),tmp_path,0,None)
    assert flags==[p.requires_grad for p in actor.parameters()]


def _heads_two_rank_worker(rank,root,rendezvous):
    # Reuse the complete adaptive fresh-core distributed/resume oracle with one
    # explicit recipe change. Its comparisons include all private Adam owners.
    import tests.test_adaptive_training_driver as oracle
    original=oracle.adaptive_recipe
    def configured():
        cfg=original();cfg['shared_update_policy']='heads_only'
        return cfg
    oracle.adaptive_recipe=configured
    oracle.worker(rank,root,rendezvous)


def test_two_rank_adaptive_heads_only_resume_and_shared_ownership(tmp_path):
    import torch.multiprocessing as mp
    from tests.test_adaptive_training_driver import adaptive_recipe
    mp.spawn(_heads_two_rank_worker,args=(tmp_path,tmp_path/'rdzv'),nprocs=2,join=True)
    full=read_joint(tmp_path/'full')
    assert_tree_equal(full,read_joint(tmp_path/'resumed'))
    cfg=adaptive_recipe();cfg['architecture']['fresh_weight_core']=True
    torch.manual_seed(cfg['actor_seed']);actor=build_actor(cfg)
    configure_shared_trainability(actor,'heads_only')
    for name,p in actor.named_parameters():
        if not p.requires_grad:assert torch.equal(p,full[0]['model'][name]),name
    assert any(not torch.equal(p,full[0]['model'][name]) for name,p in actor.named_parameters() if p.requires_grad)
    body_ids=full[0]['optimizer']['param_groups'][1]['params']
    assert all(i not in full[0]['optimizer']['state'] for i in body_ids)
    rows=[json.loads(line) for line in (tmp_path/'full/metrics.jsonl').read_text().splitlines()]
    for event in (r for r in rows if r['event']=='joint_update'):
        assert event['shared_update_policy']=='heads_only'
        for domain in ('gpt','cifar10'):
            assert sum(r['state_weight'] for r in event['rows'] if r['domain']==domain)==pytest.approx(.5)


def test_frozen_gradient_contamination_fails_before_shared_adam(tmp_path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(tmp_path/'rdzv'),rank=0,world_size=1)
    class Contaminated(ToyContexts):
        def measure(self,context,actor,private,*args,learning,**kwargs):
            row=super().measure(context,actor,private,*args,learning=learning,**kwargs)
            if learning and any(p.requires_grad for p in actor.parameters()):
                frozen=next(p for p in actor.parameters() if not p.requires_grad)
                frozen.grad=torch.ones_like(frozen)
            return row
    try:
        cfg=recipe();cfg['shared_update_policy']='heads_only'
        with pytest.raises(RuntimeError,match='frozen shared parameter unexpectedly'):
            run_training(cfg,tmp_path/'run',Contaminated(),torch.device('cpu'),metadata={'recipe':cfg})
        # The joint update cannot be published after the ownership violation.
        events=[json.loads(l) for l in (tmp_path/'run/metrics.jsonl').read_text().splitlines()]
        assert not any(r['event']=='joint_update' for r in events)
    finally:dist.destroy_process_group()
