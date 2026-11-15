"""Narrow CPU driver integration: two layouts/domains and real private Adam slots.

Toy finite logits qualify routing/accounting/resume, not NanoGPT/CIFAR numerics.
"""
import copy
import dataclasses
import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from scripts.train_coordinate_chisel import (run_training,OwnerBalancedSchedule,StateSpec,
    fitting_pass,validate_recipe,stable_seed)
from scripts.qualify_coordinate_chisel import validate_context_access
from rewriter.private_probe_store import PrivateIdentity
from rewriter.coordinate_chisel import CheckpointProbes
from tests.test_controller import make_snapshot
from tests.test_distributed_probe_store import assert_tree_equal


def recipe():
    objective=dict(mean_tolerance=1e-3,tail_tolerance=1e-2,scale=1e-3,rho=.1,ranking_weight=0.,
                   ranking_k=100,ranking_tie_policy='exact_teacher_ties')
    return dict(schema='coordinate_chisel_fullbank_v1',domain_weights={'gpt':.5,'cifar10':.5},actor_backend='eager',
        architecture=dict(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2),actor_seed=401,private_seed=402,sampler_seed=403,
        shared_optimizer=dict(head_lr=1e-4,representation_lr=1e-4,head_eps=1e-8,representation_eps=1e-8,betas=[.9,.99],clip=1.,clip_scope='separate'),
        private_optimizer=dict(lr=1e-3,eps=1e-8,betas=[.9,.99],clip=1.),
        panels=dict(gpt_sequences=16,cifar_images=128,mode='fixed',seed=404),
        objectives={d:copy.deepcopy(objective) for d in ('gpt','cifar10')},dual_lr=.01,wandb=False,
        stopping=dict(max_updates=4,full_sweep_every=2,consecutive_full_sweeps=1,stop_when_all_fit=False,
                      fitting_criteria=[dict(metric='mean_kl',operator='le',value=100.)]),
        readout=dict(gpt_sequences=16,cifar_images=128,heldout=dict(enabled=True,ids={'gpt':[17],'cifar10':[9]},
                    support_steps=1,query_gpt_sequences=16)))


class ToyReader:
    def __init__(self,domain,ntrain,naudit):
        self.manifest={'split':{'train':list(range(1,ntrain+1)),'audit':list(range(ntrain+1,ntrain+naudit+1))}}
        self.entries=[dict(completed_steps=i,local_id=i,seed=i-1,sha256=hashlib.sha256(f'{domain}:{i}'.encode()).hexdigest()) for i in range(1,ntrain+naudit+1)]
    def describe_state(self,uid):return dict(bank_seed=0,gpt_completed_steps=uid)


class ToyContexts:
    def __init__(self):
        self.catalog=SimpleNamespace(readers={'gpt':ToyReader('gpt',16,4),'cifar10':ToyReader('cifar10',8,2)})
        self.compute={};self.cifar_images=128
    def load(self,domain,local_id,*,role='train',block_offset=0,purpose='fitting',shared_actor=None,gpt_sequences=None):
        validate_context_access(self.catalog.readers[domain].manifest['split'],local_id,role=role,purpose=purpose,shared_actor=shared_actor)
        snapshot=make_snapshot()
        if domain=='cifar10':snapshot=dataclasses.replace(snapshot,matrices={'first':snapshot.matrices['first']})
        matrices={name:dataclasses.replace(obs,rows=obs.rows*(1+.001*local_id),columns=obs.columns*(1+.001*local_id)) for name,obs in snapshot.matrices.items()}
        snapshot=dataclasses.replace(snapshot,matrices=matrices)
        n=(gpt_sequences or 16) if domain=='gpt' else self.cifar_images
        generator=torch.Generator().manual_seed(stable_seed(domain,local_id,role,block_offset))
        x=torch.randn(n,5,generator=generator);weight=torch.randn(7,5,generator=generator)*.05
        return dict(domain=domain,local_id=local_id,snapshot=snapshot,inputs=x,weight=weight,
                    ids=list(range((100000 if role=='audit' else 0)+local_id*1000,(100000 if role=='audit' else 0)+local_id*1000+n)))
    def measure(self,context,actor,private,config,dual,*,learning,weight=1.):
        action=actor(context['snapshot'],private)
        logits=context['inputs']@(context['weight']+action.matrices['first'].dense()).T
        teacher=(context['inputs']@context['weight'].T).log_softmax(-1).detach()
        kl=(teacher.exp()*(teacher-logits.log_softmax(-1))).sum(-1).mean()
        norm=.01*action.norm_change(context['snapshot'].weights,context['snapshot'].vector_weights)
        loss=kl+norm
        if learning:(loss*weight).backward()
        self.compute['toy_forwards']=self.compute.get('toy_forwards',0)+1
        self.compute['toy_backwards']=self.compute.get('toy_backwards',0)+int(learning)
        return dict(domain=context['domain'],local_id=context['local_id'],context_ids=context['ids'],mean_kl=float(kl.detach()),
            mean_group_l2_percentage_reduction=-float(norm.detach()),strict_topk_changes=int((logits.argmax(-1)!=teacher.argmax(-1)).sum()),
            objective=float(loss.detach()))


def worker(rank,root,rendezvous,private_inner_steps=0,feature_stage=False,review_stage=False):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(rendezvous),rank=rank,world_size=2,
                            timeout=datetime.timedelta(seconds=120))
    try:
        config=recipe()
        if private_inner_steps:config['private_inner_steps']=private_inner_steps
        if feature_stage:
            from rewriter.coordinate_chisel import CoordinateChisel
            from rewriter.multibank import file_sha
            config['architecture']['frame_mode']='tied_orthonormal'
            config['private_optimizer']['tick_schedule']=dict(cold_steps=2,warm_lr=1e-4)
            config['private_inner_panels']={'gpt':dict(source_size=32,inner_size=16),'cifar10':dict(source_size=256,inner_size=128)}
            path=Path(root)/'imported-shared.pt'
            if rank==0:
                torch.manual_seed(919);model=CoordinateChisel(**config['architecture'])
                with torch.no_grad():model.matrix_output.weight.normal_(0,.002)
                torch.save(dict(actor=model.state_dict(),update=1024),path)
            dist.barrier()
            config['shared_initialization']=dict(mode='checkpoint_actor_only',path=str(path),sha256=file_sha(path),update=1024)
        metadata=dict(recipe=config,test_fixture='tiny finite logits;not GPT scientific qualification')
        full=run_training(config,Path(root)/'full',ToyContexts(),torch.device('cpu'),metadata=metadata)
        def interrupt(update):
            if update==2:raise RuntimeError('simulated interruption after global commit before full sweep')
        if review_stage:
            config=copy.deepcopy(config)
            config['stopping']['review_after_full_sweeps']=[1]
            metadata=dict(metadata,recipe=config)
            reviewed=run_training(config,Path(root)/'resumed',ToyContexts(),torch.device('cpu'),metadata=metadata)
            assert reviewed['update']==2 and reviewed['reason']=='review_boundary'
            if rank==0:
                events=[json.loads(x) for x in (Path(root)/'resumed/metrics.jsonl').read_text().splitlines()]
                assert not any(x['event']=='heldout_readout' for x in events)
        else:
            with pytest.raises(RuntimeError,match='simulated interruption'):
                run_training(config,Path(root)/'resumed',ToyContexts(),torch.device('cpu'),metadata=metadata,after_commit=interrupt)
        resumed=run_training(config,Path(root)/'resumed',ToyContexts(),torch.device('cpu'),metadata=metadata,resume=True)
        assert_tree_equal(full['actor'],resumed['actor'])
        assert full['visits']==resumed['visits']
        assert full['reason']==resumed['reason']=='declared_budget_review_not_convergence'
        assert full['update']==resumed['update']==4
        (Path(root)/f'rank{rank}-complete').write_text('pass\n')
    finally:dist.destroy_process_group()


def test_two_rank_driver_resume_two_layouts_and_heldout_isolation(tmp_path):
    mp.spawn(worker,args=(tmp_path,tmp_path/'rendezvous'),nprocs=2,join=True)
    assert all((tmp_path/f'rank{r}-complete').exists() for r in range(2))
    from scripts.qualify_coordinate_fullbank import read_joint
    a,b=read_joint(tmp_path/'full'),read_joint(tmp_path/'resumed')
    assert_tree_equal(a,b)
    for output in ('full','resumed'):
        rows=[json.loads(line) for line in (tmp_path/output/'metrics.jsonl').read_text().splitlines()]
        updates=[r for r in rows if r['event']=='joint_update']
        assert [r['update'] for r in updates]==[1,2,3,4]
        for update in updates:
            assert len(update['commit_seconds_by_rank'])==2
            assert all(r['private_snapshot_bytes']>0 and r['private_stage_seconds']>=0 for r in update['rows'])
            for domain in ('gpt','cifar10'):
                assert sum(r['state_weight'] for r in update['rows'] if r['domain']==domain)==pytest.approx(.5)
        assert [r['update'] for r in rows if r['event']=='full_fitting_sweep']==[2,4]
        heldout=[r for r in rows if r['event']=='heldout_readout']
        assert len(heldout)==1 and heldout[0]['shared_unchanged'] and not heldout[0]['used_for_stopping']
        assert all(r['local_id']<=16 if r['domain']=='gpt' else r['local_id']<=8 for update in updates for r in update['rows'])
        assert rows[-1]['reason']=='declared_budget_review_not_convergence'


def test_private_inner_steps_resume_and_count_every_update(tmp_path):
    mp.spawn(worker,args=(tmp_path,tmp_path/'rendezvous',2),nprocs=2,join=True)
    from scripts.qualify_coordinate_fullbank import read_joint
    assert_tree_equal(read_joint(tmp_path/'full'),read_joint(tmp_path/'resumed'))
    events=[json.loads(line) for line in (tmp_path/'full'/'metrics.jsonl').read_text().splitlines()]
    updates=[r for r in events if r['event']=='joint_update']
    for event in updates:
        for row in event['rows']:
            assert row['private_updates_this_visit']==3
            assert len(row['private_inner_rows'])==2
            assert row['private_visit']%3==0
            assert all(r['context_ids']==row['context_ids'] for r in row['private_inner_rows'])
    assert all(r['total_private_visits']==24 for r in events[-1]['per_rank'])


def test_owner_weighting_corrects_unequal_pools_and_records_repeats():
    snapshot=make_snapshot();private=CheckpointProbes(snapshot,model_dim=128,latent_tokens=2,seed=1,state_key=('gpt',0,1))
    layout=PrivateIdentity.from_private(private,state_sha256='a'*64).layout_sha256
    specs={d:[StateSpec(d,i,PrivateIdentity((d,0,i),hashlib.sha256(f'{d}{i}'.encode()).hexdigest(),layout)) for i in range(1,n+1)] for d,n in [('gpt',17),('cifar10',9)]}
    schedules=[OwnerBalancedSchedule(specs,rank=r,world_size=2,seed=13) for r in range(2)]
    assert len(schedules[0].pools['gpt'])!=len(schedules[1].pools['gpt'])
    for domain in ('gpt','cifar10'):
        for batch in range(30):
            rows=[s.visit(batch,domain)[1] for s in schedules]
            assert sum(r['state_weight'] for r in rows)==pytest.approx(.5)
        for schedule in schedules:
            count=len(schedule.pools[domain]);visited=[schedule.visit(i,domain)[0].local_id for i in range(count)]
            assert len(set(visited))==count
            assert schedule.visit(count,domain)[1]['owner_cycle']==1


def test_full_sweep_requires_uniform_complete_fitting_coverage():
    specs={'gpt':[SimpleNamespace(domain='gpt',local_id=1),SimpleNamespace(domain='gpt',local_id=2)],
           'cifar10':[SimpleNamespace(domain='cifar10',local_id=1)]}
    rows=[dict(domain=s.domain,local_id=s.local_id,mean_kl=0.) for values in specs.values() for s in values]
    rules=[dict(metric='mean_kl',operator='le',value=1e-6)]
    assert fitting_pass(rows,specs,rules)
    with pytest.raises(ValueError,match='coverage'):fitting_pass(rows[:-1],specs,rules)
    with pytest.raises(ValueError,match='coverage'):fitting_pass(rows+[rows[0]],specs,rules)
    rows[0]['mean_kl']=2e-6
    assert not fitting_pass(rows,specs,rules)


def test_stopping_and_objectives_must_be_explicit():
    config=recipe();validate_recipe(config)
    config['stopping']['stop_when_all_fit']=True;config['stopping']['fitting_criteria']=[]
    with pytest.raises(ValueError,match='cannot be inferred'):validate_recipe(config)
    config=recipe();config['actor_backend']='unqualified_compile'
    with pytest.raises(ValueError,match='unqualified'):validate_recipe(config)
    config=recipe();config['fitting_subset']={'gpt':[1,2],'cifar10':[1,2]}
    with pytest.raises(ValueError,match='qualification-only'):validate_recipe(config)
    config['qualification_only']=True;validate_recipe(config)
    config['private_snapshot_backend']='pinned_d2h_v1';validate_recipe(config)
    config['private_snapshot_backend']='owned_cpu_v1'
    with pytest.raises(ValueError,match='snapshot backend'):validate_recipe(config)


def test_tied_import_inner_disjoint_panels_tick_lr_exact_distributed_resume(tmp_path):
    mp.spawn(worker,args=(tmp_path,tmp_path/'rendezvous',2,True),nprocs=2,join=True)
    from scripts.qualify_coordinate_fullbank import read_joint
    assert_tree_equal(read_joint(tmp_path/'full'),read_joint(tmp_path/'resumed'))
    events=[json.loads(x) for x in (tmp_path/'full/metrics.jsonl').read_text().splitlines()]
    for event in events:
        if event['event']!='joint_update':continue
        for row in event['rows']:
            assert row['private_lr']==1e-4
            for inner in row['private_inner_rows']:
                assert set(inner['context_ids']).isdisjoint(row['context_ids'])
                assert inner['private_lr']==(1e-3 if inner['private_optimizer_step']<=2 else 1e-4)


def test_full_sweep_review_resume_matches_uninterrupted_tied_inner_training(tmp_path):
    mp.spawn(worker,args=(tmp_path,tmp_path/'rendezvous',2,True,True),nprocs=2,join=True)
    from scripts.qualify_coordinate_fullbank import read_joint
    assert_tree_equal(read_joint(tmp_path/'full'),read_joint(tmp_path/'resumed'))
    events=[json.loads(x) for x in (tmp_path/'resumed/metrics.jsonl').read_text().splitlines()]
    assert [x['update'] for x in events if x['event']=='full_fitting_sweep']==[2,4]
    assert [x['update'] for x in events if x['event']=='heldout_readout']==[4]
    assert [x['update'] for x in events if x['event']=='joint_update']==[1,2,3,4]


def test_review_boundaries_must_precede_budget_and_be_distinct():
    for boundaries in ([0],[2],[1,1],[-1],[True]):
        config=recipe();config['stopping']['review_after_full_sweeps']=boundaries
        with pytest.raises(ValueError,match='review'):validate_recipe(config)
