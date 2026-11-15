"""Actual trainer lifecycle with FP64 factors, FP32 finite toy model, 8 inner ticks."""
import copy,datetime,hashlib,json
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.train_coordinate_chisel import build_actor,validate_recipe,run_training
from tests.test_adaptive_training_driver import adaptive_recipe,AdaptiveToyContexts
from tests.test_coordinate_training_driver import recipe
from tests.test_distributed_probe_store import assert_tree_equal
from scripts.qualify_coordinate_fullbank import read_joint
from rewriter.adaptive_sensing import query_weight_cores


class FP64ToyContexts(AdaptiveToyContexts):
    def measure(self,context,actor,private,config,dual,*,learning,weight=1.):
        assert actor.matrix_output.weight.dtype == private.matrix_rows[0].raw.dtype == torch.float64
        assert all(p.dtype == torch.float32 for p in actor.blocks.parameters())
        action=actor(context['snapshot'],private,
            weight_cores=query_weight_cores(context['dense_weights'],private,embedding_names=()))
        base=context['weight'];assert base.dtype==torch.float32
        replacement=(base.double()+action.matrices['first'].dense()).float()
        logits=context['inputs']@replacement.T
        teacher=(context['inputs']@base.T).log_softmax(-1).detach()
        kl=(teacher.exp()*(teacher-logits.log_softmax(-1))).sum(-1).mean()
        matrices={n:w.double() for n,w in context['dense_weights'].items()}
        vectors={n:w.double() for n,w in context['snapshot'].vector_weights.items()}
        norm=.01*action.norm_change(matrices,vectors)
        loss=kl+norm
        if learning:(loss*weight).backward()
        self.compute['toy_forwards']=self.compute.get('toy_forwards',0)+1
        self.compute['toy_backwards']=self.compute.get('toy_backwards',0)+int(learning)
        return dict(domain=context['domain'],local_id=context['local_id'],context_ids=context['ids'],
            mean_kl=float(kl.detach()),mean_group_l2_percentage_reduction=-float(norm.detach()),
            strict_topk_changes=int((logits.argmax(-1)!=teacher.argmax(-1)).sum()),objective=float(loss.detach()))


def test_recipe_precision_explicit_and_family_scoped():
    cfg=adaptive_recipe();cfg['architecture']['factor_precision']='fp64';validate_recipe(cfg)
    actor=build_actor(cfg)
    assert actor.configuration()['factor_precision']=='fp64'
    assert actor.matrix_output.weight.dtype==torch.float64
    assert all(p.dtype==torch.float32 for p in actor.blocks.parameters())
    for value in ('automatic',True,None,64):
        broken=copy.deepcopy(cfg);broken['architecture']['factor_precision']=value
        with pytest.raises(ValueError,match='factor_precision'):validate_recipe(broken)
    cfg=recipe();cfg['architecture']['factor_precision']='fp64'
    with pytest.raises(ValueError,match='factor_precision'):validate_recipe(cfg)


def _worker(rank,root,rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(rendezvous),rank=rank,world_size=2,timeout=datetime.timedelta(seconds=120))
    try:
        cfg=adaptive_recipe();cfg['architecture'].update(fresh_weight_core=True,factor_precision='fp64')
        cfg['private_inner_steps']=8
        cfg['private_optimizer']['tick_schedule']=dict(cold_steps=8,warm_lr=1e-4)
        metadata=dict(recipe=cfg,scope='CPU numerical lifecycle only; no scientific GPT result')
        full=run_training(cfg,Path(root)/'full',FP64ToyContexts(),torch.device('cpu'),metadata=metadata)
        def interrupt(update):
            if update==2:raise RuntimeError('saved update2')
        with pytest.raises(RuntimeError,match='saved update2'):
            run_training(cfg,Path(root)/'split',FP64ToyContexts(),torch.device('cpu'),metadata=metadata,after_commit=interrupt)
        resumed=run_training(cfg,Path(root)/'split',FP64ToyContexts(),torch.device('cpu'),metadata=metadata,resume=True)
        assert_tree_equal(full,resumed)
        dist.barrier()
        if rank==0:
            a,b=read_joint(Path(root)/'full'),read_joint(Path(root)/'split')
            assert_tree_equal(a,b)  # all private Adam/visits and shared Adam, per-rank RNG/cursors/duals
            assert a[2]
            (Path(root)/'qualified.json').write_text(json.dumps(dict(exact=True,world_size=2,updates=4,private_inner_steps=8)))
    finally:dist.destroy_process_group()


def test_two_rank_fp64_eight_inner_updates_exact_resume(tmp_path):
    mp.spawn(_worker,args=(tmp_path,tmp_path/'rdzv'),nprocs=2,join=True)
    assert json.loads((tmp_path/'qualified.json').read_text())['exact']


def test_telemetry_defers_every_cuda_call_until_trainer(monkeypatch,tmp_path):
    import sys
    import scripts.qualify_adaptive_precision_runtime as wrapper
    r=adaptive_recipe();r.update(qualification_only=True);r['architecture']['factor_precision']='fp64'
    path=tmp_path/'recipe.json';path.write_text(json.dumps(r));out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['wrapper','--recipe',str(path),'--output',str(out)])
    ran=[]
    def trainer(*args,**kwargs):
        assert args==('scripts.train_coordinate_chisel',)
        ran.append(True)
    monkeypatch.setattr(wrapper.runpy,'run_module',trainer)
    def forbidden(*args,**kwargs):raise AssertionError('CUDA initialized before deterministic policy')
    monkeypatch.setattr(torch.cuda,'set_device',forbidden)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',forbidden)
    def initialized():assert ran;return True
    monkeypatch.setattr(torch.cuda,'is_initialized',initialized)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda _:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda _:123)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda _:456)
    wrapper.main()
    data=json.loads((out/'precision-runtime-fresh-rank0.json').read_text())
    assert data['status']=='completed' and data['peak_allocated_bytes']==123
