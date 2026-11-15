"""Same-parent private-only continuation preserves shared Adam and exact resume."""
import copy,hashlib,json
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
from scripts.train_coordinate_chisel import run_training,validate_recipe
from scripts.qualify_coordinate_fullbank import read_joint
from rewriter.distributed_probe_fork import fork_distributed_checkpoint
from tests.test_coordinate_training_driver import recipe,ToyContexts
from tests.test_distributed_probe_store import assert_tree_equal


def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def test_freeze_is_explicit_boolean_and_rejects_uncertified_graph():
    cfg=recipe();validate_recipe(cfg)
    for value in (0,1,'false',None):
        cfg['shared_optimizer_step']=value
        with pytest.raises(ValueError,match='shared_optimizer_step'):validate_recipe(cfg)
    cfg['shared_optimizer_step']=False;validate_recipe(cfg)
    cfg['actor_backend']='eager_actor_graph'
    with pytest.raises(ValueError,match='frozen shared'):validate_recipe(cfg)


def test_same_parent_frozen_shared_adam_and_private_exact_stop_resume(tmp_path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(tmp_path/'rendezvous'),rank=0,world_size=1)
    try:
        cfg=recipe();cfg['private_inner_steps']=2
        cfg['qualification_only']=True;cfg['fitting_subset']={'gpt':[1,2],'cifar10':[1,2]}
        cfg['readout']['heldout']['enabled']=False
        cfg['stopping']['review_after_full_sweeps']=[1]
        meta={'recipe':cfg,'fixture':'same-parent frozen shared lifecycle'}
        parent=tmp_path/'parent';run_training(cfg,parent,ToyContexts(),torch.device('cpu'),metadata=meta)
        before=read_joint(parent);assert before[1][0]['next_global_batch']==2
        assert before[1][0]['shared_optimizer_steps']==2
        parent_ref=json.loads((parent/'joint_store/global/CURRENT.json').read_text())
        def fork(name,freeze):
            child=tmp_path/name;child.mkdir()
            c=copy.deepcopy(cfg);c['shared_optimizer_step']=not freeze
            c['stopping'].update(max_updates=6,review_after_full_sweeps=[2])
            m=dict(meta,recipe=c);(child/'metadata.json').write_text(json.dumps(m))
            fork_distributed_checkpoint(parent/'joint_store',child/'joint_store',parent_manifest_ref=parent_ref,
                source_run_identity=identity(meta),destination_run_identity=identity(m),
                provenance={'reason':'test explicit freeze','source_code_id':'test','destination_code_id':'test'})
            return child,c,m
        full,c,m=fork('frozen',True)
        run_training(c,full,ToyContexts(),torch.device('cpu'),metadata=m,resume=True)
        split,c2,m2=fork('split',True)
        def interrupt(batch):
            if batch==3:raise RuntimeError('intentional postcommit interruption')
        with pytest.raises(RuntimeError,match='intentional'):
            run_training(c2,split,ToyContexts(),torch.device('cpu'),metadata=m2,resume=True,after_commit=interrupt)
        run_training(c2,split,ToyContexts(),torch.device('cpu'),metadata=m2,resume=True)
        after=read_joint(full);assert_tree_equal(after,read_joint(split))
        assert_tree_equal(before[0],after[0])
        assert after[1][0]['next_global_batch']==4 and after[1][0]['shared_optimizer_steps']==2
        assert any(not torch.equal(p['private'][k],before[2][token]['private'][k]) for token,p in after[2].items()
                   for k in p['private'] if torch.is_tensor(p['private'][k]))
        events=[json.loads(x) for x in (full/'metrics.jsonl').read_text().splitlines()]
        batches=[x for x in events if x['event']=='private_only_batch']
        assert [x['global_batch'] for x in batches]==[3,4]
        assert all(x['shared_optimizer_steps']==2 and not x['shared_optimizer_step_performed'] for x in batches)
        joint,j,jm=fork('joint',False)
        run_training(j,joint,ToyContexts(),torch.device('cpu'),metadata=jm,resume=True)
        final=read_joint(joint);assert final[1][0]['shared_optimizer_steps']==4
        assert any(not torch.equal(v,final[0]['model'][k]) for k,v in before[0]['model'].items() if torch.is_tensor(v))
    finally:dist.destroy_process_group()


def test_frozen_adaptive_capture_requires_explicit_family_fresh_core_and_fp32():
    cfg=recipe();cfg.update(action_family='adaptive_ucv_v1',student_backend='dense',actor_backend='adaptive_actor_graph',private_actor_backend='adaptive_actor_graph',private_inner_steps=8,shared_optimizer_step=False)
    cfg['architecture'].update(frame_mode='tied_orthonormal',fresh_weight_core=True)
    validate_recipe(cfg)
    cfg['architecture']['factor_precision']='fp64'
    with pytest.raises(ValueError,match='frozen shared'):validate_recipe(cfg)
