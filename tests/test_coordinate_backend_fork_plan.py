"""CPU metadata/fork/trainer-resume integration; CUDA math is separately gated."""
import copy,json,hashlib
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from scripts.prepare_coordinate_backend_fork import child_recipe,identity,finalize_closed_parent
from tests.test_coordinate_training_driver import recipe,ToyContexts
from tests.test_distributed_probe_store import initialize,assert_tree_equal
from scripts.train_coordinate_chisel import run_training
from rewriter.distributed_probe_fork import fork_distributed_checkpoint
from rewriter.private_probe_store import PrivateProbeStore


def test_only_two_declared_recipe_changes_and_running_parent_never_reads_current(tmp_path):
    parent=recipe();parent['stopping']['review_after_full_sweeps']=[1]
    original=copy.deepcopy(parent);child=child_recipe(parent)
    assert parent==original
    assert child.pop('private_actor_backend')=='eager_actor_graph'
    assert child['stopping'].pop('review_after_full_sweeps')==[1,2,3]
    original['stopping'].pop('review_after_full_sweeps');assert child==original
    (tmp_path/'status.json').write_text(json.dumps(dict(status='running',exit_code=None)))
    with pytest.raises(ValueError,match='not closed'):
        finalize_closed_parent(tmp_path,tmp_path/'child',parent_metadata={},child_metadata={})
    assert not (tmp_path/'results').exists()


def worker(rank,root,rendezvous):
    initialize(rank,2,rendezvous)
    try:
        # Transparent CPU dispatch exercises the real new callsite/identity and
        # freeze ownership. It does not claim CPU emulates CUDA capture.
        import rewriter.coordinate_actor_graph as graphs
        class CPUDispatch:
            def __init__(self,actor,slots,templates):self.actor=actor;self.graphs={};self.replays=0;self.eager_fallbacks=0
            def for_visit(self,domain,context,private):
                assert not any(p.requires_grad for p in self.actor.parameters())
                self.replays+=1;return self.actor
        graphs.PrivateActorDispatch=CPUDispatch
        cfg=recipe();cfg['private_inner_steps']=2;cfg['readout']['heldout']['enabled']=False
        cfg['stopping'].update(max_updates=8,review_after_full_sweeps=[1])
        meta=dict(recipe=cfg,sources={'source':'parent'},world_size=2)
        root=Path(root);parent=root/'parent';child=root/'child'
        first=run_training(cfg,parent,ToyContexts(),torch.device('cpu'),metadata=meta)
        assert first['update']==2 and first['reason']=='review_boundary'
        new=child_recipe(cfg);newmeta=dict(recipe=new,sources={'source':'qualified-private-graph'},world_size=2)
        assert identity(meta)!=identity(newmeta)
        if rank==0:
            child.mkdir();(child/'metadata.json').write_text(json.dumps(newmeta))
        dist.barrier()
        with PrivateProbeStore(parent/'joint_store/global',read_only=True) as store:_,ref=store._manifest_and_reference()
        result=fork_distributed_checkpoint(parent/'joint_store',child/'joint_store',parent_manifest_ref=ref,
            source_run_identity=identity(meta),destination_run_identity=identity(newmeta),
            provenance=dict(source_code_id='parent',destination_code_id='qualified-private-graph',reason='CPU backend-only metadata/resume fixture'))
        assert result['training_steps']==0 and result['destination_generation']==1
        full=run_training(cfg,parent,ToyContexts(),torch.device('cpu'),metadata=meta,resume=True)
        for expected in (4,6,8):
            resumed=run_training(new,child,ToyContexts(),torch.device('cpu'),metadata=newmeta,resume=True)
            assert resumed['update']==expected
            assert resumed['reason']==('review_boundary' if expected<8 else 'declared_budget_review_not_convergence')
        assert_tree_equal(full['actor'],resumed['actor']);assert full['visits']==resumed['visits']
        (root/f'rank{rank}-passed').write_text('CPU metadata+fork+actual trainer resume exact')
    finally:dist.destroy_process_group()


def test_metadata_fork_then_actual_trainer_resume_preserves_complete_trajectory(tmp_path):
    mp.spawn(worker,args=(tmp_path,tmp_path/'rendezvous'),nprocs=2,join=True)
    from scripts.qualify_coordinate_fullbank import read_joint
    assert_tree_equal(read_joint(tmp_path/'parent'),read_joint(tmp_path/'child'))
    rows=[json.loads(s) for s in (tmp_path/'child/metrics.jsonl').read_text().splitlines()]
    assert [r['update'] for r in rows if r['event']=='joint_update']==[3,4,5,6,7,8]
    assert [r['update'] for r in rows if r['event']=='full_fitting_sweep']==[4,6,8]
    assert all((tmp_path/f'rank{r}-passed').exists() for r in range(2))
