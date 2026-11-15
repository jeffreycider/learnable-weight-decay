"""Precision-only fork and exactly-once initial TRAIN readout; CPU execution."""
import copy,json
import pytest
import torch
import torch.distributed as dist
from scripts.fork_coordinate_residual_precision import execute,transition_shared
from scripts.train_coordinate_chisel import build_actor,run_training,validate_recipe
from scripts.qualify_coordinate_fullbank import read_joint
from scripts.prepare_coordinate_backend_fork import identity
from tests.test_adaptive_training_driver import adaptive_recipe,AdaptiveToyContexts
from tests.test_distributed_probe_store import assert_tree_equal


def test_fork_rejects_unrelated_numeric_setting_and_preserves_rng():
    cfg=adaptive_recipe();cfg['shared_optimizer_step']=False
    with torch.random.fork_rng(devices=[]):actor=build_actor(cfg)
    shared={'model':actor.state_dict(),'optimizer':{'state':{'existing':torch.tensor([2.])}}}
    before=torch.get_rng_state().clone();child=copy.deepcopy(cfg);child['architecture']['residual_precision']='fp32'
    result=transition_shared(shared,cfg,child)
    assert torch.equal(before,torch.get_rng_state())
    result['model']['_extra_state']=shared['model']['_extra_state'];assert_tree_equal(result,shared)
    for key in ('objectives','private_optimizer','panels'):
        bad=copy.deepcopy(child);bad[key]['injected']=123
        with pytest.raises((ValueError,TypeError)):transition_shared(shared,cfg,bad)
    bad=copy.deepcopy(child);bad['architecture']['model_dim']*=2
    with pytest.raises(ValueError,match='only BF16'):transition_shared(shared,cfg,bad)
    bad=copy.deepcopy(child);bad['mixed_index_sha256']='unapproved-data'
    with pytest.raises(ValueError,match='mixed_index'):transition_shared(shared,cfg,bad)
    for value in (-1,True,999):
        bad=copy.deepcopy(cfg);bad['pre_update_fitting_readout_at']=value
        with pytest.raises(ValueError,match='pre-update'):validate_recipe(bad)


def test_actual_store_precision_fork_initial_readout_and_resume(tmp_path,monkeypatch):
    import scripts.train_coordinate_chisel as driver
    original=driver.build_actor
    def nonzero_heads(config):
        actor=original(config)
        with torch.no_grad():
            for head in actor.output_heads:head.weight.normal_(std=.001)
        return actor
    monkeypatch.setattr(driver,'build_actor',nonzero_heads)
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(tmp_path/'rdzv'),rank=0,world_size=1)
    try:
        cfg=adaptive_recipe();cfg.update(shared_optimizer_step=False,qualification_only=True,
             fitting_subset={'gpt':[1,2],'cifar10':[1,2]})
        cfg['stopping']['review_after_full_sweeps']=[1]
        meta={'recipe':cfg,'fixture':'residual precision fork'};parent=tmp_path/'parent'
        run_training(cfg,parent,AdaptiveToyContexts(),torch.device('cpu'),metadata=meta)
        before=read_joint(parent);ref=json.loads((parent/'joint_store/global/CURRENT.json').read_text())
        assert before[1][0]['next_global_batch']==2
        def fork(name,initial_readout=True):
            child=tmp_path/name;child.mkdir();c=copy.deepcopy(cfg)
            c['architecture']['residual_precision']='fp32'
            if initial_readout:c['pre_update_fitting_readout_at']=2
            c['stopping'].update(max_updates=6,review_after_full_sweeps=[2])
            m={'recipe':c,'fixture':'residual precision fork'};(child/'metadata.json').write_text(json.dumps(m))
            plan=dict(schema='chisel_distributed_checkpoint_fork_plan_v1',world_size=1,
                source=str(parent/'joint_store'),destination=str(child/'joint_store'),parent_manifest_ref=ref,
                source_run_identity=identity(meta),destination_run_identity=identity(m),
                provenance=dict(source_code_id='cpu-parent',destination_code_id='cpu-child',reason='precision only'))
            report=execute(plan,cfg,c)
            assert report['all_numeric_owners_unchanged'] and report['training_steps']==0
            forked=read_joint(child);forked[0]['model']['_extra_state']=before[0]['model']['_extra_state']
            assert_tree_equal(before,forked)
            return child,c,m
        full,c,m=fork('full');run_training(c,full,AdaptiveToyContexts(),torch.device('cpu'),metadata=m,resume=True)
        split,c,m=fork('split')
        def stop(batch):
            if batch==3:raise RuntimeError('postcommit interruption')
        with pytest.raises(RuntimeError,match='postcommit'):
            run_training(c,split,AdaptiveToyContexts(),torch.device('cpu'),metadata=m,resume=True,after_commit=stop)
        run_training(c,split,AdaptiveToyContexts(),torch.device('cpu'),metadata=m,resume=True)
        a,b=read_joint(full),read_joint(split);assert_tree_equal(a,b)
        assert_tree_equal(read_joint(parent),before)
        for name,p in before[0]['model'].items():
            if name!='_extra_state':assert_tree_equal(a[0]['model'][name],p)
        assert_tree_equal(a[0]['optimizer'],before[0]['optimizer'])
        assert a[1][0]['shared_optimizer_steps']==0
        assert a[1][0]['pre_update_readout_completed']==2
        assert any(not torch.equal(p['private'][n],before[2][token]['private'][n])
                   for token,p in a[2].items() for n in p['private'] if torch.is_tensor(p['private'][n]))
        for output in (full,split):
            rows=[json.loads(l) for l in (output/'metrics.jsonl').read_text().splitlines()]
            assert [r['global_batch'] for r in rows if r['event']=='pre_update_fitting_readout_completed']==[2]
            assert [r['global_batch'] for r in rows if r['event']=='full_fitting_sweep']==[2,4]
        no_read,c,m=fork('no_initial_readout',False)
        run_training(c,no_read,AdaptiveToyContexts(),torch.device('cpu'),metadata=m,resume=True)
        without=read_joint(no_read)
        # The additional diagnostic changes only its marker and accounted
        # forwards. It must not perturb RNG, panels, gradients or Adam updates.
        assert a[1][0]['compute']['toy_forwards']==without[1][0]['compute']['toy_forwards']+4
        assert a[1][0]['compute']['toy_backwards']==without[1][0]['compute']['toy_backwards']
        comparable=copy.deepcopy(a)
        comparable[1][0]['compute']=without[1][0]['compute']
        del comparable[1][0]['pre_update_readout_completed']
        assert_tree_equal(comparable,without)
    finally:dist.destroy_process_group()
