"""New fixed-state driver: real two-rank reduction plus identity bindings."""
import copy,json,sys
from pathlib import Path
from types import SimpleNamespace
from datetime import timedelta
import pytest,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runs/chisel-astra'))
import qualify_pilot_replay_curve as driver
from tests.test_online_preservation import make,settings
from tests.test_transformer_online import _ordinary_step
from rewriter.online_preservation import ReplayMaintenance
from rewriter.lag_one import LagOneHypergradient
from rewriter.integration import _checkpoint_tensor_copy


def test_panel_binding_rejects_wrong_ids_hash_and_clock():
    panel=dict(ids=[1,2],id_sha256='ids',input_sha256='inputs',action_step=2)
    state=dict(controller=dict(replay=dict(last_panel=copy.deepcopy(panel))))
    case=dict(panel_sequence_ids=[1,2],panel_id_sha256='ids',panel_input_sha256='inputs',action_step=2)
    driver.validate_panel(panel,state,case)
    for key in panel:
        bad=copy.deepcopy(panel);bad[key]=[3,4] if key=='ids' else 3 if key=='action_step' else 'changed'
        with pytest.raises(ValueError):driver.validate_panel(bad,state,case)


def worker(rank,directory,rendezvous):
    torch.distributed.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=90))
    try:
        model,opt,adapter=make(directory,2)
        _ordinary_step(model,opt,adapter,2,rank,2)
        saved=adapter.state_dict();pending=saved['learner']['pending'] if rank==0 else None
        objects=[pending];torch.distributed.broadcast_object_list(objects,src=0);pending=objects[0]
        with torch.no_grad():
            for n,p in model.named_parameters():p.copy_(pending['base'][n])
        life=adapter.lifecycle
        if rank==0:
            life.action=None;life.lag=LagOneHypergradient();life.action_step=None
        settings_=copy.deepcopy(adapter.replay.settings)
        replay=ReplayMaintenance(model,settings_,batch_tokens=8,seq_len=4,embedding_names={'embed.weight'})
        panel=replay.sampler.sample(2,2)
        case=dict(panel_sequence_ids=panel['ids'],panel_id_sha256=panel['id_sha256'],panel_input_sha256=panel['input_sha256'],action_step=2)
        state=dict(step=3,controller=dict(replay=saved['replay']))
        a=SimpleNamespace(output=Path(directory)/'curve.json',curve_updates=2,case=case,
            expected_sha256='fixture',spec=Path(directory)/'data_manifest.json',settings=Path(directory)/'data_manifest.json',
            deterministic_policy='CPU gloo fixture',adam_steps_before=driver.adam_steps(life.optimizer) if rank==0 else None)
        _,proposal=driver.learning_curve(a,state,settings_,model,life,replay,
            _checkpoint_tensor_copy(pending['snapshot'],torch.device('cpu')) if rank==0 else None,pending,2,rank,torch.device('cpu'))
        if rank==0:
            result=json.loads(a.output.read_text())
            assert result['adam_steps_before']['minimum']==1 and result['adam_steps_after']['minimum']==3
            assert result['fresh_dual_initial']==0 and result['actor_forward_calls']==4
            assert len(result['rank_records'])==2
            for record in result['rank_records']:
                c=record['gpt_compute'];assert c['teacher_forwards']==1 and c['student_forwards']==4 and c['student_backwards']==2
            assert proposal is not None and life.action is None
        adapter.close()
    finally:torch.distributed.destroy_process_group()


def test_actual_two_rank_curve_fixed_teacher_and_saved_adam(tmp_path):
    settings(tmp_path,2)
    torch.multiprocessing.spawn(worker,args=(str(tmp_path),str(tmp_path/'rendezvous')),nprocs=2,join=True)
