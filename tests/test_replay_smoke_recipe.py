import copy
import importlib.util
import json
from pathlib import Path
import sys
import pytest
from tests.test_online_preservation import make
from tests.test_transformer_online import _ordinary_step

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runs/chisel-astra'))
from verify_replay_smoke import check_controller


def test_real_replay_smoke_clock_checker(tmp_path):
    model,opt,adapter=make(tmp_path)
    _ordinary_step(model,opt,adapter,2)
    pending=copy.deepcopy(adapter.state_dict())
    check_controller(pending,task_updates=0,replay_updates=1,pending=True,local_tokens=8)
    for step in (3,4,5):_ordinary_step(model,opt,adapter,step)
    final=adapter.state_dict()
    check_controller(final,task_updates=3,replay_updates=3,pending=False,local_tokens=8)
    bad=copy.deepcopy(final);bad['replay']['compute']['teacher_tokens']+=8
    with pytest.raises(AssertionError):check_controller(bad,task_updates=3,replay_updates=3,pending=False,local_tokens=8)
    adapter.close()


def test_recipe_pins_science_and_restarts_nonterminal_pending():
    root=ROOT/'runs/chisel-astra/replay-online-smoke'
    recipe=json.loads((root/'recipe.json').read_text());settings=json.loads((root/'settings.json').read_text())
    replay=settings['replay_maintenance']
    assert (replay['steps'],replay['sequences'],replay['microbatch_sequences'],replay['shrinkage_coefficient'])==(1,128,16,1)
    assert replay['teacher_cache_device']=='device'
    assert settings['preserve_first_action_checkpoint']
    commands={c['name']:c['argv'] for c in recipe['commands']}
    assert commands['convert_learned'][commands['convert_learned'].index('--max-steps')+1]=='1004'
    assert '--compile-loss' not in commands['convert_learned']
    assert 'step1001' in commands['cpu_stage_pending'][-1]
    assert commands['train_restarted'][commands['train_restarted'].index('--resume')+1].endswith('/pending_restart')
    for name,command in commands.items():
        if command[1:2]==['-c']:compile(command[2],name,'exec')


def test_peek_next_batch_uses_separate_cursor_and_handles_shard_boundary(tmp_path):
    import torch
    from tests.test_online_preservation import settings
    from baseline.data import FinewebStream
    from peek_replay_pilot_batch import peek
    settings(tmp_path)
    pattern=str(tmp_path/'fineweb_train_*.bin')
    live=FinewebStream(pattern,8,4,rank=0,world_size=1,device='cpu')
    for _ in range(12):next(live)
    before=copy.deepcopy(live.state_dict());rng=torch.get_rng_state().clone()
    state=dict(data_cursor=copy.deepcopy(before),config=dict(batch_tokens=8,seq_len=4,world_size=1))
    panel,metadata=peek(state,pattern)
    assert live.state_dict()==before and state['data_cursor']==before
    assert torch.equal(torch.get_rng_state(),rng)
    expected,_=next(live);torch.testing.assert_close(panel,expected,rtol=0,atol=0)
    assert metadata['isolated_cursor_after_read']['file_index']==1
    assert metadata['data_role']=='fresh_next_training_audit'
    assert metadata['sequence_coordinate_ids']==['fineweb_train_001.bin:token0','fineweb_train_001.bin:token4']


def test_nonroot_checkpoint_owns_compute_but_no_actor_or_dual(tmp_path):
    model,opt,adapter=make(tmp_path)
    _ordinary_step(model,opt,adapter,2)
    state=copy.deepcopy(adapter.state_dict());state['rank']=1
    state['learner']=None;state['replay']['dual']=None
    result=check_controller(state,task_updates=0,replay_updates=1,pending=True,local_tokens=8)
    assert result['actor_authority'] is False
    assert result['adam_step'] is None and result['task_updates'] is None and result['pending'] is None
    bad=copy.deepcopy(state);bad['learner']={}
    with pytest.raises(AssertionError):check_controller(bad,task_updates=0,replay_updates=1,pending=True,local_tokens=8)
    adapter.close()
