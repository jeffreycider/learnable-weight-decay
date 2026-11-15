import copy
import random
import struct
import numpy as np
import pytest
import torch
from baseline.checkpoint import build_checkpoint, restore_checkpoint, save_atomic, validate_resume_config
from baseline.data import FinewebStream


def shards(tmp_path):
    for index in range(3):
        tokens = list(range(index*20, index*20+10))
        (tmp_path/f'shard{index}.bin').write_bytes(
            struct.pack('<256i', 20240520, 1, len(tokens), *([0]*253)) + struct.pack('<10H', *tokens))
    return str(tmp_path/'shard*.bin')


def stream(pattern):
    return FinewebStream(pattern, 4, 2, rank=0, world_size=1, device='cpu')


def test_stream_resume_at_shard_tail_without_replaying_batches(tmp_path):
    pattern = shards(tmp_path)
    original = stream(pattern)
    x0, y0 = next(original)
    x1, y1 = next(original)
    assert x0.flatten().tolist() == [0, 1, 2, 3]
    assert y0.flatten().tolist() == [1, 2, 3, 4]
    assert x1.flatten().tolist() == [4, 5, 6, 7]
    state = original.state_dict()
    assert state['pos'] == 8 and state['file_index'] == 0
    resumed = stream(pattern)
    resumed.load_state_dict(state)
    assert resumed.tokens is None  # cursor restoration does not traverse old data
    for _ in range(3):
        actual, target = next(original), next(resumed)
        torch.testing.assert_close(actual[0], target[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], target[1], rtol=0, atol=0)
    assert resumed.state_dict() == original.state_dict()


def test_complete_cpu_restart_matches_uninterrupted_update(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(11); random.seed(12); np.random.seed(13)
    pattern = shards(tmp_path)
    loader = stream(pattern)
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, betas=(.8,.95))
    config = dict(max_steps=5, seed=11, schedule_steps=5, train_data=pattern)
    def step(m, o, data):
        x, y = next(data)
        loss = (m(x.float()/60) - y.float()/60 + torch.randn(2,2)*.01).square().mean()
        loss.backward(); o.step(); o.zero_grad(set_to_none=True)
    step(model,opt,loader); step(model,opt,loader)
    state = build_checkpoint(model, [opt], None, loader, config, 2, dict(forward=2,backward=2), 7.5)
    save_atomic(state, tmp_path/'saved.pt')
    step(model,opt,loader)
    expected_rng = (torch.rand(3), random.random(), np.random.rand())
    expected_model = copy.deepcopy(model.state_dict())
    expected_opt = copy.deepcopy(opt.state_dict())
    # Fresh construction perturbs RNG; restoration must undo it.
    model2 = torch.nn.Linear(2,2)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=.003, betas=(.8,.95))
    loader2 = stream(pattern)
    loaded = torch.load(tmp_path/'saved.pt', weights_only=False)
    step_index, counts, elapsed = restore_checkpoint(loaded,model2,[opt2],None,loader2,config)
    assert (step_index, counts, elapsed) == (2, dict(forward=2,backward=2), 7.5)
    step(model2,opt2,loader2)
    actual_rng = (torch.rand(3), random.random(), np.random.rand())
    torch.testing.assert_close(actual_rng[0],expected_rng[0],rtol=0,atol=0)
    assert actual_rng[1:] == expected_rng[1:]
    for name,tensor in expected_model.items():
        torch.testing.assert_close(model2.state_dict()[name],tensor,rtol=0,atol=0)
    for index, values in expected_opt['state'].items():
        for name, tensor in values.items():
            torch.testing.assert_close(opt2.state_dict()['state'][index][name],tensor,rtol=0,atol=0)
    assert not (tmp_path/'saved.pt.tmp').exists()


def test_resume_rejects_algorithm_schedule_seed_changes():
    config = dict(max_steps=3250,schedule_steps=3250,seed=0,mode='learned',world_size=1,
                  train_data='/a/*.bin',rewriter=dict(aux_lr=2e-5))
    validate_resume_config(config,dict(config,train_data='/b/*.bin'))
    for key,value in [('max_steps',3000),('schedule_steps',3000),('seed',1),('mode','baseline'),
                      ('world_size',2),('rewriter',dict(aux_lr=1e-5))]:
        with pytest.raises(ValueError,match='research configuration'):
            validate_resume_config(config,dict(config,**{key:value}))


def test_resume_rejects_optimizer_group_override(tmp_path):
    pattern = shards(tmp_path)
    loader = stream(pattern)
    model = torch.nn.Linear(2,2)
    opt = torch.optim.AdamW(model.parameters(),lr=.1,betas=(.8,.95))
    state = build_checkpoint(model,[opt],None,loader,dict(max_steps=5),0,dict(forward=0,backward=0),0.)
    wrong = torch.optim.AdamW(model.parameters(),lr=.1,betas=(.9,.95))
    with pytest.raises(ValueError,match='hyperparameter'):
        restore_checkpoint(state,model,[wrong],None,loader,dict(max_steps=5))
