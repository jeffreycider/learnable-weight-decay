from pathlib import Path
import torch
import pytest
from chisel.offline import ProbeStore
from chisel.recording import cpu_copy


def make(root, budget, device="cpu"):
    return ProbeStore(root,dict(seed=2,rank=2,cache_bytes=budget,
        optimizer=dict(_target_='torch.optim.Adam',lr=.001)),device)


def update(store,identity,step):
    probes,opt=store.load(identity,{'w':{'raw':torch.ones(8,8)}})
    sum(p.square().sum() for p in probes.parameters()).backward();opt.step()
    store.save(identity,step,probes,opt)
    return probes,opt


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required'))])
def test_cache_matches_eager_save_and_checkpoint_flush_is_immutable(tmp_path, device):
    cached,eager=make(tmp_path/'cached',1<<20,device),make(tmp_path/'eager',0,device)
    key='a'*64
    for step in range(3):
        a,ao=update(cached,key,step);b,bo=update(eager,key,step)
        torch.testing.assert_close(a.state_dict(),b.state_dict(),rtol=0,atol=0)
        torch.testing.assert_close(cpu_copy(ao.state_dict()),cpu_copy(bo.state_dict()),rtol=0,atol=0)
    assert key not in cached.latest and not list((tmp_path/'cached').glob('*.pt'))
    state=cached.state_dict();path=Path(state['latest'][key]);before=path.read_bytes()
    assert path.stem.endswith('00000002')
    old_payload=cached.cache[key][0]['optimizer']['state'][0]['exp_avg'].clone()
    live,opt=cached.load(key,{'w':{'raw':torch.ones(8,8)}})
    sum(p.square().sum() for p in live.parameters()).backward();opt.step()
    torch.testing.assert_close(cached.cache[key][0]['optimizer']['state'][0]['exp_avg'],old_payload,rtol=0,atol=0)
    update(cached,key,3);assert path.read_bytes()==before
    current=cached.state_dict();assert current['latest'][key]!=str(path)
    restored=make(tmp_path/'restored',1<<20,device);restored.load_state_dict(current)
    a,ao=restored.load(key,{'w':{'raw':torch.ones(8,8)}})
    b,bo=cached.load(key,{'w':{'raw':torch.ones(8,8)}})
    torch.testing.assert_close(a.state_dict(),b.state_dict(),rtol=0,atol=0)
    torch.testing.assert_close(cpu_copy(ao.state_dict()),cpu_copy(bo.state_dict()),rtol=0,atol=0)


def test_dirty_eviction_is_durable_and_memory_is_bounded(tmp_path):
    store=make(tmp_path,1<<20);first='a'*64;second='b'*64
    update(store,first,0);size=store.resident_bytes
    store.cache_bytes=size
    update(store,second,1)
    assert list(store.cache)==[second]
    assert store.resident_bytes==size
    assert Path(store.latest[first]).is_file()
    saved=store.state_dict()
    assert set(saved['latest'])=={first,second}


def test_metrics_distinguish_cache_hits_from_file_reloads(tmp_path):
    store=make(tmp_path,1<<20);first='a'*64;second='b'*64
    update(store,first,0)
    size=store.resident_bytes
    store.cache_bytes=size
    update(store,first,1)
    assert store.cache_metrics()==dict(private_cache_hits=1,private_disk_loads=0,
        private_disk_writes=0,private_cache_evictions=0,private_cache_bytes=size)
    update(store,second,2)  # One-entry cache must write the learned first state.
    update(store,first,3)   # Its next visit must read that file, then evict second.
    metrics=store.cache_metrics()
    assert metrics['private_cache_hits']==1 and metrics['private_disk_loads']==1
    assert metrics['private_cache_evictions']==2
    assert metrics['private_disk_writes']==len(list(tmp_path.glob('*.pt')))==2
    assert metrics['private_cache_bytes']==size
    saved=store.state_dict()
    assert store.cache_metrics()['private_disk_writes']==3
    assert 'private_cache_hits' not in saved  # Diagnostics do not alter learned restart state.


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required'))])
def test_buffered_restore_preserves_probe_adam_update(tmp_path, device):
    eager=make(tmp_path/'eager',0,device)
    key='a'*64
    update(eager,key,0)
    update(eager,key,1)
    saved=eager.state_dict()
    buffered=make(tmp_path/'buffered',0,device)
    buffered.config['buffered_load']=True
    buffered.load_state_dict(saved)
    before=Path(saved['latest'][key]).read_bytes()
    a,ao=update(eager,key,2)
    b,bo=update(buffered,key,2)
    torch.testing.assert_close(a.state_dict(),b.state_dict(),rtol=0,atol=0)
    torch.testing.assert_close(cpu_copy(ao.state_dict()),cpu_copy(bo.state_dict()),rtol=0,atol=0)
    assert Path(saved['latest'][key]).read_bytes()==before
    assert buffered.cache_metrics()['private_disk_loads']==1
