from pathlib import Path
import torch
from chisel.offline import ProbeStore


def make(root, budget):
    return ProbeStore(root,dict(seed=2,rank=2,cache_bytes=budget,
        optimizer=dict(_target_='torch.optim.Adam',lr=.001)),'cpu')


def update(store,identity,step):
    probes,opt=store.load(identity,{'w':{'raw':torch.ones(8,8)}})
    sum(p.square().sum() for p in probes.parameters()).backward();opt.step()
    store.save(identity,step,probes,opt)
    return probes,opt


def test_cache_matches_eager_save_and_checkpoint_flush_is_immutable(tmp_path):
    cached,eager=make(tmp_path/'cached',1<<20),make(tmp_path/'eager',0)
    key='a'*64
    for step in range(3):
        a,ao=update(cached,key,step);b,bo=update(eager,key,step)
        torch.testing.assert_close(a.state_dict(),b.state_dict(),rtol=0,atol=0)
        torch.testing.assert_close(ao.state_dict(),bo.state_dict(),rtol=0,atol=0)
    assert key not in cached.latest and not list((tmp_path/'cached').glob('*.pt'))
    state=cached.state_dict();path=Path(state['latest'][key]);before=path.read_bytes()
    assert path.stem.endswith('00000002')
    old_payload=cached.cache[key][0]['optimizer']['state'][0]['exp_avg'].clone()
    live,opt=cached.load(key,{'w':{'raw':torch.ones(8,8)}})
    sum(p.square().sum() for p in live.parameters()).backward();opt.step()
    torch.testing.assert_close(cached.cache[key][0]['optimizer']['state'][0]['exp_avg'],old_payload,rtol=0,atol=0)
    update(cached,key,3);assert path.read_bytes()==before
    current=cached.state_dict();assert current['latest'][key]!=str(path)
    restored=make(tmp_path/'restored',1<<20);restored.load_state_dict(current)
    a,ao=restored.load(key,{'w':{'raw':torch.ones(8,8)}})
    b,bo=cached.load(key,{'w':{'raw':torch.ones(8,8)}})
    torch.testing.assert_close(a.state_dict(),b.state_dict(),rtol=0,atol=0)
    torch.testing.assert_close(ao.state_dict(),bo.state_dict(),rtol=0,atol=0)


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
