from pathlib import Path
import pytest
import torch
from chisel.device_probe_store import DeviceProbeStore
from chisel.recording import cpu_copy
from tests.harness.test_private_cache import make, update


def device_store(path, budget, device='cpu', cpu_budget=0):
    reference=make(path,cpu_budget,device)
    return DeviceProbeStore(path,reference.config,device,cache_bytes=budget)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required'))])
def test_reused_objects_match_reconstructed_adam_and_durable_restore(tmp_path,device):
    cached=device_store(tmp_path/'cached',1<<20,device)
    reference=make(tmp_path/'reference',0,device)
    for step,key in enumerate(['a'*64,'b'*64,'a'*64,'b'*64,'a'*64]):
        a,ao=update(cached,key,step);b,bo=update(reference,key,step)
        torch.testing.assert_close(a.state_dict(),b.state_dict(),rtol=0,atol=0)
        torch.testing.assert_close(cpu_copy(ao.state_dict()),cpu_copy(bo.state_dict()),rtol=0,atol=0)
    assert cached.device_hits==3 and cached.disk_writes==0
    saved=cached.state_dict();paths={key:Path(path) for key,path in saved['latest'].items()}
    contents={key:path.read_bytes() for key,path in paths.items()}
    for step,key in enumerate(['a'*64,'b'*64],5):update(cached,key,step)
    assert all(paths[k].read_bytes()==data for k,data in contents.items())
    restored=device_store(tmp_path/'restored',1<<20,device);restored.load_state_dict(saved)
    eager=make(tmp_path/'eager',0,device);eager.load_state_dict(saved)
    for key in saved['latest']:
        a,ao=update(restored,key,8);b,bo=update(eager,key,8)
        torch.testing.assert_close(a.state_dict(),b.state_dict(),rtol=0,atol=0)
        torch.testing.assert_close(cpu_copy(ao.state_dict()),cpu_copy(bo.state_dict()),rtol=0,atol=0)


@pytest.mark.parametrize('flush_first',[False,True])
def test_device_eviction_cannot_reload_stale_cpu_weights(tmp_path,flush_first):
    store=device_store(tmp_path/'cached',1<<20,cpu_budget=1<<20)
    reference=make(tmp_path/'reference',0)
    first,second='a'*64,'b'*64
    update(store,first,0);update(reference,first,0)
    store.device_budget=store.device_bytes
    if flush_first:store.state_dict()
    update(store,second,1);update(reference,second,1)
    assert list(store.device_cache)==[second]
    assert store.device_bytes<=store.device_budget and store.device_evictions==1
    a,ao=update(store,first,2);b,bo=update(reference,first,2)
    torch.testing.assert_close(a.state_dict(),b.state_dict(),rtol=0,atol=0)
    torch.testing.assert_close(ao.state_dict(),bo.state_dict(),rtol=0,atol=0)
    assert first not in store.cache
    assert store.device_bytes<=store.device_budget
    restored=device_store(tmp_path/'restored',1<<20);restored.load_state_dict(store.state_dict())
    a,ao=update(restored,first,3);b,bo=update(reference,first,3)
    torch.testing.assert_close(a.state_dict(),b.state_dict(),rtol=0,atol=0)


def test_zero_device_budget_uses_existing_cpu_store(tmp_path):
    store=device_store(tmp_path,0,cpu_budget=1<<20)
    update(store,'a'*64,0)
    assert not store.device_cache and store.device_bytes==0
    assert store.resident_bytes>0
    assert store.state_dict()['latest']
