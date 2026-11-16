import copy
import pytest
import torch
from chisel.network import Chisel, PrivateProbes, propose
from chisel.sensor_cache import BANKS, prepare


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required'))])
def test_cached_sensors_preserve_actions_and_all_learned_gradients(device):
    torch.manual_seed(52)
    snapshot = {}
    for name,shape in [('blocks.0.attn.q.weight',(8,12)),('blocks.0.mlp.fc.weight',(12,8))]:
        snapshot[name] = {bank:torch.randn(*shape,device=device,dtype=torch.float64) for bank in BANKS}
        snapshot[name]['consolidated'].zero_()
        snapshot[name]['activity'] = dict(input=torch.rand(shape[1],device=device,dtype=torch.float64),
                                          adjoint=torch.rand(shape[0],device=device,dtype=torch.float64))
    cached = prepare(snapshot)
    assert all(not v.requires_grad for item in cached.values() for v in item['encoder_cache']['banks'])
    actor = Chisel(2,16,1,head_dim=4,expansion=2).to(device).double()
    torch.nn.init.normal_(actor.proj.weight,std=.05)
    probes = PrivateProbes({n:x['raw'].shape for n,x in snapshot.items()},2).to(device).double()
    parameters = tuple(actor.parameters()) + tuple(probes.parameters())
    for iteration in range(3):
        results=[]
        for inputs in [snapshot,cached]:
            output = propose(actor,probes,inputs)
            flat = torch.cat([v.flatten() for item in output.values() for v in item.values()])
            grads = torch.autograd.grad(flat.square().sum(),parameters)
            results.append((flat,grads))
        torch.testing.assert_close(results[0],results[1],rtol=1e-12,atol=1e-12)
        assert any(g.abs().max()>0 for g in results[0][1])
        with torch.no_grad():
            for parameter,gradient in zip(parameters,results[0][1]):parameter.add_(gradient,alpha=-.01)


def test_zero_raw_bank_keeps_zero_write_scale_and_activity_defaults():
    item = {bank:torch.zeros(8,12) for bank in BANKS}
    result = prepare({'w':item})['w']['encoder_cache']
    assert result['raw_scale']==0
    assert all(torch.isfinite(value).all() for value in (*result['banks'],*result['activity'],*result['scales']))
    assert [v.shape for v in result['activity']]==[torch.Size([12]),torch.Size([8])]
