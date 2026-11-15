import pytest
import torch
from chisel.models import CifarGPT
from chisel.observations import Observations
from chisel.muonh import MuonH

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA qualification')

def test_compiled_sensors_reuse_graphs_without_changing_gradients():
    from torch._dynamo.utils import counters
    model=CifarGPT(32,2,4,'mean',head_dim=8,causal=False,learned_qk_scale=True,expansion=2).cuda()
    sensor=Observations(model,r'blocks\.\d+\.(attn\.(q|k|v|proj)|mlp\.(fc|proj))\.weight')
    compiled=torch.compile(model,fullgraph=True,dynamic=False)
    counts=[]
    for step in range(5):
        sensor.active=True
        for squares,count in sensor.sums.values():squares.zero_();count.zero_()
        x=torch.randn(8,3,32,32,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=compiled(x).square().sum()
        loss.backward();sensor.finalize(1);model.zero_grad(set_to_none=True)
        counts.append(counters['stats']['unique_graphs'])
    assert counts[2]==counts[3]==counts[4],counts
    sensor.close()

def test_cuda_muonh_matches_literal_record():
    from test_muonh import reference
    torch.manual_seed(4)
    for shape in [(32,64),(64,32),(32,32)]:
        p=torch.nn.Parameter(torch.randn(*shape,device='cuda'));p.grad=torch.randn_like(p)
        expected=reference(p.grad,torch.zeros_like(p),p.detach(),.018,.95)
        optimizer=MuonH([p],lr=.018,mu=.95,ns_steps=12,compile=True);optimizer.step()
        torch.testing.assert_close(p,expected,rtol=1e-5,atol=2e-6)
