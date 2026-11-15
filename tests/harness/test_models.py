import subprocess, types
import torch
from baseline.model import Block
from chisel.models import CifarGPT

def test_default_block_matches_original():
    original = types.ModuleType('original')
    exec(subprocess.check_output(['git','show','56cfb7e5dec46ed4e95b58d5e90cfe209086805a:baseline/model.py'],text=True), original.__dict__)
    torch.manual_seed(5); expected=original.Block(128)
    torch.manual_seed(5); actual=Block(128)
    x=torch.randn(2,5,128,requires_grad=True)
    y=expected(x);z=actual(x)
    torch.testing.assert_close(y,z,rtol=0,atol=0)
    a=torch.autograd.grad(y.sum(),list(expected.parameters()))
    b=torch.autograd.grad(z.sum(),list(actual.parameters()))
    for u,v in zip(a,b):torch.testing.assert_close(u,v,rtol=0,atol=0)

def test_cifar_shapes_qk_scale_and_expansion():
    m=CifarGPT(32,2,4,'mean',head_dim=8,causal=False,learned_qk_scale=True,attention_scale=.24,expansion=2)
    y=m(torch.randn(3,3,32,32));assert y.shape==(3,10)
    assert m.blocks[0].mlp.fc.weight.shape==(64,32)
    assert m.blocks[0].attn.log_scale.shape==(4,)
    with torch.no_grad():m.proj.weight.normal_()
    m(torch.randn(3,3,32,32)).square().mean().backward()
    assert m.blocks[0].attn.log_scale.grad.abs().sum()>0
