import re
from chisel.harness import resolve, build

def test_cifar_recipe_partitions_all_parameters():
    cfg=resolve('recipes/cifar.yaml');m=build(cfg['model'])
    for name,p in m.named_parameters():
        assert sum(bool(re.fullmatch(g['parameters'],name)) for g in cfg['optimizers'].values())==1,name
    assert all(b.attn.log_scale is not None for b in m.blocks)
    assert all(not b.attn.causal for b in m.blocks)
    assert cfg['train_steps']==52700

def test_text_protocol():
    cfg=resolve('recipes/nanogpt.yaml')
    assert cfg['train_steps']==3325
    assert cfg['data']['batch_size']==524288
    assert cfg['runtime']['world_size']==8
    assert cfg['gradient_units']=='sum'
