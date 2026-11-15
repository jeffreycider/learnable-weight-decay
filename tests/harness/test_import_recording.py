import copy
import torch
import pytest
from chisel.harness import build,resolve
from chisel.import_recording import merge


def fixture():
    config=resolve('recipes/nanogpt.yaml')
    config['model'].update(vocab_size=32,num_layers=1,model_dim=8,head_dim=4,expansion=2)
    config['optimizers']['hidden']['optimizer']['compile']=False
    for item in config['optimizers'].values():
        if 'fused' in item['optimizer']:item['optimizer']['fused']=False
    model=build(config['model']);names=list(dict(model.named_parameters()))
    groups=[dict(optimizer='legacy',groups=[dict(params=names,lr=.1)])]
    shards=[]
    for rank in range(8):
        owned={n:p.detach().clone() for i,(n,p) in enumerate(model.named_parameters()) if i%8==rank}
        shard=dict(schema='chisel-muonh-gpt-transition-v1',rank=rank,seed=0,iteration=3,world_size=8,
                   loss_units='global_token_sum',consolidated_weights='implicit zero',weights=owned,
                   buffers=dict(model.named_buffers()) if rank==0 else {},
                   gradient={n:torch.ones_like(p) for n,p in owned.items()},optimizer_state={n:{} for n in owned},
                   optimizer_groups=groups,observations={'input_rms/blocks.0.attn.q':torch.ones(8)*(rank+1)},
                   data_cursor=dict(pos=24,batch_size=8,batches=3),torch_rng=torch.get_rng_state(),cuda_rng=torch.get_rng_state(),
                   observed_global_token_positions=[rank])
        shards.append(shard)
    return config,model,shards


def test_import_preserves_parameters_gradients_and_global_rms():
    config,model,shards=fixture();record=merge(shards,config)
    for n,p in model.state_dict().items():torch.testing.assert_close(record['weights'][n],p)
    assert record['next_data']['pos']==32
    assert record['current']['data']['pos']==24
    expected=torch.arange(1,9).float().square().mean().sqrt()
    torch.testing.assert_close(record['activity']['blocks.0.attn.q.weight']['input'],expected.expand(8))
    assert record['activation_shards']==0
    assert 'output-adjoint RMS' in record['import_provenance']['unavailable']
    assert len(record['legacy_samples'])==8


def test_import_rejects_missing_duplicate_and_wrong_phase():
    config,_,shards=fixture()
    with pytest.raises(ValueError):merge(shards[:-1],config)
    broken=copy.deepcopy(shards);broken[1]['weights'].update(broken[0]['weights'])
    with pytest.raises(ValueError):merge(broken,config)
    broken=copy.deepcopy(shards);broken[0]['schema']='chisel-muonh-gpt-terminal-v1'
    with pytest.raises(ValueError):merge(broken,config)
