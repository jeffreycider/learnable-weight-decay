import numpy as np
import torch
from baseline.data import FinewebStream
from chisel.preservation_data import cursor_at_step,training_batch_step


def test_random_past_sampling_is_repeatable_bounded_and_rng_independent():
    config={'preservation_data':{'policy':'uniform_past','seed':5}}
    before=torch.get_rng_state()
    values=[training_batch_step(config,'abc',i,40) for i in range(100)]
    assert values==[training_batch_step(config,'abc',i,40) for i in range(100)]
    assert min(values)>=0 and max(values)<=40 and len(set(values))>20
    assert torch.equal(before,torch.get_rng_state())


def test_fineweb_index_replays_exact_batches_across_shard_tails(tmp_path):
    for i in range(3):
        header=np.zeros(256,dtype=np.int32);header[:3]=[20240520,1,27]
        (tmp_path/f'shard{i}.bin').write_bytes(header.tobytes()+np.arange(i*30,i*30+27,dtype=np.uint16).tobytes())
    stream=FinewebStream(str(tmp_path/'*.bin'),8,4,rank=0,world_size=1,device='cpu')
    cursor=stream.state_dict();expected=[next(stream) for _ in range(9)]
    for index in [8,0,4,3,2,7]:
        stream.load_state_dict(cursor_at_step(stream,cursor,index))
        actual=next(stream)
        assert all(torch.equal(a,b) for a,b in zip(actual,expected[index]))


def test_sequence_sampling_covers_whole_batch_without_consuming_rng():
    from chisel.preservation_fit import sampled_inputs
    batch={'inputs':torch.arange(512)};before=torch.get_rng_state()
    a=sampled_inputs(batch,dict(preservation_examples=16,preservation_sample_seed=10))
    b=sampled_inputs(batch,dict(preservation_examples=16,preservation_sample_seed=10))
    assert torch.equal(a,b) and len(a.unique())==16 and a.max()>100
    assert torch.equal(before,torch.get_rng_state())
