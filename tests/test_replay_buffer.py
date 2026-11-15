import struct
from pathlib import Path
import numpy as np
import pytest
import torch
from baseline.data import FinewebStream
from rewriter.replay_buffer import sample_past_sequences, synthetic_sequences


def write_shards(tmp_path, lengths=(10,9,14,10)):
    paths=[]
    for i,n in enumerate(lengths):
        p=tmp_path/f'fineweb_train_{i:03d}.bin'
        p.write_bytes(struct.pack('<256i',20240520,1,n,*([0]*253))+
                      np.arange(i*100,i*100+n,dtype='<u2').tobytes())
        paths.append(p)
    return paths


def all_rows(replay):
    return [(x,y,p) for batch in (replay.train,replay.audit)
            for x,y,p in zip(batch.inputs,batch.targets,batch.provenance)]


def test_sample_matches_actual_consumed_prefix_at_every_boundary(tmp_path):
    paths=write_shards(tmp_path)
    stream=FinewebStream(str(tmp_path/'*.bin'),4,2,rank=0,world_size=1,device='cpu')
    actual=[next(stream) for _ in range(7)]
    replay=sample_past_sequences(paths,7,batch_tokens=4,seq_len=2,count=14,seed=70)
    ids=set()
    for x,y,p in all_rows(replay):
        step,seq=p['step'],p['sequence']
        torch.testing.assert_close(x,actual[step][0][seq],rtol=0,atol=0)
        torch.testing.assert_close(y,actual[step][1][seq],rtol=0,atol=0)
        assert p['id']==step*2+seq
        ids.add(p['id'])
    assert ids==set(range(14))
    # The second shard's equality tail is skipped: offset4+4+1 == 9.
    assert not any(p['shard_index']==1 and p['offset']>=4 for _,_,p in all_rows(replay))
    assert replay.train.inputs.dtype==torch.int32 and replay.audit.targets.dtype==torch.int64
    assert replay.train.inputs.device.type=='cpu'


def test_single_advance_rule_accepts_new_shard_exact_batch_plus_one(tmp_path):
    paths=write_shards(tmp_path,(5,5,10))
    stream=FinewebStream(str(tmp_path/'*.bin'),4,2,rank=0,world_size=1,device='cpu')
    actual=[next(stream) for _ in range(2)]
    out=sample_past_sequences(paths,2,batch_tokens=4,seq_len=2,count=4)
    for x,y,p in all_rows(out):
        torch.testing.assert_close(x,actual[p['step']][0][p['sequence']])
        torch.testing.assert_close(y,actual[p['step']][1][p['sequence']])
    assert {p['shard_index'] for _,_,p in all_rows(out)}=={1,2}


def test_deterministic_disjoint_sampling_no_global_rng_or_future_read(tmp_path):
    paths=write_shards(tmp_path)
    # A corrupt future shard must never be opened while only step0 is known.
    paths[-1].write_bytes(b'not yet available')
    before=torch.random.get_rng_state().clone()
    a=sample_past_sequences(paths,1,batch_tokens=4,seq_len=2,count=2,seed=19)
    b=sample_past_sequences(paths,1,batch_tokens=4,seq_len=2,count=2,seed=19)
    torch.testing.assert_close(before,torch.random.get_rng_state(),rtol=0,atol=0)
    assert a.train.provenance==b.train.provenance and a.audit.provenance==b.audit.provenance
    assert {p['id'] for p in a.train.provenance}.isdisjoint(p['id'] for p in a.audit.provenance)
    assert all(p['step']==0 and p['shard_index']==0 for _,_,p in all_rows(a))


def test_sample_is_selected_by_header_and_memmap_not_dense_loader(tmp_path,monkeypatch):
    import baseline.data
    paths=write_shards(tmp_path)
    def fail(*args,**kwargs):raise AssertionError('dense shard loader invoked')
    monkeypatch.setattr(baseline.data,'_load_data_shard',fail)
    out=sample_past_sequences(paths,2,batch_tokens=4,seq_len=2,count=3)
    assert len(all_rows(out))==3


@pytest.mark.parametrize('name',['fineweb_val_0.bin','validation.bin','test.bin','data.eval.bin'])
def test_rejects_validation_or_test_filename(tmp_path,name):
    p=write_shards(tmp_path)[0];new=tmp_path/name;p.rename(new)
    with pytest.raises(ValueError,match='prohibited'):sample_past_sequences([new],1,batch_tokens=4,seq_len=2,count=2)


@pytest.mark.parametrize('kwargs',[dict(completed_steps=0),dict(count=100),dict(seq_len=3),dict(audit_fraction=0),dict(completed_steps=99)])
def test_invalid_or_unavailable_prefix_rejected(tmp_path,kwargs):
    opts=dict(completed_steps=2,batch_tokens=4,seq_len=2,count=3);opts.update(kwargs)
    with pytest.raises(ValueError):sample_past_sequences(write_shards(tmp_path),**opts)


@pytest.mark.parametrize('kind',['uniform','repeated','alternating','empirical_unigram'])
def test_synthetic_is_seeded_shifted_and_explicitly_labeled(tmp_path,kind):
    replay=sample_past_sequences(write_shards(tmp_path),1,batch_tokens=4,seq_len=2,count=2)
    before=torch.random.get_rng_state().clone()
    a=synthetic_sequences(kind,20,8,50,seed=83,past=replay)
    b=synthetic_sequences(kind,20,8,50,seed=83,past=replay)
    torch.testing.assert_close(a.inputs,b.inputs,rtol=0,atol=0)
    torch.testing.assert_close(a.targets[:,:-1],a.inputs[:,1:].long(),rtol=0,atol=0)
    torch.testing.assert_close(before,torch.random.get_rng_state(),rtol=0,atol=0)
    assert all(p['kind']=='synthetic' and not p['all_input_coverage'] for p in a.provenance)
    if kind=='repeated':assert torch.all(a.inputs==a.inputs[:,:1])
    if kind=='alternating':
        assert torch.all(a.inputs[:,0]!=a.inputs[:,1])
        assert torch.all(a.inputs[:,::2]==a.inputs[:,:1])
    if kind=='empirical_unigram':
        assert set(a.inputs.flatten().tolist()) <= set(replay.train.inputs.flatten().tolist())
        assert not set(a.inputs.flatten().tolist()).intersection(replay.audit.inputs.flatten().tolist())


def test_synthetic_rejects_missing_past_and_invalid_alternating():
    with pytest.raises(ValueError):synthetic_sequences('empirical_unigram',2,3,5)
    with pytest.raises(ValueError):synthetic_sequences('alternating',2,3,1)
