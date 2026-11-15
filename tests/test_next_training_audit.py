import struct
import numpy as np
import torch
from baseline.data import FinewebStream
from rewriter.replay_buffer import next_training_sequences
from scripts.train_multistate_chisel import globally_heldout_next_sequences,heldout_sequence


def test_next_batch_sample_matches_prescribed_stream_across_shard_boundary(tmp_path):
    paths=[]
    for shard in range(3):
        p=tmp_path/f'fineweb_train_{shard:03}.bin';tokens=(np.arange(35)+shard*100).astype('<u2')
        p.write_bytes(struct.pack('<iii',20240520,1,len(tokens))+bytes(1012)+tokens.tobytes());paths.append(p)
    stream=FinewebStream(str(tmp_path/'*.bin'),16,seq_len=4,rank=0,world_size=1,device='cpu')
    for completed in range(5):
        expected,target=next(stream)
        sampled=next_training_sequences(paths,completed,batch_tokens=16,seq_len=4,count=4)
        order=[r['sequence'] for r in sampled.provenance]
        assert torch.equal(sampled.inputs,expected[order])
        assert torch.equal(sampled.targets,target[order])
        assert all(r['step']==completed for r in sampled.provenance)


def test_fresh_audit_stays_heldout_when_later_checkpoints_are_fitted(tmp_path):
    path=tmp_path/'fineweb_train_000.bin';tokens=np.arange(2000,dtype='<u2')
    path.write_bytes(struct.pack('<iii',20240520,1,len(tokens))+bytes(1012)+tokens.tobytes())
    config=dict(batch_tokens=256,seq_len=4)
    inputs,ids=globally_heldout_next_sequences([path],2,config,8,600001,500001)
    assert len(ids)==8 and len(set(ids))==8
    assert all(128<=sequence_id<192 and heldout_sequence(sequence_id,500001) for sequence_id in ids)
    expected=torch.stack([torch.arange(sequence_id*4,sequence_id*4+4) for sequence_id in ids]).int()
    torch.testing.assert_close(inputs,expected)
