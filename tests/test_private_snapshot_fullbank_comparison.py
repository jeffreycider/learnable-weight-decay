from scripts.compare_private_snapshot_training import numeric_rank_states
from rewriter.distributed_probe_store import _digest


def test_only_physical_bytes_are_separated_from_numeric_state():
    original={0:{'private_written_bytes':100,'next_global_batch':9,'compute':{'forward':2}}}
    candidate={0:{'private_written_bytes':120,'next_global_batch':9,'compute':{'forward':2}}}
    a,abytes=numeric_rank_states(original);b,bbytes=numeric_rank_states(candidate)
    assert _digest(a)==_digest(b) and abytes!=bbytes
    assert original[0]['private_written_bytes']==100
    candidate[0]['compute']['forward']=3
    assert _digest(a)!=_digest(numeric_rank_states(candidate)[0])
