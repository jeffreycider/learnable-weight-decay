"""Read-only reconstruction of sharded Muon moments for isolated local diagnostics.

This does not change a training checkpoint or implement a resume path. A local
diagnostic must still declare its gradient summation order and workload scope.
"""
import copy


def merge_muon_shards(shards):
    if not shards:raise ValueError('all rank optimizer shards required')
    groups=shards[0]['param_groups'];world=len(shards)
    if any(s['param_groups']!=groups for s in shards):raise ValueError('Muon shard groups differ')
    merged={};expected_all=set()
    for group in groups:
        ids=group['params']
        if expected_all.intersection(ids) or len(ids)!=len(set(ids)):raise ValueError('Muon parameter IDs are duplicated')
        expected_all.update(ids)
    for rank,shard in enumerate(shards):
        expected={pid for group in groups for offset,pid in enumerate(group['params']) if offset%world==rank}
        if set(shard['state'])!=expected:raise ValueError('Muon state ownership or coverage differs')
        for pid,state in shard['state'].items():
            if set(state)!={'momentum'}:raise ValueError('unexpected Muon state schema')
            merged[pid]=dict(state)
    if set(merged)!=expected_all:raise ValueError('incomplete Muon momentum reconstruction')
    return dict(state=merged,param_groups=copy.deepcopy(groups))
