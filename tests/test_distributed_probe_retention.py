"""Collective retention tests delete only unreachable files in temporary stores."""
import copy
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rewriter.coordinate_chisel import CheckpointProbes
from rewriter.private_probe_store import PrivateIdentity,PrivateProbeStore
from rewriter.private_probe_slot import PrivateProbeSlot
from rewriter.distributed_probe_store import DistributedProbeStore,private_owner
from tests.test_coordinate_private_invariants import setup
from tests.test_distributed_probe_store import initialize,assert_tree_equal


def worker(rank,directory,rendezvous):
    initialize(rank,2,rendezvous)
    try:
        actor,private,snapshot=setup();slot=PrivateProbeSlot(actor,private)
        optimizer=torch.optim.Adam(private.parameters(),lr=1e-4)
        layout=PrivateIdentity.from_private(private,state_sha256='a'*64).layout_sha256
        identities=[]
        for i in range(100):
            identity=PrivateIdentity(('gpt',0,i+1),f'{i+1:064x}',layout)
            if private_owner(identity,2)==rank:identities.append(identity)
            if len(identities)==2:break
        assert len(identities)==2
        with DistributedProbeStore(directory,run_identity='retention-fixture') as store:
            def save(step,updates):
                return store.commit(shared_state=dict(model={'weight':torch.tensor([float(step)])},optimizer={}),
                    rng_state={'torch':torch.get_rng_state(),'rank':rank},
                    sampler_state={'next_global_batch':step,'rank':rank},private_updates=updates)
            updates=[]
            for identity in identities:
                fresh=CheckpointProbes(snapshot,model_dim=128,latent_tokens=2,seed=71,state_key=identity.state_key)
                slot.stage(fresh.state_dict(),state_key=identity.state_key)
                updates.append(store.snapshot_slot(identity,slot,optimizer,visits=0))
            save(0,updates)
            initial=store.retain('research-initial')
            assert store.retain('live-reader',manifest_ref=initial)==initial
            reader=PrivateProbeStore(Path(directory)/'global',read_only=True)
            old=reader.restore(manifest_ref=initial)
            old_index=old['sampler']['ranks'][rank]['index']
            with PrivateProbeStore(store.shard.directory,read_only=True) as shard_reader:
                old_versions=torch.load(shard_reader._verify(old_index),weights_only=False)['private_versions']
                old_payloads={k:torch.load(shard_reader._verify(v),weights_only=False) for k,v in old_versions.items()}
            for step in range(1,5):
                updates=[]
                for identity in identities:
                    count=store.stage(identity,slot,optimizer)
                    for parameter in slot.private.parameters():parameter.grad=torch.full_like(parameter,.001)
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
                    updates.append(store.snapshot_slot(identity,slot,optimizer,visits=count+1))
                if step==2:
                    previous=(Path(directory)/'global/CURRENT.json').read_bytes()
                    original=store._prepare_index
                    if rank==1:
                        def fail(updates):raise OSError('rank publish interrupted before global commit')
                        store._prepare_index=fail
                    with pytest.raises(RuntimeError,match='transaction aborted'):save(step,updates)
                    store._prepare_index=original
                    assert (Path(directory)/'global/CURRENT.json').read_bytes()==previous
                    store.restore()
                    # The failed private files remain orphaned; recreate this
                    # visit from its committed model/Adam, never directory order.
                    updates=[]
                    for identity in identities:
                        count=store.stage(identity,slot,optimizer)
                        for parameter in slot.private.parameters():parameter.grad=torch.full_like(parameter,.001)
                        optimizer.step();optimizer.zero_grad(set_to_none=True)
                        updates.append(store.snapshot_slot(identity,slot,optimizer,visits=count+1))
                save(step,updates)
            assert not (store.shard.directory/'CURRENT.json').exists()
            # Only the latest pending version for an identity survives GC;
            # coalescing storage must not confuse this with committed state.
            count=store.stage(identities[0],slot,optimizer)
            abandoned=store.snapshot_slot(identities[0],slot,optimizer,visits=count)
            pending=store.snapshot_slot(identities[0],slot,optimizer,visits=count)
            foreign=store.shard.directory/'notes.pt';foreign.write_bytes(b'not store-owned')
            with pytest.raises(RuntimeError,match='barrier'):store.gc_plan()
            plan=store.gc_plan(at_global_barrier=True)
            namespace=f'private/rank{rank:04d}'
            local={r['file'] for r in plan['candidates'] if r['namespace']==namespace}
            assert abandoned['file'] in local and pending['file'] not in local
            assert old_index['file'] not in local
            assert all(v['file'] not in local for v in old_versions.values())
            assert plan['reclaimable_bytes']>0
            # Late readers invalidate a reviewed plan before any deletion.
            store.retain('late-reader')
            with pytest.raises(RuntimeError,match='stale or altered'):store.apply_gc(plan,at_global_barrier=True)
            store.release_pin('late-reader')
            plan=store.gc_plan(at_global_barrier=True)
            altered=copy.deepcopy(plan)
            if rank==1:altered['candidates'][0]['file']='../../outside-owned-store'
            with pytest.raises(RuntimeError,match='stale or altered'):
                store.apply_gc(altered,at_global_barrier=True)
            # A protected rank index must verify before any rank deletes data.
            if rank==1:
                path=store.shard._path(store._index_ref);original_bytes=path.read_bytes()
                damaged=bytearray(original_bytes);damaged[-1]^=1;path.write_bytes(damaged)
            dist.barrier()
            with pytest.raises(RuntimeError,match='planning failed'):
                store.gc_plan(at_global_barrier=True)
            if rank==1:path.write_bytes(original_bytes)
            dist.barrier()
            plan=store.gc_plan(at_global_barrier=True)
            # A shard deletion failure may leave garbage, never broken roots.
            original=store._delete_gc_candidates
            if rank==1:
                def fail_delete(candidates):raise OSError('rank shard deletion failure')
                store._delete_gc_candidates=fail_delete
            with pytest.raises(RuntimeError,match='transaction aborted'):store.apply_gc(plan,at_global_barrier=True)
            store._delete_gc_candidates=original
            assert (store.shard.directory/pending['file']).exists()
            recovered=store.restore();assert recovered['sampler']['next_global_batch']==4
            for identity in identities:assert store.stage(identity,slot,optimizer)==4
            # Explicit reader pin keeps the complete old shared AND private
            # state usable after partial and subsequently successful collection.
            plan=store.gc_plan(at_global_barrier=True)
            collected=store.apply_gc(plan,at_global_barrier=True)
            assert all(not (Path(directory)/r['namespace']/r['file']).exists() for r in collected['deleted'])
            assert foreign.read_bytes()==b'not store-owned'
            assert_tree_equal(reader.restore(manifest_ref=initial)['shared'],old['shared'])
            with PrivateProbeStore(store.shard.directory,read_only=True) as shard_reader:
                for token,version in old_versions.items():
                    assert_tree_equal(torch.load(shard_reader._verify(version),weights_only=False),old_payloads[token])
            store.release_pin('research-initial')
            still=store.gc_plan(at_global_barrier=True)
            assert initial['file'] in still['protected']['global']
            reader.close();store.release_pin('live-reader')
            final=store.gc_plan(at_global_barrier=True)
            assert sum(n.startswith('manifest-') for n in final['protected']['global'])==2
            assert initial['file'] in {r['file'] for r in final['candidates'] if r['namespace']=='global'}
            store.apply_gc(final,at_global_barrier=True)
            with pytest.raises(FileNotFoundError):
                store.global_store._manifest_and_reference(initial)
            assert store.restore()['sampler']['next_global_batch']==4
            assert not (store.shard.directory/'CURRENT.json').exists()
            (Path(directory)/f'retention-rank{rank}.json').write_text(json.dumps({'passed':True}))
    finally:dist.destroy_process_group()


def test_collective_gc_retains_global_roots_across_failures_and_readers(tmp_path):
    directory=tmp_path/'store'
    mp.spawn(worker,args=(directory,tmp_path/'rendezvous'),nprocs=2,join=True)
    assert all(json.loads((directory/f'retention-rank{r}.json').read_text())['passed'] for r in range(2))
