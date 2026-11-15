"""Two-rank CPU fork preserves one complete transaction; never mutates parent."""
import copy,json,random,hashlib
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rewriter.distributed_probe_store import DistributedProbeStore,private_owner
from rewriter.distributed_probe_fork import fork_distributed_checkpoint
from rewriter.private_probe_store import PrivateIdentity,PrivateProbeStore
from rewriter.private_probe_slot import PrivateProbeSlot
from tests.test_distributed_probe_store import initialize,assert_tree_equal
from tests.test_coordinate_tied_frames import actor_fixture


def fingerprint(path):
    return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file()}


def create_worker(rank,path,rendezvous):
    initialize(rank,2,rendezvous)
    try:
        snapshot,actor,private=actor_fixture();slot=PrivateProbeSlot(actor,private)
        opt=torch.optim.Adam(private.parameters(),lr=1e-4);fresh=copy.deepcopy(opt.state_dict())
        shared=torch.optim.Adam(actor.parameters(),lr=2e-4)
        layout=PrivateIdentity.from_private(private,state_sha256='a'*64).layout_sha256
        identities=[]
        for i in range(100):
            ident=PrivateIdentity(('gpt',i//20,i%20+1),f'{i+1:064x}',layout)
            if private_owner(ident,2)==rank:identities.append(ident)
            if len(identities)==2:break
        with DistributedProbeStore(path,run_identity='parent') as store:
            for tick in (1,2):
                versions=[]
                for identity in identities:
                    if tick==1:
                        new=actor.make_private(snapshot,state_key=identity.state_key,seed=42)
                        slot.stage(new.state_dict(),state_key=identity.state_key);opt.load_state_dict(copy.deepcopy(fresh))
                    else:assert store.stage(identity,slot,opt)==tick-1
                    for p in private.parameters():p.grad=torch.full_like(p,.003*tick)
                    opt.step();opt.zero_grad(set_to_none=True)
                    versions.append(store.snapshot_slot(identity,slot,opt,visits=tick))
                for p in actor.parameters():p.grad=torch.full_like(p,.002*tick)
                shared.step();shared.zero_grad(set_to_none=True)
                torch.manual_seed(400+rank+tick);random.seed(500+rank+tick)
                store.commit(shared_state={'model':actor.state_dict(),'optimizer':shared.state_dict()},
                    rng_state={'torch':torch.get_rng_state(),'python':random.getstate(),'cuda':torch.tensor([rank,tick],dtype=torch.uint8)},
                    sampler_state={'next_global_batch':tick,'rank_cursor':rank*100+tick,'duals':{rank:3.5*tick},'visits':{v['token']:tick for v in versions}},private_updates=versions)
            # A pending payload is deliberately not part of CURRENT.
            store.snapshot_slot(identities[-1],slot,opt,visits=2)
    finally:dist.destroy_process_group()


def fork_worker(rank,world,source,dest,reference,rendezvous,mode):
    initialize(rank,world,rendezvous)
    try:
        import rewriter.distributed_probe_fork as module
        if mode=='interrupt' and rank==1:
            def fail(*args):raise OSError('injected private-copy failure')
            module._copy_private=fail
        kwargs=dict(parent_manifest_ref=reference,source_run_identity='parent',destination_run_identity='child',
                    provenance={'source_code_id':'old-source','destination_code_id':'qualified-new-source','reason':'backend-only explicit fork'})
        if mode in ('interrupt','retry','world','tamper','live'):
            with pytest.raises((RuntimeError,ValueError)):
                fork_distributed_checkpoint(source,dest,**kwargs)
            assert not (Path(dest)/'global/CURRENT.json').exists()
        else:
            result=fork_distributed_checkpoint(source,dest,**kwargs)
            assert result['source_generation']==2 and result['destination_generation']==1
            with DistributedProbeStore(source,run_identity='parent') as parent,DistributedProbeStore(dest,run_identity='child') as child:
                a,b=parent.restore(),child.restore()
                for field in ('shared','rng','sampler'):assert_tree_equal(a[field],b[field])
                assert parent._versions==child._versions
                assert len(list(child.shard.directory.glob('private-*.pt')))==2
                assert len(list(parent.shard.directory.glob('private-*.pt')))==5
                torch.set_rng_state(a['rng']['torch']);ra=torch.randn(8)
                torch.set_rng_state(b['rng']['torch']);rb=torch.randn(8);assert torch.equal(ra,rb)
                random.setstate(a['rng']['python']);ra=random.random()
                random.setstate(b['rng']['python']);assert ra==random.random()
                snapshot,actor,private=actor_fixture()
                def next_shared(saved):
                    module=copy.deepcopy(actor);module.load_state_dict(saved['model'])
                    optimizer=torch.optim.Adam(module.parameters(),lr=999);optimizer.load_state_dict(saved['optimizer'])
                    for p in module.parameters():p.grad=torch.ones_like(p)*.007
                    optimizer.step();return module.state_dict(),optimizer.state_dict()
                assert_tree_equal(next_shared(a['shared']),next_shared(b['shared']))
                for token,version in parent._versions.items():
                    identity=PrivateIdentity(tuple(version['identity']['state_key']),version['identity']['state_sha256'],version['identity']['layout_sha256'])
                    pa=parent.shard.read_private(identity,version=version);pb=child.shard.read_private(identity,version=child._versions[token])
                    assert_tree_equal(pa,pb)
                    def next_private(payload):
                        module=actor.make_private(snapshot,state_key=identity.state_key,seed=5);module.load_state_dict(payload['private'])
                        optimizer=torch.optim.Adam(module.parameters(),lr=999);optimizer.load_state_dict(payload['optimizer'])
                        for p in module.parameters():p.grad=torch.ones_like(p)*.009
                        optimizer.step();return module.state_dict(),optimizer.state_dict()
                    assert_tree_equal(next_private(pa),next_private(pb))
            if rank==0:(Path(dest)/'test-passed').write_text('exact numerical restore and next Adam steps')
    finally:dist.destroy_process_group()


@pytest.fixture(scope='module')
def parent_store(tmp_path_factory):
    root=tmp_path_factory.mktemp('fork');source=root/'parent'
    mp.spawn(create_worker,args=(str(source),root/'create-rendezvous'),nprocs=2,join=True)
    return source,json.loads((source/'global/CURRENT.json').read_text())


def run_fork(source,reference,dest,mode,world=2):
    mp.spawn(fork_worker,args=(world,str(source),str(dest),reference,dest.parent/(dest.name+'-rendezvous'),mode),nprocs=world,join=True)


def test_complete_fork_preserves_every_owner_and_exact_next_update(parent_store,tmp_path):
    source,ref=parent_store;before=fingerprint(source);dest=tmp_path/'fork'
    run_fork(source,ref,dest,'success')
    assert fingerprint(source)==before
    meta=json.loads((dest/'FORK.json').read_text());assert meta['parent_manifest_ref']==ref
    assert meta['destination_run_identity']=='child' and meta['world_size']==2
    assert not list(dest.glob('FORK_PENDING*'))
    assert not any((dest/'private').rglob('CURRENT.json'))
    with pytest.raises(Exception):run_fork(source,ref,dest,'success')


def test_interrupted_rank_stays_unpublished_and_retry_rejects_pending(parent_store,tmp_path):
    source,ref=parent_store;before=fingerprint(source);dest=tmp_path/'fork'
    run_fork(source,ref,dest,'interrupt');assert fingerprint(source)==before
    assert len(list(tmp_path.glob('.fork.fork-pending-*')))==1
    run_fork(source,ref,dest,'retry');assert fingerprint(source)==before


def test_world_change_and_live_source_rejected(parent_store,tmp_path):
    source,ref=parent_store;before=fingerprint(source)
    run_fork(source,ref,tmp_path/'wrongworld','world',world=1)
    with PrivateProbeStore(source/'global'):
        run_fork(source,ref,tmp_path/'live','live')
    assert fingerprint(source)==before


def test_pinned_reference_and_private_tamper_rejected(parent_store,tmp_path):
    source,ref=parent_store;bad=copy.deepcopy(ref);bad['sha256']='0'*64
    run_fork(source,bad,tmp_path/'badref','tamper')
    with PrivateProbeStore(source/'global',read_only=True) as global_reader:state=global_reader.restore()
    indexref=state['sampler']['ranks'][1]['index'];shard=source/'private/rank0001'
    with PrivateProbeStore(shard,read_only=True) as reader:index=torch.load(reader._verify(indexref),weights_only=False)
    version=next(iter(index['private_versions'].values()));path=shard/version['file'];original=path.read_bytes()
    try:
        corrupted=bytearray(original);corrupted[-1]^=1;path.write_bytes(corrupted);before=fingerprint(source)
        run_fork(source,ref,tmp_path/'tampered','tamper');assert fingerprint(source)==before
    finally:path.write_bytes(original)
