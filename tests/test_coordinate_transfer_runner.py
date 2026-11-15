"""Actual heldout routine, paired actor control, CPU finite-logit fixture."""
import copy,json,datetime
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
from scripts.qualify_coordinate_transfer import execute_pairs,quartiles,TraceContexts
from scripts.train_coordinate_chisel import describe_states
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.private_probe_store import PrivateIdentity
from rewriter.distributed_probe_store import _digest
from tests.test_coordinate_training_driver import ToyContexts,ToyReader,recipe
from tests.test_distributed_probe_store import assert_tree_equal

@pytest.fixture
def setup(tmp_path):
    old=torch.get_num_threads();torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(tmp_path/'gloo'),rank=0,world_size=1,timeout=datetime.timedelta(seconds=60))
    yield
    dist.destroy_process_group();torch.set_num_threads(old)


def fixture():
    r=recipe();r['architecture']['frame_mode']='tied_orthonormal'
    torch.manual_seed(41);initial=CoordinateChisel(**r['architecture']);trained=copy.deepcopy(initial)
    with torch.no_grad():
        for h in trained.output_heads:h.weight.normal_(std=.005)
    contexts=ToyContexts();contexts.catalog.readers['cifar10']=ToyReader('cifar10',8,4)
    layouts={};cases=[];trained.requires_grad_(False)
    for d in ('gpt','cifar10'):
        for uid in quartiles(contexts.catalog.readers[d].manifest['split']['audit']):
            support=contexts.load(d,uid,purpose='private_support',shared_actor=trained)
            query=contexts.load(d,uid,purpose='private_query',role='audit',shared_actor=trained,gpt_sequences=16)
            private=trained.make_private(support['snapshot'],state_key=(d,'layout'),seed=42)
            layouts[d]=PrivateIdentity.from_private(private,state_sha256='0'*64).layout_sha256
            entry=contexts.catalog.readers[d].entries[uid-1]
            cases.append(dict(domain=d,local_id=uid,state_sha256=entry['sha256'],support_ids=support['ids'],query_ids=query['ids']))
    specs=describe_states(contexts.catalog,layouts)
    plan=dict(cases=cases,arms=['trained','initialized'],private_seed=42,private_optimizer=r['private_optimizer'],dual_lr=r['dual_lr'],objectives=r['objectives'],support={'steps':2},query={'gpt_sequences':16},parent={'review_update':211})
    return plan,trained,initial,contexts,specs


def states(output):
    result={}
    for arm in ('trained','initialized'):
        for p in (output/arm).glob('*.pt'):
            d=torch.load(p,weights_only=False);key=(arm,p.name.split('-')[1],p.name.split('-')[2]);result[key]=d
    return result


def test_actual_heldout_pair_exact_control_fresh_clocks_and_query_isolation(setup,tmp_path):
    plan,trained,initial,contexts,specs=fixture();events=[];before=_digest(trained.state_dict())
    # Trained fixture has exactly initial blocks: paired arms must match bitwise.
    summary=execute_pairs(plan,trained,initial,contexts,specs,torch.device('cpu'),tmp_path/'first',write=events.append)
    assert before==_digest(trained.state_dict())
    assert summary['compute']==dict(toy_forwards=64,toy_backwards=32)
    first=states(tmp_path/'first');assert len(first)==16
    for (arm,domain,uid),state in first.items():
        if arm=='trained':
            other=first[('initialized',domain,uid)]
            for key in ('private','optimizer','dual','support_ids','state_sha256'):assert_tree_equal(state[key],other[key])
            assert all(int(s['step'])==2 for s in state['optimizer']['state'].values())
    assert len([x for x in events if x['event']=='support_backward'])==32
    assert len([x for x in events if x['event']=='query'])==32
    class AlteredQueries(ToyContexts):
        def measure(self,*args,learning,**kwargs):
            row=super().measure(*args,learning=learning,**kwargs)
            if not learning:row['mean_kl']+=123.;row['objective']+=567.
            return row
    changed=AlteredQueries();changed.catalog=contexts.catalog
    execute_pairs(plan,trained,initial,changed,specs,torch.device('cpu'),tmp_path/'second')
    second=states(tmp_path/'second')
    for key,a in first.items():
        b=second[key]
        for field in ('private','optimizer','dual','support_ids','state_sha256'):assert_tree_equal(a[field],b[field])
        assert a['query']!=b['query']


def test_pair_rejects_bad_state_or_panel_and_shards_without_repeats(setup,tmp_path):
    plan,trained,initial,contexts,specs=fixture();bad=copy.deepcopy(plan);bad['cases'][0]['state_sha256']='0'*64
    with pytest.raises(ValueError,match='pinned held-out'):execute_pairs(bad,trained,initial,contexts,specs,torch.device('cpu'),tmp_path/'bad')
    bad=copy.deepcopy(plan);bad['cases'][0]['support_ids'][0]=-1
    with pytest.raises(ValueError,match='panel identity'):execute_pairs(bad,trained,initial,contexts,specs,torch.device('cpu'),tmp_path/'panel')
    seen=[]
    for shard in (0,1):
        ctx=ToyContexts();ctx.catalog=contexts.catalog
        result=execute_pairs(plan,trained,initial,ctx,specs,torch.device('cpu'),tmp_path/f'shard{shard}',shard=shard,shards=2)
        seen.extend((c['domain'],c['local_id']) for c in result['cases'])
    assert len(seen)==len(set(seen))==8


def test_cpu_prepare_pins_closed_parent_and_rejects_unclosed_or_tampered_inputs(tmp_path,monkeypatch):
    import hashlib
    import scripts.qualify_coordinate_transfer as module
    from rewriter.private_probe_store import PrivateProbeStore
    plan=json.loads(Path('configs/coordinate_transfer_qualification.json').read_text())
    _,trained,initial,contexts,_=fixture();architecture=recipe()['architecture'];architecture['frame_mode']='tied_orthonormal'
    data=tmp_path/'data';data.mkdir();torch.save(dict(fitting_indices=torch.arange(2048),heldout_indices=torch.arange(10000,12048)),data/'chisel_image_split.pt')
    index=tmp_path/'index.json';index.write_text('{}')
    base=tmp_path/'base.json';base.write_text(json.dumps(dict(mixed_index=str(index),cifar_data=str(data))))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=dict(architecture=architecture,base_recipe=str(base),base_recipe_sha256=sha(base),mixed_index_sha256=sha(index),**{k:plan[k] for k in ('private_optimizer','objectives','dual_lr','private_seed')})
    metadata={'recipe':r};identity=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    parent=tmp_path/'parent';parent.mkdir();(parent/'metadata.json').write_text(json.dumps(metadata))
    plan['initialization']['architecture']=architecture;draft=tmp_path/'draft.json';draft.write_text(json.dumps(plan))
    monkeypatch.setattr(module,'catalog_for',lambda base:contexts.catalog)
    cache={}
    for uid in contexts.catalog.readers['gpt'].manifest['split']['audit']:
        for role in ('train','audit'):cache[(uid,role)]={'sequence_ids':list(range((0 if role=='train' else 100000)+uid*1000,(0 if role=='train' else 100000)+uid*1000+128))}
    monkeypatch.setattr(module,'MultiTeacherCacheReader',lambda *a,**kw:type('Cache',(),{'expected':cache})())
    with PrivateProbeStore(parent/'joint_store/global') as store:
        state=dict(next_global_batch=211,run_identity=identity,world_size=1,ranks=[dict(rank=0,sampler=dict(next_global_batch=211,stop_reason=None))])
        store.commit(shared_state=dict(model=trained.state_dict(),optimizer={},rng={}),sampler_state=state,private_updates=[],expected_generation=0)
        with pytest.raises(ValueError,match='not closed'):module.prepare(draft,parent,tmp_path/'bad')
        state['ranks'][0]['sampler']['stop_reason']='review_boundary'
        store.commit(shared_state=dict(model=trained.state_dict(),optimizer={},rng={}),sampler_state=state,private_updates=[],expected_generation=1)
    before=(parent/'joint_store/global/CURRENT.json').read_bytes();rng=torch.get_rng_state().clone()
    result=module.prepare(draft,parent,tmp_path/'ready')
    assert result['status']=='pinned_ready_not_launched' and len(result['cases'])==8
    assert result['parent']['global_manifest_ref']==json.loads(before)
    assert torch.equal(rng,torch.get_rng_state())
    assert (parent/'joint_store/global/CURRENT.json').read_bytes()==before
    assert_tree_equal(torch.load(result['parent']['actor_file'],weights_only=False)['model'],trained.state_dict())
    with pytest.raises(FileExistsError):module.prepare(draft,parent,tmp_path/'ready')
    base.write_text('{}')
    with pytest.raises(ValueError,match='base recipe changed'):module.prepare(draft,parent,tmp_path/'changed')


def test_protocol_cannot_turn_query_into_support_or_change_declared_budget():
    from scripts.qualify_coordinate_transfer import validate_protocol
    p=json.loads(Path('configs/coordinate_transfer_qualification.json').read_text());validate_protocol(p)
    q=copy.deepcopy(p);q['query']['used_for_updates_or_selection']=True
    with pytest.raises(ValueError,match='query feedback'):validate_protocol(q)
    q=copy.deepcopy(p);q['support']['steps']=16
    with pytest.raises(ValueError,match='support budget'):validate_protocol(q)
