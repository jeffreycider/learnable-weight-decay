import ast,copy,json,types
from pathlib import Path
import pytest
import torch
from test_soap46_online import bare,initial_state
from rewriter.soap46_checkpoint import payload,restore,identity
from rewriter.soap46_prefix import fork_payload,fork_directory


def setup(variant='outer-only',lr=1e-5,decay=.1,warmup=500):
    model=torch.nn.Linear(3,2,bias=False);h=bare(model,lr=lr,variant=variant)
    h.stop=2900;h.warmup_steps=warmup;h.meta_ramp_steps=500;h.residual_decay=decay
    p=model.weight;opt=torch.optim.SGD([p],lr=.0375);opt.param_groups[0].update(mu=.85,initial_lr=.0375)
    opt.state[p]=initial_state(p);opt.step_count=0
    def base_step():
        h.propose(p,opt.state[p],opt.param_groups[0],opt.step_count,False,False);opt.step_count+=1
    opt.step=base_step
    tree=ast.parse(Path('baseline/record46/upstream.py').read_text());scope={'torch':torch,'SF_MODE':0}
    exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='EMA_Nesterov'],type_ignores=[]),'<actual record46 wrapper>','exec'),scope)
    wrapper=scope['EMA_Nesterov']([p],[opt],lookahead_stepsize=.3,lookahead_ema=.99,prefill_steps=300,rest_steps=1950)
    h.scope.update(optimizers=[wrapper],optimizer2=opt,SEED=3,mbs=64,_tailema=[torch.randn_like(p)],training_time=1.,t0=0.)
    return h


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b) and a.dtype==b.dtype and a.stride()==b.stride()
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif hasattr(a,'__dataclass_fields__'):
        for k in a.__dataclass_fields__:equal(getattr(a,k),getattr(b,k))
    else:assert a==b


@pytest.fixture
def prefix(monkeypatch,tmp_path):
    torch.set_num_threads(1);torch.manual_seed(964)
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda:1)
    monkeypatch.setattr(torch.distributed,'all_reduce',lambda x:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda:torch.tensor([2,3],dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda x:None)
    h=setup();p=h.scope['model'].weight;wrapper=h.scope['optimizers'][0]
    for step in range(500):
        wrapper.nesterov_step();p.grad=torch.randn_like(p);h.consume(step);wrapper.step();h.scope['model'].zero_grad(set_to_none=True)
    state=payload(h.scope,h,500);parent=tmp_path/'step0500';parent.mkdir()
    torch.save(state,parent/'rank0.pt')
    keys=['completed_steps','world_size','variant','seed','meta_lr','meta_clip','meta_betas','warmup_steps','warmup_meta','meta_ramp_steps','residual_decay','base_lr','cwd','stop_step','microbatch']
    (parent/'complete.json').write_text(json.dumps({k:state[k] for k in keys}))
    return parent,torch.load(parent/'rank0.pt',weights_only=False)


def next_two(h):
    wrapper=h.scope['optimizers'][0];model=h.scope['model']
    for step in (500,501):
        wrapper.nesterov_step();inputs=torch.randn(4,3);target=torch.randn(4,2)
        loss=(model(inputs)-target).square().mean()*h.scope['batch_size'];loss.backward()
        h.consume(step);wrapper.step();model.zero_grad(set_to_none=True)
    return payload(h.scope,h,502)


def test_unchanged_fork_save_restore_and_next_real_gradient_exact(prefix,tmp_path):
    parent,saved=prefix;out=tmp_path/'unchanged'
    manifest=fork_directory(parent,out,role='learned',variant='outer-only',seed=3,implementation_sha256=identity(),expected_world_size=1,expected_matrices=1)
    child=torch.load(out/'rank0.pt',weights_only=False)
    for key in saved:equal(saved[key],child[key])
    a=setup();restore(a.scope,a,copy.deepcopy(saved));expected=next_two(a)
    b=setup();restore(b.scope,b,child);actual=next_two(b)
    for key in ('model','base_optimizers','wrapper','wrapper_clock','muon_step_count','tail_ema','hyper','torch_rng','cuda_rng','data_cursor_batches'):equal(expected[key],actual[key])
    assert manifest['declared_future_changes']=={} and manifest['training_launched'] is False
    assert actual['hyper']['residual_updates']==1


def test_frozen_role_keeps_moments_pending_credit_and_zero_x(prefix,tmp_path):
    parent,saved=prefix;out=tmp_path/'frozen'
    manifest=fork_directory(parent,out,role='frozen',variant='outer-only',seed=3,implementation_sha256=identity(),expected_world_size=1,expected_matrices=1)
    child=torch.load(out/'rank0.pt',weights_only=False)
    assert child['warmup_steps']==child['hyper']['warmup_steps']==2900
    equal(saved['hyper']['optimizer'],child['hyper']['optimizer']);equal(saved['hyper']['entries'],child['hyper']['entries'])
    h=setup(warmup=2900);restore(h.scope,h,child);after=next_two(h)
    assert after['hyper']['meta_steps']==500 and after['hyper']['residual_updates']==0
    assert all(torch.count_nonzero(e[k])==0 for e in after['hyper']['entries'].values() for k in ('xl','xr'))
    assert manifest['declared_future_changes']=={'warmup_steps':{'before':500,'after':2900}}


def test_learned_fork_changes_only_declared_future_metadata(prefix):
    _,state=prefix
    child=fork_payload(state,role='learned',variant='outer-only',seed=3,parent_sha256='a'*64,implementation_sha256=identity(),meta_lr=1e-4,residual_decay=.03)
    assert child['meta_lr']==1e-4 and child['residual_decay']==child['hyper']['residual_decay']==.03
    assert state['meta_lr']==1e-5 and state['residual_decay']==.1
    for key in ('model','base_optimizers','wrapper','wrapper_clock','torch_rng','cuda_rng'):equal(state[key],child[key])
    equal(state['hyper']['optimizer'],child['hyper']['optimizer']);equal(state['hyper']['entries'],child['hyper']['entries'])


@pytest.mark.parametrize('change,match',[
    ({'seed':4},'seed'),({'variant':'inner-outer'},'variant'),({'completed_steps':501},'completed_steps'),
    ({'data_cursor_batches':499},'data_cursor'),({'implementation_sha256':'bad'},'implementation'),
])
def test_reject_wrong_parent(prefix,change,match):
    _,state=prefix
    with pytest.raises(ValueError,match=match):fork_payload({**state,**change},role='learned',variant='outer-only',seed=3,parent_sha256='a'*64,implementation_sha256=identity())


def test_reject_nonzero_x_and_missing_credit_and_existing_destination(prefix,tmp_path):
    parent,state=prefix;name=next(iter(state['hyper']['entries']))
    dirty=copy.deepcopy(state);dirty['hyper']['entries'][name]['xl'].data[0,0]=1e-30
    with pytest.raises(ValueError,match='exactly zero'):fork_payload(dirty,role='learned',variant='outer-only',seed=3,parent_sha256='a'*64,implementation_sha256=identity())
    dirty=copy.deepcopy(state);dirty['hyper']['entries'][name]['record']=None
    with pytest.raises(ValueError,match='pending credit'):fork_payload(dirty,role='learned',variant='outer-only',seed=3,parent_sha256='a'*64,implementation_sha256=identity())
    with pytest.raises(FileExistsError):fork_directory(parent,parent,role='frozen',variant='outer-only',seed=3,implementation_sha256=identity())


def test_reject_incomplete_or_duplicate_distributed_shards(prefix,tmp_path):
    parent,state=prefix;state=copy.deepcopy(state);state['world_size']=2
    header=json.loads((parent/'complete.json').read_text());header['world_size']=2
    (parent/'complete.json').write_text(json.dumps(header));torch.save(state,parent/'rank0.pt')
    with pytest.raises(FileNotFoundError):fork_directory(parent,tmp_path/'missing',role='learned',variant='outer-only',seed=3,implementation_sha256=identity(),expected_world_size=2,expected_matrices=2)
    assert not (tmp_path/'missing').exists()
    torch.save(state,parent/'rank1.pt')
    with pytest.raises(ValueError,match='duplicated controller ownership'):fork_directory(parent,tmp_path/'duplicate',role='learned',variant='outer-only',seed=3,implementation_sha256=identity(),expected_world_size=2,expected_matrices=2)
    assert not (tmp_path/'duplicate').exists()


def test_frozen_role_refuses_other_schedule_changes(prefix):
    _,state=prefix
    with pytest.raises(ValueError,match='retains nominal'):fork_payload(state,role='frozen',variant='outer-only',seed=3,parent_sha256='a'*64,implementation_sha256=identity(),meta_lr=1e-4)
    bad=copy.deepcopy(state);next(iter(bad['hyper']['optimizer']['state'].values()))['step'].fill_(497)
    with pytest.raises(ValueError,match='moments/counter'):fork_payload(bad,role='learned',variant='outer-only',seed=3,parent_sha256='a'*64,implementation_sha256=identity())
