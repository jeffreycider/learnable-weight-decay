import json
from types import SimpleNamespace
import pytest
from rewriter.mixed_domain import MixedBankCatalog,BalancedDomainBatches
from rewriter.multibank import file_sha


def catalog(tmp_path):
    items=[];readers={}
    for name,n,fit in [('gpt',20,16),('cifar10',10,8)]:
        p=tmp_path/f'{name}.json';p.write_text(json.dumps({'n':n}))
        reader=SimpleNamespace(entries=[{'local':i} for i in range(1,n+1)],manifest={'split':{'train':list(range(1,fit+1)),'audit':list(range(fit+1,n+1))}},read=lambda i,**kw:SimpleNamespace(completed_steps=i,snapshot=f'snapshot{i}',model={'unchanged':i}))
        readers[name]=lambda path,r=reader:r
        items.append(dict(name=name,index=str(p),index_sha256=file_sha(p),weight=.5))
    index=tmp_path/'mixed.json';index.write_text(json.dumps(dict(schema='chisel_mixed_domains_v1',domains=items)))
    return MixedBankCatalog(index,readers)


def test_namespace_never_relabels_child_clock(tmp_path):
    c=catalog(tmp_path);r=c.read(21)
    assert r.state_id==21 and r.domain=='cifar10' and r.local_state_id==r.completed_steps==1
    assert r.model=={'unchanged':1} and c.read(1).domain=='gpt'
    assert set(c.split['train']).isdisjoint(c.split['audit'])


def test_balanced_domains_global_weight_ddp_and_exact_restart(tmp_path):
    c=catalog(tmp_path);sampler=BalancedDomainBatches(c,{'gpt':4,'cifar10':2},world_size=2)
    visited=[]
    for b in range(sampler.steps_per_epoch):
        global_rows,clock=sampler.batch(0,b);visited+=global_rows
        joined=sum([sampler.batch(0,b,rank=r)[0] for r in range(2)],[])
        assert sorted(x['state_id'] for x in joined)==sorted(x['state_id'] for x in global_rows)
        for domain in ('gpt','cifar10'):
            assert sum(r['state_weight'] for r in joined if r['domain']==domain)==.5
        assert sampler.batch(0,b)==BalancedDomainBatches(c,{'gpt':4,'cifar10':2},world_size=2).batch(0,b)
    assert sorted(r['state_id'] for r in visited if r['domain']=='gpt')==list(range(1,17))
    assert sorted(r['state_id'] for r in visited if r['domain']=='cifar10')==list(range(21,29))
    assert sampler.batch(1,0)[1]['cifar10']['completed_cycles']==1


def test_smaller_domain_exposure_explicit_not_one_percent(tmp_path):
    c=catalog(tmp_path);sampler=BalancedDomainBatches(c,{'gpt':2,'cifar10':2})
    assert sampler.steps_per_epoch==8
    assert sampler.batch(1,0)[1]['cifar10']['completed_cycles']==2
    with pytest.raises(ValueError):BalancedDomainBatches(c,{'gpt':3,'cifar10':2})
