"""Explicit domain routing and balanced state sampling for Chisel pretraining.

This layer never interprets a GPT step as a CIFAR seed, alters a child's model
or cache, or pools image losses with token losses. Each domain's panel loss is
normalized by that domain's contexts before the fixed state weight is applied.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from .multibank import file_sha


@dataclass(frozen=True)
class DomainRecord:
    state_id: int
    domain: str
    local_state_id: int
    record: object

    @property
    def training_state_id(self):return self.state_id

    def __getattr__(self,name):
        return getattr(object.__getattribute__(self,'record'),name)


class MixedBankCatalog:
    """Pinned ordered domain indexes, with explicit adapter factories."""
    def __init__(self,path,reader_factories):
        self.path=Path(path);self.index=json.loads(self.path.read_text())
        if self.index.get('schema')!='chisel_mixed_domains_v1':raise ValueError('unsupported mixed-domain index')
        self.readers={};self.offsets={};self.entries=[];self.split={'train':[],'audit':[]};self.weights={}
        offset=0
        for item in self.index['domains']:
            name=item['name'];index=Path(item['index'])
            if name in self.readers or name not in reader_factories:raise ValueError('duplicate or unsupported domain')
            if file_sha(index)!=item['index_sha256']:raise ValueError('domain index identity differs')
            reader=reader_factories[name](index);n=len(reader.entries)
            local_ids=reader.manifest['split']['train']+reader.manifest['split']['audit']
            if len(local_ids)!=n or set(local_ids)!=set(range(1,n+1)):raise ValueError('domain split must partition all local IDs')
            weight=float(item['weight'])
            if not 0<weight<=1:raise ValueError('positive finite domain weight required')
            self.readers[name]=reader;self.offsets[name]=(offset,n);self.weights[name]=weight
            for local,entry in enumerate(reader.entries,1):
                self.entries.append(dict(state_id=offset+local,domain=name,local_state_id=local,child_entry=entry))
            for role in self.split:self.split[role].extend(offset+i for i in reader.manifest['split'][role])
            offset+=n
        if not self.readers or abs(sum(self.weights.values())-1)>1e-12:raise ValueError('domain weights must sum to one')
        self.manifest=dict(schema='chisel_mixed_domains_v1',expected_steps=offset,split=self.split,
            domain_weights=self.weights,index=self.index,state_clock='global sample identity only; child clocks unchanged')

    def route(self,state_id):
        if type(state_id) is not int or not 1<=state_id<=len(self.entries):raise IndexError('unknown mixed-domain state')
        row=self.entries[state_id-1];return row['domain'],row['local_state_id']

    def read(self,state_id,**kwargs):
        domain,local=self.route(state_id)
        return DomainRecord(state_id,domain,local,self.readers[domain].read(local,**kwargs))

    def describe_state(self,state_id):
        domain,local=self.route(state_id);child=self.readers[domain]
        details=child.describe_state(local) if hasattr(child,'describe_state') else {}
        return dict(details,state_id=state_id,domain=domain,local_state_id=local)


class BalancedDomainBatches:
    """Deterministic resumable domain cycles; one primary-domain pass per epoch.

    Each domain's batch size divides its fitting-state count and world size.
    Smaller domains cycle independently rather than receiving negligible weight.
    No sampler consumes the process/global RNG or development examples.
    """
    def __init__(self,catalog,batch_counts,*,primary_domain='gpt',world_size=1,seed=500001):
        self.catalog=catalog;self.counts=dict(batch_counts);self.primary=primary_domain;self.world=world_size;self.seed=seed
        if self.counts.keys()!=catalog.readers.keys() or primary_domain not in self.counts or world_size<1:
            raise ValueError('one batch count per domain and valid primary/world required')
        self.ids={name:[i for i in catalog.split['train'] if catalog.route(i)[0]==name] for name in self.counts}
        for name,count in self.counts.items():
            if type(count) is not int or count<1 or count%world_size or len(self.ids[name])%count:
                raise ValueError('domain batch must divide fitting coverage and be divisible by world size')
        self.steps_per_epoch=len(self.ids[primary_domain])//self.counts[primary_domain]

    def batch(self,epoch,batch_index,*,rank=None):
        if epoch<0 or not 0<=batch_index<self.steps_per_epoch or (rank is not None and not 0<=rank<self.world):
            raise ValueError('invalid epoch/batch/rank cursor')
        ordinal=epoch*self.steps_per_epoch+batch_index;rows=[];exposure={}
        for name,count in self.counts.items():
            per_cycle=len(self.ids[name])//count;cycle,within=divmod(ordinal,per_cycle)
            digest=hashlib.sha256(f'{self.seed}:{name}:{cycle}'.encode()).digest()
            order=list(self.ids[name]);random.Random(int.from_bytes(digest[:8],'little')).shuffle(order)
            selected=order[within*count:(within+1)*count]
            if rank is not None:selected=selected[rank::self.world]
            rows.extend(dict(state_id=i,domain=name,state_weight=self.catalog.weights[name]/count) for i in selected)
            exposure[name]=dict(completed_cycles=cycle,batch_in_cycle=within,global_batch_size=count)
        return rows,exposure
