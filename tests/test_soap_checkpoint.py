import io
import types
import numpy as np
import pytest
import torch
from rewriter.soap_checkpoint import loader_position
from rewriter.soap_online import SoapResidualOnline
from rewriter.soap_residual import FrozenMap,committed_proposal
from rewriter.lag_one import LagOneHypergradient


def test_data_resume_matches_original_strict_boundary(tmp_path):
    paths=[];arrays=[]
    for i,n in enumerate([18,19,22]):
        header=np.zeros(256,dtype=np.int32);header[:3]=[20240520,1,n]
        tokens=np.arange(i*100,i*100+n,dtype=np.uint16)
        p=tmp_path/f'{i}.bin';p.write_bytes(header.tobytes()+tokens.tobytes());paths.append(p);arrays.append(torch.tensor(tokens.astype(np.int32)))
    expected=[];batch=4
    for a in arrays:
        pos=0
        while pos+batch+1<len(a):expected.append(a[pos:pos+batch+1]);pos+=batch
    def load(p):return arrays[paths.index(p)]
    for step,want in enumerate(expected):
        remaining=iter(paths);tokens,pos=loader_position(remaining,batch,step,load)
        assert torch.equal(tokens[pos:pos+batch+1],want)
    with pytest.raises(ValueError):loader_position(iter(paths),batch,len(expected),load)


def bare(model):
    h=SoapResidualOnline.__new__(SoapResidualOnline)
    h.scope={'model':model,'batch_size':8};h.lr=1e-4;h.stop=2900;h.learned=True
    h.entries={};h.optimizer=None;h.meta_steps=0;h.metrics={};h.map=committed_proposal
    return h


def test_pending_replay_and_shared_adam_resume_exact(monkeypatch):
    monkeypatch.setattr(torch.distributed,'all_reduce',lambda x:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    model=torch.nn.Linear(4,3,bias=False).double();p=model.weight
    h=bare(model);e={'hl':torch.nn.Parameter(torch.zeros(3,3,dtype=torch.float64)),
                   'hr':torch.nn.Parameter(torch.zeros(4,4,dtype=torch.float64)),'lag':LagOneHypergradient()}
    h.entries[p]=e;h.optimizer=torch.optim.Adam([e['hl'],e['hr']],lr=h.lr,betas=(.9,.99),eps=1e-8)
    for v in [e['hl'],e['hr']]:v.grad=torch.randn_like(v)
    h.optimizer.step();h.meta_steps=1
    record=FrozenMap(p.detach().clone(),torch.randn_like(p),torch.eye(3,dtype=torch.float64),torch.eye(4,dtype=torch.float64),
        torch.tensor(.2,dtype=torch.float64),torch.tensor(.3,dtype=torch.float64),.01,polar_dtype=torch.float64)
    h.install_pending(e,record,1.4455,2100)
    buffer=io.BytesIO();torch.save(h.state_dict(),buffer);buffer.seek(0)
    resumed=bare(model);resumed.load_state_dict(torch.load(buffer,weights_only=False))
    p.grad=torch.randn_like(p)
    h.consume(2101);resumed.consume(2101)
    for name in ['hl','hr']:
        assert torch.equal(h.entries[p][name],resumed.entries[p][name])
        assert torch.equal(h.entries[p][name].grad,resumed.entries[p][name].grad)
    assert not resumed.entries[p]['lag'].pending
    assert resumed.meta_steps==h.meta_steps==2


def test_full_wrapper_and_external_state_checkpoint_roundtrip(monkeypatch):
    from rewriter.soap_checkpoint import payload,restore
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda:1)
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda:torch.zeros(1,dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda _:None)
    def scope():
        model=torch.nn.Linear(4,3,bias=False);opt=torch.optim.SGD(model.parameters(),lr=.03)
        class Wrapper(torch.optim.Optimizer):
            def __init__(self):
                super().__init__(model.parameters(),{});self.inner_optimizer=[opt]
                self.it=2200;self.lookahead_status=False;self.current_lookahead_stepsize=0.
        return {'model':model,'optimizers':[Wrapper()],'optimizer2':types.SimpleNamespace(step_count=2200),
                'head_eigen':types.SimpleNamespace(st={'m':torch.randn(4,4)}),
                'embed_master':types.SimpleNamespace(st={'master':torch.randn(4,4)}),
                '_tailema':[torch.randn_like(model.weight)],'SEED':0,'mbs':64,'training_time':15.}
    a=scope();ha=bare(a['model']);ha.scope=a
    buf=io.BytesIO();torch.save(payload(a,ha,2200),buf);buf.seek(0);saved=torch.load(buf,weights_only=False)
    b=scope();hb=bare(b['model']);hb.scope=b
    assert restore(b,hb,saved)==2200
    assert torch.equal(a['model'].weight,b['model'].weight)
    assert torch.equal(a['_tailema'][0],b['_tailema'][0])
    assert torch.equal(a['head_eigen'].st['m'],b['head_eigen'].st['m'])
    assert torch.equal(a['embed_master'].st['master'],b['embed_master'].st['master'])
    assert b['optimizers'][0].it==2200
    bad=dict(saved,meta_lr=.0002)
    with pytest.raises(ValueError,match='meta_lr'):restore(b,hb,bad)
