import pytest
import torch
from torch import nn
import torch.nn.functional as F
from rewriter.cifar_adapter import CifarPreservation,observe_cifar_terminal
from rewriter.algebra import Anchors,Factors,RawAction
from rewriter.transformer_chisel import TransformerChisel

class Group(nn.Module):
    def __init__(self):
        super().__init__();self.conv1=nn.Conv2d(4,4,3,padding="same",bias=False);self.norm1=nn.BatchNorm2d(4)
        self.conv2=nn.Conv2d(4,4,3,padding="same",bias=False);self.norm2=nn.BatchNorm2d(4)
    def forward(self,x):return F.gelu(self.norm2(self.conv2(F.gelu(self.norm1(self.conv1(x))))))

class TinyAirbench(nn.Module):
    def __init__(self):
        super().__init__();self.whiten=nn.Conv2d(3,4,2);self.layers=nn.Sequential(nn.GELU(),Group(),nn.AdaptiveMaxPool2d(1));self.head=nn.Linear(4,3,bias=False)
    def forward(self,x):
        x=F.conv2d(x,self.whiten.weight,self.whiten.bias);x=self.layers(x).flatten(1)
        return self.head(x)/x.shape[-1]

def fixture():
    torch.set_num_threads(1);torch.manual_seed(4)
    model=TinyAirbench().eval();engine=CifarPreservation(model);x=torch.randn(6,3,8,8);y=torch.arange(6)%3
    return model,engine,x,y

def test_identity_and_flattened_edit_gradient():
    model,engine,x,y=fixture();torch.testing.assert_close(engine._forward(engine.parameters,x),model(x),atol=0,rtol=0)
    name="layers.1.conv1.weight";delta=torch.randn_like(engine.parameters[name],requires_grad=True)
    replacement=dict(engine.parameters);replacement[name]=replacement[name]+.01*delta
    out=engine._forward(replacement,x);grad=torch.autograd.grad(out.square().sum(),delta)[0]
    parameters=dict(model.named_parameters());parameters[name]=parameters[name]+.01*delta.reshape(parameters[name].shape)
    direct=torch.func.functional_call(model,parameters,(x,));direct_grad=torch.autograd.grad(direct.square().sum(),delta)[0]
    torch.testing.assert_close(out,direct,atol=0,rtol=0);torch.testing.assert_close(grad,direct_grad,atol=0,rtol=0)
    assert not any("norm" in n for n in engine.parameters)
    assert "whiten.weight" not in engine.parameters and "whiten.bias" in engine.parameters

def test_snapshot_masks_rng_immutable_buffers_and_shared_actor():
    _,engine,x,y=fixture();rng=torch.get_rng_state().clone();buffers={n:b.clone() for n,b in engine.buffers.items()}
    snap,meta=observe_cifar_terminal(engine,x,y,momentum={},anchor_seed=87,rank=2,trajectory_images=4)
    assert torch.equal(rng,torch.get_rng_state());assert meta["observed_boundaries"]==4
    assert snap.trajectories.shape==(4,100);assert snap.trajectory_mask.tolist()==[True]*4+[False]*21
    assert all(torch.equal(v,buffers[n]) for n,v in engine.buffers.items())
    assert "whiten.weight" in snap.matrices and "layers.1.norm1.weight" in snap.vectors
    actor=TransformerChisel(rank=2,trajectory_dim=100,model_dim=256,num_layers=1,patch_size=32,trajectory_mask_features=True)
    action=actor(snap);edited=engine.edited_parameters(action)
    torch.testing.assert_close(engine._forward(edited,x),engine._forward(engine.parameters,x),rtol=0,atol=0)
    # The same shared actor accepts an ordinary GPT-shaped snapshot without a padding mask.
    from tests.test_prefix_cache import toy_record
    from dataclasses import replace
    gpt,_=toy_record();gpt=replace(gpt,trajectories=torch.zeros(4,100),trajectory_mask=None)
    actor(gpt)
    loss=sum(p.square().sum() for p in edited.values());loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in actor.parameters() if p.grad is not None)>0
    assert all(torch.equal(v,buffers[n]) for n,v in engine.buffers.items())

def test_cached_teacher_roles_and_frozen_state_guard():
    _,engine,x,y=fixture();logits=engine._forward(engine.parameters,x).detach()
    for role in ("past_training","withheld_training","fresh_audit"):
        b=engine.teacher_batch(x,logits,range(len(x)),data_role=role)
        torch.testing.assert_close(b.log_probs,logits.double().log_softmax(-1));assert b.data_role==role
    assert engine.compute["teacher_forwards"]==0
    with pytest.raises(ValueError):engine.teacher_batch(x,logits,range(len(x)),data_role="unknown")
    next(iter(engine.buffers.values())).add_(1)
    with pytest.raises(RuntimeError,match="mutated"):engine._forward(engine.parameters,x)

def test_cifar_bank_hash_and_mutation_guards(tmp_path):
    import json
    from rewriter.cifar_bank import SCHEMA,CifarBankReader,sha256_file,normalized_images,tensor_tree
    _,engine,x,y=fixture()
    snap,_=observe_cifar_terminal(engine,x,y,momentum={},anchor_seed=5,rank=2)
    payload=dict(schema=SCHEMA,local_id=1,seed=0,snapshot=tensor_tree(snap),state_dict=engine.model.state_dict(),
        teacher_logits=torch.randn(6,3),metadata={})
    file=tmp_path/'model_0001.pt';torch.save(payload,file)
    entry=dict(local_id=1,seed=0,file=file.name,sha256=sha256_file(file))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(schema=SCHEMA,entries=[entry],split=dict(train=[1],audit=[]))))
    bank=CifarBankReader(tmp_path);record=bank.read(1);assert record.assert_verified()==entry['sha256']
    record.teacher_logits.add_(1)
    with pytest.raises(ValueError,match='teacher tensor mutated'):record.assert_verified()
    record=bank.read(1);next(iter(record.model.values())).add_(1)
    with pytest.raises(ValueError,match='model tensors mutated'):record.assert_verified()
    record=bank.read(1);file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed'):record.assert_verified()
    with pytest.raises(ValueError,match='hash/path'):bank.read(1)
    normalized=normalized_images(torch.zeros(2,32,32,3,dtype=torch.uint8))
    assert normalized.dtype==torch.float32 and normalized.shape==(2,3,32,32)
    torch.testing.assert_close(normalized[0,:,0,0],-torch.tensor([.4914,.4822,.4465])/torch.tensor([.2470,.2435,.2616]))
