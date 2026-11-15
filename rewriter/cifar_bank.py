"""Verified terminal CNN records and full-image FP32 teachers for mixed Chisel fitting."""
from dataclasses import fields,is_dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from .cifar_adapter import CifarPreservation

SCHEMA='chisel-cifar-terminal-bank-v1'

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def tensor_tree(value,device='cpu'):
    if isinstance(value,torch.Tensor):return value.detach().to(device=device,copy=True)
    if is_dataclass(value):return type(value)(**{f.name:tensor_tree(getattr(value,f.name),device) for f in fields(value)})
    if isinstance(value,dict):return {k:tensor_tree(v,device) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(tensor_tree(v,device) for v in value)
    if isinstance(value,list):return [tensor_tree(v,device) for v in value]
    return value

def normalized_images(uint8_nhwc,device='cpu'):
    if uint8_nhwc.dtype!=torch.uint8 or uint8_nhwc.shape[1:]!=(32,32,3):raise ValueError('CIFAR uint8 NHWC images required')
    x=uint8_nhwc.to(device=device,dtype=torch.float32).permute(0,3,1,2)/255.
    mean=x.new_tensor((.4914,.4822,.4465))[None,:,None,None]
    std=x.new_tensor((.2470,.2435,.2616))[None,:,None,None]
    return ((x-mean)/std).contiguous(memory_format=torch.channels_last)

class CifarBankReader:
    def __init__(self,path,device='cpu'):
        self.path=Path(path);self.device=device
        self.manifest=json.loads((self.path/'manifest.json').read_text())
        if self.manifest['schema']!=SCHEMA:raise ValueError('unknown CIFAR bank schema')
        self.entries=self.manifest['entries'];self._by_id={int(e['local_id']):e for e in self.entries}
        if len(self._by_id)!=len(self.entries):raise ValueError('duplicate CIFAR local IDs')
        if set(self.manifest['split']['train'])&set(self.manifest['split']['audit']):raise ValueError('overlapping model splits')
    def read(self,local_id):
        e=self._by_id[int(local_id)];path=self.path/e['file']
        if path.parent!=self.path or sha256_file(path)!=e['sha256']:raise ValueError('CIFAR record hash/path mismatch')
        d=torch.load(path,map_location=self.device,weights_only=False)
        if d['schema']!=SCHEMA or d['local_id']!=local_id or d['seed']!=e['seed']:raise ValueError('CIFAR record identity mismatch')
        stat=path.stat();stamp=(stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        signature={n:(id(v),v._version) for n,v in d['state_dict'].items()}
        teacher_signature=(id(d['teacher_logits']),d['teacher_logits']._version)
        def verified():
            st=path.stat()
            if (st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)!=stamp:
                raise ValueError('verified CIFAR source changed')
            if {n:(id(v),v._version) for n,v in d['state_dict'].items()}!=signature:
                raise ValueError('verified CIFAR model tensors mutated')
            if (id(d['teacher_logits']),d['teacher_logits']._version)!=teacher_signature:
                raise ValueError('verified CIFAR teacher tensor mutated')
            return e['sha256']
        return SimpleNamespace(snapshot=d['snapshot'],model=d['state_dict'],teacher_logits=d['teacher_logits'],
            metadata=d['metadata'],local_id=local_id,seed=d['seed'],domain='cifar10',assert_verified=verified)
