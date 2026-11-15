"""Pinned PR267 definitions, without its executable training loop."""
import ast
import hashlib
import json
from pathlib import Path
import types
import sys


def load_record(*, attention_causal=True, attention_scale=None):
    root=Path(__file__).parent
    raw=(root/'record.txt').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=json.loads((root/'PIN.json').read_text())['record_sha256']:
        raise ValueError('PR267 record hash mismatch')
    source=raw.decode().split('\n'+'='*100+'\n',1)[0]
    tree=ast.parse(source)
    tree.body=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef))]
    if not attention_causal:
        # CIFAR-only variant authorized by the user: change exactly the SDPA mask.
        attention=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CausalSelfAttention')
        flags=[k for n in ast.walk(attention) if isinstance(n,ast.Call) for k in n.keywords if k.arg=='is_causal']
        if len(flags)!=1 or not isinstance(flags[0].value,ast.Constant) or flags[0].value.value is not True:
            raise ValueError('Pinned attention mask hunk mismatch')
        flags[0].value.value=False
    if attention_scale is not None:
        if not 0 < attention_scale < float('inf'):
            raise ValueError('attention_scale must be finite and positive')
        attention=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CausalSelfAttention')
        scales=[k for n in ast.walk(attention) if isinstance(n,ast.Call) for k in n.keywords if k.arg=='scale']
        if len(scales)!=1 or not isinstance(scales[0].value,ast.Constant) or scales[0].value.value!=0.12:
            raise ValueError('Pinned attention scale hunk mismatch')
        scales[0].value.value=float(attention_scale)
    module=types.ModuleType('chisel_pr267' if attention_causal else 'chisel_pr267_noncausal');module.__file__=str(root/'record.txt')
    sys.modules[module.__name__]=module
    exec(compile(tree,str(root/'record.txt'),'exec'),module.__dict__)
    return module


def initialize(model):
    for name,p in model.named_parameters():
        if name.endswith('.attn.proj.weight'): p.data.mul_(1.25)
        elif name.endswith('.mlp.proj.weight'): p.data.mul_(3.0)
        elif name.endswith('.mlp.fc.weight'): p.data.mul_(1.5)
        elif name=='proj.weight' or 'proj' in name: p.data.zero_()


def optimizers(model,upstream):
    import torch
    named=[(n,p) for n,p in model.named_parameters() if n.startswith('blocks.') and p.ndim>=2]
    groups=[ [p for n,p in named if n.endswith(suffix)] for suffix in ['.mlp.fc.weight','.attn.proj.weight','.mlp.proj.weight'] ]
    qkv=[p for n,p in named if any(n.endswith('.attn.'+x+'.weight') for x in ('q','k','v'))]
    adam=torch.optim.AdamW([dict(params=[model.embed.weight],lr=.3),dict(params=[model.proj.weight],lr=1/320),
                           dict(params=[p for p in model.parameters() if p.ndim<2],lr=.01)],
                          betas=(.8,.95),eps=1e-10,weight_decay=0,fused=True)
    opts=[adam]+[upstream.MuonH(ps,lr=.018) for ps in [qkv,*groups]]
    for opt in opts:
        for g in opt.param_groups:
            g['initial_lr']=g['lr'];g['schedule_type']='aux' if opt is adam else 'h'
    return opts


def set_hparams(opts,t,total=3325):
    for opt in opts:
        for g in opt.param_groups:
            cooldown=1. if g['schedule_type']=='h' else .4
            progress=t/total
            g['lr']=g['initial_lr']*(1. if progress<1-cooldown else (1-progress)/cooldown)
