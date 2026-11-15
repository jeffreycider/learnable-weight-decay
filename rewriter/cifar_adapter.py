"""Explicit Airbench CNN adapter for mixed-domain Chisel research.

Convolutions are represented as out_channels x flattened input patches. Frozen
whitening filters, all BatchNorm parameters, and running statistics remain
visible in observations but are never edited. Teacher/student inference is FP32
in eval mode. This is a separate image model, never a GPT-shaped surrogate.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
from types import MethodType
import uuid

import torch
from torch import nn
from torch.func import functional_call
import torch.nn.functional as F

from .algebra import Anchors
from .controller import ControllerSnapshot,MatrixObservation,VectorObservation,coordinate_features,vector_features


def _observable_airbench_forward(self,images,whiten_bias_grad=True):
    if not whiten_bias_grad:raise ValueError('terminal calibration/inference uses the trained whitening bias')
    x=self.layers(self.whiten(images)).flatten(1)
    return self.head(x)/x.shape[-1]


def cifar_groups(model):
    groups={name:[] for name in ('conv.entry.weight','conv.hidden.weight','head.weight','whiten.bias')}
    for name,_ in model.named_parameters():
        if name.endswith('.conv1.weight'):groups['conv.entry.weight'].append(name)
        elif name.endswith('.conv2.weight'):groups['conv.hidden.weight'].append(name)
        elif name in ('head.weight','whiten.bias'):groups[name].append(name)
    if any(not values for values in groups.values()):raise ValueError('Airbench convolution/head/whitening layout required')
    return {name:tuple(values) for name,values in groups.items()}


@dataclass(frozen=True)
class ImageTeacherBatch:
    inputs: torch.Tensor
    log_probs: torch.Tensor
    teacher_id: str
    data_role: str
    sequence_ids: tuple[int,...]  # Image IDs; retained API name, never token IDs.


class CifarPreservation:
    def __init__(self,model,*,microbatch_images=128):
        if microbatch_images<1:raise ValueError('positive image microbatch required')
        if not all(hasattr(model,n) for n in ('whiten','layers','head')):
            raise ValueError('pinned Airbench model interface required')
        self.model=deepcopy(model).float().eval().requires_grad_(False)
        self.model._compiled_call_impl=None
        # Same operations as pinned CifarNet.forward, exposing the whitening
        # module boundary to the calibration hook; no extra convolution.
        self.model.forward=MethodType(_observable_airbench_forward,self.model)
        self.device=next(self.model.parameters()).device
        self.groups=cifar_groups(self.model)
        self._all_parameters={n:p.detach().clone() for n,p in self.model.named_parameters()}
        self.buffers={n:p.detach().clone() for n,p in self.model.named_buffers()}
        self._shapes={n:p.shape for n,p in self._all_parameters.items()}
        editable={n for names in self.groups.values() for n in names}
        self.parameters={n:(p.flatten(1) if p.ndim==4 else p) for n,p in self._all_parameters.items() if n in editable}
        self.transpose_storage={n:False for n,p in self.parameters.items() if p.ndim==2}
        self._versions={n:p._version for n,p in {**self._all_parameters,**self.buffers}.items()}
        self.microbatch_sequences=microbatch_images  # Existing panel interface; counts mean images.
        self.kl_dtype=torch.float64
        self.teacher_id=uuid.uuid4().hex
        self.compute=dict(student_forwards=0,student_backwards=0,student_images=0,teacher_forwards=0,teacher_images=0)
        self.base_weight_norm_sq=sum(p.double().square().sum() for p in self.parameters.values())

    def _validate_replacements(self,replacements):
        if replacements.keys()!=self.parameters.keys():raise ValueError('replace exactly the editable flattened CNN parameters')
        if self.model.training:raise RuntimeError('terminal CNN must stay in eval mode')
        for name,p in {**self._all_parameters,**self.buffers}.items():
            if p._version!=self._versions[name]:raise RuntimeError('frozen CNN parameter/buffer mutated')
        for name,p in replacements.items():
            base=self.parameters[name]
            if (p.shape,p.dtype,p.device)!=(base.shape,base.dtype,base.device):
                raise ValueError('CNN replacement metadata differs: '+name)

    def _forward(self,replacements,images):
        self._validate_replacements(replacements)
        state=dict(self._all_parameters)
        state.update({n:p.reshape(self._shapes[n]) for n,p in replacements.items()})
        return functional_call(self.model,(state,self.buffers),(images.to(self.device,dtype=torch.float32),),strict=True)

    def edited_parameters(self,action,scale=1.):
        result={}
        for name,base in self.parameters.items():
            delta=action.matrices[name].dense() if base.ndim==2 else action.vectors[name]
            if delta.shape!=base.shape:raise ValueError('CNN anchored action shape differs: '+name)
            result[name]=base+scale*delta.float()
        return result

    @torch.no_grad()
    def teacher_batch(self,images,cached_logits,image_ids,*,data_role='past_training'):
        if data_role not in ('past_training','withheld_training','fresh_audit'):
            raise ValueError('explicit image fitting or audit role required')
        if cached_logits.shape!=(len(images),self.model.head.out_features) or len(image_ids)!=len(images):
            raise ValueError('one cached classifier-logit row and identity per image required')
        return ImageTeacherBatch(images.detach(),cached_logits.to(self.device,dtype=torch.float64).log_softmax(-1),
            self.teacher_id,data_role,tuple(map(int,image_ids)))


def observe_cifar_terminal(engine,images,labels,*,momentum,anchor_seed,rank=8,
                           momentum_loss_count=2000,trajectory_images=64,radius_fraction=.01):
    """One explicitly counted calibration backward on fitting images.

    G is the gradient of terminal eval-mode mean image CE, M is the saved
    training momentum divided by its summed-loss batch size, U=0 for the frozen
    proposal. Spatial means are projected with each known output anchor A;
    image identities are shared across all observed boundaries. Missing padded
    boundaries have an explicit mask and are not invented observations.
    """
    if len(images)!=len(labels) or not len(images) or momentum_loss_count<=0:
        raise ValueError('nonempty labeled calibration images and positive momentum loss count required')
    params={n:p.detach().clone().requires_grad_(True) for n,p in engine._all_parameters.items()}
    generator=torch.Generator().manual_seed(anchor_seed)
    oriented=lambda value:value.flatten(1) if value.ndim==4 else value
    anchors={n:Anchors.random(*oriented(p).shape,rank,generator=generator,device=engine.device)
             for n,p in params.items() if p.ndim in (2,4)}
    modules={n:engine.model.get_submodule(n.rsplit('.',1)[0]) for n in anchors}
    if not all(isinstance(m,(nn.Conv2d,nn.Linear)) for m in modules.values()):raise ValueError('unrecognized CNN matrix owner')
    if len(anchors)>25:raise ValueError('CNN trajectories exceed the declared25boundary interface')
    count=min(trajectory_images,len(images));positions=torch.randperm(len(images),generator=generator)[:count].sort().values.to(engine.device)
    observations={};handles=[];batch=len(images)
    for name,module in modules.items():
        def hook(mod,args,output,name=name):
            x=args[0].detach().float()
            if isinstance(mod,nn.Conv2d):
                padding=mod.padding
                if isinstance(padding,str):
                    if padding!='same' or mod.stride!=(1,1):raise ValueError('unsupported convolution padding')
                    padding=tuple(k//2 for k in mod.kernel_size)
                patches=F.unfold(x,mod.kernel_size,dilation=mod.dilation,padding=padding,stride=mod.stride)
                input_rms=patches.square().mean((0,2)).sqrt();del patches
            else:input_rms=x.square().mean(0).sqrt()
            values=output.detach().float()
            pooled=values.flatten(2).mean(-1) if values.ndim==4 else values
            observations[name]=dict(input_rms=input_rms,activation=pooled[positions]@anchors[name].A)
            def backward(delta):
                delta=delta.detach().float()*batch  # Per-image CE adjoint units.
                axes=tuple(i for i in range(delta.ndim) if i!=1)
                observations[name]['output_rms']=delta.square().mean(axes).sqrt()
                pooled=delta.flatten(2).mean(-1) if delta.ndim==4 else delta
                observations[name]['adjoint']=pooled[positions]@anchors[name].A
            output.register_hook(backward)
        handles.append(module.register_forward_hook(hook))
    try:
        logits=functional_call(engine.model,(params,engine.buffers),(images.to(engine.device,dtype=torch.float32),),strict=True)
        loss=F.cross_entropy(logits,labels.to(engine.device),reduction='mean');loss.backward()
    finally:
        for handle in handles:handle.remove()
    matrices={};vectors={};trajectories=[]
    with torch.no_grad():
        for index,(name,p) in enumerate(params.items()):
            w=oriented(p.detach());g=oriented(p.grad.detach())
            m=oriented(momentum.get(name,torch.zeros_like(p)).to(engine.device,dtype=torch.float32))/momentum_loss_count
            u=torch.zeros_like(w)
            n,c=w.shape if w.ndim==2 else (w.numel(),1)
            family=(10 if '.conv1.' in name else 11 if '.conv2.' in name else 12 if name=='whiten.weight'
                    else 9 if name=='head.weight' else 7 if name.endswith('weight') else 8)/10
            meta=[index/max(1,len(params)-1),math.log1p(n),math.log1p(c),1.,0.,
                  (0. if name not in momentum else .6 if p.ndim==4 else .85),family,
                  float(w.norm().clamp_min(1e-12).log()),float(g.norm().clamp_min(1e-12).log()),0.]
            if w.ndim==2:
                anchor=anchors[name];obs=observations[name]
                banks=[(v@anchor.B,v.T@anchor.A) for v in (w,g,m,u)]
                matrices[name]=MatrixObservation(anchor,coordinate_features([v[0] for v in banks],anchor.A,obs['output_rms'],meta),
                    coordinate_features([v[1] for v in banks],anchor.B,obs['input_rms'],meta),tuple(v.clone() for v in banks[0]))
                trajectories.append(torch.stack((obs['activation'],obs['adjoint']),dim=1))
            else:vectors[name]=VectorObservation(vector_features([w,g,m,u],meta),w.clone())
        trajectory=torch.zeros(count,25,2,rank,device=engine.device)
        trajectory[:,:len(trajectories)]=torch.stack(trajectories,dim=1)
        mask=torch.arange(25,device=engine.device)<len(trajectories)
        norm=engine.base_weight_norm_sq.sqrt().float()
        snapshot=ControllerSnapshot(matrices,vectors,trajectory.flatten(1),norm,norm*radius_fraction,trajectory_mask=mask)
    return snapshot,dict(domain='cifar10',calibration_forwards=1,calibration_backwards=1,calibration_images=batch,
        calibration_mean_ce=float(loss.detach()),trajectory_images=count,observed_boundaries=len(trajectories),
        trajectory_representation='per-image spatial means projected with known per-layer A anchors; padded25 with explicitmask',
        gradient_units='mean image CE',adjoint_units='individual image CE',momentum_loss_count=momentum_loss_count,
        displacement='zero frozen proposal',groups=engine.groups)
