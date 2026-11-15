"""Accounted calibration observations for a separate frozen-checkpoint experiment.

This intentionally adds GPT forward/backward passes on past training sequences.
The base optimizer does not step: W is the frozen checkpoint, U is exactly zero,
and M is its recorded momentum in the original training loss units. This is not
an observation obtained for free from the original training run.
"""
import math
import torch
from torch import nn
from .algebra import Anchors, Measurements
from .controller import ControllerSnapshot, MatrixObservation, VectorObservation, coordinate_features, vector_features
from .sensors import SensorCollector


def observe_frozen_checkpoint(model, optimizers, config, inputs, targets, *, step,
                              radius_fraction=.01, microbatch_sequences=1):
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('calibration requires cleared gradients on a standalone frozen checkpoint')
    if inputs.shape != targets.shape or inputs.ndim != 2 or not len(inputs):
        raise ValueError('calibration requires paired sequence input/target tensors')
    if microbatch_sequences < 1 or not math.isfinite(radius_fraction) or radius_fraction <= 0:
        raise ValueError('positive microbatch size and finite positive radius fraction required')
    if config['batch_tokens'] <= 0:
        raise ValueError('positive original training token count required')
    c=config.get('rewriter', {}); rank=c.get('probe_rank',8)
    device=next(model.parameters()).device
    params=dict(model.named_parameters())
    owner={id(p):(o,g) for o in optimizers for g in o.param_groups for p in g['params']}
    for name,p in params.items():
        if id(p) not in owner:
            raise ValueError('missing optimizer ownership: '+name)
        state=owner[id(p)][0].state.get(p,{})
        if state.get('momentum',state.get('exp_avg')) is None:
            raise ValueError('frozen observation needs recorded optimizer momentum: '+name)
    embeddings={n+'.weight' for n,m in model.named_modules() if isinstance(m,nn.Embedding)}
    def oriented(name,tensor): return tensor.T if name in embeddings else tensor
    generator=torch.Generator().manual_seed(c.get('anchor_seed',100001))
    anchors={name:Anchors.random(*oriented(name,p).shape,rank,generator=generator,device=device)
             for name,p in params.items() if p.ndim==2}
    collector=SensorCollector(model,{n:(a.A,a.B) for n,a in anchors.items()},
        num_sequences=len(inputs),sequence_length=inputs.shape[1],
        signs_seed=c.get('signs_seed',300001),trajectory_rank=rank,
        trajectory_positions=c.get('trajectory_positions',64))
    collector.reset({n:(a.A,a.B) for n,a in anchors.items()},step=step)
    passes=0; loss_sum=0.
    try:
        for start in range(0,len(inputs),microbatch_sequences):
            end=min(start+microbatch_sequences,len(inputs))
            batch=inputs[start:end].to(device)
            collector.set_microbatch(torch.arange(start,end,device=device),batch)
            loss=model(batch,targets[start:end].to(device))
            loss_sum+=float(loss.detach());loss.backward();passes+=1
        sensors=collector.finalize()
        matrices={};vectors={};tm={};tv={};weight_sq=torch.zeros((),device=device)
        types=['embed','.attn.q.','.attn.k.','.attn.v.','.attn.proj.','.mlp.fc.','.mlp.proj.','gains','bias','proj.weight']
        with torch.no_grad():
            for index,(name,p) in enumerate(params.items()):
                w=oriented(name,p.detach()).float()
                g=oriented(name,p.grad.detach()).float()/inputs.numel()
                opt,group=owner[id(p)];state=opt.state.get(p,{})
                momentum=state.get('momentum',state.get('exp_avg'))
                if momentum is None: raise ValueError('frozen observation needs recorded optimizer momentum: '+name)
                # The checkpoint momentum came from original token-SUM training,
                # not from this smaller calibration batch.
                m=oriented(name,momentum).float()/config['batch_tokens']
                u=torch.zeros_like(w)
                norm=w.square().sum();weight_sq+=norm
                rows,cols=w.shape if w.ndim==2 else (w.numel(),1)
                family=next((i for i,key in enumerate(types) if key in name),len(types))
                metadata=[index/max(1,len(params)-1),math.log1p(rows),math.log1p(cols),
                    step/config.get('schedule_steps',config['max_steps']),float(group['lr']),
                    float(group.get('mu',group.get('betas',(0.,))[0])),family/len(types),
                    float(norm.sqrt().clamp_min(1e-12).log()),float(g.norm().clamp_min(1e-12).log()),0.]
                if p.ndim==2:
                    a=anchors[name];sensor=sensors.matrix[name]
                    banks=[(x@a.B,x.T@a.A) for x in (w,g,m,u)]
                    matrices[name]=MatrixObservation(a,
                        coordinate_features([b[0] for b in banks],a.A,sensor.output_rms,metadata),
                        coordinate_features([b[1] for b in banks],a.B,sensor.input_rms,metadata),
                        tuple(x.clone() for x in banks[0]))
                    tm[name]=sensor.train
                else:
                    vectors[name]=VectorObservation(vector_features([w,g,m,u],metadata),w.clone())
                    tv[name]=sensors.vectors[name].train
            snapshot=ControllerSnapshot(matrices,vectors,sensors.trajectories.clone(),
                weight_sq.sqrt(),weight_sq.sqrt()*radius_fraction)
        return snapshot,Measurements(tm,tv),dict(calibration_forwards=passes,
            calibration_backwards=passes,calibration_tokens=inputs.numel(),
            calibration_mean_ce=loss_sum/inputs.numel(),base_steps=0,
            displacement='zero: frozen identity base proposal',
            momentum_normalization_tokens=config['batch_tokens'],
            task_gradient_normalization_tokens=inputs.numel(),radius_fraction=radius_fraction)
    finally:
        collector.close()
        model.zero_grad(set_to_none=True)
