"""Actor-free ordinary-pass capture for replicated PR323 weights on 1..8 ranks.

Muon owns dense momentum on parameter-index % world_size; only its two thin
projections are SUM-reduced. Adam state is replicated and rank zero contributes
once. Sensors execute eagerly at explicit compiler boundaries; GPT computation
remains eligible for torch.compile. Throughput needs qualification on hardware.
"""
import hashlib
import math
from pathlib import Path
import torch
from torch import nn
import torch.distributed as dist
from baseline.optim import Muon
from .algebra import Anchors,Measurements
from .controller import ControllerSnapshot,MatrixObservation,VectorObservation,coordinate_features,vector_features
from .sensors import SensorCollector
from .edit_scope import editable_names
from .state_bank import StateBankWriter


class CaptureCollector(SensorCollector):
    # Keep mutable IDs, buffers and hook closures out of Dynamo guards. These
    # boundaries may split GPT graphs; they never add a GPT evaluation.
    @torch.compiler.disable
    def _matrix_forward(self,*args):return super()._matrix_forward(*args)
    @torch.compiler.disable
    def _norm_forward(self,*args,**kwargs):return super()._norm_forward(*args,**kwargs)
    @torch.compiler.disable
    def _trajectory(self,*args):return super()._trajectory(*args)

    @torch.no_grad()
    def finalize(self,distributed=False):
        if not distributed:return super().finalize(distributed=False)
        if not self.active or self.ctx is None or not dist.is_initialized():
            raise RuntimeError('distributed capture needs active measurements and process group')
        coverage=torch.zeros(self.N,device=next(self.model.parameters()).device)
        coverage[list(self.seen)]=1
        arrays=[coverage,self.z]
        arrays += [x for values in self.banks.values() for x in values]
        arrays += [x for values in self.activity.values() for x in values]
        arrays += list(self.vbanks.values())
        # Hundreds of tiny collectives would dominate the 8GPU baseline.
        flat=torch.cat([value.reshape(-1) for value in arrays])
        dist.all_reduce(flat,op=dist.ReduceOp.SUM)
        offset=0
        for value in arrays:
            count=value.numel();value.copy_(flat[offset:offset+count].view_as(value));offset+=count
        if not bool((coverage==1).all()):
            raise ValueError('every global sequence must be observed exactly once')
        self.seen=set(range(self.N))
        return super().finalize(distributed=False)


class DistributedStateObserver:
    """Reusable ordinary-pass observations; no writer, actor, or task evaluations."""
    def __init__(self,model,optimizers,config,*,sensor_config,start_step=0):
        self.config=config;self.c=dict(sensor_config)
        if type(start_step) is not int or start_step<0:raise ValueError('nonnegative observer start step required')
        self.next_step=start_step;self._in_step=False
        self.distributed=dist.is_initialized();self.world=dist.get_world_size() if self.distributed else 1
        self.rank_id=dist.get_rank() if self.distributed else 0
        self.model=model;self.params=dict(model.named_parameters());self.device=next(model.parameters()).device
        self.rank=int(self.c.get('probe_rank',8));self.generator=torch.Generator().manual_seed(self.c.get('anchor_seed',100001))
        self.embedding_names={n+'.weight' for n,m in model.named_modules() if isinstance(m,nn.Embedding)}
        self.editable=editable_names(self.params,'grouped')
        self.anchors=self.make_anchors()
        # Reproduce the captured-prefix anchor RNG, without reading data or GPT.
        period=self.c.get('anchor_refresh_period',32)
        if period:
            for _ in range(max(0,(start_step-1)//period)):
                self.anchors=self.make_anchors()
        self.collector=CaptureCollector(model,{n:(a.A,a.B) for n,a in self.anchors.items()},
            num_sequences=config['batch_tokens']//config['seq_len'],sequence_length=config['seq_len'],
            signs_seed=self.c.get('signs_seed',300001),trajectory_rank=self.rank,
            trajectory_positions=self.c.get('trajectory_positions',64),
            train_probes=self.c.get('train_sequence_probes',4),audit_probes=self.c.get('audit_sequence_probes',4))
        self.owners={};self.shard_rank={}
        for opt in optimizers:
            for group in opt.param_groups:
                for index,p in enumerate(group['params']):
                    if id(p) in self.owners:raise ValueError('multiple optimizer owners')
                    self.owners[id(p)]=(opt,group)
                    self.shard_rank[id(p)]=index%self.world if isinstance(opt,Muon) else 0
        if any(id(p) not in self.owners for p in self.params.values()):raise ValueError('missing optimizer owner')
        self.metrics={};self.enabled=False;self.finished=False

    def oriented(self,name,value):return value.T if name in self.embedding_names else value

    def make_anchors(self):
        return {n:Anchors.random(*self.oriented(n,p).shape,self.rank,generator=self.generator,device=self.device)
                for n,p in self.params.items() if p.ndim==2}

    def before_step(self,t,model,optimizers):
        if self._in_step or t!=self.next_step:raise ValueError('observer training clock mismatch')
        self.enabled=True;self._in_step=True
        period=self.c.get('anchor_refresh_period',32)
        if t and period and t%period==0:self.anchors=self.make_anchors()
        self.collector.reset({n:(a.A,a.B) for n,a in self.anchors.items()},step=t)

    def before_microbatch(self,t,sequence_ids,inputs):
        if self.enabled:self.collector.set_microbatch(sequence_ids,inputs)

    @torch.no_grad()
    def after_backward(self,t,model):
        if not self.enabled:return
        self.sensors=self.collector.finalize(distributed=self.distributed)
        self.gradient={};self.gradient_norm={};self.before={}
        if self.rank_id==0:
            for n,p in self.params.items():
                g=self.oriented(n,p.grad).float()/self.config['batch_tokens']
                self.gradient_norm[n]=float(g.norm())
                self.gradient[n]=((g@self.anchors[n].B,g.T@self.anchors[n].A) if p.ndim==2 else g.clone())
                self.before[n]=p.detach().clone()

    @torch.no_grad()
    def momentum_projections(self):
        # One ownership validation and one packed thin SUM, rather than a
        # latency-heavy collective for every parameter/projection.
        local={};valid=[]
        for n,p in self.params.items():
            opt,_=self.owners[id(p)];owner=self.shard_rank[id(p)]
            state=opt.state.get(p,{}) if self.rank_id==owner else {}
            momentum=state.get('momentum',state.get('exp_avg'))
            valid.append(self.rank_id!=owner or momentum is not None)
            local[n]=momentum
        flags=torch.tensor(valid,device=self.device,dtype=torch.int32)
        if self.distributed:dist.all_reduce(flags,op=dist.ReduceOp.MIN)
        if not bool(flags.all()):raise ValueError('missing post-update optimizer momentum')
        pieces=[];spec=[]
        for n,p in self.params.items():
            momentum=local[n]
            if momentum is not None:
                m=self.oriented(n,momentum).float()/self.config['batch_tokens']
                values=(m@self.anchors[n].B,m.T@self.anchors[n].A) if p.ndim==2 else (m.clone(),)
            elif p.ndim==2:values=(torch.zeros_like(self.anchors[n].A),torch.zeros_like(self.anchors[n].B))
            else:values=(torch.zeros_like(p,dtype=torch.float32),)
            spec.append((n,[v.shape for v in values],p.ndim==2))
            pieces.extend(v.reshape(-1) for v in values)
        flat=torch.cat(pieces)
        if self.distributed:dist.all_reduce(flat,op=dist.ReduceOp.SUM)
        result={};offset=0
        for n,shapes,matrix in spec:
            values=[]
            for shape in shapes:
                count=math.prod(shape);values.append(flat[offset:offset+count].view(shape));offset+=count
            result[n]=tuple(values) if matrix else values[0]
        return result

    @torch.no_grad()
    def finish_snapshot(self,t):
        if not self._in_step or t!=self.next_step:raise ValueError('observer snapshot clock mismatch')
        momentum=self.momentum_projections()
        result=self.snapshot_from_momentum(t,momentum) if self.rank_id==0 else None
        self.release_step(t)
        return result

    def release_step(self,t):
        for name in ('sensors','gradient','gradient_norm','before'):
            if hasattr(self,name):delattr(self,name)
        self.next_step=t+1;self._in_step=False;self.enabled=False

    @torch.no_grad()
    def snapshot_from_momentum(self,t,momentum):
        if self.rank_id==0:
            matrices={};vectors={};tm={};tv={};norm=torch.zeros((),device=self.device)
            error=torch.zeros_like(norm);denom=torch.zeros_like(norm)
            types=['embed','.attn.q.','.attn.k.','.attn.v.','.attn.proj.','.mlp.fc.','.mlp.proj.','gains','bias','proj.weight']
            for index,(n,p) in enumerate(self.params.items()):
                w=self.oriented(n,p.detach()).float();u=w-self.oriented(n,self.before[n]).float()
                wn=w.square().sum()
                if n in self.editable:norm+=wn
                _,group=self.owners[id(p)];rows,cols=w.shape if w.ndim==2 else (w.numel(),1)
                family=next((i for i,key in enumerate(types) if key in n),len(types))
                meta=[index/max(1,len(self.params)-1),math.log1p(rows),math.log1p(cols),
                    t/self.config.get('schedule_steps',self.config['max_steps']),float(group['lr']),
                    float(group.get('mu',group.get('betas',(0.,))[0])),family/len(types),
                    float(wn.sqrt().clamp_min(1e-12).log()),math.log(max(self.gradient_norm[n],1e-12)),0.]
                if p.ndim==2:
                    a=self.anchors[n];s=self.sensors.matrix[n];wp=(w@a.B,w.T@a.A);up=(u@a.B,u.T@a.A)
                    banks=[wp,self.gradient[n],momentum[n],up]
                    matrices[n]=MatrixObservation(a,coordinate_features([b[0] for b in banks],a.A,s.output_rms,meta),
                        coordinate_features([b[1] for b in banks],a.B,s.input_rms,meta),tuple(x.clone() for x in wp))
                    tm[n]=s.train
                    expected=self.gradient[n];ones=s.ones
                else:
                    vectors[n]=VectorObservation(vector_features([w,self.gradient[n],momentum[n],u],meta),w.clone())
                    s=self.sensors.vectors[n];tv[n]=s.train;expected=(self.gradient[n],);ones=(s.ones,)
                for measured,g in zip(ones,expected):
                    target=g*math.sqrt(self.config['batch_tokens']/self.config['seq_len'])
                    error+=(measured-target).square().sum();denom+=target.square().sum()
            snapshot=ControllerSnapshot(matrices,vectors,self.sensors.trajectories.clone(),norm.sqrt(),
                norm.sqrt()*self.c.get('radius_fraction',.01))
            return snapshot,Measurements(tm,tv),float((error/denom.clamp_min(1e-30)).sqrt())

    def state_dict(self):
        if self._in_step:raise RuntimeError('checkpoint observations only between ordinary steps')
        from .integration import _checkpoint_tensor_copy
        return dict(schema=1,sensor_config=dict(self.c),next_step=self.next_step,
                    generator=self.generator.get_state().clone(),anchors=_checkpoint_tensor_copy(self.anchors))

    def load_state_dict(self,state):
        if self._in_step:raise RuntimeError('cannot restore mid-step observations')
        if state['schema']!=1 or state['sensor_config']!=self.c:raise ValueError('observer resume configuration changed')
        from .integration import _checkpoint_tensor_copy
        self.anchors=_checkpoint_tensor_copy(state['anchors'],self.device)
        if set(self.anchors)!={n for n,p in self.params.items() if p.ndim==2}:raise ValueError('observer anchor names changed')
        self.generator.set_state(state['generator'].cpu());self.next_step=state['next_step']
        self.collector.reset({n:(a.A,a.B) for n,a in self.anchors.items()},step=self.next_step)
        self.collector.active=False

    def close(self):
        if self.finished:return
        self.collector.close();self.finished=True


class StateCaptureAdapter(DistributedStateObserver):
    def __init__(self,model,optimizers,config,*,directory,provenance,writer_factory=StateBankWriter):
        c=config['state_bank'];self.steps=int(c.get('steps',1000))
        if not 1<=self.steps<=1000 or self.steps>config['max_steps']:
            raise ValueError('state bank must capture a prefix of 1..1000 updates within this run')
        if config.get('mode','baseline')!='baseline':raise ValueError('state capture requires unedited baseline mode')
        super().__init__(model,optimizers,config,sensor_config=c)
        self.writer=writer_factory(directory,provenance,expected_steps=self.steps,
            audit_every=c.get('audit_every',5),sync_every=c.get('sync_every',25)) if self.rank_id==0 else None

    def before_step(self,t,model,optimizers):
        self.enabled=t<self.steps
        if self.enabled:super().before_step(t,model,optimizers)

    @torch.no_grad()
    def after_step(self,t,model,optimizers):
        if not self.enabled:return
        result=self.finish_snapshot(t)
        if self.rank_id==0:
            snapshot,measurements,error=result
            self.writer.append(t+1,model.state_dict(),snapshot,measurements,
                               data_cursor_batches=t+1,identity_action=True)
            self.metrics=dict(captured_states=t+1,capture_allones_relative_error=error,
                capture_extra_gpt_forwards=0,capture_extra_gpt_backwards=0,capture_actor_parameters=0)
        if t+1==self.steps:self.close()

    def close(self):
        if self.finished:return
        super().close()
        if self.writer is not None:self.writer.close()
