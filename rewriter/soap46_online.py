"""Record46 state adapter and one pending direct basis-residual hypergradient."""
import dataclasses
import json
import math
import time
from pathlib import Path
import torch
import torch.distributed as dist
from .lag_one import LagOneHypergradient
from .soap46_map import BasisSnapshot, committed_weight, next_forward_coefficient
from .soap46_conditioning import parameter_frame_weight


class Soap46Online:
    action_diagnostic_steps = ()
    residual_frame = "basis"
    meta_epsilon = 1e-8
    reset_meta_at_activation = False
    meta_reset_done = False
    def __init__(self,scope,*,variant,meta_lr,output,stop_step,meta_clip=1.,checkpoint_every=500,meta_betas=(.9,.99),compile_map=False,warmup_steps=0,warmup_meta="moments",meta_ramp_steps=0,residual_decay=0.,residual_frame="basis",meta_epsilon=1e-8,reset_meta_at_activation=False,action_diagnostic_steps=(),action_diagnostic_names=()):
        if variant not in ('inner-outer','outer-only'):raise ValueError('unknown basis variant')
        if not 0<=meta_lr or not 0<meta_clip:raise ValueError('invalid meta optimizer settings')
        if warmup_steps<0 or meta_ramp_steps<0 or not 0<=residual_decay<=1 or warmup_meta not in ('moments','off'):raise ValueError('invalid warmup policy')
        self.warmup_steps=warmup_steps;self.warmup_meta=warmup_meta;self.warmup_meta_steps=0
        self.meta_ramp_steps=meta_ramp_steps;self.residual_decay=residual_decay;self.residual_updates=0
        if residual_frame not in ('basis','parameter') or (residual_frame=='parameter' and variant!='outer-only'):raise ValueError('invalid residual coordinate frame')
        if not math.isfinite(meta_epsilon) or meta_epsilon<=0:raise ValueError('invalid meta epsilon')
        self.residual_frame=residual_frame;self.meta_epsilon=meta_epsilon
        if reset_meta_at_activation and (warmup_steps<2 or warmup_steps>=stop_step):
            raise ValueError('meta reset requires an activation step with real delayed credit')
        self.reset_meta_at_activation=bool(reset_meta_at_activation);self.meta_reset_done=False
        from .soap46_action_diagnostic import validate_action_samples
        diagnostic_steps, diagnostic_names = validate_action_samples(
            action_diagnostic_steps, action_diagnostic_names, stop_step=stop_step,
            residual_frame=residual_frame, compile_map=compile_map)
        self.action_diagnostic_steps = frozenset(diagnostic_steps)
        self.action_diagnostic_names = diagnostic_names
        if diagnostic_steps:
            named = dict(scope['model'].named_parameters())
            eligible = {p for group in scope['optimizer2'].param_groups for p in group['params']}
            if any(name not in named or named[name] not in eligible or named[name].ndim != 2
                   for name in diagnostic_names):
                raise ValueError('action diagnostic name is not an exact Muon matrix parameter name')
            self.action_diagnostic_parameters = {named[name]: name for name in diagnostic_names}
            self.action_diagnostic_emitted = set()
        self.betas=tuple(meta_betas);self.compile_map=compile_map
        map_function=parameter_frame_weight if residual_frame=="parameter" else committed_weight
        self.map=torch.compile(map_function,fullgraph=True,dynamic=False) if compile_map else map_function
        self.scope=scope;self.variant=variant;self.lr=meta_lr;self.stop=stop_step;self.clip=meta_clip
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.entries={};self.optimizer=None;self.meta_steps=0;self.metrics={};self.checkpoint_every=checkpoint_every
        self.gpt_seconds=0.;self.optimizer_seconds=0.;self.last_time=time.perf_counter()
        self.wandb=None
        if dist.get_rank()==0:
            import wandb
            self.wandb=wandb.init(project='learnable-weight-decay',name=self.output.name,dir=str(self.output),mode='offline',
                config=dict(baseline='accepted-record46',variant=variant,meta_lr=meta_lr,meta_clip=meta_clip,seed=scope['SEED'],stop_step=stop_step,state_policy='unperturbed-Q variance/covariance; refresh every step',meta_betas=self.betas,compile_map=compile_map,warmup_steps=warmup_steps,warmup_meta=warmup_meta,meta_ramp_steps=meta_ramp_steps,residual_decay=residual_decay,residual_frame=residual_frame,meta_epsilon=meta_epsilon,reset_meta_at_activation=self.reset_meta_at_activation))

    @torch.no_grad()
    def propose(self,p,state,group,step,attention,attention_output):
        s=self.scope;g=p.grad
        state['momentum'].lerp_(g,1-group['mu'])
        d=g.lerp(state['momentum'],group['mu'])
        if p not in self.entries:
            self.entries[p]=dict(xl=torch.nn.Parameter(torch.zeros(p.shape[0],p.shape[0],device=p.device),requires_grad=self.lr>0),
                                 xr=torch.nn.Parameter(torch.zeros(p.shape[1],p.shape[1],device=p.device),requires_grad=self.lr>0),lag=LagOneHypergradient(),record=None)
        e=self.entries[p]
        if state['q_row'] is not None:
            # Exactly baseline second moment. X never changes these measurements.
            projected=state['q_row'].T@d.float()@state['q_col']
            state['exp_avg_sq'].mul_(.9).add_(projected.square(),alpha=1-.9)
        clone=lambda v:None if v is None else v.detach().clone()
        r=BasisSnapshot(clone(p),clone(d),clone(g),clone(state['q_row']),clone(state['q_col']),
                        clone(state['exp_avg_sq']) if state['q_row'] is not None else None,
                        (p.new_tensor(group['lr']) if self.compile_map else group['lr']),
                        (p.new_tensor(step) if self.compile_map else step),attention,attention_output,self.variant,
                        cwd=(p.new_tensor(s['CWD']) if self.compile_map else s['CWD']))
        if step in self.action_diagnostic_steps and p in self.action_diagnostic_parameters:
            self.measure_action(p, e, r, step)
        proposal=(committed_weight if r.q_left is None else self.map)(r,e['xl'],e['xr'])
        if not torch.isfinite(proposal).all():raise FloatingPointError('nonfinite record46 proposal')
        p.copy_(proposal)
        if self.lr>0 and r.q_left is not None and (step<self.stop-1 or step+1==self.warmup_steps) and (self.warmup_meta=='moments' or step+1>=self.warmup_steps):
            self.install_pending(e,r,next_forward_coefficient(s['optimizers'][0]),step)
        # Upstream's actual raw-G Gram and sorted QR update, AFTER this action.
        s['soap_update_preconditioner'](g,state,shampoo_beta=s['gram_beta_at'](step))

    def measure_action(self, p, entry, record, step):
        # Runs only on the rank owning this Muon matrix, before W or Q/V refresh.
        # A separate per-rank file avoids rank-zero dropping distributed samples.
        from .soap46_action_diagnostic import compare_basis_action
        name = self.action_diagnostic_parameters[p]
        key = (step, name)
        if key in self.action_diagnostic_emitted:
            raise RuntimeError('duplicate action diagnostic sample')
        rank = dist.get_rank()
        if record.q_left is None or record.q_right is None:
            row = dict(schema='soap46_finite_action_diagnostic_v1', parameter_name=name,
                       action_step=step, status='skipped_basis_free', gpt_forwards=0, gpt_backwards=0)
        else:
            row = compare_basis_action(record, entry['xl'], entry['xr'], parameter_name=name)
            row['status'] = 'measured'
        row.update(rank=rank, world_size=dist.get_world_size(),
                   action_step_convention='zero-based Muon step_count before increment',
                   selection_steps=sorted(self.action_diagnostic_steps),
                   selection_names=list(self.action_diagnostic_names))
        with (self.output / f'action_diagnostics_rank{rank}.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        self.action_diagnostic_emitted.add(key)

    def install_pending(self,e,record,coefficient,step):
        def contraction(cotangent):
            with torch.enable_grad():
                return (self.map(record,e['xl'],e['xr'])*cotangent['weight']).sum()*coefficient
        e.update(record=record,coefficient=coefficient,action_step=step)
        tensors=tuple(getattr(record,f.name) for f in dataclasses.fields(record) if isinstance(getattr(record,f.name),torch.Tensor))
        e['lag'].from_contraction(action_step=step,hyperparameters={'xl':e['xl'],'xr':e['xr']},contraction=contraction,guarded_tensors=tensors)

    def effective_lr(self,step):
        if step<self.warmup_steps:return 0.
        fraction=min(1.,max(0.,(step-self.warmup_steps)/self.meta_ramp_steps)) if self.meta_ramp_steps else 1.
        return self.lr*fraction

    def consume(self,step):
        self.metrics={}
        pending=[(p,e) for p,e in self.entries.items() if e['lag'].pending]
        if not pending:return  # no Adam update for first-pass missing SOAP basis
        start=time.perf_counter()
        params=[e[k] for e in self.entries.values() for k in ('xl','xr')]
        if self.optimizer is None:self.optimizer=torch.optim.Adam(params,lr=self.lr,betas=self.betas,eps=self.meta_epsilon)
        self.optimizer.zero_grad(set_to_none=True)
        for p,e in pending:
            result=e['lag'].consume({'weight':p.grad.detach().float()/self.scope['batch_size']},gradient_step=step)
            result.assign_gradients({'xl':e['xl'],'xr':e['xr']});e['record']=None
        normsq=sum((h.grad.double().square().sum() for h in params if h.grad is not None),params[0].new_zeros((),dtype=torch.float64))
        dist.all_reduce(normsq);norm=normsq.sqrt()
        if not torch.isfinite(norm):raise FloatingPointError('nonfinite record46 basis hypergradient')
        factor=(self.clip/(norm+1e-6)).clamp(max=1).float()
        for h in params:
            if h.grad is not None:h.grad.mul_(factor)
        warming=step<self.warmup_steps
        if warming and self.warmup_meta!='moments':raise RuntimeError('unexpected off-mode warmup credit')
        # LR zero advances moments/counter at the EXECUTED zero-X action. It is
        # not a zero action gate, whose derivative would erase this signal.
        effective_lr=self.effective_lr(step)
        applied_decay=0. if warming else self.residual_decay
        reset_this_step=False
        if self.reset_meta_at_activation:
            if step>self.warmup_steps and not self.meta_reset_done:
                raise RuntimeError('missed declared residual Adam reset boundary')
            if step==self.warmup_steps and not self.meta_reset_done:
                # Credit was computed at the executed action above. Clear only
                # residual Adam m/v/counters; preserve X, gradients, groups and
                # all baseline state. This boundary update has LR zero when
                # a ramp is configured, and initializes the new Adam history.
                self.optimizer.state.clear()
                self.meta_reset_done=True;reset_this_step=True
        # All previous-action VJPs are consumed above. This independent shrink
        # changes only residual parameters, never their Adam moments or GPT W.
        if applied_decay:
            with torch.no_grad():
                for h in params:h.mul_(1-applied_decay)
        for group in self.optimizer.param_groups:group['lr']=effective_lr
        self.optimizer.step();self.meta_steps+=1
        if effective_lr>0:self.residual_updates+=1
        if warming:
            self.warmup_meta_steps+=1
            if any(torch.count_nonzero(h).item() for h in params):raise RuntimeError('warmup changed a residual')
        torch.cuda.synchronize()
        self.metrics=dict(hypergradient_norm=norm.item(),hypergradient_clip_factor=factor.item(),meta_steps=self.meta_steps,credit_seconds=time.perf_counter()-start,warmup_meta_steps=self.warmup_meta_steps,meta_effective_lr=effective_lr,moment_only_update=effective_lr==0,applied_residual_decay=applied_decay)
        if self.reset_meta_at_activation:
            counters={float(state['step']) for state in self.optimizer.state.values()}
            if len(counters)!=1:raise RuntimeError('residual Adam counters disagree after reset')
            self.metrics.update(meta_adam_reset_this_step=reset_this_step,
                meta_optimizer_resets=int(self.meta_reset_done),meta_adam_step=counters.pop())

    def state_dict(self):
        names={p:n for n,p in self.scope['model'].named_parameters()}
        return dict(variant=self.variant,reset_meta_at_activation=self.reset_meta_at_activation,meta_reset_done=self.meta_reset_done,residual_frame=self.residual_frame,meta_epsilon=self.meta_epsilon,meta_steps=self.meta_steps,warmup_steps=self.warmup_steps,warmup_meta=self.warmup_meta,warmup_meta_steps=self.warmup_meta_steps,meta_ramp_steps=self.meta_ramp_steps,residual_decay=self.residual_decay,residual_updates=self.residual_updates,optimizer=None if self.optimizer is None else self.optimizer.state_dict(),
            entries={names[p]:{k:e.get(k) for k in ('xl','xr','record','coefficient','action_step')} for p,e in self.entries.items()})

    def load_state_dict(self,state):
        if state['variant']!=self.variant:raise ValueError('basis variant differs')
        if state.get('residual_frame','basis')!=self.residual_frame or state.get('meta_epsilon',1e-8)!=self.meta_epsilon:raise ValueError('residual frame or meta epsilon differs')
        if state.get('warmup_steps',0)!=self.warmup_steps or state.get('warmup_meta','moments')!=self.warmup_meta:raise ValueError('warmup policy differs')
        if state.get('meta_ramp_steps',0)!=self.meta_ramp_steps or state.get('residual_decay',0.)!=self.residual_decay:raise ValueError('meta ramp/decay policy differs')
        if state.get('reset_meta_at_activation',False)!=self.reset_meta_at_activation:
            raise ValueError('residual Adam reset policy differs')
        if self.entries:raise RuntimeError('restore requires fresh hyperoptimizer')
        names=dict(self.scope['model'].named_parameters());params=[]
        for name,row in state['entries'].items():
            e={k:torch.nn.Parameter(row[k].detach().clone(),requires_grad=self.lr>0) for k in ('xl','xr')}
            e.update(lag=LagOneHypergradient(),record=None);self.entries[names[name]]=e;params.extend((e['xl'],e['xr']))
            if row['record'] is not None:self.install_pending(e,row['record'],row['coefficient'],row['action_step'])
        if state['optimizer'] is not None:
            self.optimizer=torch.optim.Adam(params,lr=self.lr,betas=self.betas,eps=self.meta_epsilon);self.optimizer.load_state_dict(state['optimizer'])
        self.meta_steps=state['meta_steps'];self.warmup_meta_steps=state.get('warmup_meta_steps',0);self.residual_updates=state.get('residual_updates',self.meta_steps-self.warmup_meta_steps)
        self.meta_reset_done=state.get('meta_reset_done',False)

    def log_step(self,step,loss):
        value=loss.detach().double().clone();dist.all_reduce(value);value/=self.scope['batch_size']
        normsq=sum((e[k].double().square().sum() for e in self.entries.values() for k in ('xl','xr')),value.new_zeros(()))
        dist.all_reduce(normsq);now=time.perf_counter()
        row=dict(event='train',step=step+1,train_loss=value.item(),basis_residual_norm=normsq.sqrt().item(),
                 residual_coordinate_frame=self.residual_frame,
                 nominal_meta_lr=self.lr,warmup_moment_updates=self.warmup_meta_steps,residual_updates=self.residual_updates,zero_lr_moment_updates=self.meta_steps-self.residual_updates,
                 wall_seconds=now-self.last_time,gpt_forward_backward_seconds=self.gpt_seconds,base_optimizer_seconds=self.optimizer_seconds,**self.metrics)
        self.last_time=now
        if self.compile_map:
            from torch._dynamo.utils import counters
            row['compile_unique_graphs']=counters['stats']['unique_graphs']
            row['compile_graph_breaks']=sum(counters['graph_break'].values())
        self.write(row)

    def write(self,row):
        if dist.get_rank()==0:
            with (self.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            if self.wandb is not None:self.wandb.log(row)

    def log_validation(self,step,value,live):
        self.write(dict(event='validation',step=step,val_loss=float(value),live_val_loss=float(live),objective=float(value)))

    def maybe_checkpoint(self,completed_steps):
        if self.checkpoint_every and completed_steps%self.checkpoint_every==0 and completed_steps!=self.stop:
            self.save_checkpoint(completed_steps,False)

    def save_checkpoint(self,completed_steps,final):
        from .soap46_checkpoint import payload
        folder=self.output if final else self.output/f'step{completed_steps:04d}';folder.mkdir(parents=True,exist_ok=True)
        rank=dist.get_rank();name=f'final_rank{rank}.pt' if final else f'rank{rank}.pt'
        tmp=folder/(name+'.incomplete');torch.save(payload(self.scope,self,completed_steps),tmp);tmp.replace(folder/name);dist.barrier()
        if rank==0:
            import shutil
            shutil.copy2(self.scope['logfile'],self.output/'upstream.log')
            (folder/'complete.json').write_text(json.dumps(dict(completed_steps=completed_steps,world_size=dist.get_world_size(),variant=self.variant,
                meta_lr=self.lr,meta_clip=self.clip,meta_betas=self.betas,reset_meta_at_activation=self.reset_meta_at_activation,residual_frame=self.residual_frame,meta_epsilon=self.meta_epsilon,compile_map=self.compile_map,warmup_steps=self.warmup_steps,warmup_meta=self.warmup_meta,meta_ramp_steps=self.meta_ramp_steps,residual_decay=self.residual_decay,base_lr=self.scope['MUON_LR'],cwd=self.scope['CWD'],seed=self.scope['SEED'],stop_step=self.stop,microbatch=self.scope['mbs'])))
        dist.barrier()

    def finish(self):
        self.save_checkpoint(self.stop,True)
        if dist.get_rank()==0:
            if self.wandb is not None:self.wandb.finish()
            (self.output/'finished.json').write_text(json.dumps(dict(completed_steps=self.stop,world_size=dist.get_world_size())))
