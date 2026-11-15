"""Sharded dense residual hyperparameters and one-step optimizer-only replay."""
import dataclasses
import json
import math
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist
from .lag_one import LagOneHypergradient
from .soap_envelope import scope_envelope
from .soap_residual import FrozenMap,committed_proposal,inverse_fourth_factor,residual_scale
from .soap_pr395_residual import PublishedMap, committed_pr395_proposal


class SoapResidualOnline:
    def __init__(self,scope,*,meta_lr,output,stop_step,variant="pr395"):
        if variant not in {"pr395", "legacy-extra-polar"}:raise ValueError("unknown SOAP variant")
        self.variant=variant
        self.scope=scope;self.lr=meta_lr;self.output=Path(output);self.stop=stop_step
        self.learned=meta_lr>0;self.entries={};self.optimizer=None;self.meta_steps=0
        self.output.mkdir(parents=True,exist_ok=True)
        self.metrics={};self.last_time=time.perf_counter();self.action_difference_sq=None;self.action_cosine_stats=None;self.gpt_seconds=0.;self.optimizer_seconds=0.
        self.wandb=None
        if dist.get_rank()==0:
            import wandb
            self.wandb=wandb.init(project='learnable-weight-decay',name=self.output.name,dir=str(self.output),mode='offline',config={'meta_lr':meta_lr,'stop_step':stop_step,'variant':self.variant,'seed':scope['SEED']})
        self.map=committed_pr395_proposal if self.variant=="pr395" else committed_proposal
        if os.environ.get('SOAP_COMPILE_MAP','0')=='1':
            self.map=torch.compile(self.map,dynamic=False)
        if not 1<=stop_step<=scope['FINAL_TRAIN_STEPS']:raise ValueError('stop must fit the pinned schedule')

    @torch.no_grad()
    def propose(self,p,direction,residual,state,group,step,use_soap):
        s=self.scope
        state['shp_left'].lerp_(residual@residual.mT,1-s['SHAMPOO_BETA2'])
        state['shp_right'].lerp_(residual.mT@residual,1-s['SHAMPOO_BETA2'])
        steps=s['SHAMPOO_NS_FINAL_STEPS'] if step>=s['SHAMPOO_NS_TRANSITION_STEP'] else s['SHAMPOO_NS_STEPS']
        old_tf32=torch.backends.cuda.matmul.allow_tf32
        factors=[]
        try:
            for gram in (state['shp_left'],state['shp_right']):
                torch.backends.cuda.matmul.allow_tf32=gram.shape[-1]>s['SHAMPOO_TF32_MIN_SIDE']
                factors.append(inverse_fourth_factor(gram,steps,s['SHAMPOO_DAMPING']))
        finally:torch.backends.cuda.matmul.allow_tf32=old_tf32
        # Upstream applies the right inverse root to update.T then transposes.
        factors[1]=factors[1].mT.contiguous()
        spectral={}
        ratio=max(group['lr']/group['initial_lr'],s['SPA_RMIN'])
        if use_soap and state['q_row'] is not None and ratio<1:
            g=p.grad.float()
            if 'spa_rr' not in state:
                state['spa_rr']=g@g.mT;state['spa_cc']=g.mT@g
            else:
                state['spa_rr'].lerp_(g@g.mT,1-s['SPA_BETA']);state['spa_cc'].lerp_(g.mT@g,1-s['SPA_BETA'])
            qr,qc=state['q_row'],state['q_col']
            lr=torch.diag(qr.T@state['spa_rr']@qr).clamp_min(1e-30).log()
            lc=torch.diag(qc.T@state['spa_cc']@qc).clamp_min(1e-30).log()
            lv=lr[:,None]+lc[None,:]
            sc=((lv-lv.mean())/lv.std().clamp_min(1e-6)).clamp(-s['SPA_CLIP'],s['SPA_CLIP'])
            spectral=dict(spectral_row=qr.clone(),spectral_col=qc.clone(),spectral_scale=torch.exp(s['SPA_ALPHA']*math.log(ratio)*sc))
        base = {}
        record_type = FrozenMap
        if self.variant == "pr395":
            # Call the untouched pinned helper, preserving the direct action
            # association used by shampoo_update rather than factorizing it.
            apply = s['apply_inverse_fourth_root_left']
            raw = apply(direction, state['shp_left'], steps, s['SHAMPOO_DAMPING'])
            raw = apply(raw.mT.contiguous(), state['shp_right'], steps, s['SHAMPOO_DAMPING']).mT
            base = {'base_raw': raw.detach().clone()}
            record_type = PublishedMap
        record=record_type(p.detach().clone(),direction.detach().clone(),*factors,
                         residual_scale(state['shp_left'],s['SHAMPOO_DAMPING']),
                         residual_scale(state['shp_right'],s['SHAMPOO_DAMPING']),group['lr'],
                         target_uw=s['SOAP_TARGET_UW'] if use_soap else s['NONSOAP_TARGET_UW'],
                         outward_scale=s['RADIAL_OUTWARD_SCALE'],inward_scale=s['RADIAL_INWARD_SCALE'],
                         rowfloor=s['ROWFLOOR'],rowfloor_rho=s['ROWFLOOR_RHO'],cwd=s['CWD'],**spectral,**base)
        if p not in self.entries:
            hl=torch.nn.Parameter(torch.zeros_like(factors[0]),requires_grad=self.learned)
            hr=torch.nn.Parameter(torch.zeros_like(factors[1]),requires_grad=self.learned)
            self.entries[p]={'hl':hl,'hr':hr,'lag':LagOneHypergradient()}
        e=self.entries[p]
        e['relative_left']=(record.left_scale*e['hl']).norm()/record.left.norm().clamp_min(1e-30)
        e['relative_right']=(record.right_scale*e['hr']).norm()/record.right.norm().clamp_min(1e-30)
        proposal=self.map(record,e['hl'],e['hr'])
        if not torch.isfinite(proposal).all():raise FloatingPointError('nonfinite optimizer proposal')
        if step%16==0:
            control=self.map(record,torch.zeros_like(e['hl']),torch.zeros_like(e['hr']))
            movement=proposal.double()-record.weight.double();fixed_movement=control.double()-record.weight.double()
            cosine_stats=torch.stack(((movement*fixed_movement).sum(),movement.square().sum(),fixed_movement.square().sum()))
            self.action_cosine_stats=cosine_stats if self.action_cosine_stats is None else self.action_cosine_stats+cosine_stats
            difference=(proposal.double()-control.double()).square().sum()
            self.action_difference_sq=difference if self.action_difference_sq is None else self.action_difference_sq+difference
        p.copy_(proposal)
        if self.learned and step<self.stop-1:
            envelope=scope_envelope(s,step)
            self.install_pending(e,record,envelope.derivative,step)
            self.metrics['next_forward_cotangent_coefficient']=envelope.derivative
        if use_soap:s['soap_update_preconditioner'](p.grad,state)

    def install_pending(self,e,record,coefficient,step):
        def contraction(cotangent):
            with torch.enable_grad():
                value=self.map(record,e['hl'],e['hr'])
                return (value*cotangent['weight']).sum()*coefficient
        e['pending_record']=record;e['pending_coefficient']=coefficient;e['pending_step']=step
        e['lag'].from_contraction(action_step=step,hyperparameters={'hl':e['hl'],'hr':e['hr']},
            contraction=contraction,guarded_tensors=tuple(getattr(record,f.name) for f in dataclasses.fields(record)
            if isinstance(getattr(record,f.name),torch.Tensor)))

    def state_dict(self):
        names={p:n for n,p in self.scope['model'].named_parameters()}
        return {'variant':getattr(self,'variant','legacy-extra-polar'),'entries':{names[p]:{k:e.get(k) for k in ['hl','hr','pending_record','pending_coefficient','pending_step']}
                           for p,e in self.entries.items()},
                'optimizer':None if self.optimizer is None else self.optimizer.state_dict(),
                'meta_steps':self.meta_steps}

    def load_state_dict(self,state):
        if state.get('variant','legacy-extra-polar')!=getattr(self,'variant','legacy-extra-polar'):raise ValueError('resume SOAP variant differs')
        if self.entries or self.optimizer is not None:raise RuntimeError('restore requires fresh hyperoptimizer')
        names=dict(self.scope['model'].named_parameters());parameters=[]
        for name,row in state['entries'].items():
            p=names[name];e={'hl':torch.nn.Parameter(row['hl'].detach().clone(),requires_grad=self.learned),
                            'hr':torch.nn.Parameter(row['hr'].detach().clone(),requires_grad=self.learned),'lag':LagOneHypergradient()}
            self.entries[p]=e;parameters.extend((e['hl'],e['hr']))
            if row['pending_record'] is not None:self.install_pending(e,row['pending_record'],row['pending_coefficient'],row['pending_step'])
        if state['optimizer'] is not None:
            self.optimizer=torch.optim.Adam(parameters,lr=self.lr,betas=(.9,.99),eps=1e-8)
            self.optimizer.load_state_dict(state['optimizer'])
        self.meta_steps=state['meta_steps']

    def maybe_checkpoint(self,completed_steps):
        period=int(os.environ.get('SOAP_CHECKPOINT_EVERY','500'))
        if not period or completed_steps%period or completed_steps==self.stop:return
        self.save_checkpoint(completed_steps,final=False)

    def save_checkpoint(self,completed_steps,final):
        from .soap_checkpoint import payload
        import shutil
        folder=self.output if final else Path(os.environ.get('SOAP_CHECKPOINT_ROOT',str(self.output)))/f'step{completed_steps:04d}'
        folder.mkdir(parents=True,exist_ok=True)
        rank=dist.get_rank();name=f'final_rank{rank}.pt' if final else f'rank{rank}.pt'
        temporary=folder/(name+'.incomplete');torch.save(payload(self.scope,self,completed_steps),temporary)
        temporary.replace(folder/name);dist.barrier()
        if rank==0:
            shutil.copy2(self.scope['logfile'],self.output/'upstream.log')
            (folder/'complete.json').write_text(json.dumps({'completed_steps':completed_steps,'world_size':dist.get_world_size(),
                'variant':self.variant,'seed':self.scope['SEED'],'meta_lr':self.lr,'stop_step':self.stop,'microbatch':self.scope['mbs']}))
        dist.barrier()

    def consume(self,step):
        self.action_difference_sq=None;self.action_cosine_stats=None
        if not self.learned or step==0:return
        parameters=[v for e in self.entries.values() for v in (e['hl'],e['hr'])]
        if self.optimizer is None:self.optimizer=torch.optim.Adam(parameters,lr=self.lr,betas=(.9,.99),eps=1e-8)
        self.optimizer.zero_grad(set_to_none=True)
        start=time.perf_counter()
        for p,e in self.entries.items():
            result=e['lag'].consume({'weight':p.grad.detach().float()/self.scope['batch_size']},gradient_step=step)
            result.assign_gradients({'hl':e['hl'],'hr':e['hr']})
            e['pending_record']=None;e['pending_coefficient']=None;e['pending_step']=None
        normsq=sum((h.grad.double().square().sum() for h in parameters if h.grad is not None),torch.zeros((),device=parameters[0].device,dtype=torch.float64))
        dist.all_reduce(normsq)
        norm=normsq.sqrt()
        if not torch.isfinite(norm):raise FloatingPointError('nonfinite hypergradient')
        multiplier=(1/(norm+1e-6)).clamp(max=1).float()
        for h in parameters:
            if h.grad is not None:h.grad.mul_(multiplier)
        self.optimizer.step();self.meta_steps+=1
        torch.cuda.synchronize()
        self.metrics={'hypergradient_norm':norm.item(),'hypergradient_clip_factor':multiplier.item(),
                      'meta_steps':self.meta_steps,'credit_seconds':time.perf_counter()-start}

    def log_step(self,step,loss):
        value=loss.detach().double().clone();dist.all_reduce(value)
        # Source loss is token SUM; caller sums all prescribed microbatches.
        value=value/self.scope['batch_size']
        hs=sum((e['hl'].double().square().sum()+e['hr'].double().square().sum() for e in self.entries.values()),torch.zeros((),device=value.device,dtype=torch.float64))
        dist.all_reduce(hs)
        ratios=torch.stack([e[k] for e in self.entries.values() for k in ('relative_left','relative_right')])
        ratio_stats=torch.stack((ratios.double().sum(),ratios.double().numel()*torch.ones((),device=ratios.device,dtype=torch.float64)))
        dist.all_reduce(ratio_stats)
        ratio_max=ratios.max();dist.all_reduce(ratio_max,op=dist.ReduceOp.MAX)
        now=time.perf_counter()
        row=dict(event='train',step=step+1,train_loss=value.item(),hyperparameter_norm=hs.sqrt().item(),residual_to_base_factor_mean=(ratio_stats[0]/ratio_stats[1]).item(),residual_to_base_factor_max=ratio_max.item(),
                 wall_seconds=now-self.last_time,gpt_forward_backward_seconds=self.gpt_seconds,base_optimizer_seconds=self.optimizer_seconds,**self.metrics)
        if self.action_difference_sq is not None:
            dist.all_reduce(self.action_difference_sq)
            row['proposal_difference_vs_zero_q_norm']=self.action_difference_sq.sqrt().item()
            dist.all_reduce(self.action_cosine_stats)
            dot,a,b=self.action_cosine_stats
            row['proposal_cosine_vs_zero_q']=((dot/(a*b).sqrt().clamp_min(1e-60)) if a*b>0 else dot.new_tensor(1.)).item()
        self.last_time=now
        if dist.get_rank()==0:
            with (self.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            self.wandb.log(row)

    def log_validation(self,step,value,live_value=None):
        if dist.get_rank()==0:
            row={'event':'validation','step':step,'val_loss':float(value),'objective':float(value),'live_val_loss':float(value if live_value is None else live_value)}
            with (self.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            self.wandb.log(row)

    def finish(self):
        self.save_checkpoint(self.stop,final=True)
        if dist.get_rank()==0:
            self.wandb.finish()
            (self.output/'finished.json').write_text(json.dumps({'completed_steps':self.stop,'world_size':dist.get_world_size()}))
