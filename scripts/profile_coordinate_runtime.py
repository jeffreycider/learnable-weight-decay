"""Bounded actual continuation-driver runtime attribution, no backend changes.

CUDA-event spans are elapsed stream intervals (including host gaps), not GPU
busy-union estimates. CPU wall spans and CUDA spans are reported separately.
"""
from pathlib import Path
import argparse,collections,functools,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import scripts.continue_coordinate_chisel as continuation
import scripts.qualify_coordinate_chisel as contexts_module
import rewriter.tokenwise_reference as tokenwise
import rewriter.image_tokenwise_panel as image_panel


def backward_kind(tensors):
    values=(tensors,) if isinstance(tensors,torch.Tensor) else tuple(tensors)
    if len(values)==1 and values[0].ndim>=2 and values[0].shape[-1] in (10,50304):return 'student_backward'
    if values and values[0].ndim==0:return 'norm_edit_backward'
    return 'actor_backward'


class Recorder:
    def __init__(self):self.active=False;self.rows=[];self.spans=[];self.domain='unknown'
    def call(self,name,function,*args,**kwargs):
        if not self.active:return function(*args,**kwargs)
        first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True);first.record();start=time.perf_counter()
        try:return function(*args,**kwargs)
        finally:
            wall=1000*(time.perf_counter()-start);last.record();self.spans.append((name,wall,first,last))
    def start(self):
        if self.active:self.finish()
        torch.cuda.synchronize();self.active=True;self.spans=[];self.start_wall=time.perf_counter()
    def finish(self):
        if not self.active:return
        torch.cuda.synchronize();wall=1000*(time.perf_counter()-self.start_wall);totals={}
        for name,cpu,a,b in self.spans:
            row=totals.setdefault(name,dict(calls=0,cpu_wall_ms=0.,cuda_interval_ms=0.));row['calls']+=1;row['cpu_wall_ms']+=cpu;row['cuda_interval_ms']+=a.elapsed_time(b)
        # Proposal includes the actor subinterval: expose only its exclusive rest.
        if 'proposal_total' in totals:
            proposal=totals.pop('proposal_total');actor=totals.get('actor_forward',{})
            for key in ('cpu_wall_ms','cuda_interval_ms'):proposal[key]-=actor.get(key,0.)
            totals['proposal_norm_edit_forward']=proposal
        self.rows.append(dict(update_index=len(self.rows)+1,wall_ms=wall,phases=totals,cpu_unattributed_ms=wall-sum(v['cpu_wall_ms'] for v in totals.values())))
        self.active=False;self.spans=[]


def profile(recipe,output):
    rec=Recorder();restores=[]
    def patch(obj,name,value):restores.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
    def wrapped_phase(name,function):return lambda *a,**k:rec.call(name,function,*a,**k)
    original_load=contexts_module.FixedContexts.load
    def load(self,domain,*args,**kwargs):rec.domain=domain;return rec.call('state_load_'+domain,original_load,self,domain,*args,**kwargs)
    patch(contexts_module.FixedContexts,'load',load)
    original_measure=contexts_module.FixedContexts.measure
    def measure(self,context,actor,private,*args,**kwargs):
        domain=context['domain'];rec.domain=domain;context=dict(context);context['teacher']=wrapped_phase('teacher_'+domain,context['teacher']);engine=context['engine'];old_forward=engine._forward
        engine._forward=wrapped_phase('student_forward_'+domain,old_forward)
        try:return original_measure(self,context,wrapped_phase('actor_forward',actor),private,*args,**kwargs)
        finally:engine._forward=old_forward
    patch(contexts_module.FixedContexts,'measure',measure)
    patch(contexts_module,'build_proposal',wrapped_phase('proposal_total',contexts_module.build_proposal))
    original_loss=tokenwise.logit_terms_and_cotangent
    loss=lambda *a,**k:rec.call('loss_cotangent_'+rec.domain,original_loss,*a,**k)
    patch(tokenwise,'logit_terms_and_cotangent',loss);patch(image_panel,'logit_terms_and_cotangent',loss)
    original_backward=torch.autograd.backward
    def backward(tensors,*a,**k):return rec.call(backward_kind(tensors)+'_'+rec.domain,original_backward,tensors,*a,**k)
    patch(torch.autograd,'backward',backward)
    original_step=torch.optim.Adam.step
    def step(self,*a,**k):return rec.call('shared_adam' if any(g.get('role')=='representation' for g in self.param_groups) else 'private_adam',original_step,self,*a,**k)
    patch(torch.optim.Adam,'step',step)
    original_zero=torch.optim.Optimizer.zero_grad
    def zero(self,*a,**k):
        if any(g.get('role')=='representation' for g in self.param_groups):rec.start()
        return rec.call('zero_grad',original_zero,self,*a,**k)
    patch(torch.optim.Optimizer,'zero_grad',zero)
    patch(continuation,'clip_private_gradients',wrapped_phase('private_clip',continuation.clip_private_gradients))
    patch(continuation,'clip_shared_continuation',wrapped_phase('shared_clip',continuation.clip_shared_continuation))
    patch(continuation,'save_atomic',wrapped_phase('checkpoint',continuation.save_atomic))
    try:continuation.run(recipe,output/'driver')
    finally:
        rec.finish()
        for obj,name,old in reversed(restores):setattr(obj,name,old)
        summary=dict(scope='Actual continuation driver with resident four private maps; full fitting2GPT16+2CNN1024, cached teachers; no scientific state reused after these disposable updates. Timers add CUDA events but no per-phase synchronization. CUDA intervals are not GPU busy time. Checkpoint interval shown separately; stdout/file/W&B overhead remains in CPU-unattributed.',recipe=json.loads(Path(recipe).read_text()),updates=rec.rows)
        (output/'profile.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recipe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);profile(a.recipe,a.output)
