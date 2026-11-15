"""Saved adaptive FP64 GPT single-visit diagnostic; no training migration.

One fixed TRAIN sequence first: dense and thin start from identical saved
shared/private Adam and RNG. No private-inner updates or CNN contribution;
the .5-weighted GPT visit is explicitly a component qualification.
"""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.gpt_precision import configure_precision
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.adaptive_preservation import build_adaptive_proposal
from rewriter.adaptive_thin_student import AdaptiveExecutionCapture,adaptive_thin_tokenwise_panel
from rewriter.factorized_fp32 import factorized_logits
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual,tokenwise_panel
from rewriter.coordinate_optimization import clip_private_gradients,clip_shared_continuation
from rewriter.rank_preservation import topk_preservation_flags
from scripts.qualify_coordinate_chisel import FixedContexts
from scripts.qualify_coordinate_fullbank import read_joint
from scripts.train_coordinate_chisel import build_actor,build_shared_adam,set_private_lr,restore_rng
def relative(a,b):
 names=set(a)|set(b);sq=base=maximum=0.
 for n in names:
  x=a.get(n);y=b.get(n)
  if x is None:x=torch.zeros_like(y)
  if y is None:y=torch.zeros_like(x)
  x=x.detach().cpu().double();y=y.detach().cpu().double();delta=x-y
  sq+=float(delta.square().sum());base+=float(x.square().sum());maximum=max(maximum,float(delta.abs().max()))
 return dict(relative_l2=(sq/max(base,1e-300))**.5,absolute_l2=sq**.5,max_abs=maximum,none_mask_equal=set(a)==set(b))


def cpu_state(module):return {n:p.detach().cpu().clone() for n,p in module.named_parameters()}
def grads(module):return {n:p.grad.detach().cpu().clone() for n,p in module.named_parameters() if p.grad is not None}

def run(a):
 enable_deterministic_execution();torch.cuda.set_device(0);configure_precision('fp32');torch.set_num_threads(1)
 device=torch.device('cuda',0);out=Path(a.output);out.mkdir(exist_ok=False)
 recipe=json.loads(Path(a.recipe).read_text());assert recipe['architecture']['factor_precision']=='fp64'
 saved,ranks,payloads=read_joint(a.checkpoint);assert all(s['next_global_batch']==96 for s in ranks.values())
 base=json.loads(Path(recipe['base_recipe']).read_text());contexts=FixedContexts(base,device,gpt_sequences=16,cifar_images=128,action_family='adaptive_ucv_v1')
 ctx=contexts.load('gpt',a.state);engine=ctx['engine'];ctx['inputs']=ctx['inputs'][:a.sequences];ctx['ids']=ctx['ids'][:a.sequences]
 # Preserve the original cache generation GEMM shape, then slice exact targets.
 from rewriter.function_preservation import TeacherBatch
 block=ctx['teacher'](0,16)
 fixed_teacher=TeacherBatch(block.inputs[:a.sequences].clone(),block.log_probs[:a.sequences].clone(),block.teacher_id,block.data_role,tuple(block.sequence_ids[:a.sequences]) if block.sequence_ids is not None else None)
 del block
 engine.microbatch_sequences=a.sequences
 def teacher(start,stop):
  if start!=0 or stop!=a.sequences:raise ValueError('fixed one-microbatch teacher slice')
  return fixed_teacher
 ctx['teacher']=teacher
 actor=build_actor(recipe).to(device);key=('gpt',0,a.state);token,payload=next((k,v) for k,v in payloads.items() if tuple(v['identity']['state_key'])==key)
 private=actor.make_private(ctx['snapshot'],state_key=key,seed=19)
 shared_opt=build_shared_adam(actor,recipe['shared_optimizer']);pcfg=recipe['private_optimizer'];private_opt=torch.optim.Adam(private.parameters(),lr=pcfg['lr'],eps=pcfg['eps'],betas=tuple(pcfg['betas']))
 owner=next(s for s in ranks.values() if token in s['visits']);cfg=TokenwiseConfig(**recipe['objectives']['gpt']);dual=TokenwiseDual(list(owner['duals']['gpt']['multipliers']),config=cfg,dual_lr=recipe['dual_lr'])
 def restore():
  actor.load_state_dict(saved['model']);shared_opt.load_state_dict(copy.deepcopy(saved['optimizer']));private.load_state_dict(payload['private']);private_opt.load_state_dict(copy.deepcopy(payload['optimizer']));dual.load_state_dict(copy.deepcopy(owner['duals']['gpt']));actor.zero_grad(set_to_none=True);private.zero_grad(set_to_none=True);restore_rng(saved['rng']['0'],device)
 def proposal():
  capture=AdaptiveExecutionCapture()
  def forward(snapshot):return actor(snapshot,private,weight_cores=query_weight_cores(engine.parameters,private,embedding_names=tuple(n for n,t in engine.transpose_storage.items() if t)))
  with torch.enable_grad():params,reg,stats,_,credit=build_adaptive_proposal(forward,ctx['snapshot'],engine,ctx['grouped'],radius_fraction=.01,norm_squared=ctx['norm_squared'],execution_capture=capture)
  return params,reg,stats,credit,capture
 if a.runtime_only:
  timings=[]
  def visit(mode):
   params,reg,stats,credit,capture=proposal()
   kwargs=dict(config=cfg,dual=dual,state_id=a.state,state_weight=.5,actor_credit=credit,teacher_provider=ctx['teacher'],backward=True)
   row=(tokenwise_panel(engine,params,reg,ctx['inputs'],**kwargs) if mode=='dense' else adaptive_thin_tokenwise_panel(engine,capture,params,reg,ctx['inputs'],**kwargs))
   for p in private.parameters():
    if p.grad is not None:p.grad.div_(.5)
   clip_private_gradients(private.parameters(),pcfg['clip']);private_opt.step();private.zero_grad(set_to_none=True);dual.step([(a.state,row['mean_kl'])],expected_ids=[a.state]);clip_shared_continuation(shared_opt,'separate_representation',max_norm=recipe['shared_optimizer']['clip']);shared_opt.step();actor.zero_grad(set_to_none=True)
   return row['objective']
  for block,mode in enumerate(('dense','thin','thin','dense')):
   restore();set_private_lr(private_opt,pcfg,payload['visits']);visit(mode);torch.cuda.synchronize()
   for repeat in range(2):
    restore();set_private_lr(private_opt,pcfg,payload['visits']);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start=time.perf_counter();begin.record();objective=visit(mode);end.record();torch.cuda.synchronize()
    timings.append(dict(block=block,mode=mode,repeat=repeat,wall_seconds=time.perf_counter()-start,cuda_seconds=begin.elapsed_time(end)/1000,peak_allocated_bytes=torch.cuda.max_memory_allocated(),objective=objective))
  summary=dict(scope='Warmed saved96 GPT6 TRAIN16 single .5 weighted visit, one private/shared Adam; no inner/CNN; restores outside timing; no diagnostic gradient copies or endpoint forwards',source_update=96,sequences=a.sequences,timings=timings,warmups=4,measured=8,compute=dict(student_forwards=12,student_backwards=12,student_tokens=12*ctx['inputs'].numel(),cached_teacher_head_forwards=1,teacher_transformer_forwards=0),no_query=True,no_adoption=True)
  (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True);return
 restore();initial={'shared':cpu_state(actor),'private':cpu_state(private)};results={};states={};traces={}
 # Two initial forwards on identical realized weights, no gradients.
 params,reg,stats,credit,capture=proposal()
 with torch.no_grad():
  x=ctx['inputs'].to(device);dense=engine._forward(params,x);thin=factorized_logits(engine.view.gpt,engine.parameters,engine.buffers,x,capture.action,capture.scale,realized_parameters=params,linear_backend='ieee',dx_backend='ieee');forward=dict(bitexact=torch.equal(dense,thin),max_abs=float((dense-thin).abs().max()))
 del params,reg,stats,credit,capture,dense,thin
 setup_peak_allocated_bytes=torch.cuda.max_memory_allocated()
 for mode in a.order.split(','):
  restore();set_private_lr(private_opt,pcfg,payload['visits']);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();params,reg,stats,credit,capture=proposal()
  kwargs=dict(config=cfg,dual=dual,state_id=a.state,state_weight=.5,actor_credit=credit,teacher_provider=ctx['teacher'],backward=True)
  if mode=='dense':row=tokenwise_panel(engine,params,reg,ctx['inputs'],**kwargs)
  else:row=adaptive_thin_tokenwise_panel(engine,capture,params,reg,ctx['inputs'],**kwargs)
  for p in private.parameters():
   if p.grad is not None:p.grad.div_(.5)
  traces[mode]={'shared':grads(actor),'private':grads(private)}
  clip_private_gradients(private.parameters(),pcfg['clip']);private_opt.step();private.zero_grad(set_to_none=True);dual.step([(a.state,row['mean_kl'])],expected_ids=[a.state]);clip_shared_continuation(shared_opt,'separate_representation',max_norm=recipe['shared_optimizer']['clip']);shared_opt.step();actor.zero_grad(set_to_none=True)
  torch.cuda.synchronize();results[mode]=dict(seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),row=row,stats=stats)
  states[mode]=dict(shared=copy.deepcopy(actor.state_dict()),private=copy.deepcopy(private.state_dict()),shared_optimizer=copy.deepcopy(shared_opt.state_dict()),private_optimizer=copy.deepcopy(private_opt.state_dict()),dual=copy.deepcopy(dual.state_dict()))
  del params,reg,stats,credit,capture
 gradients={k:relative(traces['dense'][k],traces['thin'][k]) for k in initial};deltas={k:relative({n:states['dense'][k][n].detach().cpu()-v for n,v in initial[k].items()},{n:states['thin'][k][n].detach().cpu()-v for n,v in initial[k].items()}) for k in initial}
 endpoint={};logits={}
 for mode in ('dense','thin'):
  actor.load_state_dict(states[mode]['shared']);private.load_state_dict(states[mode]['private']);dual.load_state_dict(states[mode]['dual']);params,reg,stats,credit,capture=proposal()
  endpoint[mode]=tokenwise_panel(engine,params,reg,ctx['inputs'],config=cfg,dual=dual,state_id=a.state,state_weight=.5,teacher_provider=ctx['teacher'],backward=False)
  with torch.no_grad():logits[mode]=engine._forward(params,ctx['inputs'].to(device)).detach().cpu()
  del params,reg,stats,credit,capture
 x=logits['dense'].double().reshape(-1,logits['dense'].shape[-1]);y=logits['thin'].double().reshape_as(x);sumkl=maximum=0.;strict=0
 for off in range(0,len(x),64):
  u=x[off:off+64];v=y[off:off+64];lp=u.log_softmax(-1);lq=v.log_softmax(-1);kl=(lp.exp()*(lp-lq)).sum(-1);sumkl+=float(kl.sum());maximum=max(maximum,float(kl.max()));strict+=int(topk_preservation_flags(u,v,k=100)['strict_topk_changed'].sum())
 summary=dict(scope='Single savedFP64 GPT visit at .5 sharedweight/privateunweighted, no inner/CNN contribution; one matched Adamstep each, no deployment',source_checkpoint=str(a.checkpoint),source_update=96,state=a.state,sequences=a.sequences,execution_order=a.order,setup_peak_allocated_bytes=setup_peak_allocated_bytes,sequence_ids=ctx['ids'],private_tick=payload['visits'],forward=forward,gradients=gradients,Adam_deltas=deltas,endpoint=endpoint,mutual=dict(bitexact=torch.equal(logits['dense'],logits['thin']),max_logit_abs=float((x-y).abs().max()),meanKL=sumkl/len(x),maxKL=maximum,strict_changed_positions=strict),results=results,compute=dict(student_forwards=8,student_backwards=2,student_tokens=8*ctx['inputs'].numel(),student_backward_tokens=2*ctx['inputs'].numel(),cached_teacher_head_forwards=1,cached_teacher_head_sequences=16,teacher_transformer_forwards=0),no_query=True,performance_claim=False)
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');torch.save(dict(initial=initial,traces=traces,states=states),out/'numeric_evidence.pt');print(json.dumps(summary),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--state',type=int,default=6,choices=[6,499]);p.add_argument('--sequences',type=int,default=1,choices=[1,16]);p.add_argument('--runtime-only',action='store_true');p.add_argument('--order',default='dense,thin',choices=['dense,thin','thin,dense']);a=p.parse_args()
 try:run(a)
 except torch.cuda.OutOfMemoryError:
  Path(a.output).mkdir(exist_ok=True);(Path(a.output)/'failure.json').write_text(json.dumps(dict(error='CUDA OOM',sequences=a.sequences,no_automatic_retry=True))+'\n');raise
