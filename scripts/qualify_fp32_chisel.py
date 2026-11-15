"""Bounded FP32 fitting-state qualification; never a scored GPT run."""
import argparse,json,math,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import wandb
from baseline.model import GPT
from baseline.execution import enable_deterministic_execution
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from rewriter.function_preservation import FunctionalPreservation, _LogitsView
from rewriter.gpt_precision import precision_metadata
from torch.func import functional_call
from torch.nn.attention import sdpa_kernel, SDPBackend
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.chisel_factory import build_chisel
from rewriter.primal_dual import PrimalDual
from scripts.train_multistate_chisel import build_proposal,mean_kl_and_backward,evaluate_sequences,counters

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--microbatch',type=int,default=4);p.add_argument('--mode',choices=['throughput','gradient'],default='throughput');p.add_argument('--compile-gpt',action='store_true');args=p.parse_args()
torch.set_num_threads(1);enable_deterministic_execution();torch.cuda.set_device(0);device=torch.device('cuda:0');args.output.mkdir(parents=True,exist_ok=True)
wb=wandb.init(project='chisel',name='chisel-fp32-'+args.mode+'-micro'+str(args.microbatch)+('-compiled' if args.compile_gpt else ''),mode='offline',dir=str(args.output),config=vars(args))
started=time.time();rows=[]
def emit(row):
 row['elapsed_seconds']=time.time()-started;rows.append(row);print(json.dumps(row,allow_nan=False),flush=True)
 with (args.output/'measurements.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 wb.log({k:v for k,v in row.items() if isinstance(v,(int,float))})
reader=StateBankReader('/scratch/checkpoints/learnable-weight-decay/chisel-state-bank-seed0-v1',verify_once=True,mmap_model=True)
cache=TeacherCacheReader('/scratch/checkpoints/learnable-weight-decay/chisel-teacher-cache1000-v1',verify_once=True,mmap=True)
# Only inputs/IDs are reused. BF16 teacher targets are never loaded into FP32 loss.
record=reader.read(999,device=device,model_device='cpu');panel=cache.read(999,'train');inputs=panel.inputs[:16];ids=panel.sequence_ids[:16]
engine=FunctionalPreservation(GPT(**reader.manifest['provenance']['model_config']).to(device).requires_grad_(False),gpt_precision='fp32',kl_dtype=torch.float64,microbatch_sequences=args.microbatch,teacher_cache_device=device)
engine.load_teacher_state({n:record.model[n] for n in engine.parameters},{n:record.model[n] for n in engine.buffers})
# Independent unmodified baseline module path: no shared epsilon monkeypatch.
with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
 reference=functional_call(_LogitsView(engine.view.gpt,'baseline'),
     ({'gpt.'+n:v for n,v in engine.parameters.items()},
      {'gpt.'+n:v for n,v in engine.buffers.items()}),(inputs[:1].to(device),),strict=True)
 explicit=engine._forward(engine.parameters,inputs[:1].to(device))
 baseline_max_difference=float((reference-explicit).abs().max())
 assert torch.equal(reference,explicit),baseline_max_difference
 del reference,explicit
emit(dict(event='unmodified_baseline_parity',state_id=999,sequences=1,
          maximum_logit_difference=baseline_max_difference,precision=precision_metadata('fp32')))
if args.compile_gpt:engine.enable_compiled_forward(fullgraph=True,dynamic=False)
tick=time.time();teachers=engine.cache_teacher(inputs,data_role='past_training',sequence_ids=ids);torch.cuda.synchronize();teacher_seconds=time.time()-tick
provider=lambda a,b:teachers[a//args.microbatch]
with torch.no_grad():zero=evaluate_sequences(engine,engine.parameters,inputs,provider)
assert zero==0.,zero
if args.mode=='gradient':
 saved=torch.load('/scratch/checkpoints/learnable-weight-decay/chisel-primal-dual-zero-v1/controller_epoch_0040.pt',map_location='cpu',weights_only=False);architecture=saved['metadata']['architecture']
else:
 saved=None;architecture=dict(name='nanogpt_blocks_patch_io_v1',rank=8,trajectory_dim=400,model_dim=512,num_layers=4,patch_size=64)
with torch.random.fork_rng(devices=[0]):torch.manual_seed(200001);actor=build_chisel(architecture).to(device)
if saved:actor.load_state_dict(saved['controller'])
forward=torch.compile(actor,fullgraph=True,dynamic=False);params_actor=list(actor.parameters());optimizer=torch.optim.Adam(params_actor,lr=2e-6,betas=(.9,.99),eps=1e-8);dual=PrimalDual([999],tolerance=1e-6,rho=.01,dual_lr=1e-4)
grouped=AcceleratedGroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
emit(dict(event='start',mode=args.mode,microbatch=args.microbatch,compile_gpt=args.compile_gpt,zero_edit_kl=zero,teacher_generation_seconds=teacher_seconds,sequence_ids=ids,gpt_precision='fp32',tf32=torch.backends.cuda.matmul.allow_tf32,all_gpt_parameters_fp32=all(p.dtype==torch.float32 for p in engine.parameters.values())))
if args.mode=='throughput':
 for update in range(3):
  tick=time.time();optimizer.zero_grad(set_to_none=True);phase={}
  parameters,reg,stats,editable,credit=build_proposal(forward,record.snapshot,engine,grouped,radius_fraction=.01,timings=phase)
  measured=mean_kl_and_backward(engine,parameters,reg,inputs,kl_budget=1e-4,state_weight=1.,actor_credit=credit,teacher_provider=provider,primal_dual=dual,state_id=999,timings=phase)
  norm=torch.nn.utils.clip_grad_norm_(params_actor,1.);optimizer.step();dual.step([(999,measured['mean_kl'])],expected_ids=[999]);torch.cuda.synchronize()
  emit(dict(event='update',update=update+1,wall_seconds=time.time()-tick,gradient_norm=float(norm),**measured,**stats,component_seconds=phase,compile_counters=counters(),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
  del parameters,reg,credit
 before=counters();other=reader.read(997,device=device,model_device='cpu');out=forward(other.snapshot);del out;after=counters()
 emit(dict(event='changed_snapshot_compile_check',before=before,after=after))
else:
 class RawKL:
  def terms(self,value,state_id):return dict(kl_penalty=value,kl_coefficient=1.)
 tick=time.time();parameters,reg,stats,editable,credit=build_proposal(forward,record.snapshot,engine,grouped,radius_fraction=.01)
 measured=mean_kl_and_backward(engine,parameters,reg*0,inputs,kl_budget=1e-4,actor_credit=credit,teacher_provider=provider,primal_dual=RawKL(),state_id=999)
 baseline=measured['mean_kl'];gradient=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params_actor];gnorm=math.sqrt(sum(float(g.double().square().sum()) for g in gradient));direction=[-g/gnorm for g in gradient];original=[p.detach().clone() for p in params_actor]
 del parameters,reg,credit
 step_sq=0.
 for group in saved['optimizer']['param_groups']:
  b1,b2=group['betas']
  for key in group['params']:
   s=saved['optimizer']['state'].get(key)
   if not s:continue
   step=float(s['step']);m=s['exp_avg'].double()/(1-b1**step);v=s['exp_avg_sq'].double()/(1-b2**step);delta=5e-7*m/(v.sqrt()+group['eps']);step_sq+=float(delta.square().sum())
 step_norm=math.sqrt(step_sq)
 emit(dict(event='gradient_baseline',objective=baseline,mean_kl=baseline,raw_kl_gradient_norm=gnorm,last_adam_step_norm=step_norm,saved_policy_updates=160,global_scale=stats['global_scale']))
 for multiple in [-10.,-1.,-.1,-.001,.001,.1,1.,10.]:
  distance=multiple*step_norm
  with torch.no_grad():
   for p,base,d in zip(params_actor,original,direction):p.copy_(base+distance*d)
  parameters,reg,stats,editable,credit=build_proposal(forward,record.snapshot,engine,grouped,radius_fraction=.01)
  kl=evaluate_sequences(engine,{n:p.detach() for n,p in parameters.items()},inputs,provider)
  emit(dict(event='direction',multiple=multiple,parameter_distance=distance,mean_kl=kl,objective=kl,measured_change=kl-baseline,predicted_change=-gnorm*distance,global_scale=stats['global_scale']))
  del parameters,reg,credit
 with torch.no_grad():
  for p,base in zip(params_actor,original):p.copy_(base)
 parameters,reg,stats,editable,credit=build_proposal(forward,record.snapshot,engine,grouped,radius_fraction=.01);restored=evaluate_sequences(engine,{n:p.detach() for n,p in parameters.items()},inputs,provider);assert restored==baseline,(restored,baseline)
 emit(dict(event='restored',mean_kl=restored,optimizer_updates=0,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
result=dict(rows=rows,elapsed_seconds=time.time()-started,compute=engine.compute.as_dict(),compile_counters=counters(),state_id=999,role='fitting',precision=precision_metadata('fp32'))
(args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');wb.finish();print(json.dumps(dict(event='finished',elapsed_seconds=result['elapsed_seconds'],wandb=wb.dir)),flush=True)
