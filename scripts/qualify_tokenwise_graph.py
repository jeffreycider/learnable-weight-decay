"""Isolated strict tokenwise CUDA graph: full checkpoint VJP and complete update."""
from pathlib import Path
import argparse,copy,gc,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from dataclasses import asdict,replace
from baseline.execution import enable_deterministic_execution
from baseline.model import GPT
from rewriter.chisel_factory import build_chisel
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from rewriter.offline_systems import CachedBaseNormPreservation,VerifiedStateConstants,select_panel_blocks
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.edit_scope import scoped_snapshot
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual,tokenwise_panel
from rewriter.graph_tokenwise import TokenwiseThinGradientGraph
from rewriter.meta_optimizer import build_meta_adam
from scripts.train_multistate_chisel import build_proposal
from scripts.compare_tokenwise_trainer_outputs import compare

def relative(a,b):return float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-300))
def main():
 p=argparse.ArgumentParser()
 for k in ('state-bank','teacher-cache','policy','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--restore-saved-state',action='store_true');p.add_argument('--state-weight',type=float,default=None);p.add_argument('--policy-sha256');p.add_argument('--teacher-representation',choices=('log_probs','logits'),default='log_probs');p.add_argument('--experimental-dx',action='store_true')
 p.add_argument('--exact-teacher-ties',action='store_true');p.add_argument('--ranking-k',type=int,default=100)
 p.add_argument('--sequences',type=int,default=16);p.add_argument('--states',nargs='+',type=int,default=[6,999]);p.add_argument('--repeats',type=int,default=2);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 if a.experimental_dx:
  if a.exact_teacher_ties:raise ValueError('combined experimental arithmetic and new ranking is not qualified')
  from rewriter.graph_tokenwise_experimental import require_compensated_dx_environment
  require_compensated_dx_environment()
 enable_deterministic_execution();torch.set_num_threads(1);device=torch.device('cuda:0')
 import wandb
 run=wandb.init(project='chisel',mode='offline',dir=str(a.output),config={k:str(v) for k,v in vars(a).items()})
 stream=(a.output/'metrics.jsonl').open('w')
 def log(row):stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
 reader=StateBankReader(a.state_bank,verify_once=True,mmap_model=True);cache=TeacherCacheReader(a.teacher_cache,verify_once=True,mmap=True)
 engine=CachedBaseNormPreservation(GPT(**reader.manifest['provenance']['model_config']).to(device).requires_grad_(False),microbatch_sequences=16,teacher_cache_device=device,kl_dtype=torch.float64,gpt_precision='fp32')
 constants=VerifiedStateConstants(engine,AcceleratedGroupedNormObjective)
 saved=torch.load(a.policy,map_location='cpu',weights_only=False);policy_sha=hashlib.sha256(a.policy.read_bytes()).hexdigest()
 if a.policy_sha256 and policy_sha!=a.policy_sha256:raise ValueError('policy SHA mismatch')
 actor=build_chisel(saved['metadata']['architecture']).to(device);actor.load_state_dict(saved['controller']);forward=torch.compile(actor,fullgraph=True,dynamic=False)
 params=list(actor.parameters());cfg=TokenwiseConfig(1e-10,1e-10,1e-8,.01,ranking_weight=20435615.123355106,dual_update_mode='clipped_relative')
 saved_objective=copy.deepcopy(saved['metadata'].get('tokenwise_objective'))
 if a.restore_saved_state:
  cfg=TokenwiseConfig(**saved['metadata']['tokenwise_objective']);dual_state=copy.deepcopy(saved['primal_dual'])
  if a.exact_teacher_ties:
   cfg=replace(cfg,ranking_k=a.ranking_k,ranking_tie_policy='exact_teacher_ties',loss_backend='reference')
   dual_state['config']=asdict(cfg) # Explicit diagnostic objective transition; all saved values/clocks retained.
  dual=TokenwiseDual(dual_state['multipliers'],config=cfg,dual_lr=dual_state['dual_lr']);dual.load_state_dict(dual_state)
 else:
  if a.exact_teacher_ties:cfg=replace(cfg,ranking_k=a.ranking_k,ranking_tie_policy='exact_teacher_ties')
  dual=TokenwiseDual(a.states,config=cfg,dual_lr=.01);dual.multipliers={s:.01 for s in a.states}
 fast_backend='tiled_tie_fp64_v1' if a.exact_teacher_ties else 'tiled_fp64_v1'
 state_weight=1/len(a.states) if a.state_weight is None else a.state_weight
 if state_weight<=0:raise ValueError('positive checkpoint weight required')
 optimizer=build_meta_adam(actor,lr=1e-6,representation_lr_multiplier=10);initial_optimizer=copy.deepcopy(saved['optimizer'] if a.restore_saved_state else optimizer.state_dict())
 initial_duals={state:dual.multipliers[state] for state in a.states}
 def restore():actor.load_state_dict(saved['controller']);optimizer.load_state_dict(copy.deepcopy(initial_optimizer));actor.zero_grad(set_to_none=True)
 def vec():return torch.cat([p.grad.detach().flatten() for p in params]).cpu()
 def read(state):return reader.read(state,model_device='cpu',device=device),select_panel_blocks(cache.read(state,'train'),range(a.sequences//16))
 def reference(state,loss_backend='reference'):
  record,panel=read(state);constants.load(cache,record);grouped=constants.grouped();norm=constants.radius_squared(grouped);actor.zero_grad(set_to_none=True)
  edited,regularizer,stats,_,credit=build_proposal(forward,record.snapshot,engine,grouped,radius_fraction=.01,norm_squared=norm,shrinkage_coefficient=1.)
  return tokenwise_panel(engine,edited,regularizer,panel.inputs,config=replace(cfg,loss_backend=loss_backend),dual=dual,state_id=state,state_weight=state_weight,actor_credit=credit,teacher_provider=lambda start,stop:panel.teacher_batch(engine,start,stop))
 gradients={};metrics={};tiled_checks=[]
 for state in a.states:
  metrics[state]=reference(state);gradients[state]=vec();log(dict(event='reference',state=state,**metrics[state]))
  if a.exact_teacher_ties:
   fast_metrics=reference(state,fast_backend);row=dict(state=state,gradient_relative=relative(vec(),gradients[state]),objective=fast_metrics['objective'],reference_objective=metrics[state]['objective']);tiled_checks.append(row);log(dict(event='tiled_loss_qualification',**row))
 state=a.states[0];dual.multipliers[state]=initial_duals[state]+.01;reference(state);changed_gradient=vec();dual.multipliers[state]=initial_duals[state]
 accum=[torch.zeros_like(p) for p in params];timings=[]
 for repeat in range(a.repeats):
  restore();torch._foreach_zero_(accum);torch.cuda.synchronize();start=time.perf_counter()
  for state in a.states:reference(state);torch._foreach_add_(accum,[p.grad for p in params])
  for p,g in zip(params,accum):p.grad=g
  torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize();elapsed=time.perf_counter()-start
  timings.append(dict(backend='reference',seconds=elapsed));log(dict(event='timing',**timings[-1]))
 ref_policy={n:p.detach().cpu().clone() for n,p in actor.state_dict().items()};ref_optimizer=copy.deepcopy(optimizer.state_dict())
 for repeat in range(a.repeats):
  restore();torch._foreach_zero_(accum);torch.cuda.synchronize();start=time.perf_counter()
  for state in a.states:reference(state,fast_backend);torch._foreach_add_(accum,[p.grad for p in params])
  for p,g in zip(params,accum):p.grad=g
  torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize();elapsed=time.perf_counter()-start
  timings.append(dict(backend='tiled_eager',seconds=elapsed));log(dict(event='timing',**timings[-1]))
 tiled_policy={n:p.detach().cpu().clone() for n,p in actor.state_dict().items()}
 restore();actor.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
 record,panel=read(a.states[0]);constants.load(cache,record);grouped=constants.grouped();snapshot=scoped_snapshot(record.snapshot,constants.radius_squared(grouped),.01,scope='grouped')
 graph_type=TokenwiseThinGradientGraph
 if a.exact_teacher_ties:
  from rewriter.graph_tokenwise_tie import TieAwareTokenwiseGraph
  graph_type=TieAwareTokenwiseGraph
 if a.experimental_dx:
  from rewriter.graph_tokenwise_experimental import CompensatedDxTokenwiseGraph
  graph_type=CompensatedDxTokenwiseGraph
 graph=graph_type(actor,forward,snapshot,engine,grouped,panel.inputs.to(device),None,dual,a.states[0],state_weight=state_weight,teacher_features=panel.features.to(device,copy=True),dynamic_zero_groups=True,token_config=cfg,teacher_representation=a.teacher_representation)
 start=time.perf_counter();graph.capture(warmups=1);torch.cuda.synchronize();cold=time.perf_counter()-start;signature=graph.storage_signature();log(dict(event='capture',seconds=cold,peak_bytes=torch.cuda.max_memory_allocated()))
 checks=[]
 for state in a.states:
  keys=('argmax_changes','topk_order_changes','topk_membership_changes','tie_aware_argmax_changes','strict_order_changes','strict_membership_changes','strict_topk_changes') if a.exact_teacher_ties else ('argmax_changes','top10_order_changes','top10_membership_changes')
  record,panel=read(state);graph.load_checkpoint(cache,record,panel,dual,state_constants=constants);out=graph.replay();graph.check_finite();rel=relative(vec(),gradients[state]);check=dict(state=state,gradient_relative=rel,mean_kl=float(out['mean_kl']),reference_mean_kl=metrics[state]['mean_kl'],objective=float(out['objective']),reference_objective=metrics[state]['objective'],rank_counts=out['rank_change_counts'].cpu().tolist(),reference_rank_counts=[metrics[state][key] for key in keys],addresses_unchanged=signature==graph.storage_signature());checks.append(check);log(dict(event='qualification',**check))
  if rel>.003:raise RuntimeError('raw actor gradient gate failed')
 state=a.states[0];dual.multipliers[state]=initial_duals[state]+.01;record,panel=read(state);graph.load_checkpoint(cache,record,panel,dual,state_constants=constants);graph.replay();changed_relative=relative(vec(),changed_gradient);log(dict(event='changed_dual',state=state,multiplier=dual.multipliers[state],gradient_relative=changed_relative));dual.multipliers[state]=initial_duals[state]
 if changed_relative>.003:raise RuntimeError('changed dual gradient gate failed')
 for repeat in range(a.repeats):
  restore();torch._foreach_zero_(accum);torch.cuda.synchronize();start=time.perf_counter()
  for state in a.states:
   record,panel=read(state);graph.load_checkpoint(cache,record,panel,dual,state_constants=constants);graph.accumulate_into(accum);graph.check_finite()
  for p,g in zip(params,accum):p.grad=g
  torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize();elapsed=time.perf_counter()-start
  timings.append(dict(backend='graph',seconds=elapsed));log(dict(event='timing',**timings[-1]))
 graph_policy={n:p.detach().cpu().clone() for n,p in actor.state_dict().items()};graph.close();gc.collect();torch.cuda.empty_cache()
 delta=lambda policy:torch.cat([(policy[n]-saved['controller'][n]).flatten() for n in policy])
 delta_error=relative(delta(graph_policy),delta(ref_policy));tiled_delta_error=relative(delta(tiled_policy),delta(ref_policy));outputs=[]
 for state in a.states:
  record,panel=read(state);constants.load(cache,record);grouped=constants.grouped();norm=constants.radius_squared(grouped);edits=[]
  policies=(ref_policy,graph_policy,tiled_policy) if a.exact_teacher_ties else (ref_policy,graph_policy)
  for policy in policies:
   actor.load_state_dict(policy)
   with torch.enable_grad():proposal=build_proposal(forward,record.snapshot,engine,grouped,radius_fraction=.01,norm_squared=norm,shrinkage_coefficient=1.)
   edits.append({n:v.detach() for n,v in proposal[0].items()});del proposal
  for start in range(0,len(panel.inputs),16):
   with torch.no_grad():
    inputs=panel.inputs[start:start+16].to(device);x=engine._forward(edits[0],inputs)
    for candidate,edit in zip(('graph','tiled_eager'),edits[1:]):
     y=engine._forward(edit,inputs);row=dict(state=state,candidate=candidate,**compare(x,y));outputs.append(row);del y
   del x
  del edits
 gates=dict(changed_dual=changed_relative<=.003,gradient=all(x['gradient_relative']<=.003 for x in checks),adam_delta=delta_error<=.003,mutual_mean=all(max(x['reference_to_tiled_kl'],x['tiled_to_reference_kl'])<=1e-11 for x in outputs),mutual_token=all(max(x['max_reference_to_tiled_token_kl'],x['max_tiled_to_reference_token_kl'])<=1e-10 for x in outputs))
 if a.exact_teacher_ties:gates.update(tiled_gradient=all(x['gradient_relative']<=.003 for x in tiled_checks),tiled_adam_delta=tiled_delta_error<=.003,rank_counts=all(x['rank_counts']==x['reference_rank_counts'] for x in checks))
 result=dict(scope='strict tokenwise graph complete local update, verified state loading and teacher-head included; no distributed claim',policy_sha256=policy_sha,restored_saved_optimizer_duals=a.restore_saved_state,state_weight=state_weight,initial_duals=initial_duals,teacher_representation=a.teacher_representation,compilation_counters={str(k):dict(v) for k,v in torch._dynamo.utils.counters.items()},config=cfg.__dict__,states=a.states,sequences=a.sequences,gates=gates,checks=checks,changed_dual_gradient_relative=changed_relative,timings=timings,cold_capture_seconds=cold,adam_delta_relative=delta_error,direct_outputs=outputs,peak_bytes=torch.cuda.max_memory_allocated())
 result.update(experimental_dx=a.experimental_dx,arithmetic='IEEE FP32 forward; compensated TF32 activation-gradient GEMMs only' if a.experimental_dx else 'strict FP32 student',default_torch_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,default_torch_cudnn_tf32=torch.backends.cudnn.allow_tf32)
 if a.exact_teacher_ties:
  result.update(explicit_diagnostic_objective_transition=True,saved_objective=saved_objective,tiled_checks=tiled_checks,tiled_adam_delta_relative=tiled_delta_error,actor_architecture=saved['metadata']['architecture'])
 if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:raise RuntimeError('default FP32 arithmetic changed')
 if a.experimental_dx:result['scope']='Experimental compensated-dx tokenwise graph, complete local saved-state update; teacher head and state loading included; no distributed or production claim'
 (a.output/'summary.json').write_text(json.dumps(result,indent=2));torch.save(dict(reference=ref_policy,graph=graph_policy,tiled_eager=tiled_policy),a.output/'qualification_policies.pt');log(result);run.log({'objective':timings[-1]['seconds'],'passed':all(gates.values())});run.finish();stream.close()
 if not all(gates.values()):raise RuntimeError('finite update qualification failed; raw evidence preserved')
if __name__=='__main__':main()
