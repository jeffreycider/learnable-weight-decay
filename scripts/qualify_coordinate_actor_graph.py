"""Saved32 eager actor CUDA-graph shared/private VJP and complete-step gate."""
from pathlib import Path
import argparse,copy,json,sys,time
from dataclasses import replace,asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.gpt_precision import configure_precision
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.private_probe_slot import PrivateProbeSlot
from rewriter.coordinate_actor_graph import CoordinateActorGraph
from rewriter.edit_scope import editable_names
from rewriter.coordinate_optimization import configure_shared_continuation,clip_shared_continuation
from rewriter.meta_optimizer import build_meta_adam
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from rewriter.rank_preservation import topk_preservation_flags
from rewriter.multibank import file_sha
from scripts.qualify_coordinate_chisel import FixedContexts
from scripts.train_multistate_chisel import build_proposal


def relative(a,b):return float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-300))
def flat_grad(parameters):return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().flatten().cpu() for p in parameters])
def delta(after,before):return torch.cat([(after[n]-before[n]).flatten().cpu() for n in after if isinstance(after[n],torch.Tensor)])

def optimizer_comparison(candidate,reference):
 groups_equal=candidate['param_groups']==reference['param_groups'];keys_equal=candidate['state'].keys()==reference['state'].keys()
 common=sorted(candidate['state'].keys() & reference['state'].keys());errors=[]
 clocks=keys_equal and all(torch.equal(torch.as_tensor(candidate['state'][k]['step']).cpu(),torch.as_tensor(reference['state'][k]['step']).cpu()) for k in common)
 for name in ('exp_avg','exp_avg_sq'):
  aa=[candidate['state'][k][name].detach().flatten().cpu() for k in common if name in reference['state'][k]]
  bb=[reference['state'][k][name].detach().flatten().cpu() for k in common if name in reference['state'][k]]
  if aa:errors.append(relative(torch.cat(aa),torch.cat(bb)))
 return dict(param_groups_equal=groups_equal,state_keys_equal=keys_equal,candidate_only_keys=sorted(candidate['state'].keys()-reference['state'].keys()),reference_only_keys=sorted(reference['state'].keys()-candidate['state'].keys()),step_clocks_equal=bool(clocks),moment_relative=max(errors,default=0.))

@torch.no_grad()
def compare_outputs(x,y,k=100):
 x=x.reshape(-1,x.shape[-1]);y=y.reshape_as(x);count=len(x);total=torch.zeros(3,dtype=torch.float64,device=x.device);maximum=torch.zeros(2,dtype=torch.float64,device=x.device);counts={}
 for start in range(0,count,128):
  a=x[start:start+128].double();b=y[start:start+128].double();lp=a.log_softmax(-1);lq=b.log_softmax(-1)
  p=(lp.exp()*(lp-lq)).sum(-1);q=(lq.exp()*(lq-lp)).sum(-1);d=a-b;d=d-d.mean(-1,keepdim=True)
  total+=torch.stack([p.sum(),q.sum(),d.square().sum()]);maximum=torch.maximum(maximum,torch.stack([p.max(),q.max()]))
  for name,value in topk_preservation_flags(a,b,k=min(k,x.shape[-1])).items():counts[name]=counts.get(name,0)+int(value.sum())
 return dict(mean_forward_kl=float(total[0]/count),mean_reverse_kl=float(total[1]/count),max_forward_kl=float(maximum[0]),max_reverse_kl=float(maximum[1]),centered_logit_rms=float((total[2]/x.numel()).sqrt()),contexts=count,ranking_k=min(k,x.shape[-1]),rank_counts=counts)


def main():
 p=argparse.ArgumentParser()
 for name in ('checkpoint','base-recipe','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--checkpoint-sha256',required=True)
 p.add_argument('--gpt-ranking-weight',type=float,required=True);p.add_argument('--cifar-ranking-weight',type=float,required=True)
 p.add_argument('--shared-optimizer-mode',choices=['original_joint','separate_representation'],default='separate_representation')
 a=p.parse_args()
 if file_sha(a.checkpoint)!=a.checkpoint_sha256:raise ValueError('checkpoint SHA differs')
 a.output.mkdir(parents=True,exist_ok=False);enable_deterministic_execution();torch.cuda.set_device(0);torch.set_num_threads(1);configure_precision('fp32');device=torch.device('cuda:0')
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);protocol=saved['metadata']['protocol'];stage=protocol['joint_pilot'];recipe=json.loads(a.base_recipe.read_text())
 if file_sha(a.base_recipe)!=saved['metadata']['base_recipe_sha256']:raise ValueError('base recipe differs from saved fixture')
 architecture=dict(protocol['architecture']);architecture.pop('name')
 if architecture.pop('decoder_scale')!='sqrt_coordinates_over_latent_tokens':raise ValueError('decoder scale differs')
 actor=CoordinateChisel(**architecture).to(device)
 if saved['metadata'].get('frozen_residual_blocks'):actor.blocks.requires_grad_(False)
 contexts=FixedContexts(recipe,device,gpt_sequences=stage['gpt_sequences'],cifar_images=stage['cifar_images'])
 keys=list(saved['private']);weight=stage['state_weight']
 if abs(weight*len(keys)-1)>1e-12:raise ValueError('saved complete batch normalization differs')
 configs={d:replace(TokenwiseConfig(**state['config']),ranking_loss_mode='relative_gap',loss_backend='tiled_relative_fp64_v1',ranking_weight=a.gpt_ranking_weight if d=='gpt' else a.cifar_ranking_weight) for d,state in saved['duals'].items()}
 duals={}
 optimizer=build_meta_adam(actor,lr=stage['shared_head_lr'],representation_lr_multiplier=stage['shared_representation_multiplier'])
 slots={};local_optimizer={}
 for domain,uid in keys:
  if domain in slots:continue
  c=contexts.load(domain,uid);source=saved['private'][domain,uid]['parameters'];key=source['_extra_state']['state_key']
  private=actor.make_private(c['snapshot'],state_key=key,seed=991);slots[domain]=PrivateProbeSlot(actor,private)
  local_optimizer[domain]=torch.optim.Adam(private.parameters(),lr=stage['private_lr'],betas=tuple(stage['adam_betas']),eps=stage.get('private_adam_epsilon',stage.get('adam_epsilon',1e-8)))
  del c
 signatures={d:s.storage_signature() for d,s in slots.items()}
 actor.load_state_dict(saved['actor']);graphs={};capture_seconds={};staging_checks={}
 for domain,slot in slots.items():
  key=next(k for k in keys if k[0]==domain);source=saved['private'][key]['parameters'];slot.stage(source,state_key=source['_extra_state']['state_key']);c=contexts.load(*key)
  editable=editable_names(c['engine'].parameters,'grouped')-c['grouped'].frozen_parameter_names
  old_cache=actor.__dict__.pop('_static_index_cache',None)
  with torch.no_grad():original_action=actor(c['snapshot'],slot.private)
  if old_cache is not None:actor._static_index_cache=old_cache
  torch.cuda.synchronize();beg=time.perf_counter();graphs[domain]=CoordinateActorGraph(actor,slot,c['snapshot'],editable);torch.cuda.synchronize();capture_seconds[domain]=time.perf_counter()-beg
  with torch.no_grad():staged_action=actor(c['snapshot'],slot.private)
  staging_checks[domain]=all(torch.equal(original_action.matrices[n].P,staged_action.matrices[n].P) and torch.equal(original_action.matrices[n].Q,staged_action.matrices[n].Q) for n in original_action.matrices) and all(torch.equal(original_action.vectors[n],staged_action.vectors[n]) for n in original_action.vectors)
  if not staging_checks[domain]:raise RuntimeError('static index staging changed actual actor outputs')
  del c,original_action,staged_action
 import wandb
 wb=wandb.init(project='chisel',mode='offline',dir=str(a.output),config={'checkpoint_sha256':file_sha(a.checkpoint),'scope':'Eager versus CUDA-graphed coordinate actor forward/VJP; identical tiled relative loss and eager student; saved shared/private Adam; complete four-state updates'})
 def log(row):
  with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps(row),flush=True)
 def bind(key,private_states=None):
  domain,uid=key;slot=slots[domain];opt=local_optimizer[domain];opt.zero_grad(set_to_none=True)
  source=saved['private'][key] if private_states is None else private_states[key]
  state=source['parameters'];slot.stage(state,state_key=state['_extra_state']['state_key']);opt.load_state_dict(copy.deepcopy(source['optimizer']))
  return slot.require_bound(state['_extra_state']['state_key']),opt
 def arm(name):
  torch.set_rng_state(saved['rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
  actor.load_state_dict(saved['actor']);optimizer.load_state_dict(copy.deepcopy(saved['shared_optimizer']));optimizer.zero_grad(set_to_none=True)
  configure_shared_continuation(optimizer,a.shared_optimizer_mode)
  for d,state in saved['duals'].items():
   cfg=configs[d]
   dual=TokenwiseDual(state['multipliers'],config=cfg,dual_lr=state['dual_lr']);dual.multipliers=dict(state['multipliers']);dual.updates=state['updates'];duals[d]=dual
  gradients={};private_after={};rows=[];torch.cuda.synchronize();beg=time.perf_counter()
  for key in keys:
   domain,uid=key;private,opt=bind(key);c=contexts.load(domain,uid)
   if name!='reference':graphs[domain].bind_scope(private,editable_names(c['engine'].parameters,'grouped')-c['grouped'].frozen_parameter_names)
   row=contexts.measure(c,actor if name=='reference' else graphs[domain],private,configs[domain],duals[domain],learning=True,weight=weight)
   slots[domain].unweight_gradients(weight);gradients[key]=flat_grad(private.parameters());row['private_none_gradients']=[n for n,p in private.named_parameters() if p.grad is None]
   row['private_preclip_norm']=float(torch.nn.utils.clip_grad_norm_(private.parameters(),stage['private_clip'],error_if_nonfinite=True));opt.step()
   private_after[key]=dict(parameters=slots[domain].export_state(),optimizer=copy.deepcopy(opt.state_dict()));rows.append(row);del c
  shared_gradient=flat_grad(actor.parameters());shared_norm=clip_shared_continuation(optimizer,a.shared_optimizer_mode,max_norm=stage['shared_clip']);optimizer.step();torch.cuda.synchronize()
  seconds=time.perf_counter()-beg
  for domain,dual in duals.items():
   measurements=[(r['local_id'],r['mean_kl']) for r in rows if r['domain']==domain];dual.step(measurements,expected_ids=[i for i,k in measurements])
  policy={n:v.detach().cpu().clone() for n,v in actor.state_dict().items()}
  log(dict(event='backend_update',backend=name,seconds=seconds,shared_preclip_norm=shared_norm,rows=rows))
  return dict(rows=rows,duals={d:copy.deepcopy(v.state_dict()) for d,v in duals.items()},shared_none_gradients=[n for n,p in actor.named_parameters() if p.grad is None],shared_gradient=shared_gradient,private_gradients=gradients,actor=policy,private=private_after,seconds=seconds,shared_optimizer=copy.deepcopy(optimizer.state_dict()))
 reference=arm('reference');candidate=arm('graph')
 torch.save({'reference':reference,'candidate':candidate,'checkpoint_sha256':a.checkpoint_sha256},a.output/'finite_update_evidence.pt')
 shared_gradient_error=relative(candidate['shared_gradient'],reference['shared_gradient']);shared_delta_error=relative(delta(candidate['actor'],saved['actor']),delta(reference['actor'],saved['actor']))
 checks=[]
 for key in keys:
  before=saved['private'][key]['parameters'];r=reference['private'][key]['parameters'];c=candidate['private'][key]['parameters']
  checks.append(dict(domain=key[0],state=key[1],gradient_relative=relative(candidate['private_gradients'][key],reference['private_gradients'][key]),adam_delta_relative=relative(delta(c,before),delta(r,before)),gradient_norm=float(reference['private_gradients'][key].double().norm())))
 outputs=[];output_forwards={'gpt':0,'cifar10':0}
 for key in keys:
  c=contexts.load(*key);edits=[]
  for result in (reference,candidate):
   actor.load_state_dict(result['actor']);private,_=bind(key,result['private'])
   if result is candidate:graphs[key[0]].bind_scope(private,editable_names(c['engine'].parameters,'grouped')-c['grouped'].frozen_parameter_names)
   proposal=lambda snap:(actor(snap,private) if result is reference else graphs[key[0]](snap,private))
   with torch.no_grad():pr=build_proposal(proposal,c['snapshot'],c['engine'],c['grouped'],radius_fraction=.01,norm_squared=c['norm_squared'],shrinkage_coefficient=1.)
   edits.append({n:v.detach() for n,v in pr[0].items()});del pr
  micro=c['engine'].microbatch_sequences
  for start in range(0,len(c['inputs']),micro):
   with torch.no_grad():
    x=c['inputs'][start:start+micro].to(device);u=c['engine']._forward(edits[0],x);v=c['engine']._forward(edits[1],x);row=compare_outputs(u,v);row['bitwise_logits_equal']=torch.equal(u,v)
   outputs.append(dict(domain=key[0],state=key[1],**row));output_forwards[key[0]]+=2;del x,u,v
  del c,edits
 optimizer_checks={'shared':optimizer_comparison(candidate['shared_optimizer'],reference['shared_optimizer']),**{str(key):optimizer_comparison(candidate['private'][key]['optimizer'],reference['private'][key]['optimizer']) for key in keys}}
 gates=dict(static_indices_exact=all(staging_checks.values()),strict_rank_counts=all(r['rank_counts']['strict_topk_changed']==0 and r['rank_counts']['tie_aware_argmax_changed']==0 for r in outputs),unused_private_gradients=all(r['private_none_gradients']==c['private_none_gradients'] for r,c in zip(reference['rows'],candidate['rows'])),optimizer_clocks=all(r['step_clocks_equal'] and r['param_groups_equal'] and r['state_keys_equal'] for r in optimizer_checks.values()),optimizer_moments=all(r['moment_relative']<=.003 for r in optimizer_checks.values()),shared_gradient=shared_gradient_error<=.003,private_gradients=all(r['gradient_relative']<=.003 for r in checks),shared_adam_delta=shared_delta_error<=.003,private_adam_deltas=all(r['adam_delta_relative']<=.003 for r in checks),mutual_mean=all(max(r['mean_forward_kl'],r['mean_reverse_kl'])<=1e-11 for r in outputs),mutual_token=all(max(r['max_forward_kl'],r['max_reverse_kl'])<=1e-10 for r in outputs),stable_addresses=all(slots[d].storage_signature()==s for d,s in signatures.items()))
 warm=[]
 if all(gates.values()):
  for backend in ('graph','reference','reference','graph'):
   measured=arm(backend);warm.append({'backend':backend,'seconds':measured['seconds']});del measured
 summary=dict(scope='Eager versus captured coordinate actor forward/VJP; identical tiled relative loss and strictFP32 student; saved shared/private Adam and explicit separate representation settings; warm instrumented four-state updates include gradient CPU diagnostics and per-state binding, exclude restoring initial checkpoint and graph capture',checkpoint_sha256=file_sha(a.checkpoint),source_script_sha256=file_sha(__file__),architecture=architecture,keys=keys,shared_gradient_relative=shared_gradient_error,shared_adam_delta_relative=shared_delta_error,private_checks=checks,optimizer_checks=optimizer_checks,direct_outputs=outputs,gates=gates,timings={'reference_first_update':reference['seconds'],'graph_first_update':candidate['seconds']},warm_abba=warm,static_index_checks=staging_checks,capture_seconds=capture_seconds,replays={d:g.replays for d,g in graphs.items()},bitwise_output_logits=all(r['bitwise_logits_equal'] for r in outputs),shared_optimizer_mode=a.shared_optimizer_mode,objective_configs={d:asdict(c) for d,c in configs.items()},peak_bytes=torch.cuda.max_memory_allocated(),counts=contexts.compute,additional_mutual_output_student_forwards=output_forwards,reference_rows=reference['rows'],candidate_rows=candidate['rows'],post_update_duals={'reference':reference['duals'],'candidate':candidate['duals']})
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2));log(summary);wb.log({'objective':shared_gradient_error,'passed':all(gates.values())});wb.finish()
 if not all(gates.values()):raise RuntimeError('saved coordinate eager graph finite gate failed; preserve evidence')
if __name__=='__main__':main()
