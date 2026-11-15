"""Saved-state eager versus static-slot compiled coordinate actor; no scientific fit."""
from pathlib import Path
import argparse,copy,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.gpt_precision import configure_precision
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.private_probe_slot import PrivateProbeSlot
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
 if candidate['param_groups']!=reference['param_groups'] or candidate['state'].keys()!=reference['state'].keys():
  return dict(step_clocks_equal=False,moment_relative=float('inf'))
 errors=[];clocks=True
 for key,r in reference['state'].items():
  c=candidate['state'][key]
  clocks=clocks and c.keys()==r.keys() and torch.equal(torch.as_tensor(c['step']).cpu(),torch.as_tensor(r['step']).cpu())
 for name in ('exp_avg','exp_avg_sq'):
  aa=[c[name].detach().flatten().cpu() for c in candidate['state'].values() if name in c]
  bb=[r[name].detach().flatten().cpu() for r in reference['state'].values() if name in r]
  if aa:errors.append(relative(torch.cat(aa),torch.cat(bb)))
 return dict(step_clocks_equal=bool(clocks),moment_relative=max(errors,default=0.))

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
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);enable_deterministic_execution();torch.cuda.set_device(0);torch.set_num_threads(1);configure_precision('fp32');device=torch.device('cuda:0')
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);protocol=saved['metadata']['protocol'];stage=protocol['joint_pilot'];recipe=json.loads(a.base_recipe.read_text())
 if file_sha(a.base_recipe)!=saved['metadata']['base_recipe_sha256']:raise ValueError('base recipe differs from saved fixture')
 architecture=dict(protocol['architecture']);architecture.pop('name')
 if architecture.pop('decoder_scale')!='sqrt_coordinates_over_latent_tokens':raise ValueError('decoder scale differs')
 actor=CoordinateChisel(**architecture).to(device)
 if saved['metadata'].get('frozen_residual_blocks'):actor.blocks.requires_grad_(False)
 contexts=FixedContexts(recipe,device,gpt_sequences=stage['gpt_sequences'],cifar_images=stage['cifar_images'])
 keys=list(saved['private']);weight=stage['state_weight']
 if abs(weight*len(keys)-1)>1e-12:raise ValueError('saved complete batch normalization differs')
 configs={d:TokenwiseConfig(**s['config']) for d,s in saved['duals'].items()};duals={}
 for d,state in saved['duals'].items():
  dual=TokenwiseDual(state['multipliers'],config=configs[d],dual_lr=state['dual_lr']);dual.load_state_dict(copy.deepcopy(state));duals[d]=dual
 optimizer=build_meta_adam(actor,lr=stage['shared_head_lr'],representation_lr_multiplier=stage['shared_representation_multiplier'])
 slots={};local_optimizer={}
 for domain,uid in keys:
  if domain in slots:continue
  c=contexts.load(domain,uid);source=saved['private'][domain,uid]['parameters'];key=source['_extra_state']['state_key']
  private=actor.make_private(c['snapshot'],state_key=key,seed=991);slots[domain]=PrivateProbeSlot(actor,private)
  local_optimizer[domain]=torch.optim.Adam(private.parameters(),lr=stage['private_lr'],betas=tuple(stage['adam_betas']),eps=stage['adam_epsilon'])
  del c
 signatures={d:s.storage_signature() for d,s in slots.items()};compiled=torch.compile(actor,fullgraph=True,dynamic=False)
 import wandb
 wb=wandb.init(project='chisel',mode='offline',dir=str(a.output),config={'checkpoint_sha256':file_sha(a.checkpoint),'scope':'One disposable full saved four-context update perbackend; no scientific continuation'})
 def log(row):
  with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps(row),flush=True)
 def bind(key,private_states=None):
  domain,uid=key;slot=slots[domain];opt=local_optimizer[domain];opt.zero_grad(set_to_none=True)
  source=saved['private'][key] if private_states is None else private_states[key]
  state=source['parameters'];slot.stage(state,state_key=state['_extra_state']['state_key']);opt.load_state_dict(copy.deepcopy(source['optimizer']))
  return slot.require_bound(state['_extra_state']['state_key']),opt
 def arm(name):
  actor.load_state_dict(saved['actor']);optimizer.load_state_dict(copy.deepcopy(saved['shared_optimizer']));optimizer.zero_grad(set_to_none=True)
  for d,state in saved['duals'].items():duals[d].load_state_dict(copy.deepcopy(state))
  gradients={};private_after={};rows=[];torch.cuda.synchronize();beg=time.perf_counter()
  for key in keys:
   domain,uid=key;private,opt=bind(key);c=contexts.load(domain,uid)
   row=contexts.measure(c,actor if name=='eager' else compiled,private,configs[domain],duals[domain],learning=True,weight=weight)
   slots[domain].unweight_gradients(weight);gradients[key]=flat_grad(private.parameters())
   row['private_preclip_norm']=float(torch.nn.utils.clip_grad_norm_(private.parameters(),stage['private_clip'],error_if_nonfinite=True));opt.step()
   private_after[key]=dict(parameters=slots[domain].export_state(),optimizer=copy.deepcopy(opt.state_dict()));rows.append(row);del c
  shared_gradient=flat_grad(actor.parameters());shared_norm=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),stage['shared_clip'],error_if_nonfinite=True));optimizer.step();torch.cuda.synchronize()
  seconds=time.perf_counter()-beg;policy={n:v.detach().cpu().clone() for n,v in actor.state_dict().items()}
  log(dict(event='backend_update',backend=name,seconds=seconds,shared_preclip_norm=shared_norm,rows=rows,compiler_counters={str(k):dict(v) for k,v in torch._dynamo.utils.counters.items()}))
  return dict(shared_gradient=shared_gradient,private_gradients=gradients,actor=policy,private=private_after,seconds=seconds,shared_optimizer=copy.deepcopy(optimizer.state_dict()))
 reference=arm('eager');candidate=arm('compiled')
 shared_gradient_error=relative(candidate['shared_gradient'],reference['shared_gradient']);shared_delta_error=relative(delta(candidate['actor'],saved['actor']),delta(reference['actor'],saved['actor']))
 checks=[]
 for key in keys:
  before=saved['private'][key]['parameters'];r=reference['private'][key]['parameters'];c=candidate['private'][key]['parameters']
  checks.append(dict(domain=key[0],state=key[1],gradient_relative=relative(candidate['private_gradients'][key],reference['private_gradients'][key]),adam_delta_relative=relative(delta(c,before),delta(r,before)),gradient_norm=float(reference['private_gradients'][key].double().norm())))
 # Rebind A/B values through the SAME containers; warmed replay must not add graphs.
 count_before=torch._dynamo.utils.counters['stats']['unique_graphs'];replay_values=[]
 for key in reversed(keys):
  actor.load_state_dict(candidate['actor']);private,_=bind(key,candidate['private']);c=contexts.load(*key)
  action=compiled(c['snapshot'],private);replay_values.append(float(action.norm_sq().detach()));del action,c
 count_after=torch._dynamo.utils.counters['stats']['unique_graphs']
 outputs=[];output_forwards={'gpt':0,'cifar10':0}
 for key in keys:
  c=contexts.load(*key);edits=[]
  for result in (reference,candidate):
   actor.load_state_dict(result['actor']);private,_=bind(key,result['private'])
   proposal=lambda snap:(compiled(snap,private) if result is candidate else actor(snap,private))
   pr=build_proposal(proposal,c['snapshot'],c['engine'],c['grouped'],radius_fraction=.01,norm_squared=c['norm_squared'],shrinkage_coefficient=1.)
   edits.append({n:v.detach() for n,v in pr[0].items()});del pr
  micro=c['engine'].microbatch_sequences
  for start in range(0,len(c['inputs']),micro):
   with torch.no_grad():
    x=c['inputs'][start:start+micro].to(device);u=c['engine']._forward(edits[0],x);v=c['engine']._forward(edits[1],x);row=compare_outputs(u,v)
   outputs.append(dict(domain=key[0],state=key[1],**row));output_forwards[key[0]]+=2;del x,u,v
  del c,edits
 optimizer_checks={'shared':optimizer_comparison(candidate['shared_optimizer'],reference['shared_optimizer']),**{str(key):optimizer_comparison(candidate['private'][key]['optimizer'],reference['private'][key]['optimizer']) for key in keys}}
 gates=dict(optimizer_clocks=all(r['step_clocks_equal'] for r in optimizer_checks.values()),optimizer_moments=all(r['moment_relative']<=.003 for r in optimizer_checks.values()),shared_gradient=shared_gradient_error<=.003,private_gradients=all(r['gradient_relative']<=.003 for r in checks),shared_adam_delta=shared_delta_error<=.003,private_adam_deltas=all(r['adam_delta_relative']<=.003 for r in checks),mutual_mean=all(max(r['mean_forward_kl'],r['mean_reverse_kl'])<=1e-11 for r in outputs),mutual_token=all(max(r['max_forward_kl'],r['max_reverse_kl'])<=1e-10 for r in outputs),no_recompile=count_before==count_after,stable_addresses=all(slots[d].storage_signature()==s for d,s in signatures.items()))
 summary=dict(scope='Saved shared/private Adam andduals; one complete disposable four-context update; compiled actor only, eager strictFP32 GPT/CNN andsame tiled loss; no CUDAgraph/backend deploymentclaim',checkpoint_sha256=file_sha(a.checkpoint),source_script_sha256=file_sha(__file__),architecture=architecture,keys=keys,shared_gradient_relative=shared_gradient_error,shared_adam_delta_relative=shared_delta_error,private_checks=checks,optimizer_checks=optimizer_checks,direct_outputs=outputs,gates=gates,timings={'eager_first_update':reference['seconds'],'compiled_cold_update':candidate['seconds']},compiler_graphs_before_rebind=count_before,compiler_graphs_after_rebind=count_after,peak_bytes=torch.cuda.max_memory_allocated(),counts=contexts.compute,additional_mutual_output_student_forwards=output_forwards,additional_actor_only_rebind_forwards=len(keys))
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2));log(summary);wb.log({'objective':shared_gradient_error,'passed':all(gates.values())});wb.finish()
 if not all(gates.values()):raise RuntimeError('saved coordinate compiled/static-slot gate failed; preserve evidence')
if __name__=='__main__':main()
