"""Fixed-real-batch first-order gradient and optimizer sanity; never trains a run."""
import argparse,copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import scripts.train_multistate_chisel as trainer
from rewriter.accelerated_controller import PackedGlobalRewriter
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective


def main():
 p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--sequences',type=int,default=2);p.add_argument('--compile-actor',action='store_true');a=p.parse_args()
 torch.set_num_threads(4);torch.set_num_interop_threads(4);trainer.enable_deterministic_execution();device=torch.device('cuda');a.output.mkdir(parents=True,exist_ok=False)
 started=time.perf_counter()
 def log(event,**v):
  row=dict(event=event,elapsed_seconds=time.perf_counter()-started,**v);print(json.dumps(row),flush=True)
  with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 saved=torch.load(a.checkpoint,map_location=device,weights_only=False);opts=saved['metadata']['options'];reader=trainer.StateBankReader(a.bank,require_complete=False);prov=reader.manifest['provenance'];record=reader.read(24,device=device,model_device='cpu')
 model=trainer.GPT(**prov['model_config']).to(device);model.requires_grad_(False);engine=trainer.FunctionalPreservation(model,microbatch_sequences=a.sequences,teacher_cache_device=device,kl_dtype=torch.float64)
 names=set(dict(model.named_parameters()));engine.load_teacher_state({n:v for n,v in record.model.items() if n in names},{n:v for n,v in record.model.items() if n not in names});grouped=AcceleratedGroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
 rank=prov['sensor_config'].get('probe_rank',8);actor=trainer.GlobalRewriter(rank=rank,trajectory_dim=(2*prov['model_config']['num_layers']+1)*2*rank).to(device);actor.load_state_dict(saved['controller']);forward=PackedGlobalRewriter(actor)
 if a.compile_actor:forward=torch.compile(forward,fullgraph=True,dynamic=False)
 replay=trainer.state_replay(sorted(trainer.glob.glob(a.data)),24,prov['training_config'],pool_sequences=opts['replay_pool_sequences'],seed=opts['data_seed'],epoch=saved['next_epoch'],split_seed=opts['split_seed']);inputs,ids=trainer.select_sequences(replay['train'],a.sequences,seed=trainer.stable_seed('gradient-debug',24,saved['updates']))
 (a.output/'inputs.json').write_text(json.dumps(dict(state_id=24,sequence_ids=ids,controller_update=saved['updates'],options=opts,compiled_actor=a.compile_actor),indent=2));torch.save(inputs,a.output/'inputs.pt')
 teacher=engine.cache_teacher(inputs,data_role='past_training')[0]
 params=list(actor.parameters());initial=[x.detach().clone() for x in params]
 def flatgrad():return torch.cat([torch.zeros_like(x).flatten() if x.grad is None else x.grad.detach().flatten() for x in params])
 def proposal(split):
  original=trainer.detached_action_credit
  if not split:trainer.detached_action_credit=lambda action:(action,None)
  try:return trainer.build_proposal(forward,record.snapshot,engine,grouped,radius_fraction=opts['radius_fraction'])
  finally:trainer.detached_action_credit=original
 def direct(kind,native=False):
  actor.zero_grad(set_to_none=True);parameters,regularizer,stats,editable,credit=proposal(False)
  if native:
   ratios=[];edit_sq=regularizer.new_zeros(())
   for name,members in grouped.groups.items():
    base=grouped.reference_norms[name]
    if float(base)==0:ratios.append(base*0);continue
    norm_sq=sum(parameters[n].double().square().sum() for n in members)
    ratios.append(norm_sq.sqrt()/base-1)
    edit_sq=edit_sq+sum((parameters[n].double()-engine.parameters[n].double()).square().sum() for n in members)
   radius=opts['radius_fraction']*trainer.editable_weight_norm_sq(engine.parameters,editable).sqrt()
   regularizer=100*torch.stack(ratios).mean()+.01*edit_sq/radius.double().square()
   logits=engine._forward(parameters,teacher.inputs.to(device)).double();logp=torch.log_softmax(logits,dim=-1)
   kl=(teacher.log_probs.exp()*(teacher.log_probs-logp)).sum(-1).mean()
   engine.compute.student_forwards+=1;engine.compute.student_tokens+=inputs.numel()
   if kind!='norm':engine.compute.student_backwards+=1;engine.compute.student_backward_tokens+=inputs.numel()
  else:kl=engine.student_parameter_loss(parameters,teacher)
  penalty=(kl/opts['kl_budget']-1).clamp_min(0).square();loss={'norm':regularizer,'raw_kl':kl,'total':regularizer+penalty}[kind];loss.backward();gradient=flatgrad();scalars=dict(norm=float(regularizer.detach()),raw_kl=float(kl.detach()),penalty=float(penalty.detach()),objective=float((regularizer+penalty).detach()));log('direct_gradient',kind=kind,native=native,gradient_norm=float(gradient.norm()),**scalars);return gradient,scalars
 norm,scalars=direct('norm');raw_kl,_=direct('raw_kl');direct_total,_=direct('total');coefficient=trainer.deadband(scalars['raw_kl'],opts['kl_budget'])[1]
 predicted=norm+coefficient*raw_kl;log('component_sum',relative_error=float((direct_total-predicted).norm()/direct_total.norm()),norm_kl_cosine=float(torch.dot(norm,raw_kl)/(norm.norm()*raw_kl.norm())),norm_gradient_norm=float(norm.norm()),weighted_kl_gradient_norm=float((coefficient*raw_kl).norm()))
 native_norm,_=direct('norm',native=True);native_kl,_=direct('raw_kl',native=True)
 log('custom_vs_native',norm_relative_error=float((native_norm-norm).norm()/native_norm.norm()),kl_relative_error=float((native_kl-raw_kl).norm()/native_kl.norm()))
 actor.zero_grad(set_to_none=True);parameters,regularizer,stats,editable,credit=proposal(True);measured=trainer.mean_kl_and_backward(engine,parameters,regularizer,inputs,kl_budget=opts['kl_budget'],actor_credit=credit);split_total=flatgrad();log('split_vs_direct',relative_error=float((split_total-direct_total).norm()/direct_total.norm()),cosine=float(torch.dot(split_total,direct_total)/(split_total.norm()*direct_total.norm())),**measured);del parameters,regularizer,credit
 def restore():
  with torch.no_grad():
   for x,value in zip(params,initial):x.copy_(value)
 def evaluate(label):
  # Preserve the exact grad-enabled compiled actor arithmetic used by the VJP.
  # A no-grad specialization can cross BF16 rounding thresholds independently.
  with torch.enable_grad():
   parameters,regularizer,stats,editable,credit=proposal(False);kl=engine.student_parameter_loss(parameters,teacher);penalty=trainer.deadband(float(kl),opts['kl_budget'])[0];log('probe',label=label,norm=float(regularizer),raw_kl=float(kl),penalty=penalty,objective=float(regularizer)+penalty)
 evaluate('current')
 # Same checkpoint optimizer moments, fresh current-batch gradient, one explicit diagnostic Adam proposal.
 optimizer=torch.optim.Adam(params,lr=opts['lr'],betas=(.9,.99),eps=1e-8);optimizer.load_state_dict(saved['optimizer']);offset=0
 for x in params:x.grad=direct_total[offset:offset+x.numel()].view_as(x).clone();offset+=x.numel()
 preclip=float(torch.nn.utils.clip_grad_norm_(params,opts['gradient_clip']));optimizer.step();displacement=torch.cat([(x.detach()-base).flatten() for x,base in zip(params,initial)]);log('adam_alignment',preclip_gradient_norm=preclip,displacement_norm=float(displacement.norm()),gradient_dot_displacement=float(torch.dot(direct_total,displacement)),cosine=float(torch.dot(direct_total,displacement)/(direct_total.norm()*displacement.norm())));evaluate('adam_current_moments');restore()
 direction=direct_total/direct_total.norm()
 for step in (1e-6,1e-5,1e-4,1e-3):
  offset=0
  with torch.no_grad():
   for x,base in zip(params,initial):x.copy_(base-step*direction[offset:offset+x.numel()].view_as(x));offset+=x.numel()
  evaluate('negative_gradient_normstep_'+str(step));restore()
 torch.save(dict(norm_gradient=norm.cpu(),raw_kl_gradient=raw_kl.cpu(),direct_total=direct_total.cpu(),split_total=split_total.cpu(),adam_displacement=displacement.cpu()),a.output/'gradients.pt');log('finished',compute=engine.compute.as_dict())
if __name__=='__main__':main()
