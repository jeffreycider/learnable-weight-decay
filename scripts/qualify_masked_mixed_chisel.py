"""Disposable masked427 mixed-domain calibration and local graph lifecycle gate."""
from pathlib import Path
import argparse,copy,gc,json,statistics,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from dataclasses import replace
from baseline.execution import enable_deterministic_execution
from baseline.model import GPT
from baseline.fp32_execution import configure_precision
from rewriter.chisel_factory import build_chisel,ARCHITECTURE
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from rewriter.offline_systems import CachedBaseNormPreservation,VerifiedStateConstants,select_panel_blocks
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.cifar_adapter import CifarPreservation,cifar_groups
from rewriter.cifar_bank import CifarBankReader,normalized_images,tensor_tree
from rewriter.meta_optimizer import build_meta_adam
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual,tokenwise_panel,logit_terms_and_cotangent
from rewriter.image_tokenwise_panel import image_tokenwise_panel
from rewriter.graph_tokenwise_tie import TieAwareTokenwiseGraphAccumulator
from scripts.capture_cifar_chisel_bank import load_airbench
from scripts.train_multistate_chisel import build_proposal
from scripts.compare_tokenwise_trainer_outputs import compare as gpt_compare
from scripts.qualify_cifar_identity import compare as cnn_compare

def compare(x,y):
 if x.shape[-1]>10:return gpt_compare(x,y)
 r=cnn_compare(x,y)
 return dict(reference_to_tiled_kl=r["mean_kl"],tiled_to_reference_kl=r["reverse_mean_kl"],max_reference_to_tiled_token_kl=r["max_kl"],max_tiled_to_reference_token_kl=r["reverse_max_kl"],**r)

def relative(a,b):return float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-300))
def main():
 p=argparse.ArgumentParser()
 for key in ('state-bank','teacher-cache','cifar-bank','data','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--layers',type=int,default=4);p.add_argument('--seed',type=int,default=200001)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 enable_deterministic_execution();torch.cuda.set_device(0);torch.set_num_threads(1);device=torch.device('cuda:0');configure_precision('fp32')
 module=load_airbench(a.data,defer_test_readouts=True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;configure_precision('fp32')
 import wandb
 wb=wandb.init(project='chisel',mode='offline',dir=str(a.output),config={k:str(v) for k,v in vars(a).items()})
 def log(row):
  with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps(row),flush=True)
 reader=StateBankReader(a.state_bank,verify_once=True,mmap_model=True);cache=TeacherCacheReader(a.teacher_cache,verify_once=True,mmap=True)
 engine=CachedBaseNormPreservation(GPT(**reader.manifest['provenance']['model_config']).to(device).requires_grad_(False),microbatch_sequences=16,teacher_cache_device=device,kl_dtype=torch.float64,gpt_precision='fp32')
 constants=VerifiedStateConstants(engine,AcceleratedGroupedNormObjective)
 bank=CifarBankReader(a.cifar_bank);raw=torch.load(a.data/'train.pt',weights_only=False);split=torch.load(a.data/'chisel_image_split.pt',weights_only=False)
 ids=split['fitting_indices'][:128].long();images=normalized_images(raw['images'][ids],device);labels=raw['labels'][ids].to(device)
 cnn_model=module.CifarNet().to(device).float().eval().requires_grad_(False)
 architecture=dict(name=ARCHITECTURE,rank=8,trajectory_dim=400,model_dim=512,num_layers=a.layers,patch_size=64,trajectory_mask_features=True)
 torch.manual_seed(a.seed);actor=build_chisel(architecture).to(device);forward=torch.compile(actor,fullgraph=True,dynamic=False)
 optimizer=build_meta_adam(actor,lr=1e-7,representation_lr_multiplier=10);parameters=list(actor.parameters())
 cfg=TokenwiseConfig(1e-10,1e-10,1e-8,.01,ranking_weight=1.,ranking_k=100,ranking_tie_policy='exact_teacher_ties',loss_backend='tiled_tie_fp64_v1',dual_update_mode='clipped_relative')
 def context(domain,state):
  if domain=='gpt':
   record=reader.read(state,device=device,model_device='cpu');constants.load(cache,record);grouped=constants.grouped();panel=select_panel_blocks(cache.read(state,'train'),range(4))
   return dict(domain=domain,state=state,record=record,engine=engine,grouped=grouped,snapshot=record.snapshot,inputs=panel.inputs,panel=panel,norm=constants.radius_squared(grouped))
  record=bank.read(state);record.assert_verified();cnn_model.load_state_dict(record.model);active=CifarPreservation(cnn_model,microbatch_images=128)
  return dict(domain=domain,state=state,record=record,engine=active,grouped=GroupedNormObjective(active.parameters,groups=cifar_groups(active.model),zero_reference_policy='freeze'),snapshot=tensor_tree(record.snapshot,device),inputs=images,norm=None)
 def proposal(c):
  return build_proposal(forward,c['snapshot'],c['engine'],c['grouped'],radius_fraction=.01,norm_squared=c['norm'],shrinkage_coefficient=1.)
 def teacher(c,start,stop):
  if c['domain']=='gpt':return c['panel'].teacher_batch(engine,start,stop)
  return c['engine'].teacher_batch(images[start:stop],c['record'].teacher_logits[ids[start:stop]],ids[start:stop].tolist(),data_role='past_training')
 # Prespecified disposable norm-only step: no base-model forward or backward.
 optimizer.zero_grad(set_to_none=True);norm_rows=[]
 for domain,state in [('gpt',6),('cifar10',1)]:
  c=context(domain,state);edited,reg,stats,_,credit=proposal(c);(.5*reg).backward();pairs=[(o,l.grad) for o,l in zip(*credit) if o.requires_grad and l.grad is not None];torch.autograd.backward([o for o,g in pairs],[g for o,g in pairs]);norm_rows.append(dict(domain=domain,state=state,regularizer=float(reg),trajectory_input=actor.trajectory_input.in_features,token_counts=actor.token_counts(c['snapshot'])))
  del edited,reg,credit,pairs,stats,c
 gradnorm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));optimizer.step();log(dict(event='disposable_norm_step',head_lr=1e-7,representation_multiplier=10,gradient_norm=gradnorm,rows=norm_rows,base_model_forwards=0))
 initial={n:v.detach().cpu().clone() for n,v in actor.state_dict().items()};initial_opt=copy.deepcopy(optimizer.state_dict())
 torch.save(dict(controller=initial,optimizer=initial_opt,architecture=architecture,seed=a.seed,scope='Disposable norm-only calibration fixture; never science initializer'),a.output/'disposable_fixture.pt')
 balance=[]
 for domain,states in [('gpt',[6,251,499,999]),('cifar10',[1,21,41,61])]:
  for state in states:
   c=context(domain,state);edited,reg,_,_,credit=proposal(c);edited={n:v.detach() for n,v in edited.items()};del reg,credit
   sums=torch.zeros(17,dtype=torch.float64,device=device);kls=[];unit='token' if domain=='gpt' else 'image';n=c['inputs'].numel() if domain=='gpt' else len(images);micro=16 if domain=='gpt' else 128
   for start in range(0,len(c['inputs']),micro):
    stop=start+micro
    with torch.no_grad():t=teacher(c,start,stop);student=c['engine']._forward(edited,c['inputs'][start:stop].to(device))
    _,measured,kl=logit_terms_and_cotangent(student,t.log_probs,config=cfg,multiplier=0.,panel_tokens=n,backward=False,balance=True)
    sums+=measured;kls.append(kl.flatten());del t,student,measured,kl
   tail=float(sums[11].sqrt());rank=float(sums[12].sqrt());row=dict(domain=domain,state=state,seed=getattr(c['record'],'seed',None),contexts=n,unit=unit,tail_cotangent_norm=tail,unit_rank_cotangent_norm=rank,ratio=tail/rank if rank>0 else None,mean_kl=float(torch.cat(kls).mean()),tail_loss=float(sums[1]/n),unit_rank_loss=float(sums[2]/n),strict_topk_changes=int(sums[16]))
   balance.append(row);log(dict(event='balance',**row));del c,edited,sums,kls
 gamma={d:statistics.median([r['ratio'] for r in balance if r['domain']==d]) if all(r['ratio'] is not None for r in balance if r['domain']==d) else None for d in ('gpt','cifar10')}
 calibration=dict(architecture=architecture,seed=a.seed,rows=balance,gamma=gamma,selection='Median of four fixed fitting-state tail/unit-rank cotangent norms; zero denominator is unresolved, never imputed',scratch_norm_step=norm_rows)
 (a.output/'calibration.json').write_text(json.dumps(calibration,indent=2));log(dict(event='calibration_closed',gamma=gamma))
 # Diagnostic weights are explicit; an inactive rank basis uses unit weight only
 # for numerical qualification, and never resolves the scientific calibration.
 configs={d:replace(cfg,ranking_weight=gamma[d] if gamma[d] is not None else 1.) for d in gamma}
 duals={d:TokenwiseDual([6] if d=='gpt' else [1],config=configs[d],dual_lr=.01) for d in gamma}
 for dual in duals.values():dual.multipliers={s:.01 for s in dual.multipliers}
 def restore():actor.load_state_dict(initial);optimizer.load_state_dict(copy.deepcopy(initial_opt));optimizer.zero_grad(set_to_none=True)
 def eager(domain,state):
  c=context(domain,state);edited,reg,stats,_,credit=proposal(c)
  kw=dict(config=configs[domain],dual=duals[domain],state_id=state,state_weight=.5,actor_credit=credit,teacher_provider=lambda x,y:teacher(c,x,y),backward=True)
  if domain=='gpt':out=tokenwise_panel(engine,edited,reg,c['inputs'],**kw)
  else:out=image_tokenwise_panel(c['engine'],edited,reg,images,labels,**kw)
  return out
 def gradient():return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().flatten() for p in parameters]).cpu()
 restore();reference_rows=[];torch.cuda.synchronize();beg=time.perf_counter()
 for d,s in [('gpt',6),('cifar10',1)]:reference_rows.append(eager(d,s))
 refgrad=gradient();refnorm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));optimizer.step();torch.cuda.synchronize();refseconds=time.perf_counter()-beg
 refpolicy={n:v.detach().cpu().clone() for n,v in actor.state_dict().items()};log(dict(event='reference_update',seconds=refseconds,gradient_norm=refnorm))
 restore();gc.collect();torch.cuda.empty_cache();adapter=TieAwareTokenwiseGraphAccumulator(actor,forward,engine,cache,constants,radius_fraction=.01,shrinkage_coefficient=1.)
 adapter.begin_update([6],global_checkpoint_count=2);c=context('gpt',6);adapter.accumulate(c['record'],c['panel'],duals['gpt']);adapter.finish_update();before_cnn=gradient();del c
 eager('cifar10',1);fastgrad=gradient();error=relative(fastgrad,refgrad);cnnnorm=float((fastgrad-before_cnn).double().norm());torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();fastpolicy={n:v.detach().cpu().clone() for n,v in actor.state_dict().items()};capture_seconds=adapter.capture_seconds;adapter.close();del adapter;gc.collect();torch.cuda.empty_cache()
 delta=lambda policy:torch.cat([(policy[n]-initial[n]).flatten() for n in policy]);delta_error=relative(delta(fastpolicy),delta(refpolicy));outputs=[]
 for domain,state in [('gpt',6),('cifar10',1)]:
  c=context(domain,state);edits=[]
  for policy in (refpolicy,fastpolicy):
   actor.load_state_dict(policy);pr=proposal(c);edits.append({n:v.detach() for n,v in pr[0].items()});del pr
  micro=16 if domain=='gpt' else 128
  for start in range(0,len(c['inputs']),micro):
   with torch.no_grad():
    x=c['inputs'][start:start+micro].to(device);u=c['engine']._forward(edits[0],x);v=c['engine']._forward(edits[1],x);out=compare(u,v)
   outputs.append(dict(domain=domain,state=state,**out));del u,v,x
  del c,edits
 gates=dict(raw_gradient=error<=.003,adam_delta=delta_error<=.003,cnn_gradient_added=cnnnorm>0,mutual_mean=all(max(r['reference_to_tiled_kl'],r['tiled_to_reference_kl'])<=1e-11 for r in outputs),mutual_token=all(max(r['max_reference_to_tiled_token_kl'],r['max_tiled_to_reference_token_kl'])<=1e-10 for r in outputs))
 summary=dict(scope='One GPT state6×64 + one CNN seed0×128; equal domain weights; masked427 shared actor; local gradient SUM then clip/Adam, no distributed claim',architecture=architecture,calibration=calibration,configs={d:v.__dict__ for d,v in configs.items()},gradient_relative=error,adam_delta_relative=delta_error,cnn_added_gradient_norm=cnnnorm,reference_seconds=refseconds,capture_seconds=capture_seconds,direct_outputs=outputs,gates=gates,peak_bytes=torch.cuda.max_memory_allocated(),compiler_counters={str(k):dict(v) for k,v in torch._dynamo.utils.counters.items()})
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2));log(summary);wb.log({'objective':error,'passed':all(gates.values())});wb.finish()
 if not all(gates.values()):raise RuntimeError('masked mixed numerical qualification failed; preserve evidence')
if __name__=='__main__':main()
