"""Fixed128 image microbatches; larger panels reuse one Chisel forward/backward.

Read-only disposable actor gradients, no optimizer/dual/model update. All images
are from the immutable fitting split and every teacher comes from stored logits.
"""
from pathlib import Path
import argparse,json,hashlib,statistics,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from baseline.fp32_execution import configure_precision
from scripts.capture_cifar_chisel_bank import load_airbench
from scripts.train_multistate_chisel import build_proposal
from rewriter.cifar_adapter import CifarPreservation,cifar_groups
from rewriter.cifar_bank import CifarBankReader,normalized_images,tensor_tree
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.chisel_factory import build_chisel
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from rewriter.image_tokenwise_panel import image_tokenwise_panel


def architecture_of(saved):
 if 'architecture'in saved:return saved['architecture']
 metadata=saved['metadata']
 return metadata.get('architecture') or metadata['recipe']['architecture']


def main():
 p=argparse.ArgumentParser()
 for key in ('bank','data','policy','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--model',type=int,default=1);p.add_argument('--sizes',type=int,nargs='+',default=[128,1024,4096]);p.add_argument('--repeats',type=int,default=3)
 p.add_argument('--ranking-weight',type=float,default=910056964.1834549);p.add_argument('--multiplier',type=float,default=.01)
 a=p.parse_args()
 if any(n<128 or n%128 for n in a.sizes):raise ValueError('exact full128 microbatches required')
 a.output.mkdir(parents=True,exist_ok=False);enable_deterministic_execution();torch.cuda.set_device(0);torch.set_num_threads(1);device=torch.device('cuda:0')
 module=load_airbench(a.data,defer_test_readouts=True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;configure_precision('fp32')
 bank=CifarBankReader(a.bank);record=bank.read(a.model);record.assert_verified()
 raw=torch.load(a.data/'train.pt',map_location='cpu',weights_only=False);split=torch.load(a.data/'chisel_image_split.pt',map_location='cpu',weights_only=False)
 ids=split['fitting_indices'][:max(a.sizes)].long()
 if len(ids)!=max(a.sizes):raise ValueError('insufficient fitting images')
 images=normalized_images(raw['images'][ids],device);labels=raw['labels'][ids].to(device);cached=record.teacher_logits[ids].to(device)
 model=module.CifarNet().to(device).float().eval().requires_grad_(False);model.load_state_dict(record.model,strict=True)
 engine=CifarPreservation(model,microbatch_images=128);grouped=GroupedNormObjective(engine.parameters,groups=cifar_groups(engine.model),zero_reference_policy='freeze');snapshot=tensor_tree(record.snapshot,device)
 saved=torch.load(a.policy,map_location='cpu',weights_only=False);architecture=architecture_of(saved)
 if not architecture.get('trajectory_mask_features'):raise ValueError('masked427 actor required')
 actor=build_chisel(architecture).to(device);actor.load_state_dict(saved['controller']);forward=torch.compile(actor,fullgraph=True,dynamic=False)
 config=TokenwiseConfig(1e-10,1e-10,1e-8,.01,ranking_weight=a.ranking_weight,ranking_k=100,ranking_tie_policy='exact_teacher_ties',loss_backend='tiled_tie_fp64_v1',dual_update_mode='clipped_relative')
 dual=TokenwiseDual([a.model],config=config,dual_lr=.01);dual.multipliers[a.model]=a.multiplier
 import wandb
 wb=wandb.init(project='chisel',mode='offline',dir=str(a.output),config={k:str(v) for k,v in vars(a).items()})
 rows=[]
 def evaluate(n):
  actor.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();beg=time.perf_counter()
  params,regularizer,stats,_,credit=build_proposal(forward,snapshot,engine,grouped,radius_fraction=.01,shrinkage_coefficient=1.)
  torch.cuda.synchronize();actor_forward_seconds=time.perf_counter()-beg
  provider=lambda start,stop:engine.teacher_batch(images[start:stop],cached[start:stop],ids[start:stop].tolist(),data_role='past_training')
  result=image_tokenwise_panel(engine,params,regularizer,images[:n],labels[:n],config=config,dual=dual,state_id=a.model,teacher_provider=provider,actor_credit=credit,backward=True)
  torch.cuda.synchronize();elapsed=time.perf_counter()-beg
  norm=float(torch.linalg.vector_norm(torch.stack([p.grad.double().norm() for p in actor.parameters() if p.grad is not None])))
  return dict(images=n,microbatch_images=128,seconds=elapsed,actor_proposal_seconds=actor_forward_seconds,gradient_norm=norm,peak_allocated_bytes=torch.cuda.max_memory_allocated(),chisel_forwards=1,chisel_backwards=1,optimizer_updates=0,**result)
 for repeat in range(a.repeats+1):
  order=a.sizes if repeat%2==0 else list(reversed(a.sizes))
  for n in order:
   row=dict(repeat=repeat,cold=repeat==0,**evaluate(n));rows.append(row)
   with (a.output/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(row),flush=True);wb.log({'objective':row['seconds'],'images':n,'gradient_norm':row['gradient_norm']})
 record.assert_verified()
 summary=dict(scope='One frozen CNN and masked actor; complete image-panel actor forward, cached-target conversion, CNN forwards/backwards and one actor VJP; no Adam, no teacher model forwards; panel sizes change data amount, not equivalence comparison',model=a.model,policy_sha256=hashlib.sha256(a.policy.read_bytes()).hexdigest(),architecture=architecture,config=config.__dict__,multiplier=a.multiplier,microbatch_images=128,rows=rows,median_seconds={str(n):statistics.median(r['seconds'] for r in rows if r['images']==n and not r['cold']) for n in a.sizes},compiler_counters={str(k):dict(v) for k,v in torch._dynamo.utils.counters.items()})
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2));wb.finish()
if __name__=='__main__':main()
