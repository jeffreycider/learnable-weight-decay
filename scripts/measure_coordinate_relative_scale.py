"""Fixed fitting-only relative ranking scale readout; no model backward or update."""
from pathlib import Path
import argparse,json,sys,time,statistics
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.gpt_precision import configure_precision
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.tokenwise_reference import TokenwiseConfig
from rewriter.tokenwise_tiled_relative import logit_terms_and_cotangent
from rewriter.multibank import file_sha
from scripts.qualify_coordinate_chisel import FixedContexts
from scripts.train_multistate_chisel import build_proposal

def main():
 p=argparse.ArgumentParser()
 for name in ('checkpoint','base-recipe','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--checkpoint-sha256',required=True);a=p.parse_args()
 if file_sha(a.checkpoint)!=a.checkpoint_sha256:raise ValueError('fixture SHA differs')
 a.output.mkdir(parents=True,exist_ok=False);enable_deterministic_execution();torch.cuda.set_device(0);torch.set_num_threads(1);configure_precision('fp32');device=torch.device('cuda:0')
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);protocol=saved['metadata']['protocol'];architecture=dict(protocol['architecture']);architecture.pop('name')
 if architecture.pop('decoder_scale')!='sqrt_coordinates_over_latent_tokens':raise ValueError('decoder scale differs')
 if file_sha(a.base_recipe)!=saved['metadata']['base_recipe_sha256']:raise ValueError('base recipe differs')
 actor=CoordinateChisel(**architecture).to(device);actor.load_state_dict(saved['actor']);actor.requires_grad_(False)
 contexts=FixedContexts(json.loads(a.base_recipe.read_text()),device,gpt_sequences=16,cifar_images=1024)
 import wandb
 wb=wandb.init(project='chisel',mode='offline',dir=str(a.output),config={'checkpoint_sha256':a.checkpoint_sha256,'scope':'Fitting-only logit cotangent scales; zero model backwards and Adam steps'})
 rows=[];counts={'gpt_student_forwards':0,'cifar_student_forwards':0,'cached_teacher_head_forwards':0,'model_backwards':0,'optimizer_steps':0};beg=time.perf_counter()
 for key in [('gpt',6),('gpt',499),('cifar10',1),('cifar10',41)]:
  domain,uid=key;c=contexts.load(domain,uid);state=saved['private'][key]['parameters'];private=actor.make_private(c['snapshot'],state_key=state['_extra_state']['state_key'],seed=1);private.load_state_dict(state);private.requires_grad_(False)
  cfg=replace(TokenwiseConfig(**saved['duals'][domain]['config']),ranking_loss_mode='relative_gap',loss_backend='tiled_relative_fp64_v1',ranking_weight=1.)
  multiplier=saved['duals'][domain]['multipliers'][uid];count=c['inputs'].numel() if domain=='gpt' else len(c['inputs']);sums=torch.zeros(17,device=device,dtype=torch.float64);linear_sq=torch.zeros((),device=device,dtype=torch.float64);raw=[]
  with torch.enable_grad():pr=build_proposal(lambda snap:actor(snap,private),c['snapshot'],c['engine'],c['grouped'],radius_fraction=.01,norm_squared=c['norm_squared'],shrinkage_coefficient=1.)
  parameters={n:v.detach() for n,v in pr[0].items()};del pr
  for start in range(0,len(c['inputs']),c['engine'].microbatch_sequences):
   stop=min(start+c['engine'].microbatch_sequences,len(c['inputs']));teacher=c['teacher'](start,stop)
   with torch.no_grad():logits=c['engine']._forward(parameters,c['inputs'][start:stop].to(device))
   _,parts,kl=logit_terms_and_cotangent(logits,teacher.log_probs,config=cfg,multiplier=multiplier,panel_tokens=count,backward=False,balance=True)
   sums+=parts;raw.append(kl)
   with torch.no_grad():
    s=logits.flatten(0,-2);t=teacher.log_probs.to(device).reshape_as(s)
    for j in range(0,len(s),128):
     pt=t[j:j+128].exp();ps=s[j:j+128].double().softmax(-1)
     g=((ps*pt.sum(-1,keepdim=True)-pt)*(multiplier/cfg.scale/count)).float()
     linear_sq+=g.double().square().sum()
   counts['gpt_student_forwards' if domain=='gpt' else 'cifar_student_forwards']+=1
   if domain=='gpt':counts['cached_teacher_head_forwards']+=1
   del teacher,logits,parts,kl,s,t,g,pt,ps
  tail=float(sums[11].sqrt());rank=float(sums[12].sqrt());linear=float(linear_sq.sqrt())
  row=dict(domain=domain,state=uid,context_ids=c['ids'],context_count=count,mean_kl=float(torch.cat(raw).mean()),saved_multiplier=multiplier,tail_cotangent_norm=tail,unit_relative_rank_cotangent_norm=rank,saved_linear_kl_cotangent_norm=linear,tail_over_rank=tail/rank if rank else None,linear_over_rank=linear/rank if rank else None,unit_relative_rank_loss=float(sums[2]/count),strict_topk_changes=int(sums[16]))
  rows.append(row);(a.output/'rows.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True);del c,private,parameters
 medians={d:(statistics.median([r['tail_over_rank'] for r in rows if r['domain']==d]) if all(r['unit_relative_rank_cotangent_norm']>0 for r in rows if r['domain']==d) else None) for d in ('gpt','cifar10')}
 summary=dict(checkpoint_sha256=a.checkpoint_sha256,scope='Fixed saved32 fitting-only scale observations, not coefficient selection; GPT6/499 first16 train and CNN0/40 first1024 fitting images; no query',rows=rows,per_domain_tail_rank_median_if_both_nonzero=medians,compute=counts,seconds=time.perf_counter()-beg,peak_bytes=torch.cuda.max_memory_allocated())
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2));wb.log({'seconds':summary['seconds'],'gpt_ratio':medians['gpt'],'cifar_ratio':medians['cifar10']});wb.finish();print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
