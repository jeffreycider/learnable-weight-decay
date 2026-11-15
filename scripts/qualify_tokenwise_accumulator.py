"""Real-state graph adapter accumulation, zero-mask and audit recapture check."""
from pathlib import Path
import argparse,copy,gc,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from baseline.model import GPT
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from rewriter.offline_systems import CachedBaseNormPreservation,VerifiedStateConstants,select_panel_blocks
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.chisel_factory import build_chisel
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual,tokenwise_panel
from rewriter.offline_tokenwise_graph import TokenwiseGraphAccumulator
from scripts.train_multistate_chisel import build_proposal,tokenwise_aggregate

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);enable_deterministic_execution();torch.set_num_threads(1);device=torch.device('cuda:0')
 reader=StateBankReader('/scratch/checkpoints/learnable-weight-decay/chisel-state-bank-seed0-v1',verify_once=True,mmap_model=True);cache=TeacherCacheReader('/scratch/checkpoints/learnable-weight-decay/chisel-teacher-cache1000-fp32-eps23-v2',verify_once=True,mmap=True)
 engine=CachedBaseNormPreservation(GPT(**reader.manifest['provenance']['model_config']).to(device).requires_grad_(False),microbatch_sequences=16,teacher_cache_device=device,kl_dtype=torch.float64,gpt_precision='fp32');constants=VerifiedStateConstants(engine,AcceleratedGroupedNormObjective)
 saved=torch.load('/scratch/checkpoints/learnable-weight-decay/chisel-fullbank-tokenwise-v1/controller_epoch_0005.pt',map_location='cpu',weights_only=False);actor=build_chisel(saved['metadata']['architecture']).to(device);actor.load_state_dict(saved['controller']);forward=torch.compile(actor,fullgraph=True,dynamic=False);cfg=TokenwiseConfig(**saved['metadata']['tokenwise_objective']);ds=saved['primal_dual'];dual=TokenwiseDual(ds['multipliers'],config=cfg,dual_lr=ds['dual_lr']);dual.load_state_dict(ds)
 def read(s):return reader.read(s,model_device='cpu',device=device),select_panel_blocks(cache.read(s,'train'),range(4))
 def vec():return torch.cat([p.grad.detach().flatten() for p in actor.parameters()]).cpu()
 def reference(s):
  record,panel=read(s);constants.load(cache,record);grouped=constants.grouped();actor.zero_grad(set_to_none=True);edited,reg,_,_,credit=build_proposal(forward,record.snapshot,engine,grouped,radius_fraction=.01,norm_squared=constants.radius_squared(grouped),shrinkage_coefficient=1.)
  return tokenwise_panel(engine,edited,reg,panel.inputs,config=cfg,dual=dual,state_id=s,state_weight=1/200,actor_credit=credit,teacher_provider=lambda x,y:panel.teacher_batch(engine,x,y))
 states=[1,6];refs=[];grads=[]
 for s in states:refs.append(reference(s));grads.append(vec())
 expected=sum(grads);actor.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();adapter=TokenwiseGraphAccumulator(actor,forward,engine,cache,constants,radius_fraction=.01,shrinkage_coefficient=1.,capture_warmups=1);results=[]
 for cycle in range(2):
  adapter.begin_update(states,global_checkpoint_count=200);rows=[]
  for s in states:
   record,panel=read(s);measured,stats=adapter.accumulate(record,panel,dual);rows.append(dict(state=s,**measured,**stats))
  adapter.finish_update();got=vec();rel=float((got.double()-expected.double()).norm()/expected.double().norm());results.append(dict(cycle=cycle,gradient_relative=rel,rows=rows,aggregate=tokenwise_aggregate(rows),capture_count=adapter.capture_count));adapter.close();gc.collect();torch.cuda.empty_cache()
  if rel>.003:raise RuntimeError('adapter gradient differs')
  if cycle==0:reference(6);actor.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
 result=dict(scope='Two realstates1/6×64 withproductionweight1/200; noAdam/dualascent; zero-reference-mask transition and close/eager-backward/recapture',results=results,reference=refs,dual_unchanged=dual.state_dict()==ds,actor_unchanged=all(torch.equal(p.cpu(),saved['controller'][n]) for n,p in actor.state_dict().items()),compute=adapter.compute,peak_bytes=torch.cuda.max_memory_allocated())
 (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
