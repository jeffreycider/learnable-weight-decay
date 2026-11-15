"""Independent same-state full-panel derivative check; no optimizer steps."""
import argparse,copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.gpt_precision import configure_precision
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.coordinate_fitting_panels import split_fitting_panels
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from scripts.qualify_coordinate_fullbank import read_joint
from scripts.qualify_coordinate_thin_student import Contexts,grads,relative
from scripts.train_coordinate_chisel import measure_context


def run(args):
 enable_deterministic_execution();torch.cuda.set_device(0);configure_precision('fp32');torch.set_num_threads(1)
 device=torch.device('cuda',0);out=Path(args.output);out.mkdir(exist_ok=False)
 recipe=json.loads(Path(args.recipe).read_text());saved,ranks,private=read_joint(args.checkpoint)
 token,payload=next((k,v) for k,v in private.items() if tuple(v['identity']['state_key'])==('gpt',0,6))
 owner=next(r for r in ranks.values() if token in r['visits']);dual_state=owner['duals']['gpt']
 contexts=Contexts(json.loads(Path(recipe['base_recipe']).read_text()),device,gpt_sequences=128,cifar_images=2048)
 actor=CoordinateChisel(**recipe['architecture']).to(device);context=contexts.load('gpt',6)
 probe=actor.make_private(context['snapshot'],state_key=('gpt',0,6),seed=19)
 dual=TokenwiseDual(list(dual_state['multipliers']),config=TokenwiseConfig(**recipe['objectives']['gpt']),dual_lr=recipe['dual_lr'])
 _,joint=split_fitting_panels(context,inner_steps=2,inner_size=16,joint_size=64,seed=991)
 results={};vectors={}
 for mode in ('dense','thin'):
  actor.load_state_dict(saved['model']);probe.load_state_dict(payload['private']);dual.load_state_dict(copy.deepcopy(dual_state))
  actor.zero_grad(set_to_none=True);probe.zero_grad(set_to_none=True);contexts.thin=mode=='thin'
  torch.cuda.synchronize();start=time.perf_counter();row=measure_context(contexts,joint,actor,probe,dual.config,dual,learning=True,weight=.5);torch.cuda.synchronize()
  results[mode]=dict(seconds=time.perf_counter()-start,row=row)
  vectors[mode]={'shared':{n:v.cpu() for n,v in grads(actor).items()},'private':{n:v.cpu() for n,v in grads(probe).items()}}
 metrics={k:relative(vectors['dense'][k],vectors['thin'][k]) for k in ('shared','private')}
 gates=dict(same_stored_forward_metrics_exact=results['dense']['row']==results['thin']['row'],same_state_shared_private_VJP=max(v['relative_l2'] for v in metrics.values())<=.003,none_masks_equal=all(v['none_mask_equal'] for v in metrics.values()))
 summary=dict(scope='Independent same-state GPT6 full64 joint panel; same saved tied actor/private parameters and dual; no Adam updates, CNN or query. Eight GPT forward/backward microbatches total.',preset_relative_VJP=.003,gradients=metrics,gates=gates,all_gates=all(gates.values()),rows={k:v['row'] for k,v in results.items()},single_pair_seconds={k:v['seconds'] for k,v in results.items()},timing_limitation='Not matched speed qualification; excludes context loading and state restore.')
 (out/'summary.json').write_text(json.dumps(summary,indent=2));torch.save(vectors,out/'gradients.pt');print(json.dumps(summary),flush=True)
 if not summary['all_gates']:raise RuntimeError('same-state thin VJP gate failed')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);run(p.parse_args())
