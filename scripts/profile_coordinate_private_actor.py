"""Bounded private-only actor graph profile, fixed synthetic action cotangents."""
import argparse,json,time,sys,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.gpt_precision import configure_precision
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.coordinate_actor_graph import CoordinateActorGraph
from rewriter.private_probe_slot import PrivateProbeSlot
from rewriter.edit_scope import editable_names
from scripts.qualify_coordinate_thin_student import Contexts,grads,relative
from scripts.qualify_coordinate_fullbank import read_joint


def outputs(action,editable):
 return tuple(t for n,f in action.matrices.items() if n in editable for t in (f.P,f.Q))+tuple(v for n,v in action.vectors.items() if n in editable)


def run(args):
 enable_deterministic_execution();torch.cuda.set_device(0);configure_precision('fp32');torch.set_num_threads(1)
 out=Path(args.output);out.mkdir();device=torch.device('cuda',0)
 recipe=json.loads(Path(args.recipe).read_text());saved,_,payloads=read_joint(args.checkpoint)
 contexts=Contexts(json.loads(Path(recipe['base_recipe']).read_text()),device,gpt_sequences=16,cifar_images=128)
 rows={}
 for domain,uid in [('gpt',6),('cifar10',1)]:
  actor=CoordinateChisel(**recipe['architecture']).to(device);actor.load_state_dict(saved['model']);actor.requires_grad_(False)
  context=contexts.load(domain,uid);snapshot=context['snapshot'];key=('gpt',0,uid) if domain=='gpt' else ('cifar10',uid-1)
  payload=next(v for v in payloads.values() if tuple(v['identity']['state_key'])==key)
  private=actor.make_private(snapshot,state_key=key,seed=19);private.load_state_dict(payload['private'])
  editable=editable_names(context['engine'].parameters,'grouped')-context['grouped'].frozen_parameter_names
  actor.zero_grad(set_to_none=True);private.zero_grad(set_to_none=True)
  torch.cuda.synchronize();start=time.perf_counter();graph=CoordinateActorGraph(actor,PrivateProbeSlot(actor,private),snapshot,editable);torch.cuda.synchronize();capture=time.perf_counter()-start
  # Prior other-domain shared accumulation must survive private-only work.
  for p in actor.parameters():p.grad=torch.full_like(p,.125)
  sentinel={n:p.grad.clone() for n,p in actor.named_parameters()}
  refs=None;cot=None;checks=[];times={'eager':[],'graph':[]}
  for iteration,mode in enumerate(['eager','graph']+['eager','graph','graph','eager']*3):
   private.zero_grad(set_to_none=True)
   torch.cuda.synchronize();start=time.perf_counter()
   if mode=='graph':graph.bind_scope(private,editable);action=graph(snapshot,private)
   else:action=actor(snapshot,private)
   values=outputs(action,editable)
   if cot is None:
    cot=tuple(torch.full_like(v,1./max(v.numel(),1)) for v in values)
   torch.autograd.backward(values,cot)
   torch.cuda.synchronize();elapsed=time.perf_counter()-start
   if iteration>=2:times[mode].append(elapsed)
   current=grads(private)
   if refs is None:refs={'outputs':tuple(v.detach().clone() for v in values),'grads':current}
   checks.append(dict(outputs_exact=all(torch.equal(a,b) for a,b in zip(refs['outputs'],values)),private_gradient=relative(refs['grads'],current),shared_gradients_exact=all(torch.equal(p.grad,sentinel[n]) for n,p in actor.named_parameters())))
   del action,values,current
  med={k:statistics.median(v) for k,v in times.items()}
  rows[domain]=dict(capture_seconds=capture,times=times,median_seconds=med,fraction_saved=1-med['graph']/med['eager'],checks=checks,all_exact=all(c['outputs_exact'] and c['private_gradient']['absolute_l2']==0 and c['private_gradient']['none_mask_equal'] and c['shared_gradients_exact'] for c in checks),tokens=actor.token_counts(snapshot))
  del graph,actor,private,context,snapshot,refs,cot,sentinel;torch.cuda.empty_cache()
 summary=dict(scope='Actor forward and private-only VJP, shared frozen with prior accumulated gradients; saved tied-Q model, fixed synthetic action cotangents, no GPT/CNN student forwards or Adam. Includes graph binding/observation copies and callback copies.',rows=rows,all_gates=all(v['all_exact'] for v in rows.values()))
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
 if not summary['all_gates']:raise RuntimeError('private-only actor graph exactness gate failed')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);run(p.parse_args())
