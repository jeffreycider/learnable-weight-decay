"""Read-only proposed and realized group alignment; zero GPT forwards/backwards."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.train_multistate_chisel import GlobalRewriter,GPT,FunctionalPreservation,StateBankReader,build_proposal,enable_deterministic_execution
from rewriter.accelerated_controller import PackedGlobalRewriter
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.alignment_diagnostics import tensor_alignment,summarize_alignment


def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(4);torch.set_num_interop_threads(20);enable_deterministic_execution();device=torch.device('cuda');a.output.mkdir(parents=True,exist_ok=False)
 saved=torch.load(a.checkpoint,map_location=device,weights_only=False)
 if not saved.get('completed'):raise ValueError('require the declared final checkpoint, no checkpoint selection')
 options=saved['metadata']['options'];reader=StateBankReader(a.bank,require_complete=False,verify_once=True,mmap_model=True);provenance=reader.manifest['provenance'];model=GPT(**provenance['model_config']).to(device);model.requires_grad_(False)
 engine=FunctionalPreservation(model,kl_dtype=torch.float64);names=set(engine.parameters)
 def forbidden(*args,**kwargs):raise AssertionError('alignment diagnostics must never execute GPT')
 engine._forward=forbidden
 rank=provenance['sensor_config']['probe_rank'];actor=GlobalRewriter(rank=rank,trajectory_dim=(2*provenance['model_config']['num_layers']+1)*2*rank).to(device);actor.load_state_dict(saved['controller']);forward=torch.compile(PackedGlobalRewriter(actor),fullgraph=True,dynamic=False)
 holder={};original=engine.edited_parameters
 def capture(action,scale=1.):holder.update(action=action,scale=scale);return original(action,scale)
 engine.edited_parameters=capture;rows=[];started=time.perf_counter()
 for state_id in range(1,26):
  record=reader.read(state_id,device=device,model_device='cpu');engine.load_teacher_state({n:v for n,v in record.model.items() if n in names},{n:v for n,v in record.model.items() if n not in names});grouped=AcceleratedGroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
  with torch.enable_grad():parameters,regularizer,stats,editable,credit=build_proposal(forward,record.snapshot,engine,grouped,radius_fraction=options['radius_fraction'])
  measurements={};scale=holder['scale'].detach().double()
  for group,members in grouped.groups.items():
   totals={};raw_sq=raw_dot=0.
   for name in members:
    base=engine.parameters[name];new=parameters[name];anchors=record.snapshot.matrices[name].anchors if base.ndim==2 else None
    item=tensor_alignment(base.T if name=='embed.weight' else base,new.T if name=='embed.weight' else new,anchors)
    for key,value in item.items():totals[key]=totals.get(key,0.)+value
    if name in holder['action'].matrices:
     factors=holder['action'].matrices[name];P=factors.P.detach().double()*scale;Q=factors.Q.detach().double()*scale;wb,wta=record.snapshot.matrices[name].weight_probes
     raw_sq+=float(P.square().sum()+Q.square().sum());raw_dot+=float((wb.double()*P).sum()+(wta.double()*Q).sum())
    elif name in holder['action'].vectors:
     delta=holder['action'].vectors[name].detach().double()*scale;raw_sq+=float(delta.square().sum());raw_dot+=float((base.double()*delta).sum())
   item=summarize_alignment(totals);denominator=(max(item['projected_weight_squared'],0)*raw_sq)**.5
   item.update(proposed_edit_squared=raw_sq,proposed_weight_dot_edit=raw_dot,proposed_cosine_to_negative_projected_weight=-raw_dot/denominator if denominator else None)
   measurements[group]=item
  row=dict(state_id=state_id,role='heldout_states' if state_id%5==0 else 'training_states',groups=measurements,elapsed_seconds=time.perf_counter()-started)
  rows.append(row)
  with (a.output/'alignment.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps(dict(state_id=state_id,elapsed_seconds=row['elapsed_seconds'])),flush=True)
  holder.clear();del parameters,regularizer,credit,record,grouped
 (a.output/'summary.json').write_text(json.dumps(dict(completed=True,gpt_forwards=0,gpt_backwards=0,checkpoint=str(a.checkpoint),producer_metadata=saved['metadata'],rows=rows),indent=2))
if __name__=='__main__':main()
