"""Read-only saved private-span projection diagnostic; no GPT/optimizer calls."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.prefix_cache import unpack_record
from rewriter.edit_scope import editable_names,editable_weight_norm_sq,scoped_snapshot
from scripts.fit_cpu_edit_teacher import make_target,write_span_capture,sha,relative_target_error
from rewriter.adaptive_edit_target import edit_target_terms


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2)
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True);meta=c['metadata'];args=meta['args']
    if sha(args['state'])!=meta['state_sha256'] or sha(args['selection'])!=meta['selection_sha256']:raise ValueError('state/selection identity mismatch')
    data=torch.load(args['state'],map_location='cpu',weights_only=False,mmap=True)
    snapshot=unpack_record(data['observation'],device='cpu').snapshot
    names=set(snapshot.matrices)|set(snapshot.vectors)
    parameters={n:v.detach().float() for n,v in data['model'].items() if n in names}
    grouped=GroupedNormObjective(parameters,zero_reference_policy='freeze')
    editable=editable_names(parameters,'grouped')-grouped.frozen_parameter_names
    snapshot=scoped_snapshot(snapshot,editable_weight_norm_sq(parameters,editable),meta['radius_fraction'],scope='grouped')
    selection=json.loads(Path(args['selection']).read_text())
    target,_=make_target(parameters,snapshot,selection['rows'],editable)
    actor=AdaptiveCoordinateChisel(rank=8,trajectory_dim=400,model_dim=256,num_layers=2,latent_tokens=8,fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
    if actor.configuration()!=meta['architecture']:raise ValueError('architecture mismatch')
    private=actor.make_private(snapshot,state_key=('cpu-teaching',0,data['completed_steps'],meta['state_sha256']),seed=args['private_seed'])
    private.load_state_dict(c['private']);del actor
    capture=write_span_capture(private,target)
    left={};right={}
    for i,name in enumerate(private.matrix_names):
        if name not in target.matrices:continue
        f=target.matrices[name];u=private.matrix_rows[i].orthonormal_matrix().double();v=private.matrix_columns[i].orthonormal_matrix().double()
        l=(u.T@f.U)@f.C;r=f.C@(f.V.T@v)
        left[name]=float(((l@(f.V.T@f.V))*l).sum())
        right[name]=float(((f.U.T@f.U)@r*r).sum())
    for name in target.vectors:
        left[name]=right[name]=capture['captured_squared_by_parameter'][name]
    def weighted(values):
        return sum(sum(values.get(n,0.) for n in ns)/float(grouped.reference_squared[g]) for g,ns in grouped.groups.items() if float(grouped.reference_squared[g])>0)

    terms=edit_target_terms(target.scaled(0.),target);den=float(relative_target_error(terms,grouped))
    by_group={g:sum(capture['captured_squared_by_parameter'].get(n,0.) for n in ns)/float(grouped.reference_squared[g]) if float(grouped.reference_squared[g])>0 else 0. for g,ns in grouped.groups.items()}
    relative_capture=sum(by_group.values())/den
    metrics=[json.loads(s) for s in (Path(args['output'])/'metrics.jsonl').read_text().splitlines()]
    row=next(r for r in metrics if r.get('step')==c['step'])
    result=dict(step=c['step'],checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),source_sha256=sha(__file__),global_capture=capture,
        relative_capture=relative_capture,relative_error_floor=1-relative_capture,
        left_only_relative_capture=weighted(left)/den,right_only_relative_capture=weighted(right)/den,
        left_only_captured_squared=left,right_only_captured_squared=right,
        realized_relative_error=row['relative_normalized_edit_error'],relative_decoder_gap=row['relative_normalized_edit_error']-(1-relative_capture),
        realized_global_error=row['normalized_edit_error'],global_decoder_gap=row['normalized_edit_error']-capture['decoder_independent_normalized_error_floor'],
        selected_error_squared=row['selected_error_squared'],zero_target_error_squared=row['zero_target_error_squared'],
        embedding_error_squared=row['group_error_squared']['embedding.weight'],gpt_forwards=0,gpt_backwards=0,optimizer_updates=0,learned_preservation_claim=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('global_capture','left_only_captured_squared','right_only_captured_squared')},indent=2))


if __name__=='__main__':main()
