"""Read-only algebraic rank and learned-span residual audit; no optimization."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.prefix_cache import unpack_record
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.edit_scope import editable_names
from scripts.fit_cpu_edit_teacher import make_target,sha


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    root=Path('runs/cpu-edit-teacher');checkpoint=root/'relative-head1e-6-resume1000-v1/checkpoint_1000.pt'
    meta=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)['metadata'];args=meta['args']
    if sha(args['state'])!=meta['state_sha256'] or sha(args['selection'])!=meta['selection_sha256']:raise ValueError('evidence mismatch')
    state=torch.load(args['state'],map_location='cpu',weights_only=False,mmap=True)
    snapshot=unpack_record(state['observation'],device='cpu').snapshot;names=set(snapshot.matrices)|set(snapshot.vectors)
    params={n:v.detach().float() for n,v in state['model'].items() if n in names};grouped=GroupedNormObjective(params,zero_reference_policy='freeze')
    selected=json.loads(Path(args['selection']).read_text())['rows'];editable=editable_names(params,'grouped')-grouped.frozen_parameter_names
    target,_=make_target(params,snapshot,selected,editable)
    counts={}
    for row in selected:
        for choice in row['selected']:
            for key in ('first_weight','second_weight'):counts[choice[key]]=counts.get(choice[key],0)+1
    audit=json.loads((root/'span1000-baseline.json').read_text());capture=audit['global_capture']['captured_squared_by_parameter']
    name_group={n:g for g,ns in grouped.groups.items() for n in ns};rows=[];oracle_error=0.;total=0.;relative_total=0.;maxrank=0
    for name,f in target.matrices.items():
        energy=float(f.norm_sq());total+=energy;relative_total+=energy/float(grouped.reference_squared[name_group[name]])
        if energy==0:continue
        singular=torch.linalg.svdvals(f.C);threshold=float(singular[0])*torch.finfo(torch.float64).eps*max(f.C.shape)
        rank=int((singular>threshold).sum());maxrank=max(maxrank,rank)
        if rank>counts[name] or counts[name]>8:raise ValueError('teacher algebraic rank bound violated')
        gram_u=f.U.T@f.U;gram_v=f.V.T@f.V
        projected=gram_u@f.C@gram_v
        # Same orthonormal target factors give an exactly feasible write-span choice.
        ortherr=max(float((gram_u-torch.eye(8,dtype=torch.float64)).abs().max()),float((gram_v-torch.eye(8,dtype=torch.float64)).abs().max()))
        if ortherr>1e-10:raise ValueError('target factor columns not orthonormal')
        residual=float((projected-f.C).square().sum());oracle_error+=residual
        rows.append(dict(name=name,channel_rank_bound=counts[name],measured_core_rank=rank,singular_values=singular.tolist(),
            target_energy=energy,learned_capture_fraction=capture[name]/energy,
            left_only_capture_fraction=audit['left_only_captured_squared'][name]/energy,
            right_only_capture_fraction=audit['right_only_captured_squared'][name]/energy,
            missing_global_energy=energy-capture[name],missing_relative_energy=(energy-capture[name])/float(grouped.reference_squared[name_group[name]]),
            target_span_reconstruction_core_error_squared=residual,orthonormal_error=ortherr))
    for name,v in target.vectors.items():
        energy=float(v.square().sum());total+=energy;relative_total+=energy/float(grouped.reference_squared[name_group[name]])
        if energy:
            q=v/v.norm();oracle_error+=float((q*(q@v)-v).square().sum())
    for row in rows:row['global_floor_contribution']=row['missing_global_energy']/total;row['relative_floor_contribution']=row['missing_relative_energy']/relative_total
    rows.sort(key=lambda r:-r['relative_floor_contribution'])
    result=dict(scope=__doc__,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),source_sha256=sha(__file__),
        max_algebraic_rank_bound=max(counts.values()),max_measured_core_rank=maxrank,write_rank=8,
        constructive_oracle_span_error_squared=oracle_error,
        qualification='Target U/V are feasible independent orthonormal private frames; nonzero vector targets fit one direction. This does not prove shared actor decoder jointly realizes target after changing its read frames.',
        learned_global_floor=audit['global_capture']['decoder_independent_normalized_error_floor'],learned_relative_floor=audit['relative_error_floor'],
        rows=rows,gpt_forwards=0,gpt_backwards=0,optimizer_updates=0)
    (root/'rank-vs-span-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));print(json.dumps(rows[:6],indent=2))


if __name__=='__main__':main()
