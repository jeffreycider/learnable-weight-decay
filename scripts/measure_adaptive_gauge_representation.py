"""CPU-only thin-QR representation of preselected rank-capped binary gauges."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.adaptive_gauge_control import orthonormal_product


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):value.update(chunk)
    return value.hexdigest()


def measure(base, left, right, target):
    exact_delta=target.double()-base.double()
    entry=dict(shape=list(base.shape),dtype=str(base.dtype),edit_squared_norm=float(exact_delta.square().sum()),
               target_changed_elements=int((target!=base).sum()),elements=base.numel())
    changes={}
    for label,dtype in [('fp64',torch.float64),('fp32',torch.float32)]:
        f=orthonormal_product(left.to(dtype),right.to(dtype),rank=8)
        dense=f.dense()
        error=dense.double()-exact_delta
        stored=(base.to(dtype)+dense).to(base.dtype)
        stored_error=stored.double()-target.double()
        entry[label]=dict(reconstruction_error_squared=float(error.square().sum()),
            reconstruction_max_abs=float(error.abs().max()),
            factor_norm_squared=float(f.norm_sq().double()),dense_edit_squared_norm=float(dense.double().square().sum()),
            stored_mismatches=int((stored!=target).sum()),stored_max_abs=float(stored_error.abs().max()),
            stored_error_squared=float(stored_error.square().sum()),
            orthogonality_max_abs=max(float((f.U.T@f.U-torch.eye(8,dtype=dtype)).abs().max()),
                                      float((f.V.T@f.V-torch.eye(8,dtype=dtype)).abs().max())))
        delta=stored.double()-base.double()
        changes[label]=float((delta*(stored.double()+base.double())).sum())
    changes['direct']=float((exact_delta*(target.double()+base.double())).sum())
    return entry,changes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-bank',required=True,type=Path)
    p.add_argument('--selection',required=True,type=Path)
    p.add_argument('--selection-sha256',required=True)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if sha(args.selection)!=args.selection_sha256:raise ValueError('selection changed')
    torch.set_num_threads(1);started=time.perf_counter()
    selected=json.loads(args.selection.read_text())
    index={row['completed_steps']:row for row in map(json.loads,(args.state_bank/'index.jsonl').read_text().splitlines())}
    results=[]
    for sid in sorted({r['state_id'] for r in selected['rows']}):
        rows=[r for r in selected['rows'] if r['state_id']==sid]
        record=index[sid];path=args.state_bank/record['file']
        if any(r['state_sha256']!=record['sha256'] for r in rows) or sha(path)!=record['sha256']:
            raise ValueError('state source differs from pinned selection')
        saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False);model=saved['model']
        family_results=[];matrix_results=[];bias_sq=0.
        for row in rows:
            if row['rank_limit']!=8:raise ValueError('this diagnostic is explicitly rank8')
            groups={}
            for choice in row['selected']:groups.setdefault(choice['first_weight'],[]).append(choice)
            norm_deltas={label:torch.zeros(3,dtype=torch.float64) for label in ('direct','fp64','fp32')}
            for first_name,choices in groups.items():
                if len(choices)>8:raise ValueError('selected channel cap violated')
                second_name=choices[0]['second_weight'];bias_name=choices[0]['first_bias']
                indices=torch.tensor([c['channel'] for c in choices])
                scales=2.**torch.tensor([c['exponent'] for c in choices],dtype=torch.float64)
                first=model[first_name];second=model[second_name];bias=model[bias_name]
                e=torch.zeros(first.shape[0],len(indices),dtype=torch.float64)
                e[indices,torch.arange(len(indices))]=1
                target_first=first.clone();target_first[indices]=(first[indices].double()*scales[:,None]).to(first.dtype)
                target_second=second.clone();target_second[:,indices]=(second[:,indices].double()*scales.pow(-row['inverse_power'])[None,:]).to(second.dtype)
                target_bias=bias.clone();target_bias[indices]=(bias[indices].double()*scales).to(bias.dtype)
                cases=[(first_name,first,e*(scales-1),first[indices].double().T,target_first,0),
                       (second_name,second,second[:,indices].double()*(scales.pow(-row['inverse_power'])-1),e,target_second,2)]
                for name,base,left,right,target,group in cases:
                    stats,changes=measure(base,left,right,target)
                    stats.update(name=name,family=row['family'],selected_channels=len(indices))
                    matrix_results.append(stats)
                    for label in norm_deltas:norm_deltas[label][group]+=changes[label]
                delta=target_bias.double()-bias.double()
                bias_sq+=float(delta.square().sum())
                bias_change=(delta*(target_bias.double()+bias.double())).sum()
                for values in norm_deltas.values():values[1]+=bias_change
            reference=torch.tensor(row['reference_group_squared_norms'],dtype=torch.float64)
            active=reference>0
            rewards={}
            for label,delta in norm_deltas.items():
                reduction=torch.zeros(3,dtype=torch.float64)
                reduction[active]=1-((reference[active]+delta[active])/reference[active]).sqrt()
                rewards[label]=dict(mean_15group_l2_percentage_reduction=float(100*reduction.sum()/15),
                                    three_group_reductions_percent=(100*reduction).tolist())
            family_results.append(dict(family=row['family'],rewards=rewards,
                selected_channels=len(row['selected']),selection_reward=row['mean_group_l2_percentage_reduction']))
        total=sum(m['edit_squared_norm'] for m in matrix_results)
        aggregate={label:dict(
            matrix_relative_reconstruction_l2=(sum(m[label]['reconstruction_error_squared'] for m in matrix_results)/total)**.5,
            reconstruction_max_abs=max(m[label]['reconstruction_max_abs'] for m in matrix_results),
            stored_mismatches=sum(m[label]['stored_mismatches'] for m in matrix_results),
            stored_max_abs=max(m[label]['stored_max_abs'] for m in matrix_results),
            matrix_norm_sq_relative_error=(sum(m[label]['dense_edit_squared_norm'] for m in matrix_results)-total)/total,
            mean_15group_l2_percentage_reduction=sum(f['rewards'][label]['mean_15group_l2_percentage_reduction'] for f in family_results)
        ) for label in ('fp64','fp32')}
        results.append(dict(state_id=sid,state_sha256=record['sha256'],families=family_results,matrices=matrix_results,
            aggregate=aggregate,matrix_edit_squared_norm=total,bias_edit_squared_norm=bias_sq,
            reference_changed_matrix_elements=sum(m['target_changed_elements'] for m in matrix_results),
            direct_mean_15group_l2_percentage_reduction=sum(f['rewards']['direct']['mean_15group_l2_percentage_reduction'] for f in family_results)))
        del model,saved
    payload=dict(results=results,elapsed_seconds=time.perf_counter()-started,
        selection_sha256=args.selection_sha256,
        source_sha256={name:sha(Path(__file__).resolve().parents[1]/name) for name in
                      ('scripts/measure_adaptive_gauge_representation.py','rewriter/adaptive_gauge_control.py','rewriter/adaptive_action.py')},
        gpt_forwards=0,gpt_backwards=0,selection_changed=False,
        scope='Factor-family representability only. Full selected gauges, no radius interpolation. Bias edits use exact selected-coordinate vector spans. Matrix QR/GEMM measured at FP64 and FP32; additional neural-head/group-RMS normalization rounding not modeled. No prediction/rank preservation claim for this capped selection; no claim a shared actor learns these cores.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    print(json.dumps(dict(elapsed_seconds=payload['elapsed_seconds'],states=[dict(state_id=r['state_id'],aggregate=r['aggregate']) for r in results])))


if __name__=='__main__':main()
