"""Read-only raw FP64 versus stored FP32 group-norm attribution; no GPT.

Loads one immutable joint actor/private checkpoint. Uses actual fresh U.T W V
sensing, group RMS scaling and global radius; materializes only one dense matrix
edit at a time. One FP64 sum then FP32 cast, identical to adaptive proposal code.
Positive percentages mean shrinkage; matrix/head-bias/other-bias contributions
sum to the SAME equal-functional-group objective. No optimizer or model forward.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.adaptive_action import AdaptiveAction
from rewriter.adaptive_preservation import group_rms_scales
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.edit_scope import editable_names,editable_weight_norm_sq,scoped_snapshot
from rewriter.prefix_cache import unpack_record
from scripts.frozen_private_frames import FrozenPrivateFrames
from scripts.fit_cpu_edit_teacher import sha,physical_action,exact_state_equal


def percent(reference,change):
    if reference==0:return 0.
    root=math.sqrt(reference)
    return -100*change/(root*(math.sqrt(max(0.,reference+change))+root))


@torch.no_grad()
def stored_terms(base,delta,scale):
    # delta uses exact full-matrix U @ C @ V.T evaluation, as production; only
    # additions/casts/reductions are chunked so no full replacement is retained.
    stored_change=stored_edit=raw_dense_change=raw_dense_edit=0.
    nonzero=erased=changed=0
    for start in range(0,len(base),512):
        old=base[start:start+512].double();d=scale*delta[start:start+512]
        new=(old+d).to(base.dtype);actual=new.double()-old
        stored_change+=float(2*(old*actual).sum()+actual.square().sum())
        stored_edit+=float(actual.square().sum())
        raw_dense_change+=float(2*(old*d).sum()+d.square().sum());raw_dense_edit+=float(d.square().sum())
        nonzero+=int((d!=0).sum());erased+=int(((d!=0)&(actual==0)).sum());changed+=int((actual!=0).sum())
    return dict(stored_squared_norm_change=stored_change,stored_edit_squared=stored_edit,raw_dense_squared_norm_change=raw_dense_change,
        raw_dense_edit_squared=raw_dense_edit,raw_nonzero_elements=nonzero,raw_nonzero_erased_by_fp32=erased,stored_changed_elements=changed,
        zero_cast_fraction=erased/nonzero if nonzero else None,total_elements=base.numel())


@torch.no_grad()
def measure(actor,saved,case):
    tick=time.perf_counter();paths=case['paths']
    for key in ('state','checkpoint'):
        if sha(paths[key])!=case['sha256'][key]:raise ValueError('case source changed: '+key)
    state=torch.load(paths['state'],map_location='cpu',weights_only=False,mmap=True)
    seed=torch.load(paths['checkpoint'],map_location='cpu',weights_only=False,mmap=True)['metadata']
    if seed['architecture']!=actor.configuration() or state['completed_steps']!=case['state_id']:raise ValueError('case architecture/clock mismatch')
    snapshot=unpack_record(state['observation'],device='cpu').snapshot
    names=set(snapshot.matrices)|set(snapshot.vectors);parameters={n:v.detach().float() for n,v in state['model'].items() if n in names}
    grouped=GroupedNormObjective(parameters,zero_reference_policy='freeze');editable=editable_names(parameters,'grouped')-grouped.frozen_parameter_names
    snapshot=scoped_snapshot(snapshot,editable_weight_norm_sq(parameters,editable),.01,scope='grouped')
    private=actor.make_private(snapshot,state_key=('cpu-teaching',0,case['state_id'],case['sha256']['state']),seed=seed['args']['private_seed'])
    private.load_state_dict(saved['private'][case['state_id']],strict=True);private.requires_grad_(False)
    if private.get_extra_state()!=case['private_identity']:raise ValueError('private identity differs')
    phase=FrozenPrivateFrames(private);cores=phase.query_weight_cores(parameters);raw=phase.evaluate(actor,snapshot,weight_cores=cores)
    scales=group_rms_scales(grouped,dtype=torch.float64)
    action=AdaptiveAction({n:f.scaled(scales[n]) for n,f in raw.matrices.items() if n in editable},
        {n:v*scales[n] for n,v in raw.vectors.items() if n in editable})
    _,norm,guard=physical_action(raw,grouped,snapshot)
    torch.testing.assert_close(action.norm_sq().clamp_min(1e-24).sqrt(),norm,atol=0,rtol=0)
    per_parameter={}
    for n,base in parameters.items():
        if n not in editable:continue
        if base.ndim==2:
            f=action.matrices[n]
            # These cores were queried with the same fixed private U,V used by
            # actual actor outputs; use general Gram norm, not ||C||^2 shortcut.
            thin=2*guard*(cores[n]*f.C).sum()+guard.square()*f.norm_sq()
            delta=f.dense()
            if n=='embed.weight':delta=delta.T
        else:
            delta=action.vectors[n];thin=2*guard*(base.double()*delta).sum()+guard.square()*delta.square().sum()
        row=stored_terms(base,delta,guard);row['raw_thin_squared_norm_change']=float(thin)
        row['thin_vs_dense_abs_error']=abs(float(thin)-row['raw_dense_squared_norm_change'])
        if not math.isclose(float(thin),row['raw_dense_squared_norm_change'],rel_tol=1e-7,abs_tol=1e-12):raise ValueError('thin/dense norm disagreement: '+n)
        per_parameter[n]=row;del delta
    groups={}
    for g,ns in grouped.groups.items():
        rows=[per_parameter[n] for n in ns if n in per_parameter];reference=float(grouped.reference_squared[g])
        rawchange=sum(r['raw_thin_squared_norm_change'] for r in rows);storedchange=sum(r['stored_squared_norm_change'] for r in rows)
        nonzero=sum(r['raw_nonzero_elements'] for r in rows);erased=sum(r['raw_nonzero_erased_by_fp32'] for r in rows)
        groups[g]=dict(reference_squared=reference,raw_squared_norm_change=rawchange,stored_squared_norm_change=storedchange,
            raw_shrink_percent=percent(reference,rawchange),stored_shrink_percent=percent(reference,storedchange),
            raw_edit_squared=sum(r['raw_dense_edit_squared'] for r in rows),stored_edit_squared=sum(r['stored_edit_squared'] for r in rows),
            raw_nonzero_elements=nonzero,zero_cast_fraction=erased/nonzero if nonzero else None,
            stored_changed_elements=sum(r['stored_changed_elements'] for r in rows))
    summary={}
    for kind in ('raw','stored'):
        contributions={role:sum(v[kind+'_shrink_percent'] for g,v in groups.items() if
            ('matrices' if g.endswith('.weight') else 'head_bias' if g=='head.bias' else 'other_biases')==role)/len(groups)
            for role in ('matrices','head_bias','other_biases')}
        summary[kind]=dict(mean_group_shrink_percent=sum(contributions.values()),contributions_to_mean_percent=contributions,
            whole_editable_shrink_percent=percent(sum(v['reference_squared'] for v in groups.values()),sum(v[kind+'_squared_norm_change'] for v in groups.values())))
    phase.validate();phase.close()
    if not exact_state_equal(private.state_dict(),saved['private'][case['state_id']]):raise ValueError('private mutated')
    return dict(state_id=case['state_id'],source_sha256={paths[k]:case['sha256'][k] for k in ('state','checkpoint')},
        raw_action_norm=float(norm),guard=float(guard),groups=groups,summary=summary,parameters=per_parameter,
        private_unchanged=True,elapsed_seconds=time.perf_counter()-tick)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest')
    started=time.perf_counter();digest=sha(a.checkpoint);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    actor=GroupedCoordinateChisel(rank=8,trajectory_dim=400,model_dim=256,num_layers=2,latent_tokens=8,fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
    actor.load_state_dict(saved['actor'],strict=True);actor.eval().requires_grad_(False)
    rows=[]
    for case in saved['metadata']['cases']:
        row=measure(actor,saved,case);rows.append(row);gc.collect();print(json.dumps(dict(state=row['state_id'],summary=row['summary'])),flush=True)
    if not exact_state_equal(actor.state_dict(),saved['actor']) or sha(a.checkpoint)!=digest:raise ValueError('actor/checkpoint changed')
    # Pin only the complete live-log prefix through the matching measurement,
    # never claim the still-appending metrics file is an immutable input.
    metrics=a.checkpoint.parent/'metrics.jsonl';prefix=b'';reference=None
    for line in metrics.open('rb'):
        prefix+=line
        record=json.loads(line)
        if record.get('completed_updates_before_measure')==saved['completed_updates']:reference=record;break
    if reference is None:raise ValueError('matching saved-state measurement missing')
    max_error=0.
    for row in rows:
        prior=next(r for r in reference['rows'] if r['local_id']==row['state_id'])
        for g,v in row['groups'].items():
            error=abs(v['stored_shrink_percent']-prior['group_l2_percentage_reduction'][g]);max_error=max(max_error,error)
            if error>1e-12:raise ValueError('stored norm differs from actual proposal measurement')
    sources=[__file__,'rewriter/adaptive_preservation.py','rewriter/adaptive_action.py','rewriter/grouped_norm.py','scripts/frozen_private_frames.py','scripts/fit_cpu_edit_teacher.py']
    result=dict(scope=__doc__,checkpoint=str(a.checkpoint),checkpoint_sha256=digest,completed_updates=saved['completed_updates'],cases=rows,
        storage_convention='base FP32 -> base.double() + global_guard * ((U @ group_scaled_C) @ V.T), embedding transpose, one FP32 cast; streaming512 stored rows',
        raw_norm_identity='2*guard*<U.T W V,C> + guard^2*trace(C.T U.T U C V.T V); vector analog; dense expansion checked',
        observed_proposal_group_percent_max_abs_error=max_error,metrics_prefix_bytes=len(prefix),metrics_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
        zero_cast_denominator='only mathematical nonzero dense edits; readonly outputs excluded',
        elapsed_seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        gpt_forwards=0,gpt_backwards=0,optimizer_updates=0,source_sha256={f:sha(f) for f in sources})
    (a.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
