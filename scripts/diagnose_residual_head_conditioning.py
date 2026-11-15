"""Known-target frozen-feature conditioning ablation, not norm/KL learning.

Fresh shared seed940067, zero vs scaled-random residual projections. Reuses the
SAME previously target-fitted private frames AND trajectory_input on states6/7.
No GPT evaluation, optimizer updates, fitted action application or preservation
claim. The older failed teacher actor had nonzero residual outputs already; this
fresh comparison cannot attribute its failure to zero residual initialization.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.prefix_cache import unpack_record
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.adaptive_preservation import group_rms_scales
from scripts.prepare_random_grouped_cpu_seeds import fresh_actor,residual_output_spec,assert_zero_heads
from scripts.fit_cpu_teacher_heads_lstsq import matrix_design,vector_design,solve,state_equal
from scripts.fit_cpu_grouped_heads_gradient import merge_designs
from scripts.fit_cpu_edit_teacher import sha
from scripts.frozen_private_frames import FrozenPrivateFrames


@torch.no_grad()
def extract(path,actors):
    dataset=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    checkpoint=dataset['checkpoint']
    if sha(checkpoint)!=dataset['source_sha256'][checkpoint]:raise ValueError('fitted private parent changed')
    parent=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);meta=parent['metadata'];source=meta['args']
    for key in ('state','selection'):
        if sha(source[key])!=meta[key+'_sha256'] or meta[key+'_sha256']!=dataset['source_sha256'][source[key]]:raise ValueError('target/state provenance changed')
    state=torch.load(source['state'],map_location='cpu',weights_only=False,mmap=True)
    snapshot=unpack_record(state['observation'],device='cpu').snapshot
    names=set(snapshot.matrices)|set(snapshot.vectors)
    parameters={n:v.detach().float() for n,v in state['model'].items() if n in names}
    grouped=GroupedNormObjective(parameters,zero_reference_policy='freeze');scales=group_rms_scales(grouped,dtype=torch.float64)
    actor=actors['zero'];private=actor.make_private(snapshot,state_key=('cpu-teaching',0,state['completed_steps'],meta['state_sha256']),seed=source['private_seed'])
    private.load_state_dict(parent['private'],strict=True);private.requires_grad_(False)
    phase=FrozenPrivateFrames(private);cores=phase.query_weight_cores(parameters)
    outputs={};r=actor.latent_tokens
    # Group order and parameter order match the existing exporter exactly.
    for mode,actor in actors.items():
        captured={g:[] for g in grouped.groups};hooks=[]
        for g in captured:
            hooks.append(actor.output_head(g).register_forward_pre_hook(lambda module,inputs,g=g:captured[g].append(inputs[0].detach().clone())))
        raw=phase.evaluate(actor,snapshot,weight_cores=cores)
        for hook in hooks:hook.remove()
        assert all(torch.count_nonzero(f.C)==0 for f in raw.matrices.values())
        assert all(torch.count_nonzero(v)==0 for v in raw.vectors.values())
        groups={}
        for group,names_in_group in grouped.groups.items():
            matrices=[]
            if len(captured[group])!=len(names_in_group):raise ValueError('head capture inventory differs')
            for name,h in zip(names_in_group,captured[group]):
                if name in snapshot.matrices:
                    rows,cols=snapshot.matrices[name].rows.shape[0],snapshot.matrices[name].columns.shape[0]
                    h=h.reshape(2,r,-1);scale=scales[name]*math.sqrt(rows*cols/(r*r))/math.sqrt(2)
                    design=matrix_design(h[0],h[1],scale)
                    coeff=torch.linspace(-.001,.001,design.shape[1],dtype=design.dtype).reshape(r,-1)
                    output=torch.nn.functional.linear(h,coeff[:,:-1],coeff[:,-1])
                    expected=((output[0]+output[1].T)*scale).flatten()
                else:
                    h=h.reshape(r,-1);scale=scales[name]*math.sqrt(snapshot.vectors[name].weight.numel()/r)
                    design=vector_design(h,scale)
                    coeff=torch.linspace(-.001,.001,design.shape[1],dtype=design.dtype)
                    expected=torch.nn.functional.linear(h,coeff[:-1][None],coeff[-1:]).flatten()*scale
                torch.testing.assert_close(design@coeff.flatten(),expected,rtol=1e-10,atol=1e-11)
                matrices.append(design)
            A=torch.cat(matrices);old=dataset['groups'][group]
            if A.shape!=old['A'].shape or old['b'].shape!=(len(A),):raise ValueError('export shape mismatch')
            groups[group]=dict(A=A,b=old['b'].clone(),weight=old['weight'])
        outputs[mode]=dict(groups=groups)
    if not state_equal(private.state_dict(),parent['private']):raise ValueError('fitted private module changed')
    phase.validate();phase.close()
    provenance=dict(design=str(path),design_sha256=sha(path),private_checkpoint=checkpoint,private_checkpoint_sha256=sha(checkpoint),
        state=source['state'],state_sha256=meta['state_sha256'],selection=source['selection'],selection_sha256=meta['selection_sha256'],
        private_identity=private.get_extra_state(),private_reuse='exact target-fitted frames AND trajectory_input; diagnostic only',private_unchanged=True)
    return outputs,provenance


def spectrum_report(values,rank):
    s=torch.tensor(values,dtype=torch.float64);relative=s/s[0]
    probabilities=s/s.sum();positive=probabilities[probabilities>0]
    return dict(count=len(s),largest=float(s[0]),smallest=float(s[-1]),retained_smallest=float(s[rank-1]) if rank else None,
        relative_quantiles={str(q):float(torch.quantile(relative,q)) for q in (0.,.01,.1,.5,.9,.99,1.)},
        effective_ranks={str(t):int((relative>t).sum()) for t in (1e-4,1e-6,1e-8,1e-10,1e-12,1e-14)},
        stable_rank=float(s.square().sum()/s[0].square()),entropy_rank=float(torch.exp(-(positive*positive.log()).sum())))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--designs',type=Path,nargs=2,default=[Path('runs/grouped-head-diagnostic/state6-export-only-v1/designs.pt'),Path('runs/grouped-head-diagnostic/state7-export-only-v1/designs.pt')])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');started=time.perf_counter();rng=torch.get_rng_state().clone()
    actors={mode:fresh_actor(940067,residual_output_init=mode,residual_seed=960067)[0].eval().requires_grad_(False) for mode in ('zero','scaled_random')}
    allowed=set(residual_output_spec(actors['zero'],'scaled_random',960067)['changed_parameters'])
    z=actors['zero'].state_dict();s=actors['scaled_random'].state_dict()
    if any(not state_equal(v,s[n]) for n,v in z.items() if n not in allowed):raise ValueError('non-residual shared weights differ')
    for actor in actors.values():assert_zero_heads(actor)
    datasets={m:[] for m in actors};provenance=[]
    for path in a.designs:
        data,record=extract(path,actors);provenance.append(record)
        for mode in actors:datasets[mode].append(data[mode])
        gc.collect();print(json.dumps(dict(extracted=str(path),seconds=time.perf_counter()-started)),flush=True)
    results={}
    for mode in actors:
        groups=merge_designs(datasets.pop(mode));reports={}
        for name,g in groups.items():
            coef,report=solve(g['A'],g['b']);singular=report.pop('singular_values')
            outputs=8 if name.endswith('.weight') else 1;joined=coef.reshape(outputs,-1)
            reports[name]=dict(report,spectrum=spectrum_report(singular,report['rank']),head_weight_frobenius=float(joined[:,:-1].norm()),
                head_bias_l2=float(joined[:,-1].norm()),coefficient_l2=float(coef.norm()),
                relative_core_error=report['residual_squared']/report['target_squared'] if report['target_squared'] else None)
            print(json.dumps(dict(mode=mode,group=name,rank=report['rank'],head_norm=float(coef.norm()),seconds=time.perf_counter()-started)),flush=True)
        results[mode]=dict(groups=reports,projected_balanced_objective=sum(r['residual_squared'] for r in reports.values()),
            all_heads_coefficient_l2=math.sqrt(sum(r['coefficient_l2']**2 for r in reports.values())))
        del groups;gc.collect()
    if not torch.equal(rng,torch.get_rng_state()):raise ValueError('caller RNG changed')
    result=dict(scope=__doc__,shared_seed=940067,residual_seed=960067,states=provenance,arms=results,
        objective='Exact existing equal-case balanced physical SSE; target-derived fixed weights identical across arms. Fixed private projection floor omitted.',
        spectrum='SVD of actual weighted physical-core designs for one shared functional head across both states; default gelsd eps*max(shape) rank cutoff, no arbitrary truncation.',
        only_changed_shared_parameters=sorted(allowed),same_fitted_private_modules=True,zero_final_heads=True,nonzero_readout_design_parity=True,
        old_teacher_caveat='Older failed teacher actor already had nonzero residual projection norms (parent measured attention .755/.852, MLP1.689/2.105). This does not explain that failure by zero initialization.',
        gpt_forwards=0,gpt_backwards=0,optimizer_updates=0,fitted_actions_applied=0,caller_rng_unchanged=True,
        elapsed_seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        source_sha256={f:sha(f) for f in (__file__,'scripts/prepare_random_grouped_cpu_seeds.py','scripts/fit_cpu_teacher_heads_lstsq.py','scripts/fit_cpu_grouped_heads_gradient.py','scripts/frozen_private_frames.py')})
    (a.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
