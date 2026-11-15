"""Fixed many-channel binary gauges: paired finite logits, no search or training."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.model import GPT
from rewriter.function_preservation import FunctionalPreservation
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from rewriter.tokenwise_reference import TokenwiseConfig,TOKEN_SUM_FIELDS,logit_terms_and_cotangent


def reconstruct(parameters,rows):
    replacements=dict(parameters);seen=set()
    for row in rows:
        power={'attention':1,'mlp':2}[row['family']]
        if row['inverse_power']!=power:raise ValueError('gauge power differs')
        if len(row['channels'])!=len(row['channel_exponents']):raise ValueError('selection length differs')
        blocks={}
        for info,k in zip(row['channels'],row['channel_exponents']):
            if type(k) is not int or k not in (-1,0,1):raise ValueError('only selected binary scales allowed')
            names=(info['first_weight'],info['first_bias'],info['second_weight'])
            if names not in blocks:blocks[names]={}
            channel=info['channel']
            if channel in blocks[names]:raise ValueError('duplicate gauge channel')
            blocks[names][channel]=k
        for (first,bias,second),choices in blocks.items():
            names={first,bias,second}
            if names&seen:raise ValueError('overlapping gauge blocks')
            seen.update(names);n=parameters[first].shape[0]
            if set(choices)!=set(range(n)):raise ValueError('incomplete gauge channel specification')
            d=parameters[first].new_tensor([2.**choices[i] for i in range(n)])
            replacements[first]=parameters[first]*d[:,None]
            replacements[bias]=parameters[bias]*d
            replacements[second]=parameters[second]/d[None,:].pow(power)
    return replacements


@torch.no_grad()
def paired_panel(engine,replacements,panel,count,*,backend='tiled_tie_fp64_v1'):
    if count<1 or count>len(panel.inputs) or count%engine.microbatch_sequences:raise ValueError('complete cached teacher microbatches required')
    config=TokenwiseConfig(mean_tolerance=0.,tail_tolerance=0.,scale=1.,rho=1.,ranking_k=100,
                          ranking_tie_policy='exact_teacher_ties',loss_backend=backend)
    sums={arm:torch.zeros(len(TOKEN_SUM_FIELDS),device=engine.device,dtype=torch.float64) for arm in ('zero','gauge')}
    raw={arm:[] for arm in sums};different=0;max_abs=0.;logits_count=0;start_time=time.perf_counter()
    engine._validate_replacements(replacements)
    for start in range(0,count,engine.microbatch_sequences):
        stop=start+engine.microbatch_sequences;inputs=panel.inputs[start:stop].to(engine.device)
        teacher=panel.teacher_batch(engine,start,stop)
        if teacher.teacher_id!=engine.teacher_id or teacher.log_probs.dtype!=torch.float64 or not torch.equal(inputs,teacher.inputs.to(engine.device)):
            raise ValueError('teacher precision/identity/inputs differ')
        zero=engine._forward(engine.parameters,inputs);candidate=engine._forward(replacements,inputs)
        engine.compute.student_forwards+=2;engine.compute.student_tokens+=2*inputs.numel()
        if not torch.isfinite(zero).all() or not torch.isfinite(candidate).all():raise FloatingPointError('nonfinite finite logits')
        z=zero.reshape(-1,zero.shape[-1]);c=candidate.reshape_as(z)
        # Bounded temporary: compare the actual post-softcap logits directly.
        for begin in range(0,len(z),engine.chunk_tokens):
            dz=z[begin:begin+engine.chunk_tokens];dc=c[begin:begin+engine.chunk_tokens]
            different+=int(torch.count_nonzero(dz!=dc));max_abs=max(max_abs,float((dz-dc).abs().max()))
        logits_count+=zero.numel()
        for arm,logits in [('zero',zero),('gauge',candidate)]:
            _,parts,kl=logit_terms_and_cotangent(logits,teacher.log_probs,config=config,multiplier=0.,
                panel_tokens=count*inputs.shape[1],chunk_tokens=engine.chunk_tokens,backward=False)
            sums[arm]+=parts;raw[arm].append(kl)
        del teacher,zero,candidate,z,c
    results={}
    for arm in sums:
        values=torch.cat(raw[arm]);parts=sums[arm].cpu().tolist()
        if not torch.isfinite(values).all():raise FloatingPointError('nonfinite KL')
        results[arm]=dict(tokens=len(values),mean_kl=float(values.mean()),max_token_kl=float(values.max()),min_token_kl=float(values.min()),
            literal_argmax_changes=int(parts[3]),literal_top100_order_changes=int(parts[4]),literal_top100_membership_changes=int(parts[5]),
            **{name:int(parts[i]) for i,name in enumerate(TOKEN_SUM_FIELDS[13:],13)})
    return dict(sequence_ids=list(panel.sequence_ids[:count]),readouts=results,
        direct_logit_comparison=dict(equal=different==0,different_elements=different,elements=logits_count,max_abs_difference=max_abs,
                                     scope='actual FP32 post-softcap GPT logits; same ordinary eager execution'),
        elapsed_seconds=time.perf_counter()-start_time)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state-bank',type=Path,required=True);p.add_argument('--teacher-cache',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--selection-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    if sha(args.selection)!=args.selection_sha256:raise ValueError('preselected weight-only action file changed')
    selected=json.loads(args.selection.read_text());assert selected['extra_gpt_forwards']==selected['extra_gpt_backwards']==0
    assert selected['output_or_audit_access'] is False
    args.output.mkdir(exist_ok=False);torch.set_num_threads(1);start=time.perf_counter()
    reader=StateBankReader(args.state_bank,verify_once=True,mmap_model=True);cache=TeacherCacheReader(args.teacher_cache,verify_once=True,mmap=True)
    results=[];compute={}
    for sid in (6,499):
        rows=[r for r in selected['rows'] if r['state_id']==sid]
        if sorted(r['family'] for r in rows)!=['attention','mlp']:raise ValueError('exactly two preselected families required')
        record=reader.read(sid,device='cpu',model_device='cpu')
        if any(r['state_sha256']!=reader.entries[sid-1]['sha256'] for r in rows):raise ValueError('selected gauge belongs to another checkpoint')
        model=GPT(**reader.manifest['provenance']['model_config']).cuda().requires_grad_(False)
        engine=FunctionalPreservation(model,microbatch_sequences=16,teacher_cache_device='cuda',kl_dtype=torch.float64,gpt_precision='fp32')
        cache.load_engine(engine,record);parameters=reconstruct(engine.parameters,rows)
        grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze');norm=grouped.evaluate(parameters)
        metrics=norm.metrics();predicted=sum(r['best']['mean_group_l2_percentage_reduction'] for r in rows)
        if abs(metrics['mean_group_l2_percentage_reduction']-predicted)>1e-8:raise ValueError('realized gauge norm differs from weight-only selection')
        reference_sq=sum(float(x) for x in grouped.reference_squared.values());edit_sq=float(norm.edit_norm_sq)
        result=dict(state_id=sid,state_sha256=reader.entries[sid-1]['sha256'],norm=metrics,
            edit_l2_norm=math.sqrt(edit_sq),reference_editable_l2_norm=math.sqrt(reference_sq),edit_relative_l2=math.sqrt(edit_sq/reference_sq),
            guard='none; full selected power-of-two gauge, no additive interpolation',panels=[])
        for role,count in [('train',128),('audit',64)]:
            panel=cache.read(sid,role);measured=paired_panel(engine,parameters,panel,count)
            result['panels'].append(dict(role=role,**measured))
            print(json.dumps(dict(state_id=sid,role=role,**measured)),flush=True)
        result['compute']={**engine.compute.as_dict(),'cached_teacher_head_forwards':getattr(engine,'cached_teacher_head_forwards',0)}
        for key,value in result['compute'].items():compute[key]=compute.get(key,0)+value
        results.append(result);del record,model,engine,parameters,grouped,norm,panel
        torch.cuda.empty_cache()
    summary=dict(results=results,compute=compute,elapsed_seconds=time.perf_counter()-start,selection_sha256=args.selection_sha256,
        source_sha256={str(f):sha(f) for f in [Path(__file__),Path('rewriter/tokenwise_reference.py'),Path('rewriter/function_preservation.py'),Path('rewriter/gpt_precision.py')]},
        interpretation='Fixed analytic feasibility control outside the anchored/trust-limited learned family. No learning, output-based selection, audit retries or all-input finite-precision guarantee.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import wandb
    w=wandb.init(project='chisel',mode='offline',dir=str(args.output),name=args.output.name,config={'selection_sha256':args.selection_sha256,'training':False})
    for r in results:
        for row in r['panels']:
            w.log({f"state{r['state_id']}/{row['role']}/"+k:v for k,v in row['direct_logit_comparison'].items() if isinstance(v,(int,float,bool))})
    (args.output/'wandb_path.txt').write_text(str(Path(w.dir).parent)+'\n');w.finish()

if __name__=='__main__':main()
