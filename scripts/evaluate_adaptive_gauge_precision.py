"""Predeclared TRAIN-only factor-rounding diagnostic; no fitting or query."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.model import GPT
from rewriter.adaptive_gauge_control import orthonormal_product
from rewriter.function_preservation import FunctionalPreservation
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from rewriter.tokenwise_reference import TokenwiseConfig,TOKEN_SUM_FIELDS,logit_terms_and_cotangent


def reconstruct_cpu(model,rows):
    edits={label:{} for label in ('direct_binary','adaptive_fp32','adaptive_fp64_cast')}
    used=set()
    for row in rows:
        if row['rank_limit']!=8:raise ValueError('fixed rank8 selection required')
        blocks={}
        for c in row['selected']:blocks.setdefault(c['first_weight'],[]).append(c)
        for first_name,choices in blocks.items():
            if len(choices)>8:raise ValueError('channel rank cap exceeded')
            second_name=choices[0]['second_weight'];bias_name=choices[0]['first_bias']
            if used&{first_name,second_name,bias_name}:raise ValueError('overlapping selected operators')
            used.update((first_name,second_name,bias_name))
            index=torch.tensor([c['channel'] for c in choices]);d=2.**torch.tensor([c['exponent'] for c in choices],dtype=torch.float64)
            first,second,bias=(model[n].float() for n in (first_name,second_name,bias_name))
            e=torch.zeros(first.shape[0],len(index),dtype=torch.float64);e[index,torch.arange(len(index))]=1
            target=first.clone();target[index]=(first[index].double()*d[:,None]).float();edits['direct_binary'][first_name]=target
            target=second.clone();target[:,index]=(second[:,index].double()*d.pow(-row['inverse_power'])[None,:]).float();edits['direct_binary'][second_name]=target
            target=bias.clone();target[index]=(bias[index].double()*d).float()
            for case in edits.values():case[bias_name]=target
            for name,base,x,y in [(first_name,first,e*(d-1),first[index].double().T),
                                 (second_name,second,second[:,index].double()*(d.pow(-row['inverse_power'])-1),e)]:
                for label,dtype in [('adaptive_fp32',torch.float32),('adaptive_fp64_cast',torch.float64)]:
                    f=orthonormal_product(x.to(dtype),y.to(dtype),rank=8)
                    edits[label][name]=(base.to(dtype)+f.dense()).float()
    return edits


def comparison(first,second,chunk=1024):
    first=first.reshape(-1,first.shape[-1]);second=second.reshape_as(first)
    count=0;maximum=0.
    for start in range(0,len(first),chunk):
        a,b=first[start:start+chunk],second[start:start+chunk]
        count+=int((a!=b).sum());maximum=max(maximum,float((a-b).abs().max()))
    return dict(equal=count==0,different_elements=count,total_elements=first.numel(),max_abs_difference=maximum)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('state-bank','teacher-cache','selection','representation','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--selection-sha256',required=True);p.add_argument('--representation-sha256',required=True)
    a=p.parse_args();sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(a.selection)!=a.selection_sha256 or sha(a.representation)!=a.representation_sha256:raise ValueError('pinned diagnostic inputs changed')
    a.output.mkdir(exist_ok=False);torch.set_num_threads(1);started=time.perf_counter()
    selected=json.loads(a.selection.read_text());old=json.loads(a.representation.read_text())
    reader=StateBankReader(a.state_bank,verify_once=True,mmap_model=True);cache=TeacherCacheReader(a.teacher_cache,verify_once=True,mmap=True)
    cfg=TokenwiseConfig(mean_tolerance=0.,tail_tolerance=0.,scale=1.,rho=1.,ranking_k=100,
        ranking_tie_policy='exact_teacher_ties',loss_backend='tiled_tie_fp64_v1')
    results=[]
    for sid in (6,499):
        rows=[r for r in selected['rows'] if r['state_id']==sid]
        record=reader.read(sid,device='cpu',model_device='cpu')
        if len(rows)!=2 or any(r['state_sha256']!=record.assert_verified() for r in rows):raise ValueError('selected state identity differs')
        model=GPT(**reader.manifest['provenance']['model_config']).cuda().requires_grad_(False)
        engine=FunctionalPreservation(model,microbatch_sequences=16,teacher_cache_device='cuda',kl_dtype=torch.float64,gpt_precision='fp32')
        cache.load_engine(engine,record);panel=cache.read(sid,'train')
        ids=list(panel.sequence_ids[:16]);inputs=panel.inputs[:16].to(engine.device)
        (a.output/f'state{sid}-panel.json').write_text(json.dumps(dict(role='train',sequence_ids=ids,state_sha256=record.assert_verified()),indent=2)+'\n')
        edits=reconstruct_cpu(record.model,rows)
        previous=next(r for r in old['results'] if r['state_id']==sid)
        mismatches={label:sum(int((value!=edits['direct_binary'][n]).sum()) for n,value in edit.items()) for label,edit in edits.items()}
        mismatches['zero']=sum(int((value!=record.model[n].float()).sum()) for n,value in edits['direct_binary'].items())
        if mismatches['adaptive_fp64_cast']!=0 or mismatches['adaptive_fp32']!=previous['aggregate']['fp32']['stored_mismatches']:
            raise ValueError('CPU factor-rounding control differs from preserved measurement')
        teacher=panel.teacher_batch(engine,0,16)
        if teacher.log_probs.dtype!=torch.float64 or not torch.equal(teacher.inputs.to(inputs.device),inputs):raise ValueError('teacher arithmetic/panel mismatch')
        grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze');arms={};zero=binary=None
        for label in ('zero','direct_binary','adaptive_fp32','adaptive_fp64_cast'):
            params=engine.parameters if label=='zero' else {**engine.parameters,**{n:v.to(engine.device) for n,v in edits[label].items()}}
            logits=engine._forward(params,inputs)
            engine.compute.student_forwards+=1;engine.compute.student_tokens+=inputs.numel()
            if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite diagnostic logits')
            if label=='zero':zero=logits
            if label=='direct_binary':binary=logits
            _,parts,kl=logit_terms_and_cotangent(logits,teacher.log_probs,config=cfg,multiplier=0.,panel_tokens=inputs.numel(),chunk_tokens=engine.chunk_tokens,backward=False)
            parts=parts.cpu().tolist()
            arms[label]=dict(mean_kl=float(kl.mean()),max_token_kl=float(kl.max()),min_token_kl=float(kl.min()),
                literal_argmax_changes=int(parts[3]),literal_top100_order_changes=int(parts[4]),literal_top100_membership_changes=int(parts[5]),
                **{name:int(parts[i]) for i,name in enumerate(TOKEN_SUM_FIELDS[13:],13)},
                vs_zero_logits=comparison(zero,logits,engine.chunk_tokens),norm=grouped.evaluate(params).metrics(),
                stored_mismatches_vs_binary=mismatches[label])
            if label.startswith('adaptive'):arms[label]['vs_binary_logits']=comparison(binary,logits,engine.chunk_tokens)
            print(json.dumps(dict(state_id=sid,arm=label,metrics=arms[label])),flush=True)
            if label not in ('zero','direct_binary'):del logits
        results.append(dict(state_id=sid,sequence_ids=ids,role='train',tokens=inputs.numel(),arms=arms,
            compute={**engine.compute.as_dict(),'cached_teacher_head_forwards':getattr(engine,'cached_teacher_head_forwards',0)}))
        del engine,model,params,teacher,zero,binary,grouped,record,edits,panel
        torch.cuda.empty_cache()
    output=dict(results=results,elapsed_seconds=time.perf_counter()-started,selection_sha256=a.selection_sha256,
        representation_sha256=a.representation_sha256,source_sha256=sha(Path(__file__)),gpt_forwards=8,gpt_backwards=0,gpt_tokens=131072,
        cached_teacher_head_forwards=2,query_access=False,fitting_updates=0,
        interpretation='Analytic capped-gauge finite TRAIN16 diagnostic. CPU factor construction, not CUDA QR qualification. No radius interpolation, learned Chisel claim, query evaluation, or all-input proof.')
    (a.output/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    import wandb
    run=wandb.init(project='chisel',mode='offline',dir=str(a.output),name='adaptive-capped-gauge-precision-train16-v1',config=dict(selection_sha256=a.selection_sha256,training=False))
    for r in results:
        for label,metrics in r['arms'].items():run.log({f'{r["state_id"]}/{label}/{k}':v for k,v in metrics.items() if isinstance(v,(int,float))})
    (a.output/'wandb_path.txt').write_text(str(Path(run.dir).parent)+'\n');run.finish()


if __name__=='__main__':main()
