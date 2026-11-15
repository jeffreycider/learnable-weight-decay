"""Read-only comparison of two explicitly different cached GPT functions.

No transformer, Chisel, backward, optimizer update, or policy selection occurs.
Both caches are hash-verified; checkpoint, inputs, sequence IDs and head weights
must match. Their different declared normalization epsilons are the comparison.
"""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader,head_logits,tensor_hash


def accumulate(old,new,inputs,chunk=128):
    length=inputs.shape[1];o=old.flatten(0,1);n=new.flatten(0,1)
    targets=inputs.roll(-1,1).flatten().to(o.device)
    total=dict(old_to_corrected_kl_sum=0.,corrected_to_old_kl_sum=0.,old_ce_sum=0.,corrected_ce_sum=0.,centered_logit_squared_sum=0.,logprob_squared_sum=0.,ce_tokens=0,output_tokens=len(o),output_coordinates=o.numel())
    for start in range(0,len(o),chunk):
        stop=min(start+chunk,len(o));a=F.log_softmax(o[start:stop].double(),-1);b=F.log_softmax(n[start:stop].double(),-1)
        total['old_to_corrected_kl_sum']+=float((a.exp()*(a-b)).sum())
        total['corrected_to_old_kl_sum']+=float((b.exp()*(b-a)).sum())
        diff=n[start:stop].double()-o[start:stop].double();diff-=diff.mean(-1,keepdim=True)
        total['centered_logit_squared_sum']+=float(diff.square().sum());total['logprob_squared_sum']+=float((b-a).square().sum())
        valid=torch.arange(start,stop,device=o.device)%length!=length-1
        labels=targets[start:stop,None]
        total['old_ce_sum']-=float(a.gather(1,labels).squeeze(1)[valid].sum())
        total['corrected_ce_sum']-=float(b.gather(1,labels).squeeze(1)[valid].sum());total['ce_tokens']+=int(valid.sum())
    return total


def main():
    p=argparse.ArgumentParser();p.add_argument('--state-bank',type=Path,required=True);p.add_argument('--old-cache',type=Path,required=True);p.add_argument('--corrected-cache',type=Path,required=True);p.add_argument('--comparison-kind',choices=['normalization','baseline-arithmetic'],default='normalization');p.add_argument('--state-id',type=int,default=1000);p.add_argument('--role',choices=['train','audit'],default='audit');p.add_argument('--sequences',type=int,default=64);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True);assert not (args.output/'summary.json').exists()
    old=TeacherCacheReader(args.old_cache,verify_once=True,mmap=True).read(args.state_id,args.role)
    new=TeacherCacheReader(args.corrected_cache,verify_once=True,mmap=True).read(args.state_id,args.role)
    assert new.metadata['precision']['rms_epsilon']==2**-23
    if args.comparison_kind=='normalization':
        assert old.metadata['gpt_precision']=='fp32' and old.metadata['precision']['rms_epsilon']==2**-7
    else:
        assert old.metadata.get('gpt_precision','baseline')=='baseline'
        assert old.features.dtype==torch.bfloat16 and new.features.dtype==torch.float32
    for key in ['state_sha256','input_sha256','sequence_ids','microbatch_sequences','head_weight_sha256','head_bias_sha256']:
        assert old.metadata[key]==new.metadata[key],key
    assert torch.equal(old.inputs,new.inputs)
    batch=old.metadata['microbatch_sequences'];assert args.sequences<=len(old.inputs) and args.sequences%batch==0
    record=StateBankReader(args.state_bank,verify_once=True,mmap_model=True).read(args.state_id,device='cpu',model_device='cpu');assert record.assert_verified()==old.metadata['state_sha256']
    weights={n:record.model[n].float() for n in ['proj.weight','proj.bias']}
    for name,key in [('proj.weight','head_weight_sha256'),('proj.bias','head_bias_sha256')]:assert tensor_hash(weights[name])==old.metadata[key]
    engine=SimpleNamespace(device=torch.device('cuda:0'),parameters={n:v.to('cuda:0') for n,v in weights.items()})
    totals={}
    with torch.no_grad():
        for start in range(0,args.sequences,batch):
            stop=start+batch;a=head_logits(engine,old.features[start:stop]);b=head_logits(engine,new.features[start:stop])
            row=accumulate(a,b,old.inputs[start:stop])
            for key,value in row.items():totals[key]=totals.get(key,0)+value
            del a,b
    result=dict(state_id=args.state_id,role=args.role,sequence_ids=list(old.sequence_ids[:args.sequences]),sequences=args.sequences,sequence_length=old.inputs.shape[1],comparison_kind=args.comparison_kind,old_epsilon=2**-7 if args.comparison_kind=='normalization' else 2**-23,corrected_epsilon=2**-23,old_feature_dtype=str(old.features.dtype),corrected_feature_dtype=str(new.features.dtype),arithmetic_scope='Normalization-only FP32 teacher change' if args.comparison_kind=='normalization' else 'Original mixed-precision Flash teacher versus corrected FP32/math teacher; includes ordinary weight casts, activation arithmetic and attention backend',
        old_to_corrected_kl=totals['old_to_corrected_kl_sum']/totals['output_tokens'],corrected_to_old_kl=totals['corrected_to_old_kl_sum']/totals['output_tokens'],
        old_ce=totals['old_ce_sum']/totals['ce_tokens'],corrected_ce=totals['corrected_ce_sum']/totals['ce_tokens'],
        centered_logit_rms=(totals['centered_logit_squared_sum']/totals['output_coordinates'])**.5,logprob_rms=(totals['logprob_squared_sum']/totals['output_coordinates'])**.5,
        ce_scope='known next tokens inside each cached sequence; final position excluded',transformer_forwards=0,head_forwards=2*args.sequences//batch,backwards=0,optimizer_updates=0,policy_selection=False,
        old_cache_manifest_sha256=hashlib.sha256((args.old_cache/'manifest.json').read_bytes()).hexdigest(),corrected_cache_manifest_sha256=hashlib.sha256((args.corrected_cache/'manifest.json').read_bytes()).hexdigest(),checkpoint_sha256=record.assert_verified(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch_version=torch.__version__,tf32=torch.backends.cuda.matmul.allow_tf32)
    result['corrected_minus_old_ce']=result['corrected_ce']-result['old_ce']
    import wandb
    wb=wandb.init(project='chisel',mode='offline',dir=str(args.output),name=args.comparison_kind+'-teacher-comparison',config=vars(args));wb.log({k:v for k,v in result.items() if isinstance(v,(float,int))});wb.finish()
    (args.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
