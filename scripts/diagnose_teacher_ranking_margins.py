"""Teacher ranking margins and head-only arithmetic sensitivity; no Chisel edits.

Cached features are identical in both head evaluations. The FP64 head is a
numerical control, not a full FP64 transformer or an explanation of edit errors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F


@torch.no_grad()
def main():
    from baseline.execution import enable_deterministic_execution
    from baseline.model import GPT
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import TeacherCacheReader,head_logits
    from rewriter.offline_systems import CachedBaseNormPreservation
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-bank',type=Path,required=True)
    parser.add_argument('--teacher-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--states',type=int,nargs='+',default=[1,999])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    enable_deterministic_execution();torch.set_num_threads(1)
    reader=StateBankReader(args.state_bank,verify_once=True,mmap_model=True)
    cache=TeacherCacheReader(args.teacher_cache,verify_once=True,mmap=True)
    engine=CachedBaseNormPreservation(GPT(**reader.manifest['provenance']['model_config']).cuda().requires_grad_(False),
        microbatch_sequences=16,teacher_cache_device='cuda',kl_dtype=torch.float64,gpt_precision='fp32')
    rows=[]
    for state in args.states:
        record=reader.read(state);engine.load_verified_teacher(cache,record);panel=cache.read(state,'train')
        state_sha=record.assert_verified()
        assert panel.metadata['state_sha256']==state_sha
        margins=[];changes=torch.zeros(3,dtype=torch.int64,device='cuda');max_shift=0.
        weight=engine.parameters['proj.weight'].double();bias=engine.parameters['proj.bias'].double()
        for start in range(0,len(panel.inputs),16):
            features=panel.features[start:start+16].cuda()
            fp32=head_logits(engine,features).flatten(0,1)
            raw=F.linear(features.double(),weight,bias)
            fp64=(15*raw*(raw.square()+225).rsqrt()).flatten(0,1)
            values,indices=fp32.topk(11,dim=-1,sorted=True)
            double_indices=fp64.topk(10,dim=-1,sorted=True).indices
            gaps=values[:,:-1]-values[:,1:]
            margins.append(torch.stack((gaps[:,0],gaps[:,:9].amin(-1),gaps[:,9]),dim=-1).cpu())
            # Independent argmax keeps the operation's own exact-tie convention.
            changes+=torch.stack(((fp32.argmax(-1)!=fp64.argmax(-1)).sum(),
                (indices[:,:10]!=double_indices).any(-1).sum(),
                (indices[:,:10].sort(-1).values!=double_indices.sort(-1).values).any(-1).sum()))
            max_shift=max(max_shift,float((fp32.double()-fp64).abs().max()))
            del features,fp32,fp64,raw,values,indices,double_indices,gaps
        all_margins=torch.cat(margins).double()
        names=('top1_gap','smallest_adjacent_top10_gap','top10_membership_gap')
        row=dict(state=state,state_sha256=state_sha,sequence_ids=panel.sequence_ids,
            tokens=len(all_margins),margins={name:dict(exact_zero_count=int((all_margins[:,i]==0).sum()),
                quantiles=dict(zip(('min','p1','p10','median','p90','max'),
                torch.quantile(all_margins[:,i],torch.tensor([0.,.01,.1,.5,.9,1.],dtype=torch.float64)).tolist()))) for i,name in enumerate(names)},
            head_only_fp32_vs_fp64=dict(zip(('argmax_changes','top10_order_changes','top10_membership_changes'),changes.cpu().tolist())),
            maximum_head_logit_shift=max_shift,teacher_head_evaluations_fp32=len(panel.inputs)//16,
            teacher_head_evaluations_fp64=len(panel.inputs)//16,gpt_transformer_forwards=0,chisel_forwards=0)
        rows.append(row);print(json.dumps(row),flush=True)
    result=dict(scope=__doc__,rows=rows,driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import wandb
    run=wandb.init(project='chisel',mode='offline',dir=str(args.output),config={'scope':__doc__})
    for row in rows:run.log({'state':row['state'],'head_rounding_argmax_changes':row['head_only_fp32_vs_fp64']['argmax_changes']})
    run.finish()


if __name__=='__main__':main()
