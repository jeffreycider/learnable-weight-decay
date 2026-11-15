"""Attribute an eager Kineto GEMM trace to Chisel stages via CPU ancestry.

This measures a separate eager replay, not the captured graph's exact partition.
Unattributed kernels remain explicit; no sum of durations is called wall time.
"""
from pathlib import Path
import argparse,collections,json


def analyze(path):
    events=json.loads(Path(path).read_text())['traceEvents']
    cpu=[e for e in events if e.get('ph')=='X' and e.get('cat') in ('cpu_op','user_annotation')]
    by_thread=collections.defaultdict(list);parent={};external={}
    for i,e in enumerate(cpu):
        by_thread[(e['pid'],e['tid'])].append(i)
        external_id=e.get('args',{}).get('External id')
        if external_id is not None:external[external_id]=i
    for ids in by_thread.values():
        stack=[]
        for i in sorted(ids,key=lambda i:(cpu[i]['ts'],-cpu[i].get('dur',0))):
            e=cpu[i]
            while stack and e['ts']>=cpu[stack[-1]]['ts']+cpu[stack[-1]].get('dur',0):stack.pop()
            parent[i]=stack[-1] if stack else None;stack.append(i)
    runtime={e.get('args',{}).get('correlation'):e.get('args',{}).get('External id')
             for e in events if e.get('cat')=='cuda_runtime' and 'correlation' in e.get('args',{})}
    groups=collections.defaultdict(lambda:dict(microseconds=0.,count=0,examples=[]));unmatched=[]
    kernel_total=sum(e.get('dur',0) for e in events if e.get('cat')=='kernel')
    for e in events:
        if e.get('cat')!='kernel' or 'gemm' not in e.get('name','').lower():continue
        args=e.get('args',{});index=external.get(args.get('External id',runtime.get(args.get('correlation'))))
        chain=[]
        while index is not None:
            chain.append(cpu[index]);index=parent.get(index)
        names=[x['name'] for x in chain];joined=' | '.join(names)
        if 'chisel_teacher_head' in names:category='teacher_head'
        elif 'chisel_actor_forward' in names:category='actor_forward'
        elif any('CompiledFunctionBackward' in n for n in names):category='actor_backward'
        elif any(n=='aten::bmm' for n in names):category='attention_bmm'
        elif 'chisel_student_forward' in names:category='student_forward_linears'
        elif 'chisel_gpt_backward' in names or any('AnchoredLinearBackward' in n or 'AnchoredEmbeddingBackward' in n for n in names):
            mm=next((x for x in chain if x['name']=='aten::mm'),None)
            dims=mm.get('args',{}).get('Input Dims',[]) if mm else []
            if len(dims)>=2 and len(dims[1])==2:
                category='thin_factor_contractions' if dims[1][1]<=8 else 'input_gradient_dx'
            else:category='student_backward_unclassified'
        elif 'chisel_actor_proposal_vjp' in names:category='proposal_backward'
        elif chain:category='proposal_forward_or_other'
        else:category='unattributed'
        row=groups[category];row['microseconds']+=e['dur'];row['count']+=1
        if len(row['examples'])<3:row['examples'].append(dict(kernel=e['name'],cpu_chain=names[:12]))
        if not chain and len(unmatched)<5:unmatched.append(e)
    total=sum(g['microseconds'] for g in groups.values())
    for row in groups.values():
        row['milliseconds']=row.pop('microseconds')/1000
        row['fraction_of_gemm']=row['milliseconds']/(total/1000) if total else 0.
        row['fraction_of_all_kernel_time']=row['milliseconds']/(kernel_total/1000) if kernel_total else 0.
    return dict(scope='Separate profiled eager replay; GPU duration sums, not wall time or exact graph partition',
        source=str(path),gemm_ms=total/1000,all_kernel_ms=kernel_total/1000,groups=dict(groups),
        unmatched_examples=unmatched)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=analyze(args.trace);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{n:v for n,v in g.items() if n!='examples'} for k,g in result['groups'].items()},indent=2))
