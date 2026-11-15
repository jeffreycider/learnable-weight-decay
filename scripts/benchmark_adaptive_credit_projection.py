"""CPU projection-only comparison with a pinned pre-optimization implementation."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.adaptive_action import AdaptiveAction,AdaptiveFactors
from rewriter.adaptive_credit import project_adaptive_gradients

REFERENCE = '9cd2dce'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    source=subprocess.check_output(['git','show',REFERENCE+':rewriter/adaptive_credit.py'],text=True)
    module=types.ModuleType('rewriter._credit_before');module.__package__='rewriter'
    sys.modules[module.__name__]=module;exec(compile(source,'pinned_credit_before.py','exec'),module.__dict__)
    functions=dict(before=module.project_adaptive_gradients,after=project_adaptive_gradients)
    generator=torch.Generator().manual_seed(883)
    rows=[]
    for name,n,m,transposed in [('mlp_input',3072,768,False),('embedding',768,50304,True)]:
        factor=AdaptiveFactors(torch.randn(n,8,dtype=torch.float64,generator=generator)/(n**.5),
                               torch.randn(8,8,dtype=torch.float64,generator=generator),
                               torch.randn(m,8,dtype=torch.float64,generator=generator)/(m**.5))
        gradient=torch.randn((m,n) if transposed else (n,m),generator=generator)
        action=AdaptiveAction({name:factor},{})
        kwargs=dict(global_token_count=524288,loss_reduction='global_token_sum',transpose_storage=(name,) if transposed else ())
        reference=functions['before'](action,{name:gradient},**kwargs)
        candidate=functions['after'](action,{name:gradient},**kwargs)
        errors={}
        for key in ('U','C','V'):
            old,new=reference['matrices'][name][key],candidate['matrices'][name][key]
            errors[key]=float((old-new).norm()/old.norm().clamp_min(1e-300))
            torch.testing.assert_close(new,old,rtol=1e-10,atol=1e-15)
        torch.testing.assert_close(candidate['value'],reference['value'],rtol=1e-10,atol=1e-15)
        errors['contraction_absolute']=float((reference['value']-candidate['value']).abs())
        timings={key:[] for key in functions}
        for key in ('before','after','after','before')*2:
            start=time.perf_counter();result=functions[key](action,{name:gradient},**kwargs)
            timings[key].append(time.perf_counter()-start)
            del result
        rows.append(dict(name=name,computational_shape=[n,m],storage_transposed=transposed,
                         seconds=timings,median_seconds={k:statistics.median(v) for k,v in timings.items()},errors=errors))
        del gradient,action,factor,reference,candidate
    result=dict(scope=__doc__,reference_commit=REFERENCE,reference_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                candidate_source_sha256=hashlib.sha256(Path('rewriter/adaptive_credit.py').read_bytes()).hexdigest(),
                torch_version=torch.__version__,threads=1,gradient_dtype='float32',factor_dtype='float64',
                measurement_order='one warm/correctness call each, then ABBA twice per shape',rows=rows,
                limitations='Projection-only CPU timing on shared host while a separate Chisel fit runs. No end-to-end or GPU speed claim.',
                gpt_forwards=0,gpt_backwards=0,optimizer_steps=0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
