"""Telemetry wrapper for the existing fresh FP64 adaptive lifecycle qualifier.

No optimizer/model/schedule hooks; normal trainer CLI remains authoritative.
This wrapper measures process-phase allocator peak and wall time, including
setup/readouts. Per-update complete wall times remain in trainer metrics.jsonl.
"""
import argparse,json,os,runpy,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--recipe',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--resume',action='store_true')
    args,_=parser.parse_known_args()
    recipe=json.loads(Path(args.recipe).read_text())
    if (recipe.get('qualification_only') is not True or recipe.get('action_family')!='adaptive_ucv_v1'
            or recipe['architecture'].get('factor_precision')!='fp64'):
        raise ValueError('this telemetry wrapper only serves explicit FP64 adaptive qualifications')
    # The pinned determinism policy must run BEFORE first CUDA initialization.
    # This is a fresh process, so allocator peaks naturally start at zero.
    local=int(os.environ.get('LOCAL_RANK','0'))
    started=time.perf_counter();terminal='failed'
    try:
        runpy.run_module('scripts.train_coordinate_chisel',run_name='__main__')
        terminal='completed'
    except SystemExit as error:
        terminal='qualification_interrupt' if error.code==75 else 'failed'
        raise
    finally:
        initialized=torch.cuda.is_initialized()
        if initialized:torch.cuda.synchronize(local)
        directory=Path(args.output);directory.mkdir(parents=True,exist_ok=True)
        phase='resumed' if args.resume else 'fresh'
        result=dict(status=terminal,rank=int(os.environ.get('RANK','0')),phase=phase,
            total_process_seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(local) if initialized else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(local) if initialized else 0,
            timing_scope='existing trainer setup, fitting, readouts and checkpoint I/O; no extra model calls')
        target=directory/f'precision-runtime-{phase}-rank{result["rank"]}.json'
        if target.exists():raise FileExistsError('never overwrite qualifier phase telemetry')
        target.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
