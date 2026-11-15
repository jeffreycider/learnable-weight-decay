"""One complete multi-rank PR323 checkpoint, retaining every rank's RNG/cursor.

Unlike the frozen-observation merge, this bundle can reconstruct each original
format-1 rank checkpoint exactly. Model/Adam replicas and merged Muon momentum
are stored once. It does not change training configuration or world size.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.merge_baseline_checkpoint import merge_checkpoints,_equal,_sha256
from baseline.checkpoint import save_atomic

FORMAT='baseline_rank_bundle_v1'


def _record_equal(left,right,label):
    if isinstance(left,np.ndarray):
        if not (isinstance(right,np.ndarray) and left.dtype==right.dtype and np.array_equal(left,right)):
            raise ValueError('checkpoint mismatch: '+label)
    elif isinstance(left,dict):
        if not isinstance(right,dict) or left.keys()!=right.keys():raise ValueError('checkpoint mismatch: '+label)
        for key in left:_record_equal(left[key],right[key],label+'.'+str(key))
    elif isinstance(left,(list,tuple)):
        if type(left)!=type(right) or len(left)!=len(right):raise ValueError('checkpoint mismatch: '+label)
        for i,(a,b) in enumerate(zip(left,right)):_record_equal(a,b,label+'.'+str(i))
    else:_equal(left,right,label)


def bundle_checkpoints(paths,output):
    paths=[Path(p) for p in paths];output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='checkpoint-bundle-',dir=output.parent) as tmp:
        merged_path=Path(tmp)/'merged.pt'
        metadata=merge_checkpoints(paths,merged_path)
        shared=torch.load(merged_path,map_location='cpu',weights_only=False,mmap=True)
        shared={k:v for k,v in shared.items() if k not in ('optimizer_merge','rng','data_cursor','elapsed_seconds')}
        shared['format_version']=1
        per_rank={}
        for source in metadata['sources']:
            path=Path(source['path'])
            if _sha256(path)!=source['sha256']:raise ValueError('source changed after merge validation')
            state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            per_rank[source['rank']]={k:state[k] for k in ('rng','data_cursor','elapsed_seconds')}
        bundle=dict(format_version=FORMAT,shared=shared,ranks=per_rank,provenance=metadata)
        # Validate exact reconstructed records before publishing.
        for source in metadata['sources']:
            original=torch.load(source['path'],map_location='cpu',weights_only=False,mmap=True)
            _record_equal(extract_rank(bundle,source['rank']),original,'reconstructed rank checkpoint')
        save_atomic(bundle,output)
    return dict(format_version=FORMAT,resumable_world_size=len(per_rank),step=shared['step'],
                sources=metadata['sources'],bytes=output.stat().st_size,sha256=_sha256(output))


def extract_rank(bundle,rank):
    if bundle.get('format_version')!=FORMAT or rank not in bundle['ranks']:
        raise ValueError('unsupported bundle or missing rank')
    shared=bundle['shared'];world=shared['config']['world_size']
    if set(bundle['ranks'])!=set(range(world)):raise ValueError('bundle lacks source ranks')
    muon=shared['optimizers'][1]
    owned={pid for group in muon['param_groups'] for index,pid in enumerate(group['params']) if index%world==rank}
    return dict(shared,**bundle['ranks'][rank],optimizers=[shared['optimizers'][0],
        dict(muon,state={pid:muon['state'][pid] for pid in owned})])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',nargs='+',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    print(json.dumps(bundle_checkpoints(args.inputs,args.output),indent=2))

if __name__=='__main__':main()
