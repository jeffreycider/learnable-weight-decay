"""Write a separate FP32 continuation bundle and exact rank resume files."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline.fp32_transition import promote_bundle
from baseline.checkpoint import save_atomic
from scripts.bundle_baseline_checkpoint import extract_rank
from scripts.merge_baseline_checkpoint import _sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--microbatch-sequences',type=int,default=16);p.add_argument('--max-steps',type=int)
    p.add_argument('--train-shards',type=int,help='Explicitly append consecutive pinned shards without changing the captured prefix')
    p.add_argument('--eager',action='store_true');args=p.parse_args()
    digest=_sha256(args.input)
    if digest!=args.expected_sha256:raise ValueError('source bundle SHA256 mismatch')
    if args.output.exists():raise FileExistsError(args.output)
    bundle=torch.load(args.input,map_location='cpu',weights_only=False,mmap=True)
    converted=promote_bundle(bundle,source_sha256=digest,microbatch_sequences=args.microbatch_sequences,
                            max_steps=args.max_steps,compile_model=not args.eager,training_shards=args.train_shards)
    args.output.mkdir(parents=True)
    records=[]
    for rank in range(converted['shared']['config']['world_size']):
        path=args.output/f'latest_rank{rank}.pt';save_atomic(extract_rank(converted,rank),path)
        sha=_sha256(path)
        records.append(dict(rank=rank,file=path.name,sha256=sha,bytes=path.stat().st_size))
    (args.output/'config.json').write_text(json.dumps(converted['shared']['config'],indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(dict(source=str(args.input),source_sha256=digest,
        transition=converted['precision_transition'],ranks=records),indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),ranks=len(records),step=converted['shared']['step'])))
if __name__=='__main__':main()
