"""CPU check of the confidence audit's pooled-quantile path at production size.

Simulates the per-rank float32 pool files that scripts/evaluate_chisel_confidence.py
writes (default: 8 ranks, 800 states x 128 sequences x 1024 tokens = 104,857,600
tokens) and the rank-0 memory-mapped concatenation + exact FP64 NumPy quantiles.
Reports wall time and peak RSS. Synthetic values only; no GPT, no policy.
"""
import argparse,json,resource,time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rewriter.confidence_audit import pooled_quantiles,TOKEN_QUANTILES


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ranks',type=int,default=8)
    p.add_argument('--states',type=int,default=800);p.add_argument('--sequences',type=int,default=128);p.add_argument('--length',type=int,default=1024)
    a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=True)
    total=a.states*a.sequences*a.length;per_rank=total//a.ranks
    rng=np.random.default_rng(0);started=time.perf_counter()
    for r in range(a.ranks):  # lognormal KL-like values, as float32 like the audit
        np.save(a.output/f'fitting_seen_rank{r}.npy',rng.lognormal(-14.,1.,per_rank).astype(np.float32))
    write_seconds=time.perf_counter()-started;started=time.perf_counter()
    parts=[np.load(a.output/f'fitting_seen_rank{r}.npy',mmap_mode='r') for r in range(a.ranks)]
    pooled=np.concatenate(parts).astype(np.float64)
    q=pooled_quantiles(pooled,TOKEN_QUANTILES)
    quantile_seconds=time.perf_counter()-started
    exact_max=float(max(float(np.asarray(x).max()) for x in parts))
    result=dict(tokens=int(pooled.size),ranks=a.ranks,write_seconds=write_seconds,load_and_quantile_seconds=quantile_seconds,
                peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6,quantiles=q,max_matches_per_rank_max=q['p100']==exact_max,
                pool_bytes_float32=int(pooled.size*4),fp64_pool_gb=pooled.nbytes/1e9)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    for r in range(a.ranks):(a.output/f'fitting_seen_rank{r}.npy').unlink()


if __name__=='__main__':main()
