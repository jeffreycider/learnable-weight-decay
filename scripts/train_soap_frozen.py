"""Explicit within-run freeze-H diagnostic using the unchanged pinned trainer.

Example: torchrun ... scripts/train_soap_frozen.py --freeze-at 1000 [ordinary
train_soap_hyper arguments, including --resume and the original --meta-lr].
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--freeze-at',type=int,required=True)
    args,remaining=p.parse_known_args()
    if '--resume' not in remaining:raise ValueError('freeze diagnostic requires a within-run checkpoint')
    if args.freeze_at<=0:raise ValueError('freeze step must be positive')
    output=Path(remaining[remaining.index('--output')+1]);output.mkdir(parents=True,exist_ok=True)
    import rewriter.soap_checkpoint as checkpoint
    from rewriter.soap_freeze import install_checkpoint_policy
    identity=checkpoint.implementation_identity()
    record={'freeze_at_step':args.freeze_at,'baseline_implementation_sha256':identity,
            'frozen_quantity':'dimensionless H; detached Q scale continues changing',
            'hypergradient_evaluations_after_branch':0,
            'branch_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'freeze_policy_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'rewriter/soap_freeze.py').read_bytes()).hexdigest()}
    (output/'freeze_policy.json').write_text(json.dumps(record,indent=2))
    install_checkpoint_policy(checkpoint,args.freeze_at)
    sys.argv=[sys.argv[0],*remaining]
    from scripts.train_soap_hyper import main as train
    train()

if __name__=='__main__':main()
