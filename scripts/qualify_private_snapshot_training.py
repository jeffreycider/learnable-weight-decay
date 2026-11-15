"""Qualification-only store dispatch around the unchanged full-bank trainer.

The explicit backend and this wrapper's SHA belong in the recipe, so a resume
cannot silently change transport. This does not register a production backend.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rewriter.private_probe_store import PrivateProbeStore


@contextmanager
def snapshot_backend(backend):
    if backend not in ('reference','owned_cpu_v1','pinned_d2h_v1'):
        raise ValueError('unknown qualified snapshot candidate')
    original = PrivateProbeStore.__init__
    def initialize(self,directory,**kwargs):
        if 'snapshot_backend' in kwargs and kwargs['snapshot_backend'] != backend:
            raise ValueError('conflicting snapshot backend')
        return original(self,directory,**dict(kwargs,snapshot_backend=backend))
    PrivateProbeStore.__init__ = initialize
    try:yield
    finally:PrivateProbeStore.__init__ = original


def validate_qualification(recipe,backend,source):
    if recipe.get('qualification_only') is not True:
        raise ValueError('snapshot wrapper is qualification-only')
    if recipe.get('qualification_snapshot_backend') != backend:
        raise ValueError('snapshot backend differs from explicit recipe identity')
    if recipe.get('snapshot_qualification_driver_sha256') != hashlib.sha256(Path(source).read_bytes()).hexdigest():
        raise ValueError('snapshot qualifier source hash differs')


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--snapshot-backend',required=True)
    parser.add_argument('--recipe',required=True)
    args,rest=parser.parse_known_args()
    validate_qualification(json.loads(Path(args.recipe).read_text()),args.snapshot_backend,__file__)
    from scripts.train_coordinate_chisel import main
    sys.argv=[sys.argv[0],'--recipe',args.recipe,*rest]
    with snapshot_backend(args.snapshot_backend):main()
