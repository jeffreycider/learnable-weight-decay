"""CPU-only explicit branching; never launches or changes a running trainer."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rewriter.soap46_prefix import fork_directory

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--role',choices=['learned','frozen'],required=True)
    p.add_argument('--variant',choices=['inner-outer','outer-only'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--implementation-sha256',required=True)
    p.add_argument('--meta-lr',type=float);p.add_argument('--residual-decay',type=float)
    a=p.parse_args();print(json.dumps(fork_directory(**vars(a)),indent=2))
if __name__=='__main__':main()
