"""CPU-only explicit same-world distributed-store fork from a pinned JSON plan."""
import argparse,datetime,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from rewriter.distributed_probe_fork import fork_distributed_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);args=p.parse_args()
    plan=json.loads(args.plan.read_text())
    if plan.pop('schema',None)!='chisel_distributed_checkpoint_fork_plan_v1':raise ValueError('explicit fork plan required')
    world=plan.pop('world_size');torch.set_num_threads(1)
    if int(os.environ.get('WORLD_SIZE','0'))!=world:raise ValueError('launch the original world size with torchrun')
    dist.init_process_group('gloo',timeout=datetime.timedelta(minutes=30))
    try:
        record=fork_distributed_checkpoint(**plan)
        if dist.get_rank()==0:print(json.dumps(record,indent=2),flush=True)
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
