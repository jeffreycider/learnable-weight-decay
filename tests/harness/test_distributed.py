import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from chisel.harness import RunState
from chisel.training import backward
from chisel.observations import Observations


def worker(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    try:
        torch.manual_seed(12)
        model=torch.nn.Sequential(torch.nn.Linear(4,3,bias=False))
        inputs=torch.randn(6,4);targets=torch.tensor([0,1,2,0,1,2])
        full=torch.nn.functional.cross_entropy(model(inputs),targets)
        expected=torch.autograd.grad(full,list(model.parameters()))
        sensor=Observations(model,r'0.weight');sensor.active=True
        state=RunState(model=model,train_batches=iter([(inputs[3*rank:3*rank+3],targets[3*rank:3*rank+3])]),extras=dict(device=torch.device('cpu'),world_size=2,forward=model))
        cfg=dict(microbatch_size=2,gradient_units='mean',runtime=dict(autocast=None),loss=dict(name='torch.nn.functional.cross_entropy',kwargs={}))
        backward(cfg,state)
        for p,g in zip(model.parameters(),expected):torch.testing.assert_close(p.grad,g)
        activity=sensor.finalize(2)
        torch.testing.assert_close(activity['0.weight']['input'],inputs.square().mean(0).sqrt())
        sensor.close()
    finally:
        dist.destroy_process_group()


def test_global_gradient_and_activity_reduction(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/'rendezvous'),),nprocs=2,join=True)
