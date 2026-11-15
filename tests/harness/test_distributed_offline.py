import os
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from chisel.harness import RunState
from chisel.distributed_offline import owned, finish, commit, validate_resume
from chisel.recording import save_atomic


def worker(rank,path,root):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    try:
        model=torch.nn.Linear(2,1,bias=False)
        with torch.no_grad():model.weight.fill_(.5)
        state=RunState(step=2,model=model,extras=dict(rank=rank,world_size=2))
        entries=[dict(sha256=f'{i:064x}') for i in range(5)]
        metrics=dict(loss=0.)
        for i,entry in enumerate(entries):
            if owned(entry,state):
                loss=model(torch.tensor([[float(i),1.]])).sum()/len(entries)
                loss.backward();metrics['loss']+=loss.item()
        finish(model,metrics,2)
        torch.testing.assert_close(model.weight.grad,torch.tensor([[2.,1.]]))
        assert abs(metrics['loss']-1.5)<1e-6
        cfg=dict(output_dir=root,train_steps=3)
        checkpoint=Path(root)/f'checkpoint_000003_rank{rank}.pt'
        save_atomic(dict(model=model.state_dict(),step=2),checkpoint)
        commit(3)(cfg,state);dist.barrier()
        cfg['resume']=str(Path(root)/'checkpoint_000003_rank{rank}.pt')
        validate_resume()(cfg,state)
        # A rank with no examples still participates in the shared sum.
        model.zero_grad(set_to_none=True)
        if rank==0:model(torch.ones(1,2)).sum().backward()
        finish(model,dict(loss=float(rank==0)),2)
        torch.testing.assert_close(model.weight.grad,torch.ones(1,2))
    finally:dist.destroy_process_group()


def test_owned_private_work_shared_gradient_and_committed_restart(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/'join'),str(tmp_path)),nprocs=2,join=True)
