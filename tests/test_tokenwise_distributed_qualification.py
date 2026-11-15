from datetime import timedelta

import torch
import torch.distributed as dist

from scripts.qualify_tokenwise_graph_distributed import sum_actor_gradients, cpu_tree, actor_hash


def worker(rank, rendezvous):
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        actor=torch.nn.Linear(1,1)
        actor.weight.grad=torch.tensor([[1. if rank==0 else 3.]])
        actor.bias.grad=None if rank==0 else torch.tensor([.5])
        sum_actor_gradients(actor)
        assert actor.weight.grad.item()==4.  # Global 1/4 weighting already applied.
        assert actor.bias.grad.item()==.5
    finally:
        dist.destroy_process_group()


def test_production_gradient_sum_does_not_divide_by_world_size(tmp_path):
    torch.multiprocessing.spawn(worker,args=(str(tmp_path/'gloo'),),nprocs=2,join=True)


def test_cpu_snapshot_and_replica_hash_detect_finite_step_changes():
    original={'weight':torch.tensor([1.,2.])}
    saved=cpu_tree(original)
    before=actor_hash(saved)
    original['weight'].add_(.001)
    assert actor_hash(saved)==before
    assert actor_hash(cpu_tree(original))!=before
