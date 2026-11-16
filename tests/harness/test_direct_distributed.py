import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from chisel.harness import run
from tests.harness.test_direct_transfer import tiny_config, assert_tree_equal


class ShardedStream:
    def __init__(self, device, rank=0, world_size=1):
        self.device, self.rank, self.world_size = device, rank, world_size
        self.step = 0
    def __next__(self):
        generator = torch.Generator().manual_seed(101 + self.step)
        self.step += 1
        x = torch.randn(8, 4, generator=generator)
        y = torch.randint(3, (8,), generator=generator)
        return x.chunk(self.world_size)[self.rank].to(self.device), y.chunk(self.world_size)[self.rank].to(self.device)
    def state_dict(self):
        return dict(step=self.step)
    def load_state_dict(self, saved):
        self.step = saved['step']


def worker(rank, rendezvous, config):
    os.environ.update(WORLD_SIZE='2', LOCAL_RANK=str(rank))
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        result = run(config)
        torch.save(dict(model=result.model.state_dict(), optimizer=result.optimizer.state_dict()),
                   config['output_dir'] + f'/result_rank{rank}.pt')
    finally:
        dist.destroy_process_group()


def test_replicas_match_and_use_global_not_rank_local_transfer(tmp_path):
    config = tiny_config(tmp_path / 'single')
    config['model'] = dict(_target_='torch.nn.Linear', in_features=4, out_features=3)
    config.pop('initialize_model')
    config['data'] = dict(_target_=__name__ + '.ShardedStream')
    config['optimizers'] = {key: config['optimizers'][key] for key in ('hidden', 'scalars')}
    config['optimizers']['hidden']['parameters'] = 'weight'
    config['optimizers']['scalars']['parameters'] = 'bias'
    config['microbatch_size'] = 2
    single = run(config)
    config['runtime']['world_size'] = 2
    config['output_dir'] = str(tmp_path / 'distributed')
    mp.spawn(worker, args=(str(tmp_path / 'rendezvous'), config), nprocs=2, join=True)
    saved = [torch.load(tmp_path / f'distributed/result_rank{rank}.pt', weights_only=False) for rank in range(2)]
    assert_tree_equal(saved[0], saved[1])
    for name, value in single.model.state_dict().items():
        torch.testing.assert_close(saved[0]['model'][name], value, rtol=2e-5, atol=1e-6)
    for bank in ('raw', 'consolidated'):
        for name, value in getattr(single.optimizer, bank).items():
            torch.testing.assert_close(saved[0]['optimizer'][bank][name], value, rtol=2e-5, atol=1e-6)
    assert saved[0]['optimizer']['counts']['lookahead_tokens'] == single.optimizer.counts['lookahead_tokens'] == 40
