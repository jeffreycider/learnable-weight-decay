"""Full PR395 continuation state, with no learned-state transfer between trials."""
from pathlib import Path
import time
import hashlib
import torch


def loader_position(file_iter,batch_size,start_batch,load_shard):
    """Seek the original loader by headers, preserving its strict end condition."""
    if start_batch<0:raise ValueError('negative data cursor')
    for file in file_iter:
        header=torch.from_file(str(file),False,256,dtype=torch.int32)
        if int(header[0])!=20240520 or int(header[1])!=1:raise ValueError('invalid FineWeb header')
        count=(int(header[2])-2)//batch_size
        if start_batch<count:return load_shard(file),start_batch*batch_size
        start_batch-=count
    raise ValueError('checkpoint data cursor exceeds staged data')


def implementation_identity():
    root=Path(__file__).resolve().parents[1]
    files=['baseline/pr395/upstream.py','scripts/train_soap_hyper.py','rewriter/soap_checkpoint.py',
           'rewriter/soap_online.py','rewriter/soap_pr395_residual.py','rewriter/soap_residual.py','rewriter/soap_envelope.py','rewriter/lag_one.py']
    digest=hashlib.sha256()
    for name in files:digest.update(name.encode());digest.update((root/name).read_bytes())
    return digest.hexdigest()


def payload(scope,hyper,completed_steps):
    wrapper=scope['optimizers'][0]
    return dict(schema=2,implementation_sha256=implementation_identity(),completed_steps=completed_steps,world_size=torch.distributed.get_world_size(),
        model=scope['model'].state_dict(),base_optimizers=[o.state_dict() for o in wrapper.inner_optimizer],
        wrapper=wrapper.state_dict(),wrapper_clock={k:getattr(wrapper,k) for k in ['it','lookahead_status','current_lookahead_stepsize']},
        muon_step_count=scope['optimizer2'].step_count,head_eigen=scope['head_eigen'].st,
        embed_master=scope['embed_master'].st,tail_ema=scope['_tailema'],
        hyper=hyper.state_dict(),data_cursor_batches=completed_steps,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
        seed=scope['SEED'],meta_lr=hyper.lr,stop_step=hyper.stop,microbatch=scope['mbs'],
        accumulated_training_seconds=scope['training_time']+(time.perf_counter()-scope['t0'] if 't0' in scope else 0.),
        benchmark_variant=getattr(hyper,'variant','legacy-extra-polar'))


def restore(scope,hyper,state):
    if state['schema']!=2:raise ValueError('unsupported checkpoint schema')
    if state['implementation_sha256']!=implementation_identity():raise ValueError('resume implementation hash differs')
    expected={'world_size':torch.distributed.get_world_size(),'seed':scope['SEED'],
              'meta_lr':hyper.lr,'stop_step':hyper.stop,'microbatch':scope['mbs']}
    for k,v in expected.items():
        if state[k]!=v:raise ValueError('resume setting mismatch: '+k)
    if state['benchmark_variant']!=getattr(hyper,'variant','legacy-extra-polar'):raise ValueError('resume SOAP variant differs')
    scope['model'].load_state_dict(state['model'],strict=True)
    wrapper=scope['optimizers'][0]
    for optimizer,saved in zip(wrapper.inner_optimizer,state['base_optimizers'],strict=True):optimizer.load_state_dict(saved)
    wrapper.load_state_dict(state['wrapper'])
    for k,v in state['wrapper_clock'].items():setattr(wrapper,k,v)
    scope['optimizer2'].step_count=state['muon_step_count']
    scope['head_eigen'].st=state['head_eigen'];scope['embed_master'].st=state['embed_master']
    scope['_tailema']=state['tail_ema'];hyper.load_state_dict(state['hyper'])
    torch.set_rng_state(state['torch_rng'].cpu())
    torch.cuda.set_rng_state(state['cuda_rng'].cpu())
    return state['completed_steps']
