"""Complete same-trial state for accepted-record46 basis-residual experiments."""
import hashlib
import time
from pathlib import Path
import torch

def loader_position(file_iter,batch_size,start_batch,load_shard):
    if start_batch<0:raise ValueError('negative data cursor')
    for file in file_iter:
        header=torch.from_file(str(file),False,256,dtype=torch.int32)
        if int(header[0])!=20240520 or int(header[1])!=1:raise ValueError('invalid FineWeb header')
        count=(int(header[2])-2)//batch_size
        if start_batch<count:return load_shard(file),start_batch*batch_size
        start_batch-=count
    raise ValueError('checkpoint data cursor exceeds staged data')


def identity():
    root=Path(__file__).resolve().parents[1];h=hashlib.sha256()
    for name in ('baseline/record46/upstream.py','scripts/train_soap46.py','rewriter/soap46_map.py',
                 'rewriter/soap46_online.py','rewriter/soap46_checkpoint.py','rewriter/lag_one.py','rewriter/soap46_conditioning.py','rewriter/soap46_action_diagnostic.py'):
        h.update(name.encode());h.update((root/name).read_bytes())
    return h.hexdigest()


def payload(scope,hyper,step):
    wrapper=scope['optimizers'][0]
    return dict(schema='record46-basis-v1',implementation_sha256=identity(),completed_steps=step,
                world_size=torch.distributed.get_world_size(),seed=scope['SEED'],variant=hyper.variant,
                meta_lr=hyper.lr,meta_clip=hyper.clip,meta_betas=list(hyper.betas),reset_meta_at_activation=hyper.reset_meta_at_activation,residual_frame=hyper.residual_frame,meta_epsilon=hyper.meta_epsilon,compile_map=hyper.compile_map,warmup_steps=hyper.warmup_steps,warmup_meta=hyper.warmup_meta,meta_ramp_steps=hyper.meta_ramp_steps,residual_decay=hyper.residual_decay,base_lr=scope['MUON_LR'],cwd=scope['CWD'],stop_step=hyper.stop,microbatch=scope['mbs'],
                model=scope['model'].state_dict(),base_optimizers=[o.state_dict() for o in wrapper.inner_optimizer],
                wrapper=wrapper.state_dict(),wrapper_clock={k:getattr(wrapper,k) for k in ('it','lookahead_status','current_lookahead_stepsize')},
                muon_step_count=scope['optimizer2'].step_count,tail_ema=scope['_tailema'],hyper=hyper.state_dict(),
                data_cursor_batches=step,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                accumulated_training_seconds=scope['training_time']+time.perf_counter()-scope['t0'])


def restore(scope,hyper,state):
    expected=dict(schema='record46-basis-v1',implementation_sha256=identity(),world_size=torch.distributed.get_world_size(),
                  seed=scope['SEED'],variant=hyper.variant,meta_lr=hyper.lr,meta_clip=hyper.clip,meta_betas=list(hyper.betas),reset_meta_at_activation=hyper.reset_meta_at_activation,residual_frame=hyper.residual_frame,meta_epsilon=hyper.meta_epsilon,compile_map=hyper.compile_map,warmup_steps=hyper.warmup_steps,warmup_meta=hyper.warmup_meta,meta_ramp_steps=hyper.meta_ramp_steps,residual_decay=hyper.residual_decay,base_lr=scope['MUON_LR'],cwd=scope['CWD'],stop_step=hyper.stop,microbatch=scope['mbs'])
    for key,value in expected.items():
        if state.get(key)!=value:raise ValueError('record46 resume mismatch: '+key)
    wrapper=scope['optimizers'][0];scope['model'].load_state_dict(state['model'],strict=True)
    for opt,saved in zip(wrapper.inner_optimizer,state['base_optimizers'],strict=True):opt.load_state_dict(saved)
    wrapper.load_state_dict(state['wrapper'])
    for k,v in state['wrapper_clock'].items():setattr(wrapper,k,v)
    scope['optimizer2'].step_count=state['muon_step_count'];scope['_tailema']=state['tail_ema']
    hyper.load_state_dict(state['hyper']);torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
    return state['completed_steps']
