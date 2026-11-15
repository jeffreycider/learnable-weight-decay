"""Record46-only learned basis residual driver. Default state banks remain baseline."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

PIN='4ea6b937337a4889b8cfe3f38a93d120048d8f71'
SHA='35d25d769c86c227a4cbdabb583d0a95f5a6a19b87e14d0bc944649353b47188'
ACTIVE_ENV=dict(ROWFLOOR='1',ROWFLOOR_FROB='0',ROWFLOOR_RHO='1',CWD='.025',CWD_POST='1',
                TAILEMA_TAU='150',TAILEMA_LAMBDA='.6',TAILEMA_START='2400',TAILEMA_END='2900')


def configure_environment(source,*,steps,output,base_lr=.0375,cwd=.025):
    """Do not inherit dormant experiments from a caller's shell environment."""
    tree=ast.parse(source);keys=set()
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and ast.unparse(n.func)=='os.environ.get' and n.args and isinstance(n.args[0],ast.Constant):keys.add(n.args[0].value)
    for key in keys:os.environ.pop(key,None)
    os.environ.update(ACTIVE_ENV);os.environ.update(MUON_LR=str(base_lr),CWD=str(cwd));os.environ.update(STOP_STEP=str(steps),LOG_DIR=str(output),TRAIN_PROGRESS_INTERVAL='0')
    return {k:os.environ[k] for k in sorted(keys) if k in os.environ}


def replace_once(source,old,new):
    if source.count(old)!=1:raise ValueError('source patch anchor not unique: '+old[:80])
    return source.replace(old,new,1)


def patched_source(source,*,variant,meta_lr,output,steps,microbatch,meta_clip=1.,checkpoint_every=500,meta_betas=(.9,.99),compile_map=False,warmup_steps=0,warmup_meta="moments",meta_ramp_steps=0,residual_decay=0.,residual_frame="basis",meta_epsilon=1e-8,reset_meta_at_activation=False,action_diagnostic_steps=(),action_diagnostic_names=()):
    from rewriter.soap46_action_diagnostic import validate_action_samples
    action_diagnostic_steps, action_diagnostic_names = validate_action_samples(
        action_diagnostic_steps, action_diagnostic_names, stop_step=steps,
        residual_frame=residual_frame, compile_map=compile_map)
    if hashlib.sha256(source.encode()).hexdigest()!=SHA:raise ValueError('record46 source hash mismatch')
    if variant not in ('inner-outer','outer-only'):raise ValueError('unknown basis variant')
    if not 1<=steps<=2900:raise ValueError('run length must be 1..2900')
    source=replace_once(source,'MUON_LR = float(os.environ.get("MUON_LR", 0.0375))','MUON_LR = float(os.environ.get("MUON_LR", 0.0375))\nMUON_POWER_C *= MUON_LR / 0.0375')
    source=replace_once(source,'class Muon(torch.optim.Optimizer):','from rewriter.soap46_online import Soap46Online\n\nclass Muon(torch.optim.Optimizer):')
    begin=source.index('                    grad = p.grad\n',source.index('class Muon('))
    end=source.index('                dist.all_gather(params_pad',begin)
    source=source[:begin]+'''                    hyper.propose(p, state, group, self.step_count,
                                  p in self.attn_soap_params, p in self.attn_proj_soap_params)
'''+source[end:]
    source=replace_once(source,'def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):','def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024, start_batch=0):')
    source=replace_once(source,'    tokens, pos = _load_data_shard(next(file_iter)), 0\n    while True:',
        '    from rewriter.soap46_checkpoint import loader_position\n    tokens, pos = loader_position(file_iter,batch_size,start_batch,_load_data_shard)\n    while True:')
    source=replace_once(source,'train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)',
        'train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size, start_batch=int(os.environ.get("SOAP46_RESUME_BATCH","0")))')
    source=replace_once(source,'mbs = 64',f'mbs = {microbatch}')
    source=replace_once(source,'_fork_start = 0\n',f'''hyper = Soap46Online(globals(),variant={variant!r},meta_lr={meta_lr!r},output={str(output)!r},stop_step={steps},meta_clip={meta_clip!r},checkpoint_every={checkpoint_every},meta_betas={tuple(meta_betas)!r},compile_map={compile_map!r},warmup_steps={warmup_steps!r},warmup_meta={warmup_meta!r},meta_ramp_steps={meta_ramp_steps!r},residual_decay={residual_decay!r},residual_frame={residual_frame!r},meta_epsilon={meta_epsilon!r},reset_meta_at_activation={reset_meta_at_activation!r},action_diagnostic_steps={action_diagnostic_steps!r},action_diagnostic_names={action_diagnostic_names!r})
_fork_start = 0
if os.environ.get('SOAP46_RESUME'):
    from rewriter.soap46_checkpoint import restore
    checkpoint=torch.load(Path(os.environ['SOAP46_RESUME'])/f'rank{{dist.get_rank()}}.pt',map_location=device,weights_only=False)
    _fork_start=restore(globals(),hyper,checkpoint)
    training_time=checkpoint['accumulated_training_seconds']
    del checkpoint
''')
    source=replace_once(source,'    for i in range(len(inputs) // mbs):\n',
        '    torch.cuda.synchronize()\n    gpt_start=time.perf_counter()\n    step_loss=torch.zeros((),device=device,dtype=torch.float64)\n    for i in range(len(inputs) // mbs):\n')
    source=replace_once(source,'        loss.backward()\n','        step_loss.add_(loss.detach().double())\n        loss.backward()\n')
    source=replace_once(source,'    # set optimization hyperparameters and take a step\n',
        '    torch.cuda.synchronize()\n    hyper.gpt_seconds=time.perf_counter()-gpt_start\n    hyper.consume(step)\n    # set optimization hyperparameters and take a step\n')
    source=replace_once(source,'    for opt in optimizers:\n        opt.step()\n',
        '    torch.cuda.synchronize()\n    optimizer_start=time.perf_counter()\n    for opt in optimizers:\n        opt.step()\n    torch.cuda.synchronize()\n    hyper.optimizer_seconds=time.perf_counter()-optimizer_start\n')
    source=replace_once(source,'    model.zero_grad(set_to_none=True)\n','    hyper.log_step(step,step_loss)\n    model.zero_grad(set_to_none=True)\n')
    source=replace_once(source,'        model.train()\n','        hyper.log_validation(step,val_ema_loss.item(),val_loss.item())\n        model.train()\n')
    source=replace_once(source,'    if TRAIN_PROGRESS_INTERVAL > 0 and (step + 1) % TRAIN_PROGRESS_INTERVAL == 0:',
        '    hyper.maybe_checkpoint(step+1)\n    if TRAIN_PROGRESS_INTERVAL > 0 and (step + 1) % TRAIN_PROGRESS_INTERVAL == 0:')
    source=replace_once(source,'dist.destroy_process_group()','hyper.finish()\ndist.destroy_process_group()')
    return source


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['inner-outer','outer-only'],required=True)
    p.add_argument('--meta-lr',type=float,default=0.);p.add_argument('--meta-clip',type=float,default=1.)
    p.add_argument('--base-lr',type=float,default=.0375);p.add_argument('--cwd',type=float,default=.025)
    p.add_argument('--meta-betas',type=float,nargs=2,default=[.9,.99]);p.add_argument('--compile-map',action='store_true')
    p.add_argument('--warmup-steps',type=int,default=0);p.add_argument('--warmup-meta',choices=['moments','off'],default='moments')
    p.add_argument('--meta-ramp-steps',type=int,default=0);p.add_argument('--residual-decay',type=float,default=0.)
    p.add_argument('--residual-frame',choices=['basis','parameter'],default='basis');p.add_argument('--meta-epsilon',type=float,default=1e-8)
    p.add_argument('--reset-meta-at-activation',action='store_true')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=2900)
    p.add_argument('--microbatch',type=int,default=64);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--checkpoint-every',type=int,default=500)
    p.add_argument('--action-diagnostic-steps',type=int,nargs='+',default=[],
                   help='opt-in zero-based Muon action indices; max 32 steps and 128 step/name pairs')
    p.add_argument('--action-diagnostic-names',nargs='+',default=[],
                   help='exact model.named_parameters names of Muon matrices; max 8, no patterns')
    p.add_argument('--generate-only',action='store_true');a=p.parse_args()
    if not 0<=a.warmup_steps<=2900 or a.meta_ramp_steps<0 or not 0<=a.residual_decay<=1 or not 0<=a.meta_lr or not 0<a.meta_clip or not a.base_lr>0 or not a.cwd>=0 or not all(0<=b<1 for b in a.meta_betas):raise ValueError('invalid meta optimizer settings')
    if not __import__('math').isfinite(a.meta_epsilon) or a.meta_epsilon<=0 or (a.residual_frame=='parameter' and a.variant!='outer-only'):raise ValueError('invalid residual conditioning')
    if a.reset_meta_at_activation and not 2<=a.warmup_steps<a.steps:raise ValueError('invalid residual Adam reset boundary')
    from rewriter.soap46_action_diagnostic import validate_action_samples
    validate_action_samples(a.action_diagnostic_steps, a.action_diagnostic_names,
        stop_step=a.steps, residual_frame=a.residual_frame, compile_map=a.compile_map)
    root=Path(__file__).resolve().parents[1];source=(root/'baseline/record46/upstream.py').read_text()
    a.output.mkdir(parents=True,exist_ok=True);env=configure_environment(source,steps=a.steps,output=a.output,base_lr=a.base_lr,cwd=a.cwd)
    if a.resume:
        header=json.loads((a.resume/'complete.json').read_text())
        for k,v in dict(seed=a.seed,variant=a.variant,meta_lr=a.meta_lr,meta_clip=a.meta_clip,meta_betas=a.meta_betas,residual_frame=a.residual_frame,meta_epsilon=a.meta_epsilon,reset_meta_at_activation=a.reset_meta_at_activation,compile_map=a.compile_map,warmup_steps=a.warmup_steps,warmup_meta=a.warmup_meta,meta_ramp_steps=a.meta_ramp_steps,residual_decay=a.residual_decay,base_lr=a.base_lr,cwd=a.cwd,stop_step=a.steps,microbatch=a.microbatch,world_size=int(os.environ.get('WORLD_SIZE','1'))).items():
            if header.get(k)!=v:raise ValueError('resume setting differs: '+k)
        os.environ['SOAP46_RESUME']=str(a.resume.resolve());os.environ['SOAP46_RESUME_BATCH']=str(header['completed_steps'])
    else:
        os.environ.pop('SOAP46_RESUME',None);os.environ.pop('SOAP46_RESUME_BATCH',None)
    source=patched_source(source,variant=a.variant,meta_lr=a.meta_lr,output=a.output.resolve(),steps=a.steps,microbatch=a.microbatch,meta_clip=a.meta_clip,checkpoint_every=a.checkpoint_every,meta_betas=a.meta_betas,residual_frame=a.residual_frame,meta_epsilon=a.meta_epsilon,reset_meta_at_activation=a.reset_meta_at_activation,compile_map=a.compile_map,warmup_steps=a.warmup_steps,warmup_meta=a.warmup_meta,meta_ramp_steps=a.meta_ramp_steps,residual_decay=a.residual_decay,action_diagnostic_steps=a.action_diagnostic_steps,action_diagnostic_names=a.action_diagnostic_names)
    target=a.output/f'executed_record46_rank{os.environ.get("RANK","0")}.py';target.write_text(source)
    if int(os.environ.get('RANK','0'))==0:
        from rewriter.soap46_checkpoint import identity
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
        config.update(baseline='accepted-record46',baseline_commit=PIN,source_sha256=SHA,implementation_sha256=identity(),
            executed_sha256=hashlib.sha256(source.encode()).hexdigest(),environment=env,meta_epsilon=a.meta_epsilon,
            gradient_units='base token SUM; hypergradient global mean nats/token',state_policy='baseline unperturbed Q variance/covariance; no gradients through statistics',basis_refresh_every=1,hidden_decay_coefficient=3.3169534699576625e-6*a.base_lr/.0375,hidden_schedule_scaling='initial LR and decay coefficient scaled together')
        (a.output/'config.json').write_text(json.dumps(config,indent=2))
        import shutil
        record=a.output/'source_record'
        for name in ('baseline/record46/upstream.py','baseline/record46/pin.json','scripts/train_soap46.py','rewriter/soap46_map.py','rewriter/soap46_online.py','rewriter/soap46_checkpoint.py','rewriter/lag_one.py','rewriter/soap46_conditioning.py','rewriter/soap46_action_diagnostic.py'):
            dst=record/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dst)
    if a.generate_only:return
    if a.compile_map:raise RuntimeError('Full-map compilation failed GPU numerical parity; use eager training. The isolated qualification script remains available.')
    sys.argv=[str(target),'--seed',str(a.seed)]
    import torch
    torch.set_num_interop_threads(20)
    exec(compile(source,str(target),'exec'),{'__name__':'__main__','__file__':str(target)})

if __name__=='__main__':main()
