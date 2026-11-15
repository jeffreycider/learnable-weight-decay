"""Execute a reviewable, minimally patched pinned PR395 pilot source.

All original model/init/data/base optimizer/schedule/evaluation code is retained.
Only the hidden optimizer map, credit/logging/checkpoint hooks, and configurable
execution microbatch are changed. The default preserves the published PR395
zero-residual path. The historical extra-polar variant requires an explicit flag.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

PIN='20dfc9150e0e99f83881c5a468fc2d8d0106e9b8'
SHA='c852286ca93859902ffb8c120fde013e088a1886d7a8293dcf8c2dbf9cbe917f'


def replace_once(source,old,new):
    if source.count(old)!=1:raise ValueError(f'upstream patch anchor not unique: {old[:80]}')
    return source.replace(old,new,1)


def patched_source(source,*,meta_lr,output,steps,microbatch,variant="pr395"):
    if variant not in {"pr395", "legacy-extra-polar"}:raise ValueError("unknown SOAP variant")
    if not 1<=steps<=2900:raise ValueError('run length must be 1..2900')
    if hashlib.sha256(source.encode()).hexdigest()!=SHA:raise ValueError('upstream source hash mismatch')
    source=replace_once(source,'class Muon(torch.optim.Optimizer):',
        'from rewriter.soap_online import SoapResidualOnline\n\nclass Muon(torch.optim.Optimizer):')
    start=source.index('                    ns_steps = SHAMPOO_NS_FINAL_STEPS')
    stop=source.index('                dist.all_gather(params_pad',start)
    source=source[:start]+'                    hyper.propose(p, direction, residual, state, group, self.step_count, use_soap)\n'+source[stop:]
    source=replace_once(source,'def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):','def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024, start_batch=0):')
    source=replace_once(source,'    tokens, pos = _load_data_shard(next(file_iter)), 0\n    while True:','    from rewriter.soap_checkpoint import loader_position\n    tokens, pos = loader_position(file_iter, batch_size, start_batch, _load_data_shard)\n    while True:')
    source=replace_once(source,'train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)','train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size, start_batch=int(os.environ.get("SOAP_RESUME_BATCH","0")))')
    source=replace_once(source,'mbs = 64',f'mbs = {microbatch}')
    source=replace_once(source,'_tailema = None\n\nfor p in model.parameters():',
        f'_tailema = None\nhyper = SoapResidualOnline(globals(), meta_lr={meta_lr!r}, output={str(output)!r}, stop_step={steps}, variant={variant!r})\n\nfor p in model.parameters():')
    source=replace_once(source,'    # set optimization hyperparameters and take a step\n',
        '    torch.cuda.synchronize()\n    hyper.gpt_seconds=time.perf_counter()-gpt_start\n    hyper.consume(step)\n    # set optimization hyperparameters and take a step\n')
    source=replace_once(source,'    for i in range(len(inputs) // mbs):\n',
        '    torch.cuda.synchronize()\n    gpt_start=time.perf_counter()\n    step_loss = torch.zeros((),device=device,dtype=torch.float64)\n    for i in range(len(inputs) // mbs):\n')
    source=replace_once(source,'        loss.backward()\n','        step_loss.add_(loss.detach().double())\n        loss.backward()\n')
    source=replace_once(source,'    for opt in optimizers:\n        opt.step()\n', '    torch.cuda.synchronize()\n    optimizer_start=time.perf_counter()\n    for opt in optimizers:\n        opt.step()\n    torch.cuda.synchronize()\n    hyper.optimizer_seconds=time.perf_counter()-optimizer_start\n')
    source=replace_once(source,'    model.zero_grad(set_to_none=True)\n',
        '    hyper.log_step(step, step_loss)\n    model.zero_grad(set_to_none=True)\n')
    source=replace_once(source,'        model.train()\n','        hyper.log_validation(step, val_ema_loss.item(), val_loss.item())\n        model.train()\n')
    source=replace_once(source,'for step in range(train_steps + 1):', '''start_step=0
if os.environ.get('SOAP_RESUME'):
    from rewriter.soap_checkpoint import restore
    checkpoint=torch.load(Path(os.environ['SOAP_RESUME'])/f'rank{dist.get_rank()}.pt',map_location=device,weights_only=False)
    start_step=restore(globals(),hyper,checkpoint)
    training_time=checkpoint['accumulated_training_seconds']
    del checkpoint
for step in range(start_step, train_steps + 1):''')
    source=replace_once(source,'    if TRAIN_PROGRESS_INTERVAL > 0 and (step + 1) % TRAIN_PROGRESS_INTERVAL == 0:', '    hyper.maybe_checkpoint(step+1)\n    if TRAIN_PROGRESS_INTERVAL > 0 and (step + 1) % TRAIN_PROGRESS_INTERVAL == 0:')
    source=replace_once(source,'dist.destroy_process_group()','hyper.finish()\ndist.destroy_process_group()')
    return source


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['pr395','legacy-extra-polar'],default='pr395')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=256)
    parser.add_argument('--meta-lr',type=float,default=0.)
    parser.add_argument('--microbatch',type=int,default=64)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--checkpoint-every',type=int,default=500)
    parser.add_argument('--generate-only',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.resume:
        header=json.loads((args.resume/'complete.json').read_text())
        for key,value in {'variant':args.variant,'seed':args.seed,'meta_lr':args.meta_lr,'stop_step':args.steps,'microbatch':args.microbatch,'world_size':int(os.environ.get('WORLD_SIZE','1'))}.items():
            if header[key]!=value:raise ValueError('resume setting mismatch: '+key)
        os.environ['SOAP_RESUME']=str(args.resume.resolve())
        os.environ['SOAP_RESUME_BATCH']=str(header['completed_steps'])
    os.environ['SOAP_CHECKPOINT_EVERY']=str(args.checkpoint_every)
    original=Path(__file__).resolve().parents[1]/'baseline/pr395/upstream.py'
    source=patched_source(original.read_text(),meta_lr=args.meta_lr,output=args.output.resolve(),steps=args.steps,microbatch=args.microbatch,variant=args.variant)
    target=args.output/f"executed_pr395_hyper_rank{os.environ.get('RANK','0')}.py"
    target.write_text(source)
    (args.output/'config.json').write_text(json.dumps(dict(vars(args),output=str(args.output),resume=None if args.resume is None else str(args.resume),
        upstream_commit=PIN,upstream_sha256=SHA,executed_sha256=hashlib.sha256(source.encode()).hexdigest(),
        optimizer_description=('Published PR395 direct Shampoo plus additive learned residuals; no outer polar' if args.variant=='pr395' else 'Historical unintended explicit Shampoo factors plus outer polar'),
        gradient_units='base token sum; hypergradient global mean nats/token',
        hyperparameters='dense dimensionless residual matrices on all 72 hidden matrices',
        meta_betas=[.9,.99],gradient_clip=1.,original_schedule_steps=2900),indent=2))
    if args.generate_only:return
    os.environ['STOP_STEP']=str(args.steps)
    os.environ['TRAIN_PROGRESS_INTERVAL']='1'
    sys.argv=[str(target),'--seed',str(args.seed)]
    import torch
    torch.set_num_interop_threads(20)
    exec(compile(source,str(target),'exec'),{'__name__':'__main__','__file__':str(target)})

if __name__=='__main__':main()
