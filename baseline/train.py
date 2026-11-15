"""PR323 training with explicit observation/optimizer hooks; no extra GPT passes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch
import torch.distributed as dist
from .data import distributed_data_generator
from .model import GPT, initialize_parameters_like_record
from .optim import Muon
from .verify_data import data_manifest
from .checkpoint import build_checkpoint, restore_checkpoint, save_atomic, validate_resume_config


def make_optimizers(model):
    adam = torch.optim.AdamW([
        dict(params=[model.embed.weight], lr=0.7),
        dict(params=[model.proj.weight], lr=0.004),
        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.015),
    ], betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
    muon = Muon([p for p in model.blocks.parameters() if p.ndim >= 2], lr=0.025, weight_decay=0.05)
    for opt in [adam, muon]:
        for group in opt.param_groups:
            group['initial_lr'] = group['lr']
    return [adam, muon]


def set_hparams(optimizers, step, schedule_steps, cooldown_frac=0.7):
    progress = step / schedule_steps
    assert 0 <= progress < 1
    eta = 1.0 if progress < 1 - cooldown_frac else (1 - progress) / cooldown_frac
    for opt in optimizers:
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * eta


def validate_config(config):
    if config.get('gpt_precision', 'baseline') not in ('baseline', 'fp32'):
        raise ValueError('gpt_precision must be baseline or fp32')
    if config.get('gpt_precision') == 'fp32' and config.get('state_bank'):
        raise ValueError('FP32 prefix capture is not integrated; preserve the existing captured bank')
    if config.get('transformer_chisel'):
        from .chisel_transition import validate_online_config
        validate_online_config(config)
    if config.get('compile_loss', False) and config.get('compile', True):
        raise ValueError('compile_loss requires compile:false; hooks must remain eager')
    if config.get('state_bank'):
        bank=config['state_bank']
        steps=bank.get('steps',1000)
        if not isinstance(steps,int) or not 1<=steps<=min(1000,config['max_steps']):
            raise ValueError('state-bank capture requires 1..1000 steps within the run')
        if config.get('schedule_steps')!=3250:
            raise ValueError('state-bank capture must preserve the explicit original 3250-step schedule')
        if not bank.get('baseline_commit') or not bank.get('directory'):
            raise ValueError('state-bank source commit and destination directory must be pinned')
    if config['max_steps'] < 1:
        raise ValueError('max_steps must be positive')
    if config.get('schedule_steps', config['max_steps']) < config['max_steps']:
        raise ValueError('schedule_steps cannot be smaller than max_steps')
    if config['batch_tokens'] % config['seq_len']:
        raise ValueError('batch must contain whole sequences')
    if not config.get('unscored', False):
        required = dict(batch_tokens=524288, seq_len=1024, val_tokens=10485760,
                        model=dict(vocab_size=50304, num_layers=12, model_dim=768))
        for key, value in required.items():
            if config[key] != value:
                raise ValueError(f'scored protocol requires {key}={value}')


def checkpoint_due(config, completed_steps):
    """Explicit retained actions are saved even between regular checkpoints."""
    every = config.get('checkpoint_every', 250)
    return completed_steps < config['max_steps'] and bool(
        (every and completed_steps % every == 0) or
        completed_steps in config.get('preserve_checkpoint_steps', []))


def iteration_workload(config):
    """Name the measured ordinary GPT workload; never compare to offline panels."""
    fp32=config.get('gpt_precision','baseline')=='fp32'
    modern=bool(config.get('transformer_chisel'))
    mode=config.get('mode','baseline')
    if modern:
        procedure='fp32_observer_identity' if mode=='shadow' else 'fp32_chisel_task_lag_one'
        if config['transformer_chisel'].get('family') == 'adaptive_grouped_lag_one_v1':
            procedure='fp32_adaptive_observer_identity' if mode=='shadow' else 'fp32_adaptive_grouped_task_lag_one'
        replay=config['transformer_chisel'].get('replay_maintenance') or {}
        if mode=='learned' and replay.get('mode','off')!='off':
            procedure='fp32_chisel_task_lag_one_with_past_replay_primal_dual'
    elif mode!='baseline' or config.get('state_bank'):
        procedure='instrumented_fp32_gpt' if fp32 else 'instrumented_original_precision_gpt'
    else:
        procedure='ordinary_fp32_gpt' if fp32 else 'ordinary_original_precision_gpt'
    world=config.get('world_size',1)
    denominator=world*config['seq_len']*config['microbatch_sequences']
    if config['batch_tokens']%denominator:raise ValueError('global batch does not divide configured microbatches')
    return dict(training_procedure=procedure,global_tokens_per_iteration=config['batch_tokens'],
                global_sequences_per_iteration=config['batch_tokens']//config['seq_len'],
                sequence_length=config['seq_len'],microbatch_sequences=config['microbatch_sequences'],
                world_size=world,expected_ordinary_microbatches_per_rank=config['batch_tokens']//denominator,
                iteration_timing_scope='slowest rank synchronized wall interval: data read, GPT passes, reductions, optimizer, observers, actor, edit broadcast and configured diagnostics; excludes validation, checkpoint I/O and timing aggregation')


def source_record(config, output, filename="source_record.txt"):
    """Record all local Python/JSON sources in one reproducible text artifact."""
    root = Path(__file__).resolve().parents[1]
    sources = sorted(p for p in root.rglob('*') if p.suffix in {'.py', '.json'}
                     and not any(part.startswith('.') or part in {'runs', 'results', '__pycache__'}
                                 for part in p.relative_to(root).parts)
                     and output not in p.parents)
    with (output / filename).open('w') as f:
        f.write(json.dumps(config, sort_keys=True, indent=2) + '\n')
        for p in sources:
            f.write(f'\n===== {p.relative_to(root)} SHA256 {hashlib.sha256(p.read_bytes()).hexdigest()} =====\n')
            f.write(p.read_text())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='baseline/config.json')
    parser.add_argument('--mode', choices=['baseline', 'shadow', 'learned'], default='baseline')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', help='checkpoint file, or directory containing latest_rankN.pt')
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    if config.get('deterministic', False):
        from .execution import enable_deterministic_execution
        enable_deterministic_execution()
    if config.get('gpt_precision') == 'fp32':
        from rewriter.gpt_precision import configure_precision
        configure_precision('fp32')
        if args.mode != 'baseline' and not config.get('transformer_chisel'):
            raise ValueError('the legacy online rewriter is not the FP32 TransformerChisel adapter')
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world not in (1, 2, 4, 8):
        raise ValueError('PR323 supports 1,2,4,8 ranks')
    if args.mode != 'baseline' and world != 1 and not config.get('transformer_chisel'):
        raise ValueError('rewriter distributed synchronization is not yet certified; use one GPU')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = dict(config, mode=args.mode, world_size=world)
    if config.get('transformer_chisel', {}).get('family') == 'adaptive_grouped_lag_one_v1':
        from .adaptive_transition import validate_adaptive_config
        validate_adaptive_config(config)
        if world > 1 and dist.get_backend() != 'gloo':
            raise ValueError('adaptive distributed staging is CPU/gloo only; NCCL remains unqualified')
    checkpoint = None
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            resume_path = resume_path / f'latest_rank{rank}.pt'
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        validate_resume_config(checkpoint['config'], config)
    if rank == 0:
        source_record(config, output, f"source_record_resume_step{checkpoint['step']}.txt" if checkpoint else 'source_record.txt')
        (output / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
        manifest = data_manifest(config, verify=not config.get('unscored', False))
        (output / 'data_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        from .execution import execution_environment
        env = dict(execution=execution_environment(), torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                   python=platform.python_version(), readout='live', gradient_units='token_sum',
                   compilation_scope='pure_PR323_kernels_with_eager_sensors' if config.get('state_bank') and config.get('compile',True) else 'baseline')
        from rewriter.gpt_precision import precision_metadata
        env['gpt_precision'] = precision_metadata(config.get('gpt_precision', 'baseline'))
        if config.get('gpt_precision') == 'fp32':
            env['compilation_scope'] = ('full_FP32_GPT' if config.get('compile', True)
                                        else 'FP32_pure_loss' if config.get('compile_loss', False)
                                        else 'eager_FP32_GPT')
        (output / 'environment.json').write_text(json.dumps(env, indent=2)+'\n')
    wandb_run = None
    if rank == 0 and config.get('wandb', False):
        import wandb
        wandb_run = wandb.init(project='learnable-weight-decay', mode='offline', dir=str(output), config=config)
    logfile = (output / 'metrics.jsonl').open('a' if checkpoint else 'w') if rank == 0 else None
    def log(record):
        if rank == 0:
            line = json.dumps(record, allow_nan=False)
            print(line, flush=True)
            logfile.write(line+'\n'); logfile.flush()
            if wandb_run:
                wandb_run.log(record)

    model = GPT(**config['model']).to(device)
    if config.get('state_bank') and config.get('compile',True):
        from .capture_execution import enable_capture_execution
        enable_capture_execution(model)
    elif config.get('compile_loss', False) and config.get('gpt_precision') != 'fp32':
        from .execution import enable_compiled_loss
        enable_compiled_loss(model)
    if config.get('compile', True) and not config.get('state_bank') and config.get('gpt_precision') != 'fp32':
        # Compilation occurs on prescribed passes; there is no synthetic training warmup.
        model.compile(dynamic=False)
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    initialize_parameters_like_record(model)
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)
    if config.get('gpt_precision') == 'fp32':
        # Initialize in the original stored dtypes first, so promotion does not
        # alter the embedding's random initialization or the baseline RNG stream.
        from .fp32_execution import enable_fp32_execution
        model.float()
        enable_fp32_execution(model, compile_loss=config.get('compile_loss', False))
        if config.get('compile', True):
            model.compile(dynamic=False)
    optimizers = make_optimizers(model)
    adapter = None
    capture = None
    if config.get('state_bank'):
        if args.mode != 'baseline' or args.resume:
            raise ValueError('state-bank capture requires a fresh baseline run')
        from rewriter.state_capture import StateCaptureAdapter
        bank=config['state_bank']
        provenance=dict(baseline_commit=bank['baseline_commit'],model_config=config['model'],
            dataset=json.loads((output/'data_manifest.json').read_text()),seed=config['seed'],
            schedule_steps=config.get('schedule_steps',config['max_steps']),capture_world_size=world,
            sensor_config=bank,training_config=config,editable_scope='grouped',
            capture_code_sha256=hashlib.sha256(Path(__file__).resolve().parents[1].joinpath('rewriter/state_capture.py').read_bytes()).hexdigest())
        capture=StateCaptureAdapter(model,optimizers,config,directory=bank['directory'],provenance=provenance)
    if config.get('transformer_chisel'):
        if checkpoint is None:raise ValueError('modern Chisel requires an explicit prefix transition or resume')
        if config['transformer_chisel'].get('family') == 'adaptive_grouped_lag_one_v1':
            from rewriter.adaptive_adapter import AdaptiveChiselAdapter
            adapter = AdaptiveChiselAdapter(model, optimizers, config, mode=args.mode)
        else:
            from rewriter.transformer_online import TransformerChiselAdapter
            adapter = TransformerChiselAdapter(model, optimizers, config, mode=args.mode)
        model._fp32_sensors = adapter.collector
    elif args.mode != 'baseline':
        from rewriter.integration import RewriterAdapter
        adapter = RewriterAdapter(model, optimizers, config=config, mode=args.mode)
    mbs, seq_len = config['microbatch_sequences'], config['seq_len']
    val_inputs, val_targets = next(distributed_data_generator(config['val_data'], config['val_tokens'], seq_len))
    train_loader = distributed_data_generator(config['train_data'], config['batch_tokens'], seq_len)
    total_forward = total_backward = 0
    start_step, elapsed_offset = 0, 0.
    if checkpoint is not None:
        start_step, counters, elapsed_offset = restore_checkpoint(checkpoint, model, optimizers, adapter, train_loader, config)
        clock = torch.tensor([start_step, -start_step], device=device)
        dist.all_reduce(clock, op=dist.ReduceOp.MAX)
        if int(clock[0]) != -int(clock[1]):
            raise ValueError('rank checkpoints have different training clocks')
        total_forward, total_backward = counters['forward'], counters['backward']
        from .fp32_transition import training_forward_count
        expected = training_forward_count(config,start_step)
        if total_forward != expected or total_backward != expected:
            raise ValueError('checkpoint training-computation counts disagree with step')
        log(dict(event='resume', step=start_step, checkpoint=str(resume_path), elapsed_seconds=elapsed_offset))
        del checkpoint
    num_parameters = sum(p.numel() for p in model.parameters())
    log(dict(event='start', parameters=num_parameters,
             matrix_parameters=sum(p.numel() for p in model.parameters() if p.ndim == 2),
             matrix_count=sum(p.ndim == 2 for p in model.parameters()),
             interface_coordinates=sum(sum(p.shape) for p in model.parameters() if p.ndim == 2)))
    started = time.perf_counter()
    def elapsed():
        return elapsed_offset + time.perf_counter() - started
    def log_weight_norms(completed_steps):
        if config.get('transformer_chisel',{}).get('checkpoint_group_norm_logging',False):
            if rank == 0:
                from rewriter.transformer_online import checkpoint_group_norms
                log(dict(event='weight_group_norms',step=completed_steps,**checkpoint_group_norms(dict(model.named_parameters()))))
            # Keep checkpoint/readout skew outside the next timed iteration.
            dist.barrier()
    def checkpoint_at(completed_steps, filename):
        state = build_checkpoint(model, optimizers, adapter, train_loader, config, completed_steps,
                                 dict(forward=total_forward, backward=total_backward), elapsed())
        save_atomic(state, output / filename)
        log_weight_norms(completed_steps)
    fingerprint_steps = set(config.get('fingerprint_steps', []))
    def fingerprint_at(step):
        if step in fingerprint_steps:
            from .fingerprint import model_fingerprint
            fingerprint = model_fingerprint(model)
            log(dict(event='fingerprint', step=step, rank=rank, **fingerprint))
    fingerprint_at(start_step)  # initialized/restored state, before validation
    log_weight_norms(start_step)
    workload = iteration_workload(config)
    for step in range(start_step, config['max_steps']+1):
        # A captured prefix keeps the original validation cadence as well as
        # the original LR schedule; shortening capture must not invent tail evaluations.
        evaluation_horizon = config.get('schedule_steps', config['max_steps']) if capture else config['max_steps']
        every = config.get('val_every', 125) if step / evaluation_horizon < 0.9 else config.get('final_val_every', 25)
        if step == config['max_steps'] or (every > 0 and step % every == 0):
            model.eval()
            val_loss = torch.zeros((), device=device)
            with torch.no_grad():
                if len(val_inputs) % mbs:
                    raise ValueError('validation sequences not divisible by microbatch')
                for i in range(0, len(val_inputs), mbs):
                    val_loss += model(val_inputs[i:i+mbs], val_targets[i:i+mbs])
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            log(dict(event='validation', step=step, val_loss=float(val_loss/config['val_tokens']),
                     objective=float(val_loss/config['val_tokens']), elapsed_seconds=elapsed()))
            model.train()
        if step == config['max_steps']:
            break
        torch.cuda.synchronize(device)
        iteration_started = time.perf_counter()
        forwards_before, backwards_before = total_forward, total_backward
        set_hparams(optimizers, step, config.get('schedule_steps', config['max_steps']))
        if adapter:
            adapter.before_step(step, model, optimizers)
        if capture:
            capture.before_step(step, model, optimizers)
        inputs, targets = next(train_loader)
        if len(inputs) % mbs:
            raise ValueError('training sequences not divisible by microbatch')
        train_loss = torch.zeros((), device=device)
        for i in range(0, len(inputs), mbs):
            # IDs are globally unique within a step, identical at every layer/use.
            ids = torch.arange(rank * len(inputs)+i, rank * len(inputs)+i+mbs, device=device)
            if adapter:
                adapter.before_microbatch(step, ids, inputs[i:i+mbs])
            if capture:
                capture.before_microbatch(step, ids, inputs[i:i+mbs])
            loss = model(inputs[i:i+mbs], targets[i:i+mbs])
            train_loss += loss.detach()
            loss.backward()
            total_forward += 1; total_backward += 1
        for name, p in model.named_parameters():
            if p.grad is None:
                raise RuntimeError(f'missing gradient: {name}')
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        if adapter:
            adapter.after_backward(step, model)
        if capture:
            capture.after_backward(step, model)
        for opt in optimizers:
            opt.step()
        if adapter:
            adapter.after_step(step, model, optimizers)
        if capture:
            capture.after_step(step, model, optimizers)
        model.zero_grad(set_to_none=True)
        fingerprint_at(step+1)  # complete base step and optional committed rewrite
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        iteration_seconds = torch.tensor(time.perf_counter()-iteration_started,device=device,dtype=torch.float64)
        dist.all_reduce(iteration_seconds,op=dist.ReduceOp.MAX)
        iteration_forwards=total_forward-forwards_before
        iteration_backwards=total_backward-backwards_before
        if iteration_forwards!=workload['expected_ordinary_microbatches_per_rank'] or iteration_backwards!=iteration_forwards:
            raise RuntimeError('actual ordinary GPT pass counts differ from declared workload')
        record = dict(event='train', step=step+1, train_loss=float(train_loss/config['batch_tokens']),
                      elapsed_seconds=elapsed(), training_forwards_per_rank=total_forward,
                      training_backwards_per_rank=total_backward, peak_memory_bytes=torch.cuda.max_memory_allocated(),
                      iteration_wall_seconds_max_rank=float(iteration_seconds),
                      ordinary_training_forwards_per_rank_this_iteration=iteration_forwards,
                      ordinary_training_backwards_per_rank_this_iteration=iteration_backwards,**workload)
        if adapter:
            record.update(adapter.metrics)
        if capture:
            record.update(capture.metrics)
        if config.get('log_compile_counters',False):
            record['compile_counters']={str(k):dict(v) for k,v in torch._dynamo.utils.counters.items() if v}
        log(record)
        if checkpoint_due(config, step+1):
            checkpoint_at(step+1, f'latest_rank{rank}.pt')
            log(dict(event='checkpoint', step=step+1, path=f'latest_rank{rank}.pt'))
            if step+1 in config.get('preserve_checkpoint_steps', []):
                import shutil
                retained = Path(config.get('preserved_checkpoint_dir', str(output / 'retained'))) / f'step{step+1}'
                retained.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(output / f'latest_rank{rank}.pt', retained / f'rank{rank}.pt')
                dist.barrier()
                if rank == 0:
                    (retained / 'manifest.json').write_text(json.dumps(dict(step=step+1, world_size=world,
                        ranks=[f'rank{r}.pt' for r in range(world)], config=config), indent=2)+'\n')
                    import subprocess
                    subprocess.run(['sync', str(retained)], check=True)
                dist.barrier()
                log(dict(event='checkpoint_preserved', step=step+1, directory=str(retained), world_size=world))
    if capture:
        capture.close()
    if config.get('transformer_chisel'):
        adapter.close()
    # The terminal checkpoint has consumed previous task credit and applied identity.
    checkpoint_at(config['max_steps'], f'final_rank{rank}.pt')
    log(dict(event='finished', steps=config['max_steps'], training_forwards_per_rank=total_forward,
             training_backwards_per_rank=total_backward, elapsed_seconds=elapsed()))
    if wandb_run:
        wandb_run.finish()
    if logfile:
        logfile.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
