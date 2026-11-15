"""Isolated exact-forward/compensated-backward qualification; never changes a fit.

Reports raw KL actor gradients, disposable Adam update responses, warm complete
checkpoint-panel timings, and profiler data. This is explicitly a hybrid backend.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution, execution_environment
from baseline.model import GPT
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.chisel_factory import build_chisel
from rewriter.compensated_dx import experimental_dx_backend
from rewriter.edit_scope import editable_names, editable_weight_norm_sq, scoped_snapshot
from rewriter.function_preservation import FunctionalPreservation, _ChunkedTeacherKL
from rewriter.meta_optimizer import build_meta_adam
from rewriter.primal_dual import PrimalDual
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from scripts.benchmark_factorized_fp32 import thin_step
from scripts.train_multistate_chisel import build_proposal, mean_kl_and_backward


class UnitKL:
    def terms(self, kl, state_id=None):
        return dict(kl_penalty=float(kl), kl_coefficient=1.)


def comparison(actual, reference):
    a, b = actual.double().flatten(), reference.double().flatten()
    an, bn = a.norm(), b.norm()
    cosine=1. if an==0 and bn==0 else float(torch.dot(a,b)/(an*bn).clamp_min(1e-30))
    return dict(relative_error=float((a-b).norm()/bn.clamp_min(1e-30)),
                absolute_l2_error=float((a-b).norm()),
                cosine=cosine,
                max_absolute_error=float((a-b).abs().max()),
                reference_norm=float(bn), candidate_norm=float(an))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('state-bank', 'teacher-cache', 'policy', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--state-id', type=int, default=999)
    parser.add_argument('--sequences', type=int, choices=(16,32,64), default=16)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--adam-lr-scale', type=float, default=.1)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or not 0 < args.adam_lr_scale <= 1:
        parser.error('require1..10 repeats and0<Adam LR scale<=1')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    enable_deterministic_execution()
    device = torch.device('cuda')
    reader = StateBankReader(args.state_bank, verify_once=True, mmap_model=True)
    if args.state_id not in reader.manifest['split']['train']:
        raise ValueError('qualification must use a fitting state')
    cache = TeacherCacheReader(args.teacher_cache, verify_once=True, mmap=True)
    record = reader.read(args.state_id, model_device='cpu', device=device)
    engine = FunctionalPreservation(
        GPT(**reader.manifest['provenance']['model_config']).to(device).requires_grad_(False),
        microbatch_sequences=16, teacher_cache_device=device, kl_dtype=torch.float64,
        gpt_precision='fp32')
    cache.load_engine(engine, record)
    grouped = AcceleratedGroupedNormObjective(engine.parameters, zero_reference_policy='freeze')
    editable = editable_names(engine.parameters, 'grouped') - grouped.frozen_parameter_names
    snapshot = scoped_snapshot(record.snapshot, editable_weight_norm_sq(engine.parameters, editable), .01, scope='grouped')
    saved = torch.load(args.policy, map_location='cpu', weights_only=False)
    actor = build_chisel(saved['metadata']['architecture']).to(device)
    actor.load_state_dict(saved['controller'])
    forward = torch.compile(actor, fullgraph=True, dynamic=False)
    options = saved['metadata']['options']
    optimizer = build_meta_adam(actor, lr=options['lr'],
                               representation_lr_multiplier=options.get('representation_lr_multiplier',1.))
    panel = cache.read(args.state_id, 'train')
    inputs = panel.inputs[:args.sequences]
    dual_state = saved['primal_dual']
    dual = PrimalDual(dual_state['multipliers'], tolerance=dual_state['tolerance'],
                     rho=dual_state['rho'], dual_lr=dual_state['dual_lr'])
    dual.load_state_dict(dual_state)
    parameters = list(actor.parameters())
    import wandb
    run = wandb.init(project='chisel', mode='offline', dir=str(args.output), config={
        'backend':'experimental_hybrid_tf32x3_dx', 'state_id':args.state_id,
        'sequences':args.sequences, 'microbatch':16, 'production_changed':False})
    stream = (args.output/'metrics.jsonl').open('w')
    def log(row):
        stream.write(json.dumps(row)+'\n'); stream.flush()
        print(json.dumps(row), flush=True)
        run.log({k:v for k,v in row.items() if isinstance(v,(int,float,str,bool))})
    def vector():
        return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().flatten()
                          for p in parameters]).cpu()
    def step(backend, raw=False, gradients=False, update=False):
        actor.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        constraint = UnitKL() if raw else dual
        with experimental_dx_backend('tf32x3' if backend=='thin_tf32x3' else 'ieee'):
            if backend == 'dense':
                edited, regularizer, _, _, credit = build_proposal(forward,snapshot,engine,grouped,radius_fraction=.01)
                result = mean_kl_and_backward(engine,edited,regularizer*(0 if raw else 1),inputs,
                    kl_budget=dual.tolerance,actor_credit=credit,
                    teacher_provider=lambda a,b:panel.teacher_batch(engine,a,b),
                    primal_dual=constraint,state_id=args.state_id)
            else:
                result = thin_step(forward,snapshot,engine,grouped,inputs,panel,constraint,
                                   args.state_id,norm_enabled=not raw,dense_forward=True)
        if update:
            for p in parameters:
                if p.grad is not None: p.grad.div_(200.)
            torch.nn.utils.clip_grad_norm_(parameters,1.)
            optimizer.step()
        torch.cuda.synchronize()
        row = dict(event='step',backend=backend,raw_kl=raw,seconds=time.perf_counter()-started,
                   peak_bytes=torch.cuda.max_memory_allocated(),local_adam=update,**result)
        return row, vector() if gradients else None
    raw = {}; full = {}; rows=[]
    for backend in ('dense','thin_ieee','thin_tf32x3'):
        row, grad = step(backend,raw=True,gradients=True);log(row)
        raw[backend]=(row,grad)
        row, grad = step(backend,gradients=True);log(row)
        full[backend]=(row,grad)
    qualification = {}
    for backend in ('thin_ieee','thin_tf32x3'):
        metric = comparison(raw[backend][1],raw['dense'][1])
        metric['absolute_kl_difference']=abs(raw[backend][0]['mean_kl']-raw['dense'][0]['mean_kl'])
        metric['passed']=metric['relative_error']<.003 and metric['cosine']>.99999 and metric['absolute_kl_difference']<1e-9
        qualification[backend]=metric
        log(dict(event='raw_kl_gradient',backend=backend,**metric))
    # Use the same saved Adam moments and a reduced LR for a disposable one-state
    # contribution. This is not the full distributed200-checkpoint update.
    def logits():
        with torch.enable_grad():
            proposal=build_proposal(forward,snapshot,engine,grouped,radius_fraction=.01)
            edited={n:v.detach() for n,v in proposal[0].items()}
            del proposal
        with torch.no_grad():
            output=engine._forward(edited,inputs[:16].to(device))
            movement={n:float((edited[n]-engine.parameters[n]).norm()/engine.parameters[n].norm().clamp_min(1e-30))
                      for n in editable if edited[n].ndim==2}
        return output,movement
    actor.zero_grad(set_to_none=True)
    before, base_movement = logits()
    teacher = panel.teacher_batch(engine,0,16)
    with torch.no_grad():
        identity_kl=float(engine.student_parameter_loss(engine.parameters,teacher))
        initial_response_kl=float(_ChunkedTeacherKL.apply(before,teacher.log_probs.to(device),engine.chunk_tokens))
    del teacher
    log(dict(event='identity',identity_kl=identity_kl,policy_kl_first16=initial_response_kl))
    response = None; thin_response=None; adam={}; reference_update=None
    for backend in ('dense','dense_repeat','thin_ieee','thin_tf32x3'):
        actor.load_state_dict(saved['controller'])
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        for group in optimizer.param_groups:group['lr']*=args.adam_lr_scale
        cursor=0
        for p in parameters:
            source='dense' if backend=='dense_repeat' else backend
            p.grad=full[source][1][cursor:cursor+p.numel()].reshape_as(p).to(device)/200.
            cursor+=p.numel()
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.))
        optimizer.step();actor.zero_grad(set_to_none=True)
        after,movement=logits()
        change=after-before
        teacher=panel.teacher_batch(engine,0,16)
        with torch.no_grad():
            post_kl=float(_ChunkedTeacherKL.apply(after,teacher.log_probs.to(device),engine.chunk_tokens))
        del teacher
        update=torch.cat([(p.detach().cpu()-saved['controller'][name]).flatten()
                          for name,p in actor.named_parameters()])
        if backend=='dense':response=change;reference_update=update
        metric=dict(response=comparison(change,response),actor_update=comparison(update,reference_update),
                    preclip_gradient_norm=norm,kl_before=initial_response_kl,kl_after=post_kl,
                    kl_change=post_kl-initial_response_kl)
        if backend=='thin_ieee':thin_response=change.clone()
        if backend=='thin_tf32x3':metric['response_vs_thin_ieee']=comparison(change,thin_response)
        adam[backend]=metric
        log(dict(event='disposable_adam_response',backend=backend,**metric))
        del after,change
    del before,response,thin_response
    actor.load_state_dict(saved['controller']);actor.zero_grad(set_to_none=True)
    # Restore the exact initial actor/optimizer for each timed local update.
    for repeat in range(args.repeats+1):
        backends=('dense','thin_ieee','thin_tf32x3')
        if repeat%2:backends=tuple(reversed(backends))
        for backend in backends:
            actor.load_state_dict(saved['controller'])
            optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
            for group in optimizer.param_groups:group['lr']*=args.adam_lr_scale
            row,_=step(backend,update=True)
            row.update(event='timing',repeat=repeat,warmup=repeat==0)
            log(row)
            if repeat:rows.append(row)
    if args.profile:
        for backend in ('thin_ieee','thin_tf32x3'):
            actor.load_state_dict(saved['controller'])
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                step(backend)
            (args.output/(backend+'-profile.txt')).write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=50))
            prof.export_chrome_trace(str(args.output/(backend+'-trace.json')))
    import triton
    summary=dict(scope='one real GPT checkpoint x fixed panel; micro16; exactFP32 forward; hybrid backward; local Adam on1/200 gradient; no distributed reduction',
        qualification=qualification,adam_response=adam,adam_lr_scale=args.adam_lr_scale,response_actor_grad_enabled=True,
        base_matrix_relative_edits=base_movement,
        identity_kl=identity_kl,policy_kl_first16=initial_response_kl,
        median_seconds={b:statistics.median(r['seconds'] for r in rows if r['backend']==b)
                        for b in ('dense','thin_ieee','thin_tf32x3')},rows=rows,
        state_id=args.state_id,sequences=args.sequences,sequence_ids=panel.sequence_ids[:args.sequences],
        policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),state_sha256=record.assert_verified(),
        torch_version=torch.__version__,triton_version=triton.__version__,environment=execution_environment(),
        production_changed=False,single_launch=False,
        compilation_counters={k:dict(v) for k,v in torch._dynamo.utils.counters.items()})
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    log(dict(event='finished',**summary['median_seconds']))
    run.finish();stream.close()


if __name__=='__main__':
    main()
