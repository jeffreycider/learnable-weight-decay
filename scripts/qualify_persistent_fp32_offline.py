"""Real-policy literal FP32 offline raw-KL gradient qualification.

One checkpoint panel and one cooperative launch; no production update. Targets
are the unchanged FP32 cached teacher. Both paths use resident FP64 probabilities.
"""
from pathlib import Path
import argparse,copy,hashlib,json,statistics,sys,time,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from baseline.model import GPT
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.chisel_factory import build_chisel
from rewriter.edit_scope import editable_names,editable_weight_norm_sq,scoped_snapshot
from rewriter.function_preservation import FunctionalPreservation,TeacherBatch
from rewriter.persistent_offline import PersistentOfflineTrainer,export_torch_adam
from rewriter.meta_optimizer import build_meta_adam
from rewriter.primal_dual import PrimalDual
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from scripts.train_multistate_chisel import build_proposal,mean_kl_and_backward
from scripts.qualify_compensated_dx import UnitKL,comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('state-bank','teacher-cache','policy','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--state-id',type=int,default=999)
    parser.add_argument('--sequences',type=int,choices=(1,16,64),default=16)
    parser.add_argument('--head-chunk',type=int,default=256)
    parser.add_argument('--microbatch',type=int,default=16)
    parser.add_argument('--compensated-backward',action='store_true')
    parser.add_argument('--thin-backward',action='store_true')
    parser.add_argument('--state-weight',type=float,default=1/200)
    parser.add_argument('--full-objective',action='store_true')
    parser.add_argument('--actor-vjp-diagnostic',action='store_true',help='read-only actor VJP with the same literal action cotangents; excludes Adam')
    parser.add_argument('--adam',action='store_true',help='disposable local saved-Adam update, requires full objective')
    parser.add_argument('--adam-lr-scale',type=float,default=1.)
    parser.add_argument('--register-sweep',default='',help='comma-separated NVRTC register caps; reuse identical plan buffers')
    parser.add_argument('--policy-scale',type=float,default=1.,help='fixed output-head amplitude for numerical qualification only')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    microbatch=min(args.sequences,args.microbatch)
    if microbatch<=0 or args.sequences%microbatch:parser.error('panel must contain complete microbatches')
    if args.actor_vjp_diagnostic and args.adam:parser.error('actor VJP diagnostic requires no Adam')
    if args.adam and not args.full_objective:parser.error('Adam qualification requires full objective')
    if not 0<args.adam_lr_scale<=1:parser.error('Adam LR scale must be in (0,1]')
    if not 0<=args.policy_scale<100 or (args.adam and args.policy_scale!=1):parser.error('scaled-policy fixtures require no Adam and finite scale in[0,100)')
    torch.set_num_threads(1);enable_deterministic_execution()
    reader=StateBankReader(args.state_bank,verify_once=True,mmap_model=True)
    if args.state_id not in reader.manifest['split']['train']:raise ValueError('fitting states only')
    record=reader.read(args.state_id,model_device='cpu',device='cuda')
    cache=TeacherCacheReader(args.teacher_cache,verify_once=True,mmap=True)
    engine=FunctionalPreservation(GPT(**reader.manifest['provenance']['model_config']).cuda().requires_grad_(False),
        gpt_precision='fp32',microbatch_sequences=microbatch,kl_dtype=torch.float64,teacher_cache_device='cuda')
    cache.load_engine(engine,record);model=engine.view.gpt.float()
    model.load_state_dict({**engine.parameters,**engine.buffers})
    grouped=AcceleratedGroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
    editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
    snapshot=scoped_snapshot(record.snapshot,editable_weight_norm_sq(engine.parameters,editable),.01,scope='grouped')
    saved=torch.load(args.policy,map_location='cuda',weights_only=False)
    actor=build_chisel(saved['metadata']['architecture']).cuda();actor.load_state_dict(saved['controller'])
    with torch.no_grad():
        for head in actor.output_heads:
            for p in head.parameters():p.mul_(args.policy_scale)
    actual_actor=copy.deepcopy(actor)
    forward=torch.compile(actor,fullgraph=True,dynamic=False)
    dual=UnitKL();optimizer=None;optimizer_state=None;rates=None;betas=(.9,.99);eps=1e-8
    if args.full_objective:
        ds=saved['primal_dual']
        dual=PrimalDual(ds['multipliers'],tolerance=ds['tolerance'],rho=ds['rho'],dual_lr=ds['dual_lr'])
        dual.load_state_dict(ds)
    if args.adam:
        options=saved['metadata']['options']
        optimizer=build_meta_adam(actor,lr=options['lr'],representation_lr_multiplier=options.get('representation_lr_multiplier',1.))
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        for group in optimizer.param_groups:group['lr']*=args.adam_lr_scale
        optimizer_state,rates,betas,eps=export_torch_adam(actor,optimizer)
        initial_optimizer=copy.deepcopy(optimizer.state_dict())
    def restore_reference():
        actor.load_state_dict(saved['controller'])
        if optimizer is not None:optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
    panel=cache.read(args.state_id,'train');inputs=panel.inputs[:args.sequences].cuda()
    teacher=torch.empty((args.sequences,inputs.shape[1],engine.parameters['proj.weight'].shape[0]),device='cuda',dtype=torch.float64)
    # Reconstruct the immutable labels at their original generation shape,
    # then slice resident probabilities for independently chosen student batches.
    generation_batch=panel.metadata['microbatch_sequences']
    engine.microbatch_sequences=generation_batch
    for start in range(0,args.sequences,generation_batch):
        batch=panel.teacher_batch(engine,start,start+generation_batch)
        count=min(generation_batch,args.sequences-start);teacher[start:start+count].copy_(batch.log_probs[:count]);del batch
    engine.microbatch_sequences=microbatch
    def teacher_batch(start,stop):
        return TeacherBatch(inputs[start:stop],teacher[start:stop],engine.teacher_id,'past_training',panel.sequence_ids[start:stop])
    import wandb
    run=wandb.init(project='chisel',mode='offline',dir=str(args.output),config={'backend':'literal_fp32_offline','sequences':args.sequences,'state_id':args.state_id})
    stream=(args.output/'metrics.jsonl').open('w')
    def log(row):
        print(json.dumps(row),flush=True);stream.write(json.dumps(row)+'\n');stream.flush()
        run.log({k:v for k,v in row.items() if isinstance(v,(int,float,str,bool))})
    def reference():
        actor.zero_grad(set_to_none=True)
        edited,regularizer,_,_,credit=build_proposal(forward,snapshot,engine,grouped,radius_fraction=.01)
        result=mean_kl_and_backward(engine,edited,regularizer if args.full_objective else regularizer*0,inputs,
            kl_budget=dual.tolerance if args.full_objective else 1e-6,
            state_weight=args.state_weight,actor_credit=credit,teacher_provider=teacher_batch,primal_dual=dual,state_id=args.state_id)
        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
            optimizer.step()
        return result,edited
    def response():
        edited,_,_,_,_=build_proposal(forward,snapshot,engine,grouped,radius_fraction=.01)
        detached={n:p.detach() for n,p in edited.items()};del edited
        with torch.no_grad():
            logits=engine._forward(detached,inputs[:1])
            from rewriter.function_preservation import _ChunkedTeacherKL
            kl=float(_ChunkedTeacherKL.apply(logits,teacher[:1],engine.chunk_tokens))
        return logits.cpu(),kl
    before_response,before_kl=response() if args.adam else (None,None)
    result,edited=reference();torch.cuda.synchronize()
    reference_gradient=torch.cat([p.grad.flatten() for p in actor.parameters()]).cpu()
    reference_edits={n:value.detach().cpu() for n,value in edited.items() if n in editable}
    reference_update=torch.cat([(p.detach()-saved['controller'][n]).flatten() for n,p in actor.named_parameters()]).cpu() if args.adam else None
    reference_response,reference_after_kl=response() if args.adam else (None,None)
    reference_times=[]
    for _ in range(3):
        if args.adam:restore_reference()
        torch.cuda.synchronize();start=time.perf_counter();result,edited=reference();torch.cuda.synchronize()
        reference_times.append(time.perf_counter()-start)
    del edited;actor.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    started=time.perf_counter()
    program=PersistentOfflineTrainer(actual_actor,model,snapshot,inputs,teacher,gpt_precision='fp32',
        raw_kl_only=not args.full_objective,trust_weight=0.,apply_optimizer=args.adam,
        kl_budget=dual.tolerance if args.full_objective else 1e-6,state_weight=args.state_weight,
        kl_multiplier=dual.multipliers[args.state_id] if args.full_objective else None,
        augmented_rho=dual.rho if args.full_objective else .01,
        optimizer_state=optimizer_state,parameter_lrs=rates,betas=betas,eps=eps,
        microbatch_sequences=microbatch,head_chunk_tokens=args.head_chunk,
        compensated_backward=args.compensated_backward,thin_backward=args.thin_backward)
    torch.cuda.synchronize();setup=time.perf_counter()-started
    log(dict(event='setup',seconds=setup,allocated_bytes=torch.cuda.memory_allocated(),stages=len(program.plan.instructions)))
    program.step();torch.cuda.synchronize()
    actual_gradient=torch.cat([program.gradients[n].flatten() for n,_ in actual_actor.named_parameters()]).cpu()
    grad=comparison(actual_gradient,reference_gradient)
    edited_metric=comparison(torch.cat([program.parameters[n].flatten() for n in sorted(editable)]).cpu(),
                             torch.cat([reference_edits[n].flatten() for n in sorted(editable)]))
    literal_kl=float(program.kl);absolute_kl_error=abs(literal_kl-result['mean_kl'])
    log(dict(event='qualification',reference_kl=result['mean_kl'],literal_kl=literal_kl,
             absolute_kl_error=absolute_kl_error,gradient=grad,edited_parameters=edited_metric))
    adam_response=None
    if args.adam:
        actual_update=torch.cat([(p.detach()-saved['controller'][n]).flatten() for n,p in actual_actor.named_parameters()]).cpu()
        actual_updated_state={n:p.detach().cpu().clone() for n,p in actual_actor.state_dict().items()}
    def restore_literal():
        if not args.adam:return
        actual_actor.load_state_dict(saved['controller'])
        # Restore outside the timed region; each measured update has the same
        # policy, group learning rates and historical Adam moments.
        restore_reference()
        initial,_,_,_=export_torch_adam(actor,optimizer)
        state=program.optimizer_state();state['step'].copy_(initial['step'])
        for key in ('exp_avg','exp_avg_sq'):
            for target,source in zip(state[key],initial[key]):target.copy_(source)
    literal_times=[]
    for _ in range(3):
        restore_literal()
        torch.cuda.synchronize();start=time.perf_counter();program.step();torch.cuda.synchronize()
        literal_times.append(time.perf_counter()-start)
    ticks=program.plan.clocks.cpu().tolist();stages=[];by_opcode={}
    for index,instruction in enumerate(program.plan.instructions):
        elapsed=ticks[index+1]-ticks[index];op=str(instruction[0]);by_opcode[op]=by_opcode.get(op,0)+elapsed
        stages.append(dict(index=index,opcode=int(op),ticks=elapsed,dimensions=instruction[6:9]))
    (args.output/'stage-profile.json').write_text(json.dumps(sorted(stages,key=lambda r:-r['ticks']),indent=2)+'\n')
    restore_literal();torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        program.step();torch.cuda.synchronize()
    trace=args.output/'trace.json';prof.export_chrome_trace(str(trace))
    launches=[e['name'] for e in json.loads(trace.read_text())['traceEvents'] if e.get('cat')=='kernel']
    register_sweep=[]
    if args.register_sweep:
        from rewriter.persistent_mlp import CooperativeTrainingKernel
        original_kernel=program.plan.kernel
        original_cap=os.environ.get('CHISEL_PERSISTENT_MAX_REGISTERS')
        for cap in map(int,args.register_sweep.split(',')):
            if not 32<=cap<=255:raise ValueError('register cap must be32..255')
            os.environ['CHISEL_PERSISTENT_MAX_REGISTERS']=str(cap)
            start=time.perf_counter();candidate=CooperativeTrainingKernel();compile_seconds=time.perf_counter()-start
            program.plan.kernel=candidate;restore_literal();program.step();torch.cuda.synchronize()
            observed=torch.cat([program.gradients[n].flatten() for n,_ in actual_actor.named_parameters()]).cpu()
            samples=[]
            for _ in range(3):
                restore_literal();torch.cuda.synchronize();start=time.perf_counter();program.step();torch.cuda.synchronize()
                samples.append(time.perf_counter()-start)
            row=dict(cap=cap,registers=candidate.registers,blocks=candidate.blocks,compile_seconds=compile_seconds,
                     seconds=statistics.median(samples),gradient=comparison(observed,reference_gradient),kl=float(program.kl))
            register_sweep.append(row);log(dict(event='register_sweep',**row))
        program.plan.kernel=original_kernel
        if original_cap is None:os.environ.pop('CHISEL_PERSISTENT_MAX_REGISTERS',None)
        else:os.environ['CHISEL_PERSISTENT_MAX_REGISTERS']=original_cap
    actor_diagnostic=None
    if args.actor_vjp_diagnostic:
        actor.zero_grad(set_to_none=True)
        native_action=forward(snapshot)
        def leaves(action):
            return ([v for name in sorted(action.matrices) for v in (action.matrices[name].P,action.matrices[name].Q)]
                    +[action.vectors[name] for name in sorted(action.vectors)])
        native_leaves=leaves(native_action);literal_leaves=leaves(program.action)
        action_comparison=comparison(torch.cat([v.detach().flatten() for v in literal_leaves]).cpu(),
                                     torch.cat([v.detach().flatten() for v in native_leaves]).cpu())
        torch.autograd.backward(native_leaves,[v.detach() for v in leaves(program.action_cotangents)])
        torch.cuda.synchronize()
        native_vjp=torch.cat([p.grad.flatten() for p in actor.parameters()]).cpu()
        actor_diagnostic=dict(action_values=action_comparison,
            literal_actor_vs_native_actor_same_cotangent=comparison(actual_gradient,native_vjp),
            native_actor_literal_cotangent_vs_full_reference=comparison(native_vjp,reference_gradient),
            scope='Native compiled actor VJP receives exactly the literal action cotangent. Diagnostic is outside timings and is not a single-launch training path.')
    summary=dict(actor_vjp_diagnostic=actor_diagnostic,reference_kl=result['mean_kl'],literal_kl=literal_kl,absolute_kl_error=absolute_kl_error,
        actor_gradient=grad,raw_kl_actor_gradient=None if args.full_objective else grad,
        gradient_clipped=args.adam,full_objective=args.full_objective,adam_response=adam_response,register_sweep=register_sweep,policy_scale=args.policy_scale,
        reference_objective=result['objective'],literal_objective=float(program.objective[4]),
        edited_parameters=edited_metric,reference_seconds=statistics.median(reference_times),
        literal_seconds=statistics.median(literal_times),setup_seconds=setup,kernel_names=launches,
        nvrtc_cache_hit=program.plan.kernel.compile_cache_hit,nvrtc_cache_key=program.plan.kernel.compile_cache_key,
        single_launch=launches==['persistent_training'],stage_clock_ticks=by_opcode,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),state_id=args.state_id,sequences=args.sequences,
        sequence_ids=list(panel.sequence_ids[:args.sequences]),sequence_length=inputs.shape[1],data_role='fixed fitting panel',
        state_weight=args.state_weight,head_chunk=args.head_chunk,compensated_backward=args.compensated_backward,
        thin_backward=args.thin_backward,microbatch=microbatch,
        optimizer_applied=args.adam,scope='one checkpoint panel; resident exact FP64 teacher; compiledactor/fusednorm/chunkedKL reference; local Adam only if requested, no global reduction',
        literal_gemm_math=('experimental '+os.environ['CHISEL_FP32_EMULATION']+' FP32-input/output emulation; decomposition and reassembly inside timed launch') if os.environ.get('CHISEL_FP32_EMULATION') else 'strict IEEE FP32 forward; optional compensated dx',
        runtime_environment={k:v for k,v in os.environ.items() if k.startswith('CHISEL_')},
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path('rewriter/gpt_precision.py'),*Path('rewriter').glob('persistent_*.py'),*Path('rewriter/kernels').glob('persistent*')]},
        teacher_cache_manifest_sha256=hashlib.sha256((args.teacher_cache/'manifest.json').read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),state_sha256=record.assert_verified())
    if args.adam:
        # The full64/micro16 plan fits by itself. Release its GPT activation
        # arena before the independent exact-reference response evaluation.
        del program
        import gc
        gc.collect();torch.cuda.empty_cache()
        actor.load_state_dict(actual_updated_state);actual_response,actual_after_kl=response()
        adam_response=dict(actor_update=comparison(actual_update,reference_update),
            logit_response=comparison(actual_response-before_response,reference_response-before_response),
            response_scope='first sequence of fixed panel',kl_before=before_kl,
            reference_kl_after=reference_after_kl,literal_kl_after=actual_after_kl,
            lr_scale=args.adam_lr_scale)
        summary['adam_response']=adam_response
        log(dict(event='disposable_adam_response',**adam_response))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');log(dict(event='finished',**summary))
    run.finish();stream.close()


if __name__=='__main__':main()
