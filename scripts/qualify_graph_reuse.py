"""Two-checkpoint graph refill/accumulation qualification, with exact cached teacher features."""
from pathlib import Path
import argparse,copy,gc,hashlib,json,math,statistics,sys,time
from contextlib import contextmanager
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution,execution_environment
from baseline.model import GPT
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.chisel_factory import build_chisel
from rewriter.edit_scope import editable_names,editable_weight_norm_sq,scoped_snapshot
from rewriter.function_preservation import FunctionalPreservation,TeacherBatch,_ChunkedTeacherKL
from rewriter.graph_offline import NativeThinGradientGraph,clone_static_snapshot
from rewriter.offline_systems import CachedBaseNormPreservation,VerifiedStateConstants,select_panel_blocks
from rewriter.meta_optimizer import build_meta_adam
from rewriter.primal_dual import PrimalDual
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from scripts.qualify_compensated_dx import comparison,UnitKL
from scripts.qualify_graph_offline import record_qualification_source
from scripts.train_multistate_chisel import build_proposal,mean_kl_and_backward


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('state-bank','teacher-cache','policy','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--state-id',type=int,default=999);p.add_argument('--reuse-states',type=int,nargs='+',default=[998,999]);p.add_argument('--sequences',type=int,choices=(16,32,64),default=16)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--dx-backend',choices=('ieee','tf32x3','cublas3_large','cublas3_cached'),default='ieee')
    p.add_argument('--linear-backend',choices=('ieee','tf32x3','packed3'),default='ieee')
    p.add_argument('--fused-relu-square',action='store_true')
    p.add_argument('--capture-only',action='store_true')
    p.add_argument('--shrinkage-coefficient',type=float,default=1.)
    p.add_argument('--fresh-optimizer-dual',action='store_true',help='Explicit new-stage initialization; original policy, fresh Adam and zero multipliers')
    p.add_argument('--kl-tolerance-override',type=float,default=None,help='Explicit benchmark-only tolerance override; retains saved multipliers and Adam')
    p.add_argument('--teacher-mode',choices=('resident','features'),default='resident')
    p.add_argument('--fused-softcap',action='store_true')
    p.add_argument('--dynamic-zero-groups',action='store_true')
    p.add_argument('--kl-backend',choices=('chunked','tiled'),default='chunked')
    p.add_argument('--training-adapter',action='store_true',help='Also qualify production accumulation lifecycle and complete metric readout')
    p.add_argument('--adapter-only',action='store_true',help='Skip already-qualified bare graph replay tests')
    p.add_argument('--prefetch-checkpoints',action='store_true',help='Stage exact-layout pinned host records while the previous graph executes')
    p.add_argument('--compare-prefetch',action='store_true',help='Alternate synchronous and prefetched graph steps in the same capture')
    p.add_argument('--compare-gemm-backends',action='store_true',help='Qualify raw gradients for strict/hybrid then time strict-hybrid-hybrid-strict with fresh capture and identical saved state')
    p.add_argument('--post-update-kl',action='store_true',help='Audit strict FP32 finite-update KL outside timers; gate absolute KL at0.1tau and relative objective')
    p.add_argument('--profile-bind',action='store_true',help='Record host rebind profiles during qualification, outside throughput timing')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if args.adapter_only and not args.training_adapter:p.error('adapter-only requires training-adapter')
    if args.compare_gemm_backends and (args.training_adapter or args.adapter_only or args.compare_prefetch):p.error('GEMM comparison is a separate bare-graph qualification')
    if args.prefetch_checkpoints and args.training_adapter:p.error('prefetch qualification currently targets bare graph binding, not the production adapter')
    if args.compare_prefetch and not args.prefetch_checkpoints:p.error('compare-prefetch requires prefetch-checkpoints')
    record_qualification_source(args.output,vars(args))
    torch.set_num_threads(1);enable_deterministic_execution();device=torch.device('cuda:0')
    reader=StateBankReader(args.state_bank,verify_once=True,mmap_model=True)
    if args.state_id not in reader.manifest['split']['train']:raise ValueError('fitting state required')
    cache=TeacherCacheReader(args.teacher_cache,verify_once=True,mmap=True)
    record=reader.read(args.state_id,model_device='cpu',device=device)
    engine=CachedBaseNormPreservation(GPT(**reader.manifest['provenance']['model_config']).to(device).requires_grad_(False),microbatch_sequences=16,teacher_cache_device=device,kl_dtype=torch.float64,gpt_precision='fp32')
    constants=VerifiedStateConstants(engine,AcceleratedGroupedNormObjective);constants.load(cache,record);grouped=constants.grouped()
    editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
    norm_sq=constants.radius_squared(grouped)
    snapshot=scoped_snapshot(record.snapshot,norm_sq,.01,scope='grouped')
    saved=torch.load(args.policy,map_location='cpu',weights_only=False)
    actor=build_chisel(saved['metadata']['architecture']).to(device);actor.load_state_dict(saved['controller'])
    forward=torch.compile(actor,fullgraph=True,dynamic=False);parameters=list(actor.parameters())
    options=saved['metadata']['options'];optimizer=build_meta_adam(actor,lr=options['lr'],representation_lr_multiplier=options.get('representation_lr_multiplier',1.))
    initial_optimizer=copy.deepcopy(optimizer.state_dict() if args.fresh_optimizer_dual else saved['optimizer'])
    ds=copy.deepcopy(saved['primal_dual'])
    if args.fresh_optimizer_dual:
        ds['multipliers']={k:0. for k in ds['multipliers']};ds['updates']=0
    if args.kl_tolerance_override is not None:
        if not math.isfinite(args.kl_tolerance_override) or args.kl_tolerance_override<=0:p.error('positive finite tolerance required')
        ds['tolerance']=args.kl_tolerance_override
    dual=PrimalDual(ds['multipliers'],tolerance=ds['tolerance'],rho=ds['rho'],dual_lr=ds['dual_lr']);dual.load_state_dict(ds)
    panel=select_panel_blocks(cache.read(args.state_id,'train'),range(args.sequences//16));inputs=panel.inputs.to(device);features=panel.features.to(device,copy=True)
    import wandb
    run=wandb.init(project='chisel',mode='offline',dir=str(args.output),config=dict(backend='native_thin_gradient_cuda_graph',sequences=args.sequences,production_changed=False))
    stream=(args.output/'metrics.jsonl').open('w')
    active_backend='reference'
    def log(row):
        if args.compare_gemm_backends:row=dict(gemm_backend=active_backend,**row)
        stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
        run.log({k:v for k,v in row.items() if isinstance(v,(int,float,str,bool))})
    if not 2<=len(args.reuse_states)<=4 or any(n not in reader.manifest['split']['train'] for n in args.reuse_states):
        raise ValueError('bounded reuse qualification requires two to four fitting states')
    def vector():return torch.cat([(torch.zeros_like(v) if v.grad is None else v.grad).detach().flatten() for v in parameters]).cpu()
    def restore():
        actor.load_state_dict(saved['controller']);optimizer.load_state_dict(copy.deepcopy(initial_optimizer));actor.zero_grad(set_to_none=True)
    def read(state):
        record=reader.read(state,model_device='cpu',device=device)
        panel=select_panel_blocks(cache.read(state,'train'),range(args.sequences//16))
        return record,panel
    stage_rows=[]
    if args.prefetch_checkpoints:
        # Independent worker verification caches avoid concurrent mutation of
        # the synchronous reference reader's bookkeeping.
        worker_reader=StateBankReader(args.state_bank,verify_once=True,mmap_model=True)
        worker_cache=TeacherCacheReader(args.teacher_cache,verify_once=True,mmap=True)
    @contextmanager
    def candidate_records(prefetch=None):
        if prefetch is None:prefetch=args.prefetch_checkpoints
        if not prefetch:
            yield ((state,*read(state)) for state in args.reuse_states)
            return
        from rewriter.graph_stager import GraphCheckpointStager
        stager=GraphCheckpointStager(worker_reader,worker_cache,pin=True,
            select_panel=lambda panel,state:select_panel_blocks(panel,range(args.sequences//16)))
        try:
            # First wait, staging, worker creation/close all remain inside the
            # complete-step timer. Capture itself uses the normal GPU record;
            # only verified rebinds consume these pinned host observations.
            stager.schedule(args.reuse_states)
            def records():
                for state in args.reuse_states:
                    record,panel,seconds=stager.take(state)
                    stage_rows.append(dict(state_id=state,**seconds))
                    yield state,record,panel
            yield records()
        finally:stager.close()
    reference_stats={}
    def reference(state,raw=False):
        record,panel=read(state);constants.load(cache,record);grouped=constants.grouped()
        norm=constants.radius_squared(grouped);snapshot=scoped_snapshot(record.snapshot,norm,.01,scope='grouped')
        actor.zero_grad(set_to_none=True)
        edited,regularizer,stats,_,credit=build_proposal(forward,snapshot,engine,grouped,radius_fraction=.01,
            norm_squared=norm,shrinkage_coefficient=args.shrinkage_coefficient)
        reference_stats[state]=stats
        return mean_kl_and_backward(engine,edited,regularizer*(0 if raw else 1),panel.inputs,kl_budget=dual.tolerance,
            state_weight=1/200,actor_credit=credit,teacher_provider=lambda a,b:panel.teacher_batch(engine,a,b),
            primal_dual=UnitKL() if raw else dual,state_id=state)
    def post_update_norms():
        result={}
        for state in args.reuse_states:
            record,panel=read(state);constants.load(cache,record);grouped=constants.grouped()
            with torch.enable_grad():
                proposal=build_proposal(forward,record.snapshot,engine,grouped,radius_fraction=.01,
                    norm_squared=constants.radius_squared(grouped),shrinkage_coefficient=args.shrinkage_coefficient)
            stats=proposal[2]
            result[state]={k:stats[k] for k in ('mean_group_l2_percentage_reduction','group_l2_percentage_reduction','realized_edit_norm')}
            if args.post_update_kl:
                with torch.no_grad():
                    total=0.
                    edited={n:v.detach() for n,v in proposal[0].items()}
                    for start in range(0,len(panel.inputs),engine.microbatch_sequences):
                        end=min(start+engine.microbatch_sequences,len(panel.inputs))
                        teacher=panel.teacher_batch(engine,start,end)
                        total+=float(engine.student_parameter_loss(edited,teacher))*(end-start)/len(panel.inputs)
                        del teacher
                result[state]['mean_kl']=total
                result[state]['objective']=float(proposal[1].detach())+dual.terms(total,state)['kl_penalty']
                del edited
            del proposal
        return result
    layout_checks={}
    for state in args.reuse_states:
        record,panel=read(state);constants.load(cache,record);grouped=constants.grouped()
        snapshot=scoped_snapshot(record.snapshot,constants.radius_squared(grouped),.01,scope='grouped')
        def action_vector(action):
            return torch.cat([v.detach().flatten() for f in action.matrices.values() for v in (f.P,f.Q)]+[v.detach().flatten() for v in action.vectors.values()])
        for grad_enabled in (False,True):
            with torch.set_grad_enabled(grad_enabled):
                original=action_vector(forward(snapshot));contiguous=action_vector(forward(clone_static_snapshot(snapshot)))
                metric=comparison(contiguous,original);layout_checks[f'{state}:{grad_enabled}']=metric
            log(dict(event='original_vs_static_action',state=state,grad_enabled=grad_enabled,**metric))
            if metric['relative_error']>1e-7:raise RuntimeError('static observation layout changed actor proposal')
            del original,contiguous
    gradients={};ref_outputs={};rows=[];accum=[torch.zeros_like(p) for p in parameters]
    for state in args.reuse_states:
        for raw in (True,False):
            result=reference(state,raw);key=f'{state}:{raw}';gradients[key]=vector();ref_outputs[key]=result
            log(dict(event='reference',state=state,raw=raw,**result))
    reference_updates=[];reference_accum=None
    for repeat in range(args.repeats+1):
        restore();torch._foreach_zero_(accum);torch.cuda.synchronize();start=time.perf_counter()
        for state in args.reuse_states:
            reference(state);torch._foreach_add_(accum,[p.grad for p in parameters])
        for p,g in zip(parameters,accum):p.grad=g
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
        row=dict(event='timing',backend='dense',seconds=time.perf_counter()-start,repeat=repeat,warmup=repeat==0)
        rows.append(row);log(row)
        if repeat==args.repeats:
            reference_updates=[p.detach().cpu().clone() for p in parameters]
    reference_post_norms=post_update_norms()
    # Every captured contribution is1/200. Two states are only a qualification
    # subset; their sum is deliberately NOT renormalized to1/2.
    qualifications={};cold={};model_addresses={n:v.data_ptr() for n,v in engine.parameters.items()}
    if args.training_adapter:
        from rewriter.offline_graph_backend import OfflineGraphAccumulator
        adapter=OfflineGraphAccumulator(actor,forward,engine,cache,constants,radius_fraction=.01,
                                        shrinkage_coefficient=args.shrinkage_coefficient)
        restore();adapter.begin_update(args.reuse_states,global_checkpoint_count=200)
        metric_checks={};expected_accum=torch.zeros_like(gradients[f'{args.reuse_states[0]}:False'])
        for state in args.reuse_states:
            record,panel=read(state);measured,stats=adapter.accumulate(record,panel,dual)
            metric=comparison(vector(),gradients[f'{state}:False'])
            metric['absolute_kl_difference']=abs(measured['mean_kl']-ref_outputs[f'{state}:False']['mean_kl'])
            log(dict(event='adapter_gradient',state=state,**metric))
            if metric['relative_error']>.003 or metric['cosine']<.99999 or metric['absolute_kl_difference']>1e-9:
                raise RuntimeError('training adapter gradient gate failed')
            expected_accum+=gradients[f'{state}:False']
            differences={}
            for key in ('mean_group_l2_percentage_reduction','norm_term','unweighted_norm_term','raw_action_norm','global_scale','hard_radius','realized_edit_norm','proposed_edit_norm'):
                differences[key]=abs(stats[key]-reference_stats[state][key])
                if not math.isclose(stats[key],reference_stats[state][key],rel_tol=1e-7,abs_tol=1e-10):
                    raise RuntimeError('training adapter metric differs: '+key)
            for key in ('group_l2_percentage_reduction','group_new_l2_norm','proposed_group_l2_percentage_reduction'):
                differences[key]=max(abs(stats[key][n]-reference_stats[state][key][n]) for n in stats[key])
                for n,value in stats[key].items():
                    if not math.isclose(value,reference_stats[state][key][n],rel_tol=1e-7,abs_tol=1e-10):
                        raise RuntimeError('training adapter group metric differs: '+key+':'+n)
            metric_checks[state]=differences
        adapter.finish_update();metric=comparison(vector(),expected_accum)
        if metric['relative_error']>.003 or metric['cosine']<.99999:raise RuntimeError('adapter accumulation gate failed')
        qualifications['adapter_accumulation']=metric;qualifications['adapter_metrics']=metric_checks
        compiled=dict(torch._dynamo.utils.counters['stats'])
        for repeat in range(args.repeats+1):
            restore();torch.cuda.synchronize();tick=time.perf_counter()
            adapter.begin_update(args.reuse_states,global_checkpoint_count=200)
            preupdate_measurements=[]
            for state in args.reuse_states:
                record,panel=read(state);measured,_=adapter.accumulate(record,panel,dual)
                preupdate_measurements.append((state,measured['mean_kl']))
            adapter.finish_update();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize()
            row=dict(event='timing',backend='training_adapter',seconds=time.perf_counter()-tick,repeat=repeat,warmup=repeat==0)
            rows.append(row);log(row)
        actual=torch.cat([(p.detach().cpu()-saved['controller'][n]).flatten() for n,p in actor.named_parameters()])
        expected=torch.cat([(v-saved['controller'][n]).flatten() for (n,_),v in zip(actor.named_parameters(),reference_updates)])
        metric=comparison(actual,expected);qualifications['adapter_adam']=metric
        if metric['relative_error']>.003 or metric['cosine']<.99999:raise RuntimeError('adapter Adam gate failed')
        if dict(torch._dynamo.utils.counters['stats'])!=compiled:raise RuntimeError('adapter reuse recompiled actor')
        qualifications['adapter_compute']=dict(adapter.compute)
        qualifications['adapter_capture_count']=adapter.capture_count
        log(dict(event='adapter_qualified',**metric,capture_count=adapter.capture_count))
        # A real optimizer/dual boundary followed by an eager audit must not
        # leave stale GPT weights, scalar inputs or gradient aliases on resume.
        adapter.close();gc.collect();torch.cuda.empty_cache()
        dual.step(preupdate_measurements,expected_ids=args.reuse_states)
        from scripts.train_multistate_chisel import evaluate_sequences
        record,panel=read(args.reuse_states[0]);constants.load(cache,record);grouped=constants.grouped()
        edited,_,_,_,credit=build_proposal(forward,record.snapshot,engine,grouped,radius_fraction=.01,
            norm_squared=constants.radius_squared(grouped),shrinkage_coefficient=0.)
        with torch.no_grad():audit_kl=evaluate_sequences(engine,edited,panel.inputs,teacher_provider=lambda a,b:panel.teacher_batch(engine,a,b))
        del edited,credit
        after_audit_expected=torch.zeros_like(expected_accum)
        for state in args.reuse_states:
            reference(state);after_audit_expected+=vector()
        adapter.begin_update(args.reuse_states,global_checkpoint_count=200)
        for state in args.reuse_states:
            record,panel=read(state);adapter.accumulate(record,panel,dual)
        adapter.finish_update();metric=comparison(vector(),after_audit_expected)
        qualifications['adapter_after_adam_dual_audit']=metric
        log(dict(event='adapter_after_adam_dual_audit',audit_kl=audit_kl,**metric))
        if metric['relative_error']>.003 or metric['cosine']<.99999:raise RuntimeError('adapter audit/resume gradient gate failed')
        if dict(torch._dynamo.utils.counters['stats'])!=compiled:raise RuntimeError('adapter audit/resume recompiled actor')
        adapter.close();del adapter;restore();dual.load_state_dict(ds);gc.collect();torch.cuda.empty_cache()
    graph_specs=[] if args.adapter_only else [(raw,'selected',args.linear_backend,args.dx_backend,'fused' if args.fused_relu_square else 'native') for raw in (True,False)]
    if args.compare_gemm_backends:
        variants={'strict':('ieee','ieee','native'),'hybrid':('tf32x3','cublas3_cached','fused')}
        graph_specs=[(True,name,*variants[name]) for name in ('strict','hybrid')]
        graph_specs += [(False,name,*variants[name]) for name in ('strict','hybrid','hybrid','strict')]
    for arm_index,(raw,active_backend,linear_backend,dx_backend,activation_backend) in enumerate(graph_specs):
        qualification_prefix=f'{arm_index}:{active_backend}:' if args.compare_gemm_backends else ''
        restore();record,panel=read(args.state_id);constants.load(cache,record);grouped=constants.grouped()
        norm=constants.radius_squared(grouped);snapshot=scoped_snapshot(record.snapshot,norm,.01,scope='grouped')
        inputs=panel.inputs.to(device,copy=True);features=panel.features.to(device,copy=True)
        graph=NativeThinGradientGraph(actor,forward,snapshot,engine,grouped,inputs,None,dual,args.state_id,
            state_weight=1/200,raw_kl_only=raw,teacher_features=features,dx_backend=dx_backend,
            shrinkage_coefficient=args.shrinkage_coefficient,fused_softcap=args.fused_softcap,dynamic_zero_groups=args.dynamic_zero_groups,kl_backend=args.kl_backend,
            linear_backend=linear_backend,activation_backend=activation_backend)
        torch.cuda.synchronize();tick=time.perf_counter();graph.capture();torch.cuda.synchronize();cold[qualification_prefix+str(raw)]=time.perf_counter()-tick
        signature=graph.storage_signature();compiled_after_capture=dict(torch._dynamo.utils.counters['stats'])
        log(dict(event='capture',raw=raw,seconds=cold[qualification_prefix+str(raw)],compilation=compiled_after_capture,storage_count=len(signature)))
        torch._foreach_zero_(accum)
        expected_accum=torch.zeros_like(gradients[f'{args.reuse_states[0]}:{raw}'])
        with candidate_records() as records:
            for state,record,panel in records:
                torch.cuda.synchronize();tick=time.perf_counter()
                profile=None
                if args.profile_bind and not raw:
                    import cProfile
                    profile=cProfile.Profile();profile.enable()
                binding=graph.load_checkpoint(cache,record,panel,dual,state_constants=constants)
                if profile is not None:
                    import pstats
                    profile.disable();profile.dump_stats(str(args.output/f'bind-{state}.prof'))
                    with (args.output/f'bind-{state}.txt').open('w') as report:
                        pstats.Stats(profile,stream=report).strip_dirs().sort_stats('cumulative').print_stats(40)
                if graph.storage_signature()!=signature:raise RuntimeError('captured tensor address/layout changed')
                torch.cuda.synchronize();bind_seconds=time.perf_counter()-tick
                graph.accumulate_into(accum);graph.check_finite();actual=vector();key=f'{state}:{raw}'
                metric=comparison(actual,gradients[key]);metric['absolute_kl_difference']=abs(float(graph.output['mean_kl'])-ref_outputs[key]['mean_kl'])
                qualifications[qualification_prefix+key]=metric;expected_accum.add_(gradients[key])
                log(dict(event='rebind_qualification',state=state,raw=raw,bind_seconds=bind_seconds,compilation=dict(torch._dynamo.utils.counters['stats']),storage_unchanged=graph.storage_signature()==signature,zero_reference_groups=list(graph.grouped.zero_reference_groups),editable_parameters=len(graph.editable),**metric))
                if not all(math.isfinite(v) for v in metric.values()) or metric['relative_error']>.003 or metric['cosine']<.99999 or metric['absolute_kl_difference']>1e-9:raise RuntimeError('rebound graph gradient gate failed')
        actual_accum=torch.cat([v.flatten() for v in accum]).cpu();metric=comparison(actual_accum,expected_accum)
        qualifications[qualification_prefix+'accum:'+str(raw)]=metric;log(dict(event='accumulation',raw=raw,**metric))
        if not all(math.isfinite(v) for v in metric.values()) or metric['relative_error']>.003 or metric['cosine']<.99999:raise RuntimeError('weighted accumulation gate failed')
        if not raw:
            for repeat in range(args.repeats+1):
                modes=([False,True] if repeat%2==0 else [True,False]) if args.compare_prefetch else [args.prefetch_checkpoints]
                for prefetch in modes:
                    restore();torch._foreach_zero_(accum);torch.cuda.synchronize();tick=time.perf_counter()
                    with candidate_records(prefetch) as records:
                        for state,record,panel in records:
                            graph.load_checkpoint(cache,record,panel,dual,state_constants=constants)
                            graph.accumulate_into(accum);graph.check_finite()
                    for p,g in zip(parameters,accum):p.grad=g
                    torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
                    backend=('graph_prefetch' if prefetch else 'graph_sync') if args.compare_prefetch else 'graph'
                    if args.compare_gemm_backends:backend='graph_'+active_backend
                    row=dict(event='timing',backend=backend,seconds=time.perf_counter()-tick,repeat=repeat,warmup=repeat==0)
                    rows.append(row);log(row)
                    # Outside timing, qualify each alternate arm against the
                    # same saved-Adam reference, rather than only the last arm.
                    if args.compare_prefetch:
                        actual_delta=torch.cat([(p.detach().cpu()-saved['controller'][n]).flatten() for n,p in actor.named_parameters()])
                        expected_delta=torch.cat([(v-saved['controller'][n]).flatten() for (n,_),v in zip(actor.named_parameters(),reference_updates)])
                        metric=comparison(actual_delta,expected_delta)
                        log(dict(event='alternating_adam',backend=backend,repeat=repeat,**metric))
                        if metric['relative_error']>.003 or metric['cosine']<.99999:raise RuntimeError('alternating prefetch Adam gate failed')
            actual=torch.cat([(p.detach().cpu()-saved['controller'][n]).flatten() for n,p in actor.named_parameters()])
            expected=torch.cat([(v-saved['controller'][n]).flatten() for (n,_),v in zip(actor.named_parameters(),reference_updates)])
            metric=comparison(actual,expected);qualifications[qualification_prefix+'adam_update']=metric;log(dict(event='accumulated_adam',**metric))
            if not all(math.isfinite(v) for v in metric.values()) or metric['relative_error']>.003 or metric['cosine']<.99999:raise RuntimeError('accumulated Adam update gate failed')
            candidate_post_norms=post_update_norms()
            norm_differences={state:abs(candidate_post_norms[state]['mean_group_l2_percentage_reduction']-
                                      reference_post_norms[state]['mean_group_l2_percentage_reduction']) for state in args.reuse_states}
            qualifications[qualification_prefix+'post_adam_realized_norms']=dict(reference=reference_post_norms,candidate=candidate_post_norms,
                                                           mean_percentage_point_differences=norm_differences)
            log(dict(event='post_adam_realized_norms',maximum_mean_percentage_point_difference=max(norm_differences.values())))
            if args.post_update_kl:
                for state in args.reuse_states:
                    ref=reference_post_norms[state];actual=candidate_post_norms[state]
                    kl_error=abs(actual['mean_kl']-ref['mean_kl'])
                    objective_error=abs(actual['objective']-ref['objective'])
                    objective_limit=1e-4+1e-3*abs(ref['objective'])
                    log(dict(event='post_adam_strict_function',state=state,reference=ref,candidate=actual,
                             absolute_kl_difference=kl_error,kl_limit=.1*dual.tolerance,
                             absolute_objective_difference=objective_error,objective_limit=objective_limit))
                    if not math.isfinite(kl_error+objective_error) or kl_error>.1*dual.tolerance or objective_error>objective_limit:
                        raise RuntimeError('post-Adam physical KL/relative objective gate failed')
            if any(not math.isfinite(v) or v>1e-4 for v in norm_differences.values()):
                raise RuntimeError('post-Adam mean realized norm differs beyond objective tolerance')
            # Earliest fitting state may have a different exact-zero bias mask.
            # If so, binding must reject and replay must remain invalidated.
            early=min(reader.manifest['split']['train']);record,panel=read(early)
            rejected=False
            try:graph.load_checkpoint(cache,record,panel,dual,state_constants=constants)
            except ValueError as error:
                if 'zero-reference' not in str(error):raise
                rejected=True
                try:graph.replay()
                except RuntimeError:pass
                else:raise RuntimeError('failed binding did not invalidate replay')
            qualifications[qualification_prefix+'early_zero_mask']=dict(state_id=early,rejected_changed_mask=rejected,mode='dynamic' if args.dynamic_zero_groups else 'reject',zero_reference_groups=list(graph.grouped.zero_reference_groups))
            log(dict(event='early_zero_mask',**qualifications[qualification_prefix+'early_zero_mask']))
            record,panel=read(args.reuse_states[-1]);graph.load_checkpoint(cache,record,panel,dual,state_constants=constants)
            if {n:v.data_ptr() for n,v in engine.parameters.items()}!=model_addresses:raise RuntimeError('model storage changed')
        if dict(torch._dynamo.utils.counters['stats'])!=compiled_after_capture:raise RuntimeError('checkpoint rebind recompiled actor')
        graph.close();del graph;restore();gc.collect();torch.cuda.empty_cache()
    summary=dict(scope='multiple real GPT checkpoints x fixed sequence panels;1/200 each;one local accumulated clip+Adam;state reads/verifiedload/exactteacherhead included;no distributed SUM',
        states=args.reuse_states,sequences=args.sequences,shrinkage_coefficient=args.shrinkage_coefficient,
        optimizer_dual_initialization='fresh' if args.fresh_optimizer_dual else 'saved',
        kl_tolerance=dual.tolerance,saved_kl_tolerance=saved['primal_dual']['tolerance'],kl_tolerance_override=args.kl_tolerance_override,
        linear_backend=args.linear_backend,dx_backend=args.dx_backend,fused_relu_square=args.fused_relu_square,
        activation_backend='fused' if args.fused_relu_square else 'native',
        compare_gemm_backends=args.compare_gemm_backends,graph_backend_order=[v[1] for v in graph_specs],
        prefetch_checkpoints=args.prefetch_checkpoints,compare_prefetch=args.compare_prefetch,staging=stage_rows,
        qualification=qualifications,original_layout_action=layout_checks,median_seconds={b:statistics.median(r['seconds'] for r in rows if not r['warmup'] and r['backend']==b) for b in sorted({r['backend'] for r in rows})},rows=rows,
        capture_seconds=cold,policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),peak_bytes=torch.cuda.max_memory_allocated(),production_changed=False,single_kernel=False,fused_softcap=args.fused_softcap,dynamic_zero_groups=args.dynamic_zero_groups,kl_backend=args.kl_backend,
        compilation_counters={k:dict(v) for k,v in torch._dynamo.utils.counters.items()})
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');log(dict(event='finished',**summary['median_seconds']));run.finish();stream.close()


if __name__=='__main__':main()
