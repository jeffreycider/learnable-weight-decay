"""Fixed-state native thin-gradient CUDA graph qualification; never modifies a fit."""
from pathlib import Path
import argparse,copy,gc,hashlib,json,math,statistics,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution,execution_environment
from baseline.model import GPT
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.chisel_factory import build_chisel
from rewriter.edit_scope import editable_names,editable_weight_norm_sq,scoped_snapshot
from rewriter.function_preservation import FunctionalPreservation,TeacherBatch,_ChunkedTeacherKL
from rewriter.graph_offline import NativeThinGradientGraph
from rewriter.meta_optimizer import build_meta_adam
from rewriter.primal_dual import PrimalDual
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from scripts.qualify_compensated_dx import comparison,UnitKL
from scripts.train_multistate_chisel import build_proposal,mean_kl_and_backward


def record_qualification_source(output,configuration):
    """Small exact source bundle, excluding notebook/report/data directories."""
    import tarfile
    root=Path(__file__).resolve().parents[1]
    files=[]
    for folder in ('baseline','rewriter','scripts','tests','configs'):
        files.extend(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.cu','.cuh','.cpp','.h','.hpp','.md') and '__pycache__' not in p.parts)
    files.extend(root/name for name in ('README.md','pyproject.toml') if (root/name).exists())
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
    archive=output/'source.tar.gz'
    with tarfile.open(archive,'w:gz') as bundle:
        for name in hashes:bundle.add(root/name,arcname=name,recursive=False)
    (output/'source_manifest.json').write_text(json.dumps(dict(files=hashes,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        command=sys.argv,configuration=configuration),indent=2,default=str)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('state-bank','teacher-cache','policy','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--state-id',type=int,default=999);p.add_argument('--sequences',type=int,choices=(16,32,64),default=16)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--dx-backend',choices=('ieee','tf32x3','cublas3_large','cublas3_cached'),default='ieee')
    p.add_argument('--capture-only',action='store_true')
    p.add_argument('--shrinkage-coefficient',type=float,default=1.)
    p.add_argument('--teacher-mode',choices=('resident','features'),default='resident')
    p.add_argument('--fused-softcap',action='store_true')
    p.add_argument('--dynamic-zero-groups',action='store_true')
    p.add_argument('--kl-backend',choices=('chunked','tiled'),default='chunked')
    p.add_argument('--linear-backend',choices=('ieee','tf32x3','packed3'),default='ieee')
    p.add_argument('--fused-relu-square',action='store_true')
    p.add_argument('--profile',action='store_true',help='Profile one extra replay outside all timing samples')
    p.add_argument('--profile-eager',action='store_true',help='Separate eager attribution trace; no profile samples enter timings')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    record_qualification_source(args.output,vars(args))
    torch.set_num_threads(1);enable_deterministic_execution();device=torch.device('cuda:0')
    reader=StateBankReader(args.state_bank,verify_once=True,mmap_model=True)
    if args.state_id not in reader.manifest['split']['train']:raise ValueError('fitting state required')
    cache=TeacherCacheReader(args.teacher_cache,verify_once=True,mmap=True)
    record=reader.read(args.state_id,model_device='cpu',device=device)
    engine=FunctionalPreservation(GPT(**reader.manifest['provenance']['model_config']).to(device).requires_grad_(False),microbatch_sequences=16,teacher_cache_device=device,kl_dtype=torch.float64,gpt_precision='fp32')
    cache.load_engine(engine,record);grouped=AcceleratedGroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
    editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
    norm_sq=editable_weight_norm_sq(engine.parameters,editable)
    snapshot=scoped_snapshot(record.snapshot,norm_sq,.01,scope='grouped')
    saved=torch.load(args.policy,map_location='cpu',weights_only=False)
    actor=build_chisel(saved['metadata']['architecture']).to(device);actor.load_state_dict(saved['controller'])
    forward=torch.compile(actor,fullgraph=True,dynamic=False);parameters=list(actor.parameters())
    options=saved['metadata']['options'];optimizer=build_meta_adam(actor,lr=options['lr'],representation_lr_multiplier=options.get('representation_lr_multiplier',1.))
    ds=saved['primal_dual'];dual=PrimalDual(ds['multipliers'],tolerance=ds['tolerance'],rho=ds['rho'],dual_lr=ds['dual_lr']);dual.load_state_dict(ds)
    panel=cache.read(args.state_id,'train');inputs=panel.inputs[:args.sequences].to(device)
    import wandb
    run=wandb.init(project='chisel',mode='offline',dir=str(args.output),config=dict(backend='native_thin_gradient_cuda_graph',sequences=args.sequences,production_changed=False))
    stream=(args.output/'metrics.jsonl').open('w')
    def log(row):
        stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
        run.log({k:v for k,v in row.items() if isinstance(v,(int,float,str,bool))})
    torch.cuda.synchronize();start=time.perf_counter();teacher=None;teacher_id=None;features=None
    if args.teacher_mode=='resident':
        for a in range(0,args.sequences,16):
            batch=panel.teacher_batch(engine,a,a+16)
            if teacher is None:teacher=torch.empty((args.sequences,*batch.log_probs.shape[1:]),dtype=torch.float64,device=device)
            teacher[a:a+16].copy_(batch.log_probs);teacher_id=batch.teacher_id;del batch
    else:
        # Validate original cache binding/precision/head before capture; replay
        # reconstructs the head itself and does not call Python validation.
        batch=panel.teacher_batch(engine,0,16);teacher_id=batch.teacher_id;del batch
        features=panel.features[:args.sequences].to(device=device,copy=True)
    torch.cuda.synchronize();head_seconds=time.perf_counter()-start
    target=teacher if teacher is not None else features
    log(dict(event='teacher_setup',mode=args.teacher_mode,seconds=head_seconds,bytes=target.numel()*target.element_size()))
    def provider(a,b):
        if teacher is None:return panel.teacher_batch(engine,a,b)
        return TeacherBatch(inputs[a:b],teacher[a:b],teacher_id,'past_training',tuple(panel.sequence_ids[a:b]))
    def vector():return torch.cat([(torch.zeros_like(v) if v.grad is None else v.grad).detach().flatten() for v in parameters]).cpu()
    def dense(raw=False):
        actor.zero_grad(set_to_none=True)
        edited,regularizer,_,_,credit=build_proposal(forward,snapshot,engine,grouped,radius_fraction=.01,norm_squared=norm_sq,shrinkage_coefficient=args.shrinkage_coefficient)
        return mean_kl_and_backward(engine,edited,regularizer*(0 if raw else 1),inputs,kl_budget=dual.tolerance,state_weight=1/200,actor_credit=credit,teacher_provider=provider,primal_dual=UnitKL() if raw else dual,state_id=args.state_id)
    def make(raw=False):return NativeThinGradientGraph(actor,forward,snapshot,engine,grouped,inputs,teacher,dual,args.state_id,state_weight=1/200,raw_kl_only=raw,dx_backend=args.dx_backend,shrinkage_coefficient=args.shrinkage_coefficient,teacher_features=features,fused_softcap=args.fused_softcap,dynamic_zero_groups=args.dynamic_zero_groups,kl_backend=args.kl_backend,linear_backend=args.linear_backend,activation_backend='fused' if args.fused_relu_square else 'native')
    def scalar(result):return {k:float(v) for k,v in result.items() if isinstance(v,torch.Tensor) and v.numel()==1}
    qualification={};eagers={};reference_gradients={};eager_gradients={};captured_gradients={};cold={};references={};rows=[]
    for raw in (True,False):
        label='raw_kl' if raw else 'objective'
        references[label]=dense(raw);reference_gradients[label]=vector();log(dict(event='dense_reference',kind=label,**references[label]))
        eager=make(raw);out=eager.eager();eager_gradients[label]=vector();eagers[label]=eager
        log(dict(event='native_eager',kind=label,**scalar(out)))
    # Eager and captured backprop pools cannot coexist on one80GB device.
    # Measure eager first; replay phases below use the identical initial weights.
    eager=eagers['objective'];old_multiplier=dual.multipliers[args.state_id]
    dual.multipliers[args.state_id]=old_multiplier*1.013+.001;eager.bind_dual(dual,args.state_id)
    eager.eager();changed_dual_gradient=vector()
    dual.multipliers[args.state_id]=old_multiplier;eager.bind_dual(dual,args.state_id)
    def restore():
        actor.load_state_dict(saved['controller']);optimizer.load_state_dict(copy.deepcopy(saved['optimizer']));actor.zero_grad(set_to_none=True)
    for repeat in range(args.repeats+1):
        for backend in (('dense','native_eager') if repeat%2==0 else ('native_eager','dense')):
            restore();torch.cuda.synchronize();start=time.perf_counter()
            if backend=='dense':dense()
            else:eager.eager();eager.check_finite()
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
            row=dict(event='timing',backend=backend,seconds=time.perf_counter()-start,repeat=repeat,warmup=repeat==0);rows.append(row);log(row)
    # Also test that updated actor values, not capture-time weights, are used.
    updated_actor={n:v.detach().cpu().clone() for n,v in actor.state_dict().items()}
    eager.eager();updated_actor_gradient=vector();restore()
    if args.profile_eager:
        import rewriter.graph_offline as graph_module
        from functools import wraps
        def annotated(fn,label):
            @wraps(fn)
            def wrapped(*a,**k):
                with torch.profiler.record_function(label):return fn(*a,**k)
            return wrapped
        original_forward=eager.forward;original_teacher=eager._teacher_slice
        original_logits=graph_module.factorized_logits;original_backward=torch.autograd.backward
        def backward(tensors,*a,**k):
            label='chisel_actor_proposal_vjp' if isinstance(tensors,(tuple,list)) and len(tensors)>1 else 'chisel_gpt_backward'
            with torch.profiler.record_function(label):return original_backward(tensors,*a,**k)
        eager.forward=annotated(original_forward,'chisel_actor_forward')
        eager._teacher_slice=annotated(original_teacher,'chisel_teacher_head')
        graph_module.factorized_logits=annotated(original_logits,'chisel_student_forward')
        torch.autograd.backward=backward
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as profile:
                eager.eager();torch.cuda.synchronize()
            profile.export_chrome_trace(str(args.output/'eager-profile.json'))
            (args.output/'eager-profile.txt').write_text(profile.key_averages().table(sort_by='self_cuda_time_total',row_limit=80))
        finally:
            eager.forward=original_forward;eager._teacher_slice=original_teacher
            graph_module.factorized_logits=original_logits;torch.autograd.backward=original_backward
        restore()
    for raw in (True,False):
        label='raw_kl' if raw else 'objective';graph=make(raw)
        torch.cuda.synchronize();start=time.perf_counter();graph.capture();torch.cuda.synchronize()
        cold[label]=time.perf_counter()-start;log(dict(event='capture',kind=label,seconds=cold[label],peak_bytes=torch.cuda.max_memory_allocated()))
        churn=[torch.empty(4096,dtype=torch.int64,pin_memory=True) for _ in range(64)]
        for allocation in churn:allocation.fill_(-1)
        del churn
        replay_checks=[]
        for repeat in range(2):
            actor.zero_grad(set_to_none=True);graph.replay();graph.check_finite();cg=vector()
            replay_checks.append(comparison(cg,eager_gradients[label]))
        metric=comparison(cg,reference_gradients[label]);metric['absolute_kl_difference']=abs(float(graph.output['mean_kl'])-references[label]['mean_kl'])
        qualification[label]=dict(vs_dense=metric,replay_vs_eager=replay_checks)
        log(dict(event='qualification',kind=label,**qualification[label]));captured_gradients[label]=cg
        if max(m['relative_error'] for m in replay_checks)>1e-7:raise RuntimeError('capture changed eager gradient')
        if raw:
            with torch.no_grad():
                for head in actor.output_heads:
                    head.weight.zero_();head.bias.zero_()
            graph.replay();graph.check_finite();identity_kl=float(graph.output['mean_kl'])
            log(dict(event='zero_edit_identity',mean_kl=identity_kl))
            limit=0. if args.linear_backend=='ieee' else 1e-10
            if not math.isfinite(identity_kl) or (args.linear_backend=='ieee' and identity_kl!=0.) or identity_kl < -1e-13 or identity_kl>limit:raise RuntimeError('zero-edit KL failed declared backend identity gate')
            restore()
        if not all(math.isfinite(v) for v in metric.values()) or metric['relative_error']>.003 or metric['cosine']<.99999 or metric['absolute_kl_difference']>1e-9:raise RuntimeError('native graph failed dense numerical gate')
        if not raw:
            dual.multipliers[args.state_id]=old_multiplier*1.013+.001;graph.bind_dual(dual,args.state_id)
            graph.replay();graph.check_finite();changed_check=comparison(vector(),changed_dual_gradient)
            log(dict(event='changed_dual_replay',**changed_check))
            if changed_check['relative_error']>1e-7:raise RuntimeError('stale dual after graph replay')
            dual.multipliers[args.state_id]=old_multiplier;graph.bind_dual(dual,args.state_id)
            actor.load_state_dict(updated_actor);graph.replay();graph.check_finite();changed_check=comparison(vector(),updated_actor_gradient)
            log(dict(event='changed_actor_replay',**changed_check))
            if changed_check['relative_error']>1e-7:raise RuntimeError('stale actor after graph replay')
            for repeat in range(args.repeats+1):
                restore();torch.cuda.synchronize();start=time.perf_counter()
                graph.replay();graph.check_finite();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
                row=dict(event='timing',backend='graph',seconds=time.perf_counter()-start,repeat=repeat,warmup=repeat==0);rows.append(row);log(row)
            if args.profile:
                restore();graph.replay();torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False) as profile:
                    graph.replay();torch.cuda.synchronize()
                profile.export_chrome_trace(str(args.output/'graph-profile.json'))
                (args.output/'graph-profile.txt').write_text(profile.key_averages().table(sort_by='self_cuda_time_total',row_limit=80))
        graph.close();del graph;out=None;restore();gc.collect();torch.cuda.empty_cache()
    responses={}
    def response_readout():
        # Use the actual grad-enabled compiled actor specialization; disable
        # gradients only after materializing/detaching its finite proposal.
        with torch.enable_grad():
            proposal=build_proposal(forward,snapshot,engine,grouped,radius_fraction=.01,norm_squared=norm_sq,shrinkage_coefficient=args.shrinkage_coefficient)
            edited={n:v.detach() for n,v in proposal[0].items()}
            stats=proposal[2]
            del proposal
        total=0.;first=None;first_kl=None
        with torch.no_grad():
            for start in range(0,len(inputs),engine.microbatch_sequences):
                stop=min(start+engine.microbatch_sequences,len(inputs))
                output=engine._forward(edited,inputs[start:stop])
                target=provider(start,stop).log_probs
                value=float(_ChunkedTeacherKL.apply(output,target,engine.chunk_tokens))
                total+=value*((stop-start)/len(inputs))
                if start==0:first=output;first_kl=value
                del output,target
        return first,first_kl,total,stats
    before,before_kl,before_full_kl,before_stats=response_readout()
    for lr_scale in ((1.,.1) if not args.capture_only else ()):
        response=None;reference_update=None
        for backend in ('dense','graph'):
            restore()
            for group in optimizer.param_groups:group['lr']*=lr_scale
            gradient=reference_gradients['objective'] if backend=='dense' else captured_gradients['objective'];cursor=0
            for parameter in parameters:
                parameter.grad=gradient[cursor:cursor+parameter.numel()].reshape_as(parameter).to(device);cursor+=parameter.numel()
            preclip=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));optimizer.step();actor.zero_grad(set_to_none=True)
            after,after_kl,after_full_kl,after_stats=response_readout();change=after-before
            update=torch.cat([(v.detach().cpu()-saved['controller'][n]).flatten() for n,v in actor.named_parameters()])
            if backend=='dense':response=change.clone();reference_update=update
            metric=dict(response=comparison(change,response),actor_update=comparison(update,reference_update),preclip_gradient_norm=preclip,kl_before=before_kl,kl_after=after_kl,kl_change=after_kl-before_kl)
            metric.update(response_sequences=min(16,len(inputs)),full_panel_sequences=len(inputs),
                full_panel_kl_before=before_full_kl,full_panel_kl_after=after_full_kl,
                full_panel_objective_after=after_stats['norm_term']+dual.terms(after_full_kl,args.state_id)['kl_penalty'],
                mean_group_l2_percentage_reduction_before=before_stats['mean_group_l2_percentage_reduction'],
                mean_group_l2_percentage_reduction_after=after_stats['mean_group_l2_percentage_reduction'],
                group_l2_percentage_reduction_after=after_stats['group_l2_percentage_reduction'],
                realized_edit_norm_after=after_stats['realized_edit_norm'])
            metric['response']['absolute_logit_rms_error']=metric['response']['absolute_l2_error']/math.sqrt(before.numel())
            responses[f'{lr_scale}:{backend}']=metric;log(dict(event='adam_response',backend=backend,lr_scale=lr_scale,**metric))
            if backend=='dense':
                reference_after_kl=after_kl;reference_full_kl=after_full_kl
                reference_post_objective=metric['full_panel_objective_after']
            elif (metric['actor_update']['relative_error']>.003 or abs(after_kl-reference_after_kl)>1e-9
                  or abs(after_full_kl-reference_full_kl)>1e-9
                  or abs(metric['full_panel_objective_after']-reference_post_objective)>1e-4):
                raise RuntimeError('Adam delta or full-panel post-update KL/objective failed declared gate')
        del response,reference_update
    del before
    summary=dict(identity_kl=identity_kl,scope='one GPT checkpoint x fixed panel;micro16;state_weight1/200;local clip+Adam;no distributed SUM;fixed-state graph',response_actor_grad_enabled=True,teacher_mode=args.teacher_mode,teacher_reconstruction_in_timer=args.teacher_mode=='features',qualification=qualification,adam_response=responses,capture_seconds=cold,teacher_head_setup_seconds=head_seconds,
        median_seconds={b:statistics.median(r['seconds'] for r in rows if not r['warmup'] and r['backend']==b) for b in ('dense','native_eager','graph')},rows=rows,state_id=args.state_id,sequences=args.sequences,policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),state_sha256=record.assert_verified(),sequence_ids=panel.sequence_ids[:args.sequences],environment=execution_environment(),production_changed=False,single_kernel=False,fused_softcap=args.fused_softcap,dynamic_zero_groups=args.dynamic_zero_groups,kl_backend=args.kl_backend,linear_backend=args.linear_backend,fused_relu_square=args.fused_relu_square,checkpoint_rebinding=False,peak_bytes=torch.cuda.max_memory_allocated(),shrinkage_coefficient=args.shrinkage_coefficient)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');log(dict(event='finished',**summary['median_seconds']));run.finish();stream.close()


if __name__=='__main__':main()
