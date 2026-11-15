"""Experimental dense student CUDA graph: saved shared/private Adam qualification."""
import argparse,copy,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.gpt_precision import configure_precision
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.coordinate_dense_student_graph import DenseStudentGraph
from rewriter.tokenwise_reference import tokenwise_panel
from rewriter.coordinate_fitting_panels import split_fitting_panels
from rewriter.coordinate_optimization import clip_private_gradients,clip_shared_continuation
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from scripts.qualify_coordinate_chisel import FixedContexts
from scripts.qualify_coordinate_fullbank import read_joint
from rewriter.distributed_probe_store import _digest
from scripts.train_coordinate_chisel import build_shared_adam,set_private_lr,measure_context
from scripts.train_multistate_chisel import build_proposal,norm_preservation_diagnostics


class Contexts(FixedContexts):
    graph_enabled=False
    student_graph=None
    capture_seconds=None
    def measure(self,context,actor,private,config,dual,*,learning,weight=1.):
        if not self.graph_enabled or context['domain']!='gpt' or not learning:
            return super().measure(context,actor,private,config,dual,learning=learning,weight=weight)
        engine=context['engine']
        with torch.enable_grad():
            parameters,regularizer,stats,_,credit=build_proposal(lambda s:actor(s,private),context['snapshot'],engine,context['grouped'],radius_fraction=.01,norm_squared=context['norm_squared'])
        if self.student_graph is None:
            torch.cuda.synchronize();start=time.perf_counter()
            self.student_graph=DenseStudentGraph(engine,parameters,context['inputs'][:16].to(engine.device))
            torch.cuda.synchronize();self.capture_seconds=time.perf_counter()-start
        else:
            if self.student_graph.engine is not engine:raise ValueError('one stable engine required')
            self.student_graph.bind_checkpoint()
        before=getattr(engine,'cached_teacher_head_forwards',0)
        result=tokenwise_panel(self.student_graph,parameters,regularizer,context['inputs'],config=config,dual=dual,state_id=context['local_id'],state_weight=weight,actor_credit=credit,teacher_provider=context['teacher'],backward=True)
        self.add_counts(engine.compute.as_dict());self.add_counts(dict(cached_teacher_head_forwards=getattr(engine,'cached_teacher_head_forwards',0)-before))
        stats.update(norm_preservation_diagnostics(stats,context['grouped']))
        return {**result,**stats,'domain':context['domain'],'local_id':context['local_id'],'context_ids':context['ids']}


def tensor_state(module):return {n:p.detach().cpu().clone() for n,p in module.named_parameters()}
def actor_state(module):return {n:p.detach().cpu().clone() for n,p in module.state_dict().items()}
def grads(module):return {n:p.grad.detach().clone() for n,p in module.named_parameters() if p.grad is not None}
def relative(a,b):
    names=set(a)|set(b);sq=base=maximum=0.
    for n in names:
        x=a.get(n);y=b.get(n)
        if x is None:x=torch.zeros_like(y)
        if y is None:y=torch.zeros_like(x)
        x=x.detach().cpu().double();y=y.detach().cpu().double();delta=x-y
        sq+=float(delta.square().sum());base+=float(x.square().sum());maximum=max(maximum,float(delta.abs().max()))
    return dict(relative_l2=(sq/max(base,1e-300))**.5,absolute_l2=sq**.5,max_abs=maximum,none_mask_equal=set(a)==set(b))


def validate_candidate_recipe(recipe):
    if recipe.get('action_family','fixed_anchored_v1')!='fixed_anchored_v1':
        raise ValueError('this qualifier only supports the existing anchored family')
    if recipe['architecture'].get('frame_mode')!='tied_orthonormal':
        raise ValueError('the declared saved fixture uses tied orthonormal probes')
    for domain in ('gpt','cifar10'):
        cfg=TokenwiseConfig(**recipe['objectives'][domain])
        if (cfg.loss_backend,cfg.ranking_loss_mode,cfg.ranking_tie_policy,cfg.ranking_k)!=(
                'tiled_relative_fp64_v1','relative_gap','exact_teacher_ties',100):
            raise ValueError('the declared qualifier requires the unchanged relative100 objective')


def run(args):
    recipe=json.loads(Path(args.recipe).read_text());validate_candidate_recipe(recipe)
    enable_deterministic_execution();torch.cuda.set_device(0);configure_precision('fp32');torch.set_num_threads(1)
    device=torch.device('cuda',0);root=Path(args.output);root.mkdir(exist_ok=False)
    saved,rank_states,private_payloads=read_joint(args.checkpoint)
    base=json.loads(Path(recipe['base_recipe']).read_text());contexts=Contexts(base,device,gpt_sequences=128,cifar_images=2048)
    actor=CoordinateChisel(**recipe['architecture']).to(device);shared_opt=build_shared_adam(actor,recipe['shared_optimizer'])
    selected=[('gpt',6),('cifar10',1)];private={};optimizers={};duals={};payloads={};dual_states={}
    for domain,uid in selected:
        key=('gpt',0,uid) if domain=='gpt' else ('cifar10',uid-1)
        token,payload=next((k,v) for k,v in private_payloads.items() if tuple(v['identity']['state_key'])==key)
        payloads[domain]=payload;context=contexts.load(domain,uid)
        private[domain]=actor.make_private(context['snapshot'],state_key=key,seed=19)
        opt=torch.optim.Adam(private[domain].parameters(),lr=recipe['private_optimizer']['lr'],eps=recipe['private_optimizer']['eps'],betas=tuple(recipe['private_optimizer']['betas']));optimizers[domain]=opt
        owner=next(v for v in rank_states.values() if token in v['visits']);dual_states[domain]=copy.deepcopy(owner['duals'][domain])
        cfg=TokenwiseConfig(**recipe['objectives'][domain]);duals[domain]=TokenwiseDual(list(dual_states[domain]['multipliers']),config=cfg,dual_lr=recipe['dual_lr'])
    def restore():
        actor.load_state_dict(saved['model']);shared_opt.load_state_dict(copy.deepcopy(saved['optimizer']));actor.zero_grad(set_to_none=True)
        for domain in private:
            private[domain].load_state_dict(payloads[domain]['private']);optimizers[domain].load_state_dict(copy.deepcopy(payloads[domain]['optimizer']));private[domain].zero_grad(set_to_none=True);duals[domain].load_state_dict(copy.deepcopy(dual_states[domain]))
    restore();initial={'shared':tensor_state(actor),**{d:tensor_state(p) for d,p in private.items()}}
    diagnostic_compute=dict(gpt_forwards=0,gpt_backwards=0,cifar10_forwards=0,cifar10_backwards=0)
    # Same-weight forward plus dense VJP preflight, outside update timers.
    context=contexts.load('gpt',6);engine=context['engine']
    proposal=build_proposal(lambda s:actor(s,private['gpt']),context['snapshot'],engine,context['grouped'],radius_fraction=.01,norm_squared=context['norm_squared'])
    params={n:v.detach().requires_grad_(v.requires_grad) for n,v in proposal[0].items()}
    del proposal
    inputs=context['inputs'][:16].to(device)
    dense=engine._forward(params,inputs)
    # Deterministic bounded nonzero cotangent exercises every logit, no new RNG.
    cotangent=torch.ones_like(dense)/dense.numel()
    dense.backward(cotangent)
    diagnostic_compute['gpt_forwards']+=1;diagnostic_compute['gpt_backwards']+=1
    dense_cpu=dense.detach().cpu()
    dense_grads={n:v.grad.detach().cpu() for n,v in params.items() if v.grad is not None}
    del context,params,dense,cotangent,inputs
    results={};traces={};states={}
    for mode in ('dense','graph'):
        restore();contexts.graph_enabled=mode=='graph';contexts.compute={};trace={};rows=[]
        if mode=='graph':
            # Finish the eager arm before retaining a private graph pool. Keeping
            # both full F/B activation pools resident can exceed one 80GB GPU.
            torch.cuda.empty_cache()
            context=contexts.load('gpt',6);engine=context['engine']
            proposal=build_proposal(lambda s:actor(s,private['gpt']),context['snapshot'],engine,context['grouped'],radius_fraction=.01,norm_squared=context['norm_squared'])
            params={n:v.detach().requires_grad_(v.requires_grad) for n,v in proposal[0].items()}
            del proposal
            inputs=context['inputs'][:16].to(device)
            capture_cuda_rng=torch.cuda.get_rng_state(device);capture_cpu_rng=torch.get_rng_state()
            torch.cuda.synchronize();capture_start=time.perf_counter()
            graph=DenseStudentGraph(engine,params,inputs)
            torch.cuda.synchronize();contexts.capture_seconds=time.perf_counter()-capture_start
            capture_rng_equal=torch.equal(capture_cuda_rng,torch.cuda.get_rng_state(device)) and torch.equal(capture_cpu_rng,torch.get_rng_state())
            capture_gradients_clean=all(v.grad is None for v in params.values())
            candidate=graph._forward(params,inputs);candidate_cpu=candidate.detach().cpu()
            forward=dict(bitexact=torch.equal(dense_cpu,candidate_cpu),max_abs=float((dense_cpu-candidate_cpu).abs().max()),capture_CPU_CUDA_RNG_bitexact=capture_rng_equal,capture_live_gradients_untouched=capture_gradients_clean)
            candidate.backward(torch.ones_like(candidate)/candidate.numel())
            diagnostic_compute['gpt_forwards']+=1;diagnostic_compute['gpt_backwards']+=1
            preflight_replays=graph.replays
            forward['dense_parameter_VJP']=relative(dense_grads,{n:v.grad for n,v in params.items() if v.grad is not None})
            contexts.student_graph=graph
            del context,params,inputs,candidate,candidate_cpu,dense_cpu,dense_grads
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize();start=time.perf_counter()
        for domain,uid in selected:
            context=contexts.load(domain,uid);cfg=recipe['private_inner_panels'][domain]
            inner,joint=split_fitting_panels(context,inner_steps=2,inner_size=cfg['inner_size'],joint_size=recipe['panels']['gpt_sequences' if domain=='gpt' else 'cifar_images'],seed=991)
            p=private[domain];opt=optimizers[domain];dual=duals[domain];tick=payloads[domain]['visits']
            actor.requires_grad_(False)
            for i,part in enumerate(inner):
                p.zero_grad(set_to_none=True);set_private_lr(opt,recipe['private_optimizer'],tick+i)
                row=measure_context(contexts,part,actor,p,dual.config,dual,learning=True,weight=1.)
                trace[f'{domain}/inner{i}']=grads(p);clip_private_gradients(p.parameters(),1.);opt.step();p.zero_grad(set_to_none=True)
                dual.step([(uid,row['mean_kl'])],expected_ids=[uid])
            actor.requires_grad_(True);set_private_lr(opt,recipe['private_optimizer'],tick+2)
            row=measure_context(contexts,joint,actor,p,dual.config,dual,learning=True,weight=.5)
            for parameter in p.parameters():
                if parameter.grad is not None:parameter.grad.div_(.5)
            trace[domain+'/joint']=grads(p);clip_private_gradients(p.parameters(),1.);opt.step();p.zero_grad(set_to_none=True)
            dual.step([(uid,row['mean_kl'])],expected_ids=[uid]);rows.append(row)
        trace['shared']=grads(actor);clip_shared_continuation(shared_opt,'separate_representation',max_norm=1.);shared_opt.step();actor.zero_grad(set_to_none=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        traces[mode]={k:{n:v.cpu() for n,v in value.items()} for k,value in trace.items()}
        states[mode]=dict(shared=actor_state(actor),private={d:copy.deepcopy(p.state_dict()) for d,p in private.items()},optimizer=copy.deepcopy(shared_opt.state_dict()),private_optimizer={d:copy.deepcopy(o.state_dict()) for d,o in optimizers.items()},duals={d:v.state_dict() for d,v in duals.items()})
        results[mode]=dict(seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),rows=rows,compute=dict(contexts.compute))
    gradients={k:relative(traces['dense'][k],traces['graph'][k]) for k in traces['dense']}
    deltas={}
    for family in initial:
        values=[states[m]['shared'] if family=='shared' else states[m]['private'][family] for m in ('dense','graph')]
        deltas[family]=relative({n:values[0][n].detach().cpu()-v for n,v in initial[family].items()},{n:values[1][n].detach().cpu()-v for n,v in initial[family].items()})
    (root/'preliminary.json').write_text(json.dumps(dict(status='endpoint gates not yet checked',forward=forward,gradients=gradients,Adam_deltas=deltas,runtimes={k:v['seconds'] for k,v in results.items()}),indent=2))
    torch.save(dict(initial=initial,traces=traces,states=states),root/'numeric_evidence.pt')
    # True post-update teacher metrics on the same joint fitting panels.
    endpoint={}
    for mode in ('dense','graph'):
        actor.load_state_dict(states[mode]['shared']);contexts.graph_enabled=False;rows=[]
        for domain,uid in selected:
            private[domain].load_state_dict(states[mode]['private'][domain]);duals[domain].load_state_dict(states[mode]['duals'][domain])
            context=contexts.load(domain,uid);cfg=recipe['private_inner_panels'][domain]
            _,joint=split_fitting_panels(context,inner_steps=2,inner_size=cfg['inner_size'],joint_size=recipe['panels']['gpt_sequences' if domain=='gpt' else 'cifar_images'],seed=991)
            before=contexts.compute.get('student_forwards',0)
            rows.append(contexts.measure(joint,actor,private[domain],duals[domain].config,duals[domain],learning=False))
            diagnostic_compute[domain+'_forwards']+=contexts.compute.get('student_forwards',0)-before
        endpoint[mode]=rows
    # Direct mutual edited-output comparison, independent of distance to teacher.
    from rewriter.rank_preservation import topk_preservation_flags
    mutual={}
    for domain,uid in selected:
        context=contexts.load(domain,uid);cfg=recipe['private_inner_panels'][domain]
        _,joint=split_fitting_panels(context,inner_steps=2,inner_size=cfg['inner_size'],joint_size=recipe['panels']['gpt_sequences' if domain=='gpt' else 'cifar_images'],seed=991)
        stored={}
        for mode in ('dense','graph'):
            actor.load_state_dict(states[mode]['shared']);private[domain].load_state_dict(states[mode]['private'][domain])
            with torch.enable_grad():
                proposal=build_proposal(lambda s:actor(s,private[domain]),joint['snapshot'],joint['engine'],joint['grouped'],radius_fraction=.01,norm_squared=joint['norm_squared'])
                stored[mode]={n:v.detach() for n,v in proposal[0].items()}
            del proposal
        total=0;ab=ba=0.;maximum=0.;rank_changes=0;centered_square=0.;elements=0
        micro=16 if domain=='gpt' else 128
        with torch.no_grad():
            for start in range(0,len(joint['inputs']),micro):
                data=joint['inputs'][start:start+micro].to(device)
                a=joint['engine']._forward(stored['dense'],data);a=a.reshape(-1,a.shape[-1])
                b=joint['engine']._forward(stored['graph'],data).reshape_as(a)
                diagnostic_compute[domain+'_forwards']+=2
                for offset in range(0,len(a),128):
                    x=a[offset:offset+128].double();y=b[offset:offset+128].double()
                    lp=x.log_softmax(-1);lq=y.log_softmax(-1)
                    ka=(lp.exp()*(lp-lq)).sum(-1);kb=(lq.exp()*(lq-lp)).sum(-1)
                    ab+=float(ka.sum());ba+=float(kb.sum());total+=len(x);maximum=max(maximum,float(ka.max()),float(kb.max()))
                    gap=x-y;gap-=gap.mean(-1,keepdim=True);centered_square+=float(gap.square().sum());elements+=gap.numel()
                    rank_changes+=int(topk_preservation_flags(x,y,k=min(100,x.shape[-1]))['strict_topk_changed'].sum())
                del a,b
        mutual[domain]=dict(dense_to_graph_KL=ab/total,graph_to_dense_KL=ba/total,max_token_KL=maximum,strict_changed_positions=rank_changes,positions=total,centered_logit_RMS=(centered_square/elements)**.5)
    teacher_differences={a['domain']:dict(mean_kl_difference=b['mean_kl']-a['mean_kl'],strict_count_difference=b['strict_topk_changes']-a['strict_topk_changes'],dense_strict=a['strict_topk_changes'],graph_strict=b['strict_topk_changes']) for a,b in zip(endpoint['dense'],endpoint['graph'])}
    gates=dict(capture_RNG_preserved=forward['capture_CPU_CUDA_RNG_bitexact'],capture_live_gradients_untouched=forward['capture_live_gradients_untouched'],gradient_none_masks_equal=forward['dense_parameter_VJP']['none_mask_equal'] and all(v['none_mask_equal'] for v in gradients.values()),shared_private_parameters_Adam_duals_bitexact=_digest(states['dense'])==_digest(states['graph']),mutual_strict_positions_zero=all(v['strict_changed_positions']==0 for v in mutual.values()),mutual_output_KL=all(max(abs(v['dense_to_graph_KL']),abs(v['graph_to_dense_KL']))<=1e-11 and v['max_token_KL']<=1e-10 for v in mutual.values()),stored_forward_bitexact=forward['bitexact'],raw_gradients=max(v['relative_l2'] for v in gradients.values())<=.003,dense_parameter_VJP=forward['dense_parameter_VJP']['relative_l2']<=.003,Adam_delta=max(v['relative_l2'] for v in deltas.values())<=.003,endpoint_strict_counts_equal=all(v['strict_count_difference']==0 for v in teacher_differences.values()))
    summary=dict(actual_actor_backend='eager',actual_student_backends=['eager_dense','cuda_graph_dense'],actor_graph_pool_coexistence_qualified=False,scope='One saved tied-Q shared/private update with two private inner steps on GPT6/CNN0; same saved Adam/duals, .5 domain weights, strict FP32; strict dense GPT graph execution qualification; no distributed or science claim',preset=dict(raw_gradient_relative_l2=.003,Adam_delta_relative_l2=.003,endpoint_strict_counts_equal=True,mutual_mean_KL=1e-11,mutual_max_token_KL=1e-10),forward=forward,gradients=gradients,Adam_deltas=deltas,teacher_differences=teacher_differences,mutual_outputs=mutual,rank_identical_between_backends=all(v['strict_changed_positions']==0 for v in mutual.values()),endpoint=endpoint,runtimes={k:v['seconds'] for k,v in results.items()},capture_seconds=contexts.capture_seconds,captured_student_training_replays=contexts.student_graph.replays-preflight_replays,captured_student_preflight_replays=preflight_replays,peak_allocated_bytes_by_arm={k:v['peak_allocated_bytes'] for k,v in results.items()},timing_scope='Includes warm context load, teacher reconstruction,2inner+joint student/actor,gradient clones,bothAdam; excludes checkpoint/restore/endpoint diagnostics. Single pair, not speed qualification.',extra_diagnostic_GPT_forwards=diagnostic_compute['gpt_forwards'],extra_diagnostic_GPT_backwards=diagnostic_compute['gpt_backwards'],capture_compute_excluded_from_training_counts=True,capture_warmup_pairs=2,capture_recorded_pairs=1,extra_diagnostic_CNN_forwards=diagnostic_compute['cifar10_forwards'],checkpoint_rebind_scope='Repeated verified reloads of GPT6 and changed edited parameters; no second GPT checkpoint values. Cross-checkpoint CUDA reuse remains unqualified.',gates=gates,all_gates=all(gates.values()),compute={k:v['compute'] for k,v in results.items()})
    (root/'summary.json').write_text(json.dumps(summary,indent=2));torch.save(dict(initial=initial,traces=traces,states=states),root/'numeric_evidence.pt')
    print(json.dumps(summary),flush=True)
    if not summary['all_gates']:raise RuntimeError('dense graph student finite-update qualification failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);run(p.parse_args())
