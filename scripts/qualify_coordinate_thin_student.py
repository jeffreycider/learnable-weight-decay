"""Saved tied-program shared/private Adam comparison; no production migration."""
import argparse,copy,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.gpt_precision import configure_precision
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.coordinate_thin_student import ProposalCapture,thin_tokenwise_panel
from rewriter.coordinate_fitting_panels import split_fitting_panels
from rewriter.coordinate_optimization import clip_private_gradients,clip_shared_continuation
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from rewriter.factorized_fp32 import factorized_logits
from scripts.qualify_coordinate_chisel import FixedContexts
from scripts.qualify_coordinate_fullbank import read_joint
from scripts.train_coordinate_chisel import build_shared_adam,set_private_lr,measure_context
from scripts.train_multistate_chisel import build_proposal,norm_preservation_diagnostics


class Contexts(FixedContexts):
    thin=False
    def measure(self,context,actor,private,config,dual,*,learning,weight=1.):
        if not self.thin or context['domain']!='gpt' or not learning:
            return super().measure(context,actor,private,config,dual,learning=learning,weight=weight)
        engine=context['engine'];capture=ProposalCapture(engine)
        with torch.enable_grad():
            parameters,regularizer,stats,_,credit=build_proposal(lambda s:actor(s,private),context['snapshot'],capture,context['grouped'],radius_fraction=.01,norm_squared=context['norm_squared'])
        before=getattr(engine,'cached_teacher_head_forwards',0)
        result=thin_tokenwise_panel(capture,parameters,regularizer,context['inputs'],config=config,dual=dual,state_id=context['local_id'],state_weight=weight,actor_credit=credit,teacher_provider=context['teacher'],backward=True)
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


def run(args):
    enable_deterministic_execution();torch.cuda.set_device(0);configure_precision('fp32');torch.set_num_threads(1)
    device=torch.device('cuda',0);root=Path(args.output);root.mkdir(exist_ok=False)
    recipe=json.loads(Path(args.recipe).read_text());saved,rank_states,private_payloads=read_joint(args.checkpoint)
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
    # Explicit same-stored-weight forward identity; two extra GPT forwards, no backward.
    context=contexts.load('gpt',6);capture=ProposalCapture(context['engine'])
    params,_,_,_,_=build_proposal(lambda s:actor(s,private['gpt']),context['snapshot'],capture,context['grouped'],radius_fraction=.01,norm_squared=context['norm_squared'])
    with torch.no_grad():
        inputs=context['inputs'][:16].to(device);dense=context['engine']._forward(params,inputs)
        thin=factorized_logits(context['engine'].view.gpt,context['engine'].parameters,context['engine'].buffers,inputs,capture.action,capture.scale,realized_parameters=params,linear_backend='ieee',dx_backend='ieee')
        forward=dict(bitexact=torch.equal(dense,thin),max_abs=float((dense-thin).abs().max()))
    del context,capture,params,dense,thin,_
    results={};traces={};states={}
    for mode in ('dense','thin'):
        restore();contexts.thin=mode=='thin';contexts.compute={};trace={};rows=[]
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
        results[mode]=dict(seconds=seconds,rows=rows,compute=dict(contexts.compute))
    gradients={k:relative(traces['dense'][k],traces['thin'][k]) for k in traces['dense']}
    deltas={}
    for family in initial:
        values=[states[m]['shared'] if family=='shared' else states[m]['private'][family] for m in ('dense','thin')]
        deltas[family]=relative({n:values[0][n].detach().cpu()-v for n,v in initial[family].items()},{n:values[1][n].detach().cpu()-v for n,v in initial[family].items()})
    (root/'preliminary.json').write_text(json.dumps(dict(status='endpoint gates not yet checked',forward=forward,gradients=gradients,Adam_deltas=deltas,runtimes={k:v['seconds'] for k,v in results.items()}),indent=2))
    torch.save(dict(initial=initial,traces=traces,states=states),root/'numeric_evidence.pt')
    # True post-update teacher metrics on the same joint fitting panels.
    endpoint={}
    for mode in ('dense','thin'):
        actor.load_state_dict(states[mode]['shared']);contexts.thin=False;rows=[]
        for domain,uid in selected:
            private[domain].load_state_dict(states[mode]['private'][domain]);duals[domain].load_state_dict(states[mode]['duals'][domain])
            context=contexts.load(domain,uid);cfg=recipe['private_inner_panels'][domain]
            _,joint=split_fitting_panels(context,inner_steps=2,inner_size=cfg['inner_size'],joint_size=recipe['panels']['gpt_sequences' if domain=='gpt' else 'cifar_images'],seed=991)
            rows.append(contexts.measure(joint,actor,private[domain],duals[domain].config,duals[domain],learning=False))
        endpoint[mode]=rows
    # Direct mutual edited-output comparison, independent of distance to teacher.
    from rewriter.rank_preservation import topk_preservation_flags
    mutual={}
    for domain,uid in selected:
        context=contexts.load(domain,uid);cfg=recipe['private_inner_panels'][domain]
        _,joint=split_fitting_panels(context,inner_steps=2,inner_size=cfg['inner_size'],joint_size=recipe['panels']['gpt_sequences' if domain=='gpt' else 'cifar_images'],seed=991)
        stored={}
        for mode in ('dense','thin'):
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
                b=joint['engine']._forward(stored['thin'],data).reshape_as(a)
                for offset in range(0,len(a),128):
                    x=a[offset:offset+128].double();y=b[offset:offset+128].double()
                    lp=x.log_softmax(-1);lq=y.log_softmax(-1)
                    ka=(lp.exp()*(lp-lq)).sum(-1);kb=(lq.exp()*(lq-lp)).sum(-1)
                    ab+=float(ka.sum());ba+=float(kb.sum());total+=len(x);maximum=max(maximum,float(ka.max()),float(kb.max()))
                    gap=x-y;gap-=gap.mean(-1,keepdim=True);centered_square+=float(gap.square().sum());elements+=gap.numel()
                    rank_changes+=int(topk_preservation_flags(x,y,k=min(100,x.shape[-1]))['strict_topk_changed'].sum())
                del a,b
        mutual[domain]=dict(dense_to_thin_KL=ab/total,thin_to_dense_KL=ba/total,max_token_KL=maximum,strict_changed_positions=rank_changes,positions=total,centered_logit_RMS=(centered_square/elements)**.5)
    teacher_differences={a['domain']:dict(mean_kl_difference=b['mean_kl']-a['mean_kl'],strict_count_difference=b['strict_topk_changes']-a['strict_topk_changes'],dense_strict=a['strict_topk_changes'],thin_strict=b['strict_topk_changes']) for a,b in zip(endpoint['dense'],endpoint['thin'])}
    gates=dict(mutual_strict_positions_zero=all(v['strict_changed_positions']==0 for v in mutual.values()),mutual_output_KL=all(max(abs(v['dense_to_thin_KL']),abs(v['thin_to_dense_KL']))<=1e-11 and v['max_token_KL']<=1e-10 for v in mutual.values()),stored_forward_bitexact=forward['bitexact'],raw_gradients=max(v['relative_l2'] for v in gradients.values())<=.003,Adam_delta=max(v['relative_l2'] for v in deltas.values())<=.003,endpoint_strict_counts_equal=all(v['strict_count_difference']==0 for v in teacher_differences.values()))
    summary=dict(scope='One saved tied-Q shared/private update with two private inner steps on GPT6/CNN0; same saved Adam/duals, .5 domain weights, strict FP32; component execution qualification, no distributed or science claim',preset=dict(raw_gradient_relative_l2=.003,Adam_delta_relative_l2=.003,endpoint_strict_counts_equal=True,mutual_mean_KL=1e-11,mutual_max_token_KL=1e-10),forward=forward,gradients=gradients,Adam_deltas=deltas,teacher_differences=teacher_differences,mutual_outputs=mutual,rank_identical_between_backends=all(v['strict_changed_positions']==0 for v in mutual.values()),endpoint=endpoint,runtimes={k:v['seconds'] for k,v in results.items()},timing_scope='Includes warm context load, teacher reconstruction,2inner+joint student/actor,gradient clones,bothAdam; excludes checkpoint/restore/endpoint diagnostics. Single pair, not speed qualification.',extra_diagnostic_GPT_forwards=18,extra_diagnostic_CNN_forwards=32,gates=gates,all_gates=all(gates.values()),compute={k:v['compute'] for k,v in results.items()})
    (root/'summary.json').write_text(json.dumps(summary,indent=2));torch.save(dict(initial=initial,traces=traces,states=states),root/'numeric_evidence.pt')
    print(json.dumps(summary),flush=True)
    if not summary['all_gates']:raise RuntimeError('thin student finite-update qualification failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);run(p.parse_args())
