"""Actual adaptive query+actor graph eight-inner qualification/timing."""
import argparse,copy,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.gpt_precision import configure_precision
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.coordinate_fitting_panels import split_fitting_panels
from rewriter.coordinate_optimization import clip_private_gradients,clip_shared_continuation
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from scripts.qualify_coordinate_chisel import FixedContexts
from scripts.qualify_coordinate_fullbank import read_joint
from scripts.train_coordinate_chisel import build_shared_adam,set_private_lr,measure_context
from scripts.train_multistate_chisel import build_proposal,norm_preservation_diagnostics


from scripts.train_coordinate_chisel import build_actor
from rewriter.adaptive_actor_graph import AdaptiveActorGraph,sensed_parameters
from rewriter.private_probe_slot import PrivateProbeSlot
from rewriter.edit_scope import editable_names
from rewriter.distributed_probe_store import _digest
import statistics

def relative(a,b):
    names=set(a)|set(b);sq=base=maximum=0.
    for n in names:
        x=a.get(n);y=b.get(n)
        if x is None:x=torch.zeros_like(y)
        if y is None:y=torch.zeros_like(x)
        x=x.detach().cpu().double();y=y.detach().cpu().double();delta=x-y
        sq+=float(delta.square().sum());base+=float(x.square().sum());maximum=max(maximum,float(delta.abs().max()))
    return dict(relative_l2=(sq/max(base,1e-300))**.5,absolute_l2=sq**.5,max_abs=maximum,none_mask_equal=set(a)==set(b))


class Contexts(FixedContexts):
    graph_enabled=False
    graphs=None
    capture_seconds=None
    rng_equal=True
    forward_equal=True
    def prepare(self,actor,private,selected):
        actor.zero_grad(set_to_none=True)
        for p in private.values():p.zero_grad(set_to_none=True)
        slots={d:PrivateProbeSlot(actor,p) for d,p in private.items()}
        graphs={};start=time.perf_counter();cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state(self.device)
        for domain,uid in selected:
            c=super().load(domain,uid);e=c['engine'];weights=sensed_parameters(e)
            embeddings=tuple(n for n,t in e.transpose_storage.items() if t)
            editable=editable_names(e.parameters,'grouped')-c['grouped'].frozen_parameter_names
            for shared in (False,True):
                actor.requires_grad_(shared)
                graph=AdaptiveActorGraph(actor,slots[domain],c['snapshot'],editable,weights,embedding_names=embeddings)
                graphs[(domain,shared)]=graph
                graph.bind_scope(private[domain],editable)
                with torch.no_grad():
                    expected=graph.wrapper.action(type(graph.wrapper).forward(graph.wrapper,graph.dummy))
                    actual=graph(c['snapshot'],private[domain])
                    self.forward_equal=self.forward_equal and all(torch.equal(getattr(actual.matrices[n],f),getattr(expected.matrices[n],f)) for n in editable if n in actual.matrices for f in ('U','C','V')) and all(torch.equal(actual.vectors[n],expected.vectors[n]) for n in editable if n in actual.vectors)
                del expected,actual
                actor.zero_grad(set_to_none=True);private[domain].zero_grad(set_to_none=True)
        actor.requires_grad_(True);torch.cuda.synchronize();self.capture_seconds=time.perf_counter()-start
        self.rng_equal=torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state(self.device));self.graphs=graphs
    def load(self,domain,*a,**kw):
        c=super().load(domain,*a,**kw)
        if self.graphs:
            e=c['engine'];weights=sensed_parameters(e);embeddings=tuple(n for n,t in e.transpose_storage.items() if t)
            for shared in (False,True):self.graphs[(domain,shared)].bind_weights(weights,embedding_names=embeddings)
        return c
    def measure(self,c,actor,private,config,dual,*,learning,weight=1.):
        if not self.graph_enabled or not learning:return super().measure(c,actor,private,config,dual,learning=learning,weight=weight)
        graph=self.graphs[(c['domain'],any(p.requires_grad for p in actor.parameters()))]
        editable=editable_names(c['engine'].parameters,'grouped')-c['grouped'].frozen_parameter_names
        graph.bind_scope(private,editable)
        row=super().measure(c,graph,private,config,dual,learning=learning,weight=weight)
        return row

def run(args):
    recipe=json.loads(Path(args.recipe).read_text());assert recipe['action_family']=='adaptive_ucv_v1' and recipe['architecture']['fresh_weight_core']
    setup_start=time.perf_counter()
    enable_deterministic_execution();torch.cuda.set_device(0);configure_precision('fp32');torch.set_num_threads(1)
    device=torch.device('cuda',0);root=Path(args.output);root.mkdir(exist_ok=False)
    saved,rank_states,private_payloads=read_joint(args.checkpoint)
    base=json.loads(Path(recipe['base_recipe']).read_text());contexts=Contexts(base,device,gpt_sequences=128,cifar_images=2048,action_family=recipe['action_family'])
    actor=build_actor(recipe).to(device);shared_opt=build_shared_adam(actor,recipe['shared_optimizer'])
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
    restore()
    setup_seconds=time.perf_counter()-setup_start
    rows=[];traces={}
    modes=['eager','graph'] if args.mode=='qualify' else [args.mode,args.mode]
    for index,mode in enumerate(modes):
        restore()
        if mode=='graph' and contexts.graphs is None:
            contexts.prepare(actor,private,selected);restore()
        contexts.graph_enabled=mode=='graph';contexts.compute={};outcomes=[];trace=[]
        def capture_grads(parameters):
            if args.mode=='qualify':trace.append({str(i):p.grad.detach().cpu().clone() for i,p in enumerate(parameters) if p.grad is not None})
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize();start=time.perf_counter()
        for domain,uid in selected:
            context=contexts.load(domain,uid);cfg=recipe['private_inner_panels'][domain]
            inner,joint=split_fitting_panels(context,inner_steps=8,inner_size=cfg['inner_size'],joint_size=recipe['panels']['gpt_sequences' if domain=='gpt' else 'cifar_images'],seed=991)
            p=private[domain];opt=optimizers[domain];dual=duals[domain];tick=payloads[domain]['visits']
            actor.requires_grad_(False)
            for i,part in enumerate(inner):
                p.zero_grad(set_to_none=True);set_private_lr(opt,recipe['private_optimizer'],tick+i)
                row=measure_context(contexts,part,actor,p,dual.config,dual,learning=True,weight=1.)
                capture_grads(p.parameters());clip_private_gradients(p.parameters(),1.);opt.step();p.zero_grad(set_to_none=True)
                dual.step([(uid,row['mean_kl'])],expected_ids=[uid])
            actor.requires_grad_(True);set_private_lr(opt,recipe['private_optimizer'],tick+8)
            active=actor
            row=measure_context(contexts,joint,active,p,dual.config,dual,learning=True,weight=.5)
            for parameter in p.parameters():
                if parameter.grad is not None:parameter.grad.div_(.5)
            capture_grads(p.parameters());clip_private_gradients(p.parameters(),1.);opt.step();p.zero_grad(set_to_none=True)
            dual.step([(uid,row['mean_kl'])],expected_ids=[uid]);outcomes.append(row)
        capture_grads(actor.parameters())
        clip_shared_continuation(shared_opt,'separate_representation',max_norm=1.);shared_opt.step();actor.zero_grad(set_to_none=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        traces[mode]=trace
        rows.append(dict(mode=mode,warmup=index==0 and args.mode!='qualify',seconds=seconds,compute=dict(contexts.compute),joint_metrics=outcomes,peak_allocated_bytes=torch.cuda.max_memory_allocated(),state_digest=_digest(dict(actor=actor.state_dict(),shared_optimizer=shared_opt.state_dict(),private={d:p.state_dict() for d,p in private.items()},private_optimizers={d:o.state_dict() for d,o in optimizers.items()},duals={d:v.state_dict() for d,v in duals.items()}))))
        (root/'progress.json').write_text(json.dumps(rows,indent=2))
    summary=dict(scope='Immutable adaptive fresh-core qualifier4; local GPT6+CNN0,8privateinner per domain,jointGPT64/CNN1024,saved shared/privateAdam+duals. Eager versus captured live core+actor; strict dense GPT and relative100 loss unchanged.',
        exclusions='No distributed SUM, private staging/store/commit, cold source/setup/capture in measured update, endpoint audits. Qualification timers include gradient exports and are not performance comparisons. Cached-head reconstruction and repeated context loads included.',
        mode=args.mode,setup_seconds=setup_seconds,capture_seconds=contexts.capture_seconds or 0.,capture_RNG_bitexact=contexts.rng_equal,rows=rows,
        numerical_repeat_bitexact=rows[0]['state_digest']==rows[1]['state_digest'])
    if args.mode=='qualify':
        errors=[relative(a,b) for a,b in zip(traces['eager'],traces['graph'])]
        gates=dict(same_weight_action_forward_bitexact=contexts.forward_equal,raw_gradients=len(traces['eager'])==len(traces['graph']) and max(x['relative_l2'] for x in errors)<=.003,none_masks=all(x['none_mask_equal'] for x in errors),all_final_owners_bitexact=rows[0]['state_digest']==rows[1]['state_digest'],compute_counts_equal=rows[0]['compute']==rows[1]['compute'],capture_RNG_bitexact=contexts.rng_equal,strict_counts_equal=all(a['strict_topk_changes']==b['strict_topk_changes'] for a,b in zip(rows[0]['joint_metrics'],rows[1]['joint_metrics'])))
        summary.update(gradient_errors=errors,gates=gates,all_gates=all(gates.values()))
        torch.save(traces,root/'gradient_evidence.pt')
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    if args.mode=='qualify' and not summary['all_gates']:raise RuntimeError('adaptive actor graph qualification gate failed')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--mode',choices=['qualify','eager','graph'],required=True);run(p.parse_args())
