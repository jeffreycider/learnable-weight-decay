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


from scripts.qualify_coordinate_thin_student import Contexts
from scripts.train_coordinate_chisel import ActorDispatch
from rewriter.private_probe_slot import PrivateProbeSlot
import statistics
from rewriter.coordinate_actor_graph import PrivateActorDispatch
from scripts.qualify_coordinate_thin_student import grads,relative,actor_state
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
    restore()
    templates={d:contexts.load(d,u) for d,u in selected}
    dispatch=ActorDispatch(actor,{d:PrivateProbeSlot(actor,p) for d,p in private.items()},templates,'eager_actor_graph')
    private_dispatch=PrivateActorDispatch(actor,{d:PrivateProbeSlot(actor,p) for d,p in private.items()},templates)
    del templates
    restore()
    rows=[];states={};traces={}
    for index,mode in enumerate(['dense','graph']):
        restore();contexts.thin=False;contexts.compute={};trace={};metrics=[]
        torch.cuda.synchronize();start=time.perf_counter()
        for domain,uid in selected:
            context=contexts.load(domain,uid);cfg=recipe['private_inner_panels'][domain]
            inner,joint=split_fitting_panels(context,inner_steps=8,inner_size=cfg['inner_size'],joint_size=recipe['panels']['gpt_sequences' if domain=='gpt' else 'cifar_images'],seed=991)
            p=private[domain];opt=optimizers[domain];dual=duals[domain];tick=payloads[domain]['visits']
            actor.requires_grad_(False)
            for i,part in enumerate(inner):
                p.zero_grad(set_to_none=True);set_private_lr(opt,recipe['private_optimizer'],tick+i)
                active=private_dispatch.for_visit(domain,part,p) if mode=='graph' else actor
                row=measure_context(contexts,part,active,p,dual.config,dual,learning=True,weight=1.)
                trace[f'{domain}/inner{i}']=grads(p);metrics.append(row)
                clip_private_gradients(p.parameters(),1.);opt.step();p.zero_grad(set_to_none=True)
                dual.step([(uid,row['mean_kl'])],expected_ids=[uid])
            actor.requires_grad_(True);set_private_lr(opt,recipe['private_optimizer'],tick+8)
            active=dispatch.for_visit(domain,joint,p)
            row=measure_context(contexts,joint,active,p,dual.config,dual,learning=True,weight=.5)
            for parameter in p.parameters():
                if parameter.grad is not None:parameter.grad.div_(.5)
            trace[domain+'/joint']=grads(p);metrics.append(row)
            clip_private_gradients(p.parameters(),1.);opt.step();p.zero_grad(set_to_none=True)
            dual.step([(uid,row['mean_kl'])],expected_ids=[uid])
        trace['shared']=grads(actor)
        clip_shared_continuation(shared_opt,'separate_representation',max_norm=1.);shared_opt.step();actor.zero_grad(set_to_none=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        rows.append(dict(mode=mode,seconds=seconds,compute=dict(contexts.compute),metrics=metrics))
        traces[mode]={k:{n:v.cpu() for n,v in value.items()} for k,value in trace.items()}
        states[mode]=dict(shared=actor_state(actor),private={d:copy.deepcopy(p.state_dict()) for d,p in private.items()},optimizer=copy.deepcopy(shared_opt.state_dict()),private_optimizer={d:copy.deepcopy(o.state_dict()) for d,o in optimizers.items()},duals={d:v.state_dict() for d,v in duals.items()})
        (root/'progress.json').write_text(json.dumps(rows,indent=2))
    def exact(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
        return a==b
    gradient_checks={k:relative(traces['dense'][k],traces['graph'][k]) for k in traces['dense']}
    gates=dict(all_numeric_owners_exact=exact(states['dense'],states['graph']),all_raw_gradients_exact=exact(traces['dense'],traces['graph']),all_loss_and_rank_metrics_exact=rows[0]['metrics']==rows[1]['metrics'],compute_exact=rows[0]['compute']==rows[1]['compute'])
    summary=dict(scope='One local GPT6/CNN0 complete visit, eight private inner Adam+dual steps/domain, joint GPT64/CNN1024, shared+private Adam; joint actor graph both, private actor eager versus graph. Same saved numeric state; no production mutation.',gates=gates,all_gates=all(gates.values()),gradient_checks=gradient_checks,rows=rows,timing_limitation='One ordered pair including trace cloning; not matched speed qualification.')
    torch.save(dict(states=states,traces=traces),root/'numeric_evidence.pt')
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    if not summary['all_gates']:raise RuntimeError('private-inner complete lifecycle exactness gate failed')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);run(p.parse_args())
