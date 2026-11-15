"""Isolated immutable adaptive fixture CPU/CUDA profile; no science migration."""
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
Contexts=FixedContexts
from rewriter.distributed_probe_store import _digest
import statistics

def install_ranges():
    from contextlib import contextmanager
    import functools
    import rewriter.adaptive_coordinate_chisel as ac
    import rewriter.adaptive_sensing as sensing
    import rewriter.adaptive_preservation as preservation
    import rewriter.tokenwise_reference as token
    import rewriter.image_tokenwise_panel as image
    from torch.profiler import record_function
    active={'domain':'setup','tensor_backward':False}
    def wrap(obj,name,label):
        old=getattr(obj,name)
        @functools.wraps(old)
        def call(*a,**kw):
            with record_function('phase/'+active['domain']+'/'+label):return old(*a,**kw)
        setattr(obj,name,call)
    wrap(ac.AdaptiveCoordinateChisel,'forward','actor_forward')
    wrap(sensing,'query_weight_cores','fresh_core_query_QR')
    wrap(preservation,'build_adaptive_proposal','proposal_norm_materialization')
    wrap(token,'logit_terms_and_cotangent','loss_metrics_cotangent')
    wrap(image,'logit_terms_and_cotangent','loss_metrics_cotangent')
    wrap(torch.optim.Adam,'step','Adam')
    oldload=FixedContexts.load
    def load(self,domain,*a,**kw):
        active['domain']=domain
        with record_function('phase/'+domain+'/context_load'):
            c=oldload(self,domain,*a,**kw)
        e=c['engine']
        if not getattr(e,'_profile_wrapped',False):
            wrap(e,'_forward','student_forward');e._profile_wrapped=True
        original=c['teacher']
        def teacher(*a,**kw):
            with record_function('phase/'+domain+'/teacher_head_logprobs'):return original(*a,**kw)
        c['teacher']=teacher
        return c
    FixedContexts.load=load
    oldmeasure=FixedContexts.measure
    def measure(self,c,*a,**kw):
        active['domain']=c['domain']
        with record_function('phase/'+c['domain']+'/finite_measure'):return oldmeasure(self,c,*a,**kw)
    FixedContexts.measure=measure
    oldtb=torch.Tensor.backward;oldab=torch.autograd.backward
    def tb(self,*a,**kw):
        with record_function('phase/'+active['domain']+'/student_backward'):
            active['tensor_backward']=True
            try:return oldtb(self,*a,**kw)
            finally:active['tensor_backward']=False
    def ab(*a,**kw):
        if active['tensor_backward']:return oldab(*a,**kw)
        with record_function('phase/'+active['domain']+'/actor_materialization_backward'):return oldab(*a,**kw)
    torch.Tensor.backward=tb;torch.autograd.backward=ab

def run(args):
    recipe=json.loads(Path(args.recipe).read_text());assert recipe['action_family']=='adaptive_ucv_v1' and recipe['architecture']['fresh_weight_core']
    install_ranges()
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
    rows=[]
    for index in range(2):
        mode='adaptive_eager'
        restore();contexts.compute={};outcomes=[]
        prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) if index==1 else None
        if prof:prof.start()
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
                clip_private_gradients(p.parameters(),1.);opt.step();p.zero_grad(set_to_none=True)
                dual.step([(uid,row['mean_kl'])],expected_ids=[uid])
            actor.requires_grad_(True);set_private_lr(opt,recipe['private_optimizer'],tick+8)
            active=actor
            row=measure_context(contexts,joint,active,p,dual.config,dual,learning=True,weight=.5)
            for parameter in p.parameters():
                if parameter.grad is not None:parameter.grad.div_(.5)
            clip_private_gradients(p.parameters(),1.);opt.step();p.zero_grad(set_to_none=True)
            dual.step([(uid,row['mean_kl'])],expected_ids=[uid]);outcomes.append(row)
        clip_shared_continuation(shared_opt,'separate_representation',max_norm=1.);shared_opt.step();actor.zero_grad(set_to_none=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        if prof:
            prof.stop();prof.export_chrome_trace(str(root/'trace.json'))
            (root/'operators.json').write_text(json.dumps([dict(key=e.key,count=e.count,cpu_total_us=e.cpu_time_total,cpu_self_us=e.self_cpu_time_total,device_total_us=e.device_time_total,device_self_us=e.self_device_time_total) for e in prof.key_averages()]))
        rows.append(dict(mode=mode,warmup=index==0,seconds=seconds,compute=dict(contexts.compute),joint_metrics=outcomes,peak_allocated_bytes=torch.cuda.max_memory_allocated(),state_digest=_digest(dict(actor=actor.state_dict(),shared_optimizer=shared_opt.state_dict(),private={d:p.state_dict() for d,p in private.items()},private_optimizers={d:o.state_dict() for d,o in optimizers.items()},duals={d:v.state_dict() for d,v in duals.items()}))))
        (root/'progress.json').write_text(json.dumps(rows,indent=2))
    summary=dict(scope='Closed adaptive fresh-core qualifier update4 fixture; one local GPT6+CNN0 update, eight private-inner fits each, joint GPT64/CNN1024; actual saved shared/private Adam and duals. One unprofiled warmup then same-state profiler replay. All adaptive/GPT arithmetic unchanged; profiled timing is diagnostic, not speedup evidence.',
        exclusions='No distributed SUM or private store/commit. Private saved-state restore occurs before timer; repeated verified context loads and cached-head reconstruction inside. Profiler overhead included in second row.',
        mode=mode,setup_seconds=setup_seconds,capture_seconds=0.,rows=rows,
        numerical_repeat_bitexact=rows[0]['state_digest']==rows[1]['state_digest'])
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);run(p.parse_args())
