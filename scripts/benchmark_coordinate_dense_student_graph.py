"""One isolated arm of a matched dense-student-graph ABBA; no production migration."""
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


from scripts.qualify_coordinate_dense_student_graph import Contexts,validate_candidate_recipe
from rewriter.distributed_probe_store import _digest
import statistics
def run(args):
    qualification=json.loads(Path(args.qualification).read_text())
    if not qualification.get('all_gates'):raise ValueError('closed dense graph qualification must pass before timing')
    recipe=json.loads(Path(args.recipe).read_text());validate_candidate_recipe(recipe)
    setup_start=time.perf_counter()
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
    restore()
    setup_seconds=time.perf_counter()-setup_start
    rows=[]
    for index in range(2):
        mode=args.mode
        restore();contexts.graph_enabled=mode=='graph';contexts.compute={};outcomes=[]
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
        rows.append(dict(mode=mode,warmup=index==0,seconds=seconds,compute=dict(contexts.compute),joint_metrics=outcomes,peak_allocated_bytes=torch.cuda.max_memory_allocated(),state_digest=_digest(dict(actor=actor.state_dict(),shared_optimizer=shared_opt.state_dict(),private={d:p.state_dict() for d,p in private.items()},private_optimizers={d:o.state_dict() for d,o in optimizers.items()},duals={d:v.state_dict() for d,v in duals.items()}))))
        (root/'progress.json').write_text(json.dumps(rows,indent=2))
    summary=dict(scope='One local GPT6 + CNN0 update, eight private-inner fits per domain, joint GPT64/CNN1024, both Adam owners/duals. Eager coordinate actor in both arms; strict dense student eager versus CUDA graph. This worker performs one warmup and one measured update, restoring identical saved state before each.',
        exclusions='No distributed SUM, private staging/store/commit, cold source/setup/capture in measured update, endpoint audits. Cached-head reconstruction and repeated context loads included.',
        mode=args.mode,setup_seconds=setup_seconds,capture_seconds=contexts.capture_seconds or 0.,rows=rows,
        numerical_repeat_bitexact=rows[0]['state_digest']==rows[1]['state_digest'])
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--mode',choices=['dense','graph'],required=True);p.add_argument('--qualification',required=True);run(p.parse_args())
