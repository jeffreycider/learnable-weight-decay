"""Bounded GPU paired qualification only; no fitting or speed claim.

Fresh Adam per repetition, private eps=1e-8 in BOTH arms. GPU-generated teacher
cached once for the supplied checkpoint state; source paths are explicit and
verified against each actor checkpoint's embedded metadata. Timed updates include
live sensing/proposal/norm, GPT forward/backward, normalization, clipping, Adam
and dual ascent. Restores, setup, diagnostic CPU copies and writes are excluded.
"""
import argparse,dataclasses,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.model import GPT
from rewriter.adaptive_preservation import build_adaptive_proposal
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.edit_scope import editable_names,editable_weight_norm_sq
from rewriter.experimental_factor_panel import FactorPanelCapture
from rewriter.experimental_factor_norm import prepare_factor_panel
from rewriter.function_preservation import FunctionalPreservation
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.prefix_cache import unpack_record
from rewriter.tokenwise_reference import TokenwiseDual,tokenwise_panel
from scripts.diagnose_cpu_norm_preservation import load_bound_state,objective_config,pin_sources,tensor_sha
from scripts.fit_cpu_edit_teacher import sha
from scripts.prepare_cpu_teacher_state import validate_train_panel


def cpu_parameters(module,grad=False):
    return {n:(None if p.grad is None else p.grad.detach().cpu().clone()) if grad else p.detach().cpu().clone() for n,p in module.named_parameters()}


def compare(reference,candidate,seeds):
    report={};passed=reference['loss']==candidate['loss'] and reference['compute']==candidate['compute']
    for role in ('heads','body','private'):
        section='private' if role=='private' else 'actor'
        names=[n for n in seeds[section] if role=='private' or n.startswith(('matrix_output.','vector_output.'))==(role=='heads')]
        for kind,limit in [('gradients',1e-4),('delta',1e-3)]:
            err=0.;den=0.;maxabs=0.;none_match=True
            for n in names:
                if kind=='gradients':a=reference[kind][section][n];b=candidate[kind][section][n]
                else:a=reference['post'][section][n].double()-seeds[section][n].double();b=candidate['post'][section][n].double()-seeds[section][n].double()
                if a is None or b is None:
                    none_match &= (a is None)==(b is None)
                    continue
                d=a.double()-b.double();err+=float(d.square().sum());den+=float(a.double().square().sum());maxabs=max(maxabs,float(d.abs().max()))
            relative=(err/max(den,1e-300))**.5
            ok=none_match and relative<=limit;passed &= ok
            report[role+'_'+kind]=dict(relative_l2=relative,max_abs=maxabs,none_pattern_equal=none_match,threshold=limit,passed=ok)
    return dict(passed=passed,exact_loss=reference['loss']==candidate['loss'],exact_compute=reference['compute']==candidate['compute'],roles=report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,action='append',required=True,help='Repeat for active and near-preserving actor snapshots')
    for name in ('state','panel','panel-metadata','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case-id',type=int,default=6);p.add_argument('--warmups',type=int,default=1);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--factor-chunk-tokens',type=int,default=128)
    p.add_argument('--streamed-norm',action='store_true',help='Direct arm only: stream rounded norm cotangents and discard dense proposal graphs')
    p.add_argument('--norm-chunk-rows',type=int,default=128)
    args=p.parse_args()
    if not 1<=args.norm_chunk_rows<=2048:raise ValueError('bounded positive norm chunk rows required')
    if not 1<=args.warmups<=2 or not 1<=args.repeats<=3 or not 1<=len(args.checkpoint)<=2:raise ValueError('bounded paired qualification required')
    if not 1<=args.factor_chunk_tokens<=2048:raise ValueError('bounded positive factor chunk size required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; script does not lease GPUs')
    torch.set_num_threads(1);torch.manual_seed(930067)
    args.output.mkdir(parents=True,exist_ok=False);sources=pin_sources(args.output)
    hashes={k:sha(getattr(args,k)) for k in ('state','panel','panel_metadata')}
    state=torch.load(args.state,map_location='cpu',weights_only=False,mmap=True)
    panel=torch.load(args.panel,map_location='cpu',weights_only=False,mmap=True);pm=json.loads(args.panel_metadata.read_text())
    validate_train_panel(panel,pm,hashes['state'],hashes['panel'])
    inputs=panel['inputs'][:2].clone();ids=pm['sequence_ids'][:2];del panel
    if len(inputs)!=2 or state['completed_steps']!=args.case_id:raise ValueError('exactly two sequences and matching state required')
    snapshot=unpack_record(state['observation'],device='cuda').snapshot
    base=state['model'];depth=len({n.split('.')[1] for n in base if n.startswith('blocks.')})
    model=GPT(base['embed.weight'].shape[0],depth,base['embed.weight'].shape[1]).float().cuda().eval().requires_grad_(False)
    engine=FunctionalPreservation(model,microbatch_sequences=1,chunk_tokens=128,teacher_cache_device='cuda',kl_dtype=torch.float64,gpt_precision='fp32')
    load_bound_state(engine,base,hashes['state'])
    teachers=engine.cache_teacher(inputs,data_role='past_training',sequence_ids=ids)
    teacher_setup=engine.compute.as_dict()
    grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
    editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
    norm_squared=editable_weight_norm_sq(engine.parameters,editable);config=objective_config()
    metadata=dict(scope=__doc__,norm_modes=dict(dense='dense',direct='streamed' if args.streamed_norm else 'dense'),norm_chunk_rows=args.norm_chunk_rows,source_sha256=sources,input_sha256=hashes,selected_inputs_sha256=tensor_sha(inputs),sequence_ids=ids,
        checkpoints=[dict(path=str(c),sha256=sha(c)) for c in args.checkpoint],torch_version=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
        teacher_setup_compute=teacher_setup,objective=dataclasses.asdict(config),state_weight=.5,private_eps=1e-8,shared_eps=1e-8,
        policy='full',optimizer='fresh Adam each arm/repetition; no inherited moments',lrs=dict(heads=1e-8,body=1e-7,private=1e-6),betas=[.9,.99],clip=1.,warmups=args.warmups,repeats=args.repeats,factor_chunk_tokens=args.factor_chunk_tokens)
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    all_rows=[]
    for checkpoint_index,path in enumerate(args.checkpoint):
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        case=next(c for c in saved['metadata']['cases'] if c['state_id']==args.case_id)
        if any(case['sha256'][k]!=v for k,v in hashes.items()) or case['sequence_ids']!=ids or case['input_sha256']!=tensor_sha(inputs):raise ValueError('embedded checkpoint state/panel identity mismatch')
        if saved['metadata']['objective']!=dataclasses.asdict(config):raise ValueError('checkpoint objective differs')
        actor=GroupedCoordinateChisel(rank=8,trajectory_dim=400,model_dim=256,num_layers=2,latent_tokens=8,fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32').cuda().eval()
        actor.load_state_dict(saved['actor'],strict=True)
        private=actor.make_private(snapshot,state_key=tuple(saved['private'][args.case_id]['_extra_state']['state_key']),seed=0)
        private.load_state_dict(saved['private'][args.case_id],strict=True)
        seeds=dict(actor=cpu_parameters(actor),private=cpu_parameters(private))
        heads=[p for n,p in actor.named_parameters() if n.startswith(('matrix_output.','vector_output.'))]
        body=[p for n,p in actor.named_parameters() if not n.startswith(('matrix_output.','vector_output.'))]
        reference=None
        for iteration in range(args.warmups+args.repeats):
            order=('dense','direct') if iteration%2==0 else ('direct','dense')
            for backend in order:
                if any(sha(n)!=h for n,h in sources.items()):raise ValueError('pinned source changed')
                actor.load_state_dict(saved['actor']);private.load_state_dict(saved['private'][args.case_id]);actor.zero_grad(set_to_none=True);private.zero_grad(set_to_none=True)
                shared_opt=torch.optim.Adam([dict(params=heads,lr=1e-8),dict(params=body,lr=1e-7)],betas=(.9,.99),eps=1e-8)
                private_opt=torch.optim.Adam(private.parameters(),lr=1e-6,betas=(.9,.99),eps=1e-8)
                dual=TokenwiseDual([args.case_id],config=config,dual_lr=.01);dual.multipliers[args.case_id]=saved['dual']['multipliers'][args.case_id]
                capture=FactorPanelCapture(chunk_tokens=args.factor_chunk_tokens) if backend=='direct' else None
                before=engine.compute.as_dict();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                def proposal(scoped):return actor(scoped,private,weight_cores=query_weight_cores(engine.parameters,private))
                parameters,regularizer,stats,_,credit=build_adaptive_proposal(proposal,snapshot,engine,grouped,radius_fraction=.01,norm_squared=norm_squared,execution_capture=capture)
                if backend=='direct' and args.streamed_norm:
                    parameters,regularizer=prepare_factor_panel(parameters,grouped,capture,transpose_storage=engine.transpose_storage,chunk_rows=args.norm_chunk_rows)
                run=tokenwise_panel if capture is None else capture.run
                loss=run(engine,parameters,regularizer,inputs,config=config,dual=dual,state_id=args.case_id,state_weight=.5,actor_credit=credit,teacher_provider=lambda a,b:teachers[a],backward=True)
                for parameter in private.parameters():
                    if parameter.grad is not None:parameter.grad.mul_(2.)
                torch.cuda.synchronize();fw_bw_norm_seconds=time.perf_counter()-start
                grads=dict(actor=cpu_parameters(actor,True),private=cpu_parameters(private,True))
                torch.cuda.synchronize();start=time.perf_counter()
                sn=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True));pn=float(torch.nn.utils.clip_grad_norm_(private.parameters(),1.,error_if_nonfinite=True))
                shared_opt.step();private_opt.step();dual.step([(args.case_id,loss['mean_kl'])],expected_ids=[args.case_id])
                torch.cuda.synchronize();optimizer_seconds=time.perf_counter()-start
                compute={k:v-before[k] for k,v in engine.compute.as_dict().items()}
                artifact=dict(loss=loss,compute=compute,gradients=grads,post=dict(actor=cpu_parameters(actor),private=cpu_parameters(private)))
                row=dict(checkpoint_index=checkpoint_index,backend=backend,norm_mode='streamed' if backend=='direct' and args.streamed_norm else 'dense',iteration=iteration,warmup=iteration<args.warmups,
                    update_seconds=fw_bw_norm_seconds+optimizer_seconds,fw_bw_normalization_seconds=fw_bw_norm_seconds,optimizer_seconds=optimizer_seconds,
                    fresh_weight_core_queries=1,compute=compute,loss=loss,proposal=stats,shared_preclip_norm=sn,private_preclip_norm=pn,
                    max_memory_allocated=torch.cuda.max_memory_allocated(),max_memory_reserved=torch.cuda.max_memory_reserved())
                if reference is None:reference=artifact
                gate=compare(reference,artifact,seeds);row['gate']=gate
                torch.save(artifact,args.output/f'checkpoint{checkpoint_index}_{backend}_last.pt')
                all_rows.append(row)
                with (args.output/'metrics.jsonl').open('a') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
                if not gate['passed']:raise RuntimeError('numerical qualification failed; stop before further timing repetitions')
                del parameters,regularizer,credit,capture,grads,artifact,shared_opt,private_opt,run
        del actor,private,heads,body,seeds,reference,saved
    (args.output/'complete.json').write_text(json.dumps(dict(metadata=metadata,rows=all_rows,compute=engine.compute.as_dict(),qualified=True),indent=2)+'\n')


if __name__=='__main__':main()
