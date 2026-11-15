"""Matched real-checkpoint CPU qualification, not Chisel fitting or GPU timing.

Every repetition restores the same active-loss actor/private state and starts
fresh Adam. The measured update includes live sensing, proposal/norm, student
forwards/backwards, gradient clipping and Adam. Teacher-cache verification,
initialization, restoration, diagnostic copies and artifact writes are excluded.
"""
import argparse
import dataclasses
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baseline.model import GPT
from rewriter.adaptive_preservation import build_adaptive_proposal
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.edit_scope import editable_names, editable_weight_norm_sq
from rewriter.experimental_factor_panel import FactorPanelCapture
from rewriter.experimental_factor_norm import prepare_factor_panel
from rewriter.function_preservation import FunctionalPreservation
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.prefix_cache import unpack_record
from rewriter.tokenwise_reference import TokenwiseDual, tokenwise_panel
from scripts.diagnose_cpu_norm_preservation import DiskTeacherCache, load_bound_state, objective_config, pin_sources
from scripts.fit_cpu_edit_teacher import sha
from scripts.prepare_cpu_teacher_state import validate_train_panel


def gradients(module):
    return {n: None if p.grad is None else p.grad.detach().clone() for n,p in module.named_parameters()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['dense','direct'],required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case-id',type=int,default=6)
    parser.add_argument('--teacher-cache-root',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--warmups',type=int,default=1)
    parser.add_argument('--private-eps',type=float,default=1e-12)
    parser.add_argument('--streamed-norm',action='store_true',help='Direct arm only: stream rounded norm cotangents and discard dense proposal graphs')
    parser.add_argument('--norm-chunk-rows',type=int,default=128)
    args=parser.parse_args()
    if not 1<=args.norm_chunk_rows<=2048:raise ValueError('bounded positive norm chunk rows required')
    if args.streamed_norm and args.backend!='direct':raise ValueError('streamed norm requires direct backend')
    if not 1<=args.repeats<=5 or not 0<=args.warmups<=2:raise ValueError('bounded qualification required')
    if not math.isfinite(args.private_eps) or args.private_eps<=0:raise ValueError('positive finite private epsilon required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(930067)
    sources=pin_sources(args.output)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    case=next(c for c in saved['metadata']['cases'] if c['state_id']==args.case_id)
    for key,path in case['paths'].items():
        if sha(path)!=case['sha256'][key]:raise ValueError('changed input: '+path)
    state=torch.load(case['paths']['state'],map_location='cpu',weights_only=False,mmap=True)
    panel=torch.load(case['paths']['panel'],map_location='cpu',weights_only=False,mmap=True)
    panel_meta=json.loads(Path(case['paths']['panel_metadata']).read_text())
    validate_train_panel(panel,panel_meta,case['sha256']['state'],case['sha256']['panel'])
    sequences=saved['metadata']['args']['sequences']
    inputs=panel['inputs'][:sequences].clone();ids=panel_meta['sequence_ids'][:sequences]
    if ids!=case['sequence_ids']:raise ValueError('panel selection changed')
    del panel
    snapshot=unpack_record(state['observation'],device='cpu').snapshot
    actor=GroupedCoordinateChisel(rank=8,trajectory_dim=400,model_dim=256,num_layers=2,
        latent_tokens=8,fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32').eval()
    actor.load_state_dict(saved['actor'],strict=True)
    seed=torch.load(case['paths']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    private=actor.make_private(snapshot,state_key=('cpu-teaching',0,args.case_id,case['sha256']['state']),seed=seed['metadata']['args']['private_seed'])
    private.load_state_dict(saved['private'][args.case_id],strict=True);del seed
    base=state['model'];depth=len({n.split('.')[1] for n in base if n.startswith('blocks.')})
    model=GPT(base['embed.weight'].shape[0],depth,base['embed.weight'].shape[1]).float().eval().requires_grad_(False)
    engine=FunctionalPreservation(model,microbatch_sequences=1,chunk_tokens=128,teacher_cache_device='cpu',kl_dtype=torch.float64,gpt_precision='fp32')
    load_bound_state(engine,base,case['sha256']['state'])
    provider=DiskTeacherCache(args.teacher_cache_root/f'teacher_state{args.case_id}',engine,inputs,ids,case['sha256']['state'])
    grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
    editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
    norm_squared=editable_weight_norm_sq(engine.parameters,editable)
    heads=[p for n,p in actor.named_parameters() if n.startswith(('matrix_output.','vector_output.'))]
    body=[p for n,p in actor.named_parameters() if not n.startswith(('matrix_output.','vector_output.'))]
    config=objective_config()
    metadata=dict(scope=__doc__,backend=args.backend,norm_mode='streamed' if args.streamed_norm else 'dense',norm_chunk_rows=args.norm_chunk_rows,checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),
        case=case,sequences=sequences,objective=dataclasses.asdict(config),source_sha256=sources,
        warmups=args.warmups,repeats=args.repeats,torch_version=torch.__version__,threads=2,
        initial_optimizer='fresh Adam each repetition; no inherited moments',state_weight=.5,
        optimizer_hyperparameters=dict(head_lr=1e-8,body_lr=1e-7,private_lr=1e-6,
            shared_eps=1e-8,private_eps=args.private_eps,betas=[.9,.99],clip=1.),
        teacher_cache_manifest_sha256=sha(args.teacher_cache_root/f'teacher_state{args.case_id}'/'manifest.json'))
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    rows=[]
    for repetition in range(args.warmups+args.repeats):
        if any(sha(p)!=digest for p,digest in sources.items()):raise ValueError('execution source changed')
        actor.load_state_dict(saved['actor']);private.load_state_dict(saved['private'][args.case_id])
        actor.zero_grad(set_to_none=True);private.zero_grad(set_to_none=True)
        shared_opt=torch.optim.Adam([dict(params=heads,lr=1e-8),dict(params=body,lr=1e-7)],betas=(.9,.99),eps=1e-8)
        private_opt=torch.optim.Adam(private.parameters(),lr=1e-6,betas=(.9,.99),eps=args.private_eps)
        dual=TokenwiseDual([args.case_id],config=config,dual_lr=.01)
        dual.multipliers[args.case_id]=saved['dual']['multipliers'][args.case_id]
        capture=FactorPanelCapture(chunk_tokens=128) if args.backend=='direct' else None
        before=engine.compute.as_dict();start=time.perf_counter()
        def proposal(scoped):
            return actor(scoped,private,weight_cores=query_weight_cores(engine.parameters,private))
        parameters,regularizer,stats,_,credit=build_adaptive_proposal(proposal,snapshot,engine,grouped,
            radius_fraction=.01,norm_squared=norm_squared,execution_capture=capture)
        if args.streamed_norm:
            parameters,regularizer=prepare_factor_panel(parameters,grouped,capture,transpose_storage=engine.transpose_storage,chunk_rows=args.norm_chunk_rows)
        run=tokenwise_panel if capture is None else capture.run
        row=run(engine,parameters,regularizer,inputs,config=config,dual=dual,state_id=args.case_id,
            state_weight=.5,actor_credit=credit,teacher_provider=provider,backward=True)
        forward_backward_seconds=time.perf_counter()-start
        # Match private normalization in the existing two-state trainer.
        start=time.perf_counter()
        for p in private.parameters():
            if p.grad is not None:p.grad.mul_(2.)
        normalization_seconds=time.perf_counter()-start
        raw_gradients=dict(actor=gradients(actor),private=gradients(private))
        start=time.perf_counter()
        shared_norm=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True))
        private_norm=float(torch.nn.utils.clip_grad_norm_(private.parameters(),1.,error_if_nonfinite=True))
        shared_opt.step();private_opt.step()
        dual.step([(args.case_id,row['mean_kl'])],expected_ids=[args.case_id])
        optimizer_seconds=time.perf_counter()-start
        row.update(stats)
        row.update(repetition=repetition,warmup=repetition<args.warmups,
            forward_backward_seconds=forward_backward_seconds,optimizer_seconds=optimizer_seconds,
            private_normalization_seconds=normalization_seconds,
            update_seconds=forward_backward_seconds+normalization_seconds+optimizer_seconds,
            shared_preclip_norm=shared_norm,private_preclip_norm=private_norm,
            dual_after=dual.state_dict(),fresh_weight_core_queries=1,
            compute={k:v-before[k] for k,v in engine.compute.as_dict().items()})
        rows.append(row)
        with (args.output/'metrics.jsonl').open('a') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
        torch.save(dict(raw_gradients=raw_gradients,actor=actor.state_dict(),private=private.state_dict(),
            shared_optimizer=shared_opt.state_dict(),private_optimizer=private_opt.state_dict(),dual=dual.state_dict()),args.output/'last_update.pt')
        del parameters,regularizer,credit,capture,raw_gradients
    (args.output/'complete.json').write_text(json.dumps(dict(metadata=metadata,rows=rows,compute=engine.compute.as_dict()),indent=2)+'\n')


if __name__=='__main__':main()
