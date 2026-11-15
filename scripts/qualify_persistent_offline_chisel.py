"""Qualify one literal complete offline Chisel update against original semantics.

Teacher logits are produced once with the unchanged baseline before either
backend runs and retained as frozen FP64 log probabilities. This is an engineering
fixture, not the multi-checkpoint learning experiment.
"""
import argparse,copy,json,math,sys,time,os,hashlib
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline.model import GPT
from rewriter.transformer_chisel import TransformerChisel
from rewriter.algebra import Factors,RawAction
from rewriter.function_preservation import FunctionalPreservation,gpt_logits,chunked_teacher_kl
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.edit_scope import editable_names,editable_weight_norm_sq,scoped_snapshot,scoped_action,keep_readonly_parameters
from rewriter.persistent_offline import PersistentOfflineTrainer
from scripts.benchmark_chisel_kernels import synthetic_snapshot
from scripts.benchmark_persistent_mlp import discrepancy


def reference_step(actor,snapshot,engine,teacher,inputs,optimizer,*,radius_fraction,budget,clip,microbatch_sequences=None,state_weight=1.):
    grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
    editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
    snapshot=scoped_snapshot(snapshot,editable_weight_norm_sq(engine.parameters,editable),radius_fraction,scope='grouped')
    action=actor(snapshot);scaled=scoped_action(action,editable);rank=actor.rank;scales={}
    for group,names in grouped.groups.items():
        rms=(grouped.reference_norms[group]/math.sqrt(sum(engine.parameters[n].numel() for n in names))).float()
        for name in names:
            value=engine.parameters[name];dimension=rank*(sum(value.shape)-rank) if value.ndim==2 else value.numel()
            scales[name]=rms*math.sqrt(value.numel()/dimension)
    scaled=RawAction({n:Factors(f.P*scales.get(n,0),f.Q*scales.get(n,0),f.anchors) for n,f in scaled.matrices.items()},
        {n:v*scales.get(n,0) for n,v in scaled.vectors.items()})
    norm=scaled.norm_sq().clamp_min(1e-24).sqrt();scale=torch.minimum(norm.new_ones(()),snapshot.radius/(norm+1e-12))
    replacements=keep_readonly_parameters(engine.edited_parameters(scaled,scale),engine.parameters,editable)
    norm_result=grouped.evaluate(replacements)
    regularizer=100*norm_result.mean_relative_change+.01*norm_result.edit_norm_sq/snapshot.radius.double().square()
    leaves={n:p.detach().requires_grad_(n in editable) for n,p in replacements.items()}
    from torch.func import functional_call
    microbatch_sequences=microbatch_sequences or len(inputs);kl_value=0.
    for start in range(0,len(inputs),microbatch_sequences):
        stop=min(len(inputs),start+microbatch_sequences);fraction=(stop-start)/len(inputs)
        logits=functional_call(engine.view,({f'gpt.{n}':p for n,p in leaves.items()},
            {f'gpt.{n}':p for n,p in engine.buffers.items()}),(inputs[start:stop],),strict=True)
        kl=chunked_teacher_kl(logits,teacher[start:stop]);kl_value+=float(kl.detach())*fraction
        (kl*fraction).backward()
    excess=max(kl_value/budget-1,0.);coefficient=2*excess/budget
    outputs=[regularizer];cotangents=[regularizer.new_tensor(state_weight)]
    for name in replacements:
        if name in editable:outputs.append(replacements[name]);cotangents.append(leaves[name].grad*(coefficient*state_weight))
    torch.autograd.backward(outputs,cotangents)
    gradient_norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),clip)
    gradients={n:p.grad.detach().clone() for n,p in actor.named_parameters()}
    values=dict(kl=kl_value,penalty=excess**2,norm_term=float(100*norm_result.mean_relative_change.detach()),
                trust=float(.01*norm_result.edit_norm_sq.detach()/snapshot.radius.double().square()),
                objective=float(regularizer.detach())+excess**2,gradient_norm=float(gradient_norm),radius_scale=float(scale))
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    return values,gradients,logits.detach(),{n:p.detach() for n,p in replacements.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--bank',type=Path);p.add_argument('--teacher-cache',type=Path);p.add_argument('--state',type=int,default=100)
    p.add_argument('--tokens',type=int,default=17);p.add_argument('--updates',type=int,default=2)
    p.add_argument('--reference-first',action='store_true')
    p.add_argument('--batch',type=int,default=1);p.add_argument('--microbatch',type=int)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--state-weight',type=float,default=1.)
    p.add_argument('--head-std',type=float,default=.001);p.add_argument('--radius-fraction',type=float,default=.01)
    p.add_argument('--kl-budget',type=float,default=1e-4);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'config.json').write_text(json.dumps(vars(args),default=str,indent=2)+'\n')
    import wandb
    run=wandb.init(project='chisel-persistent-kernel',mode='offline',dir=str(args.output))
    from baseline.execution import enable_deterministic_execution
    enable_deterministic_execution()
    torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=False
    model=GPT(50304,12,768).cuda() if args.checkpoint or args.bank else GPT(257,1,128).cuda()
    if args.checkpoint:model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=False)['model'])
    model.requires_grad_(False)
    record=None
    if args.bank:
        from rewriter.state_bank import StateBankReader
        record=StateBankReader(args.bank).read(args.state,device='cuda',model_device='cpu')
        record.load_model(model);snapshot=record.snapshot
    else:snapshot=synthetic_snapshot(model)
    original=TransformerChisel(model_dim=512,patch_size=64).cuda()
    with torch.no_grad():
        for head in original.output_heads:head.weight.normal_(std=args.head_std)
    actual=copy.deepcopy(original)
    inputs=torch.randint(0,model.embed.weight.shape[0],(args.batch,args.tokens),device='cuda')
    engine=FunctionalPreservation(model,teacher_cache_device='cuda',kl_dtype=torch.float64)
    if args.teacher_cache:
        from rewriter.teacher_cache import TeacherCacheReader
        cache=TeacherCacheReader(args.teacher_cache);cache.load_engine(engine,record)
        panel=cache.read(args.state,'train');batch=panel.metadata['microbatch_sequences']
        engine.microbatch_sequences=batch
        if args.batch>len(panel.inputs):raise ValueError('fixture batch exceeds cached panel')
        inputs=panel.inputs[:args.batch].cuda();teacher=torch.empty((args.batch,inputs.shape[1],model.proj.weight.shape[0]),device='cuda',dtype=torch.float64)
        for start in range(0,args.batch,batch):
            teacher_batch=panel.teacher_batch(engine,start,start+batch)
            stop=min(start+batch,args.batch);teacher[start:stop].copy_(teacher_batch.log_probs[:stop-start]);del teacher_batch
    else:
        with torch.no_grad():teacher=gpt_logits(model,inputs).double().log_softmax(-1)
    optimizer=torch.optim.Adam(original.parameters(),lr=args.lr,betas=(.9,.99),eps=1e-8)
    reference_cache=[];reference_peak=None
    if args.reference_first:
        # Sequential execution avoids charging either backend for the other's
        # retained activation buffers. CPU copies happen outside timed regions.
        for iteration in range(args.updates):
            torch.cuda.synchronize();reference_start=time.perf_counter()
            values,grads,logits,parameters=reference_step(original,snapshot,engine,teacher,inputs,optimizer,
                radius_fraction=args.radius_fraction,budget=args.kl_budget,clip=1.,microbatch_sequences=args.microbatch,state_weight=args.state_weight)
            torch.cuda.synchronize();duration=time.perf_counter()-reference_start
            count=int(os.environ.get('CHISEL_HEAD_CHUNK_TOKENS','0')) or logits.numel()//logits.shape[-1]
            reference_cache.append(dict(values=values,seconds=duration,
                gradients=torch.cat([g.flatten() for g in grads.values()]).cpu(),
                logits=logits.flatten(0,1)[-count:].cpu(),
                actor=torch.cat([p.detach().flatten() for p in original.parameters()]).cpu(),
                parameters=torch.cat([p.float().flatten() for p in parameters.values()]).cpu()))
            del grads,logits,parameters
        reference_peak=torch.cuda.max_memory_allocated();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();program=PersistentOfflineTrainer(actual,model,snapshot,inputs,teacher,
        radius_fraction=args.radius_fraction,kl_budget=args.kl_budget,lr=args.lr,microbatch_sequences=args.microbatch,state_weight=args.state_weight)
    torch.cuda.synchronize();setup=time.perf_counter()-start;checks=[]
    (args.output/'setup.json').write_text(json.dumps(dict(allocated_bytes=torch.cuda.memory_allocated(),stages=len(program.plan.stages),
        batch=args.batch,microbatch=args.microbatch,head_chunk_tokens=os.environ.get('CHISEL_HEAD_CHUNK_TOKENS','0'),
        timing_scope='one checkpoint sequence panel; resident exact FP64 teacher targets for both backends; excludes reconstruction and loading'))+'\n')
    for iteration in range(args.updates):
        if args.reference_first:
            cached=reference_cache[iteration];reference=cached['values'];reference_seconds=cached['seconds']
            reference_gradient=cached['gradients'];reference_logits=cached['logits'];reference_actor=cached['actor'];reference_parameters=cached['parameters']
        else:
            torch.cuda.synchronize();reference_start=time.perf_counter()
            reference,gradients,logits,parameters=reference_step(original,snapshot,engine,teacher,inputs,optimizer,
                radius_fraction=args.radius_fraction,budget=args.kl_budget,clip=1.,microbatch_sequences=args.microbatch,state_weight=args.state_weight)
            torch.cuda.synchronize();reference_seconds=time.perf_counter()-reference_start
            reference_gradient=torch.cat([g.flatten() for g in gradients.values()])
            reference_logits=logits.flatten(0,1)[-program.logits.shape[0]:]
            reference_actor=torch.cat([p.detach().flatten() for p in original.parameters()])
            reference_parameters=torch.cat([p.float().flatten() for p in parameters.values()])
        start=time.perf_counter();program.step();torch.cuda.synchronize();duration=time.perf_counter()-start
        actual_objective=program.objective.tolist()
        actual_gradient=torch.cat([g.flatten() for g in program.gradients.values()])
        target=reference_gradient.device
        check=dict(iteration=iteration,reference_seconds=reference_seconds,reference=reference,actual=dict(zip(('kl','penalty','norm_term','trust','objective'),actual_objective)),
            action_gradient=discrepancy(reference_gradient,actual_gradient.to(target)),logits=discrepancy(reference_logits,program.logits.to(target)),logit_comparison_tokens=program.logits.shape[0],
            updated_actor=discrepancy(reference_actor,torch.cat([p.detach().flatten() for p in actual.parameters()]).to(target)),
            edited_parameters=discrepancy(reference_parameters,torch.cat([p.flatten() for p in program.parameters.values()]).to(target)),
            kernel_seconds=duration,radius_scale=float(program.radius_info[1]))
        checks.append(check);print(json.dumps(check),flush=True);(args.output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        assert torch.isfinite(actual_gradient).all() and all(math.isfinite(x) for x in actual_objective)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        program.step();torch.cuda.synchronize()
    trace=args.output/'trace.json';prof.export_chrome_trace(str(trace))
    names=[e['name'] for e in json.loads(trace.read_text())['traceEvents'] if e.get('cat')=='kernel']
    result=dict(coverage='full Chisel edits, student GPT, cached-target FP64 KL, 15-group norm/trust objective, all backward, gradient clip and Adam',
        numerical_acceptance='not implied by this execution check; evaluate unchanged-teacher zero-edit floor and same-input derivatives separately',
        comparison_scope='one checkpoint sequence panel, local optimizer; not the distributed200-checkpoint step',
        timing_scope='resident exact FP64 teacher probabilities for both paths; excludes loading/reconstruction',
        batch=args.batch,microbatch=args.microbatch,state_weight=args.state_weight,reference_first=args.reference_first,reference_peak_bytes=reference_peak,
        runtime_environment={k:v for k,v in os.environ.items() if k.startswith('CHISEL_')},
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),*Path('rewriter').glob('persistent_*.py'),*Path('rewriter/kernels').glob('persistent*')]},
        full_pr323_shapes=bool(args.checkpoint or args.bank),teacher_forward_inside_kernel=False,kernel_count=len(names),kernel_names=names,
        stages=len(program.plan.stages),setup_seconds=setup,peak_allocated_bytes=torch.cuda.max_memory_allocated(),checks=checks)
    ticks=program.plan.clocks.cpu().tolist();descriptors=program.plan.stages.cpu().tolist()
    stages=[dict(index=i,opcode=d[0],ticks=ticks[i+1]-ticks[i],dimensions=d[6:9]) for i,d in enumerate(descriptors)]
    (args.output/'stage-profile.json').write_text(json.dumps(sorted(stages,key=lambda x:-x['ticks']),indent=2)+'\n')
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
    run.log(dict(objective=actual_objective[-1],kernel_count=len(names),gradient_relative_error=checks[-1]['action_gradient']['relative_l2']));run.finish()
    assert names==['persistent_training']


if __name__=='__main__':main()
