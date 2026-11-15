"""Time the current scientific trainer's one-checkpoint panel path.

Fresh engineering actor only. Both this and the literal comparator begin from
resident exact FP64 teacher targets; checkpoint loading/target reconstruction
and compilation are reported separately. This is not a distributed update.
"""
import argparse,json,sys,time,hashlib
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline.model import GPT
from baseline.execution import enable_deterministic_execution
from rewriter.transformer_chisel import TransformerChisel
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from rewriter.function_preservation import FunctionalPreservation,TeacherBatch
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from scripts.train_multistate_chisel import build_proposal,mean_kl_and_backward

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--teacher-cache',type=Path,required=True)
    p.add_argument('--state',type=int,default=100);p.add_argument('--batch',type=int,default=64)
    p.add_argument('--microbatch',type=int,default=16);p.add_argument('--iterations',type=int,default=5)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--fused-kl',action='store_true')
    p.add_argument('--signatures',action='store_true');p.add_argument('--compare-signatures',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import wandb
    wb=wandb.init(project='chisel-persistent-kernel',mode='offline',dir=str(args.output))
    enable_deterministic_execution();torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=False
    model=GPT(50304,12,768).cuda().requires_grad_(False)
    record=StateBankReader(args.bank).read(args.state,device='cuda',model_device='cpu');record.load_model(model)
    actor=TransformerChisel(model_dim=512,patch_size=64).cuda()
    # Consume the same initial RNG sequence as literal qualification with std0.
    with torch.no_grad():
        for head in actor.output_heads:head.weight.normal_(std=0.)
    engine=FunctionalPreservation(model,microbatch_sequences=args.microbatch,teacher_cache_device='cuda',kl_dtype=torch.float64)
    cache=TeacherCacheReader(args.teacher_cache);cache.load_engine(engine,record)
    panel=cache.read(args.state,'train');original_batch=panel.metadata['microbatch_sequences']
    inputs=panel.inputs[:args.batch].cuda()
    teacher=torch.empty((args.batch,inputs.shape[1],model.proj.weight.shape[0]),device='cuda',dtype=torch.float64)
    torch.cuda.synchronize();started=time.perf_counter()
    for start in range(0,args.batch,original_batch):
        value=panel.teacher_batch(engine,start,start+original_batch);stop=min(args.batch,start+original_batch)
        teacher[start:stop].copy_(value.log_probs[:stop-start]);del value
    torch.cuda.synchronize();teacher_seconds=time.perf_counter()-started
    grouped=AcceleratedGroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
    forward=torch.compile(actor,fullgraph=True,dynamic=False)
    optimizer=torch.optim.Adam(actor.parameters(),lr=1e-5,betas=(.9,.99),eps=1e-8)
    def provider(start,stop):return TeacherBatch(inputs[start:stop],teacher[start:stop],engine.teacher_id,'past_training')
    if args.fused_kl:
        import rewriter.function_preservation as preservation
        from rewriter.accelerated_kl import fused_teacher_kl
        preservation.chunked_teacher_kl=lambda logits,targets,chunk_tokens:fused_teacher_kl(logits,targets)
    rows=[];signatures=[];comparison=[]
    expected=torch.load(args.compare_signatures,weights_only=True) if args.compare_signatures else None
    for iteration in range(args.iterations):
        torch.cuda.synchronize();started=time.perf_counter();phases={}
        parameters,regularizer,stats,editable,credit=build_proposal(forward,record.snapshot,engine,grouped,radius_fraction=.01,timings=phases)
        result=mean_kl_and_backward(engine,parameters,regularizer,inputs,kl_budget=1e-4,actor_credit=credit,timings=phases,teacher_provider=provider)
        gradient=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.))
        retained_gradients=[v.grad for v in actor.parameters()] if iteration<3 and (args.signatures or expected is not None) else None
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        row=dict(iteration=iteration,seconds=seconds,phases=phases,gradient_norm=gradient,**result)
        rows.append(row);print(json.dumps(row),flush=True);wb.log(row)
        if retained_gradients is not None:
            signature=dict(gradient=torch.cat([v.flatten() for v in retained_gradients]).cpu(),
                parameters=torch.cat([v.detach().flatten() for v in actor.parameters()]).cpu())
            if args.signatures:signatures.append(signature)
            if expected is not None:
                errors={key:float((value-expected[iteration][key]).double().norm()/expected[iteration][key].double().norm().clamp_min(1e-30)) for key,value in signature.items()}
                comparison.append(dict(iteration=iteration,**errors));print(json.dumps(comparison[-1]),flush=True)
            del retained_gradients,signature
        del parameters,regularizer,credit
    files=['scripts/train_multistate_chisel.py','rewriter/function_preservation.py','rewriter/transformer_chisel.py','rewriter/accelerated_grouped_norm.py']
    result=dict(scope='one checkpoint panel and local Adam; no distributed200-checkpoint accumulation',
        timing_scope='resident exact FP64 teacher targets; excludes state load, grouped-reference setup and teacher reconstruction',
        backend='compiled actual actor, eager native Flash GPT, accelerated grouped norm, '+('fused KL experimental variant' if args.fused_kl else 'production chunked KL'),
        state=args.state,batch=args.batch,microbatch=args.microbatch,rows=rows,
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        teacher_reconstruction_seconds=teacher_seconds,peak_bytes=torch.cuda.max_memory_allocated(),comparison=comparison)
    if args.signatures:torch.save(signatures,args.output/'signatures.pt')
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');wb.finish()

if __name__=='__main__':main()
