"""Bounded dense-versus-factorized FP32 KL backward qualification; no optimizer steps."""
from pathlib import Path
import argparse,hashlib,json,statistics,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.model import GPT
from baseline.execution import enable_deterministic_execution
from rewriter.algebra import Factors,RawAction
from rewriter.chisel_factory import build_chisel
from rewriter.edit_scope import editable_names,editable_weight_norm_sq,scoped_snapshot,scoped_action,keep_readonly_parameters
from rewriter.factorized_fp32 import factorized_logits
from rewriter.function_preservation import FunctionalPreservation,_ChunkedTeacherKL
from rewriter.primal_dual import PrimalDual
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from scripts.train_multistate_chisel import build_proposal,mean_kl_and_backward,group_action_scales


def thin_step(forward,snapshot,engine,grouped,inputs,panel,dual,state_id,*,norm_enabled=True,dense_forward=False):
    editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
    action=scoped_action(forward(snapshot),editable)
    scales=group_action_scales(grouped,next(iter(action.matrices.values())).P.shape[1])
    action=RawAction({n:Factors(f.P*scales[n],f.Q*scales[n],f.anchors) for n,f in action.matrices.items()},
                     {n:v*scales[n] for n,v in action.vectors.items()})
    raw_norm=action.norm_sq().clamp_min(1e-24).sqrt()
    scale=torch.minimum(raw_norm.new_ones(()),snapshot.radius/(raw_norm+1e-12))
    parameters=keep_readonly_parameters(engine.edited_parameters(action,scale),engine.parameters,editable)
    regularizer=100*grouped.evaluate(parameters).mean_relative_change
    if not norm_enabled:regularizer=regularizer*0
    originals=[];leaves=[]
    def leaf(value):
        result=value.detach().requires_grad_(value.requires_grad);originals.append(value);leaves.append(result);return result
    measured=RawAction({n:Factors(leaf(f.P),leaf(f.Q),f.anchors) for n,f in action.matrices.items()},
                       {n:leaf(v) for n,v in action.vectors.items()})
    scale_leaf=leaf(scale);total=0.
    for start in range(0,len(inputs),engine.microbatch_sequences):
        stop=min(start+engine.microbatch_sequences,len(inputs));teacher=panel.teacher_batch(engine,start,stop)
        logits=factorized_logits(engine.view.gpt,engine.parameters,engine.buffers,inputs[start:stop].to(engine.device),measured,scale_leaf,realized_parameters=parameters if dense_forward else None)
        loss=_ChunkedTeacherKL.apply(logits,teacher.log_probs.to(engine.device),engine.chunk_tokens)
        fraction=(stop-start)/len(inputs);total+=float(loss.detach())*fraction
        (loss*fraction).backward()
        engine.compute.student_forwards+=1;engine.compute.student_tokens+=inputs[start:stop].numel()
        engine.compute.student_backwards+=1;engine.compute.student_backward_tokens+=inputs[start:stop].numel()
        del loss,logits,teacher
    terms=dual.terms(total,state_id);outputs=[regularizer];cotangents=[torch.ones_like(regularizer)]
    for original,detached in zip(originals,leaves):
        if original.requires_grad and detached.grad is not None:
            outputs.append(original);cotangents.append(detached.grad*terms['kl_coefficient'])
    torch.autograd.backward(outputs,cotangents)
    return dict(mean_kl=total,objective=float(regularizer.detach())+terms['kl_penalty'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['state-bank','teacher-cache','policy','output']:p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--state-id',type=int,default=999);p.add_argument('--sequences',type=int,default=64)
    p.add_argument('--dense-forward',action='store_true')
    p.add_argument('--microbatch',type=int,default=16);p.add_argument('--repeats',type=int,default=5)
    args=p.parse_args();torch.set_num_threads(1);enable_deterministic_execution();device=torch.device('cuda')
    if not 1<=args.repeats<=10 or args.sequences not in (16,32,64) or args.microbatch!=16:p.error('bounded qualification requires16/32/64sequences,micro16,1..10repeats')
    args.output.mkdir(parents=True,exist_ok=False)
    reader=StateBankReader(args.state_bank,verify_once=True,mmap_model=True)
    if args.state_id not in reader.manifest['split']['train']:raise ValueError('use a fitting checkpoint')
    cache=TeacherCacheReader(args.teacher_cache,verify_once=True,mmap=True);record=reader.read(args.state_id,model_device='cpu',device=device)
    engine=FunctionalPreservation(GPT(**reader.manifest['provenance']['model_config']).to(device).requires_grad_(False),microbatch_sequences=args.microbatch,teacher_cache_device=device,kl_dtype=torch.float64,gpt_precision='fp32')
    cache.load_engine(engine,record)
    from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
    grouped=AcceleratedGroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
    editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
    snapshot=scoped_snapshot(record.snapshot,editable_weight_norm_sq(engine.parameters,editable),.01,scope='grouped')
    saved=torch.load(args.policy,map_location=device,weights_only=False);actor=build_chisel(saved['metadata']['architecture']).to(device);actor.load_state_dict(saved['controller']);del saved
    forward=torch.compile(actor,fullgraph=True,dynamic=False)
    panel=cache.read(args.state_id,'train');inputs=panel.inputs[:args.sequences]
    dual=PrimalDual([args.state_id],tolerance=1e-6,rho=.01,dual_lr=.1);dual.multipliers[args.state_id]=.1
    import wandb
    wb=wandb.init(project='chisel',mode='offline',dir=str(args.output),config=dict(state_id=args.state_id,sequences=args.sequences,microbatch=args.microbatch,optimizer_updates=0,backend='factorized_fp32_prototype',policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest()))
    stream=(args.output/'metrics.jsonl').open('w')
    def log(row):stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True);wb.log(row)
    def step(backend,*,qualify=False):
        actor.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        if backend=='dense':
            parameters,regularizer,stats,_,credit=build_proposal(forward,snapshot,engine,grouped,radius_fraction=.01)
            result=mean_kl_and_backward(engine,parameters,regularizer*(0 if qualify else 1),inputs,kl_budget=1e-6,actor_credit=credit,teacher_provider=lambda a,b:panel.teacher_batch(engine,a,b),primal_dual=dual,state_id=args.state_id)
        else:result=thin_step(forward,snapshot,engine,grouped,inputs,panel,dual,args.state_id,norm_enabled=not qualify,dense_forward=args.dense_forward)
        torch.cuda.synchronize();result.update(backend=backend,qualify=qualify,seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),optimizer_updates=0)
        gradient=torch.cat([torch.zeros_like(v).flatten() if v.grad is None else v.grad.detach().flatten() for v in actor.parameters()]).cpu() if qualify else None
        return result,gradient
    # Raw KL-only actor gradients prevent the norm objective from hiding KL discrepancies.
    dense,dg=step('dense',qualify=True);thin,tg=step('factorized',qualify=True);log(dense);log(thin)
    tg=tg.double();dg=dg.double();relative=float((tg-dg).norm()/dg.norm());cosine=float(torch.dot(tg,dg)/(tg.norm()*dg.norm()));difference=abs(thin['mean_kl']-dense['mean_kl'])
    qualification=dict(gradient_relative_error=relative,gradient_cosine=cosine,absolute_kl_difference=difference,passed=relative<.003 and cosine>.99999 and difference<1e-9)
    log(dict(event='qualification',**qualification));del dg,tg
    if not qualification['passed']:raise RuntimeError('factorized backend failed fixed-policy gradient qualification')
    measured=[]
    for repeat in range(args.repeats+1):
        for backend in (('dense','factorized') if repeat%2==0 else ('factorized','dense')):
            row,_=step(backend);row.update(event='timing',repeat=repeat,warmup=repeat==0);log(row)
            if repeat:measured.append(row)
    summary=dict(qualification=qualification,median_seconds={b:statistics.median(r['seconds'] for r in measured if r['backend']==b) for b in ['dense','factorized']},rows=measured,optimizer_updates=0,production_backend_changed=False,dense_forward=args.dense_forward,policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),state_sha256=record.assert_verified(),sequence_ids=panel.sequence_ids[:args.sequences],compute=vars(engine.compute))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');log(dict(event='finished',**{k:v for k,v in summary.items() if k not in ['rows','sequence_ids']}));wb.finish()


if __name__=='__main__':main()
