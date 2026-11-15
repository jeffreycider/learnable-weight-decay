"""Read-only real-policy GEMM finite-response diagnostic; no fit or Adam step."""
from pathlib import Path
import argparse,hashlib,json,math,statistics,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from baseline.execution import enable_deterministic_execution
from baseline.model import GPT
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.chisel_factory import build_chisel
from rewriter.compensated_linear import compensated_linear,packed_compensated_linear,pack_compensated_weight,cublas_three_product_linear,bf16_six_product_linear
from rewriter.edit_scope import editable_names,editable_weight_norm_sq,scoped_snapshot
from rewriter.function_preservation import FunctionalPreservation,_ChunkedTeacherKL
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from scripts.qualify_graph_offline import record_qualification_source
from scripts.train_multistate_chisel import build_proposal


@torch.no_grad()
def compare(r0,r1,c0,c1):
    arrays=[x.reshape(-1,x.shape[-1]) for x in (r0,r1,c0,c1)]
    totals=torch.zeros(6,dtype=torch.float64,device=r0.device)
    for start in range(0,len(arrays[0]),128):
        a,b,c,d=[x[start:start+128] for x in arrays]
        # Finite responses are subtracted in stored FP32, then measured in FP64.
        response=(b-a).double();error=(d-c).double()-response
        output_error=d.double()-b.double()
        totals[:4].add_(torch.stack((response.square().sum(),error.square().sum(),
                                    b.double().square().sum(),output_error.square().sum())))
        totals[4]=torch.maximum(totals[4],error.abs().max())
        totals[5]=torch.maximum(totals[5],output_error.abs().max())
    rr,ee,oo,dd,em,dm=totals.cpu().tolist()
    return dict(response_relative_error=math.sqrt(ee/max(rr,1e-60)),
        response_rms=math.sqrt(rr/r0.numel()),response_error_rms=math.sqrt(ee/r0.numel()),
        response_error_max=em,output_relative_error=math.sqrt(dd/max(oo,1e-60)),
        output_error_rms=math.sqrt(dd/r0.numel()),output_error_max=dm)


@torch.no_grad()
def timing(fn):
    for _ in range(3):fn()
    values=[]
    for _ in range(9):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();fn();end.record();end.synchronize();values.append(start.elapsed_time(end))
    return dict(median_ms=statistics.median(values),samples_ms=values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('state-bank','teacher-cache','policy','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--state-id',type=int,default=999)
    p.add_argument('--backend',choices=('triton','packed','triple','bf16x6','bf16x6_kahan'),default='triton')
    p.add_argument('--direction',choices=('forward','dx'),default='forward')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    record_qualification_source(args.output,vars(args));torch.set_num_threads(1);enable_deterministic_execution()
    from functools import partial
    method={'triton':compensated_linear,'packed':packed_compensated_linear,'triple':cublas_three_product_linear,'bf16x6':bf16_six_product_linear,'bf16x6_kahan':partial(bf16_six_product_linear,compensate=True)}[args.backend]
    device=torch.device('cuda:0');reader=StateBankReader(args.state_bank,verify_once=True,mmap_model=True)
    cache=TeacherCacheReader(args.teacher_cache,verify_once=True,mmap=True)
    record=reader.read(args.state_id,model_device='cpu',device=device)
    model=GPT(**reader.manifest['provenance']['model_config']).to(device).requires_grad_(False)
    engine=FunctionalPreservation(model,microbatch_sequences=16,teacher_cache_device=device,kl_dtype=torch.float64,gpt_precision='fp32')
    cache.load_engine(engine,record);grouped=AcceleratedGroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
    editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
    norm=editable_weight_norm_sq(engine.parameters,editable);snapshot=scoped_snapshot(record.snapshot,norm,.01,scope='grouped')
    saved=torch.load(args.policy,map_location='cpu',weights_only=False)
    actor=build_chisel(saved['metadata']['architecture']).to(device);actor.load_state_dict(saved['controller'])
    forward=torch.compile(actor,fullgraph=True,dynamic=False)
    proposal=build_proposal(forward,snapshot,engine,grouped,radius_fraction=.01,norm_squared=norm,shrinkage_coefficient=0.)
    edited={n:v.detach() for n,v in proposal[0].items()};del proposal
    names=('blocks.0.attn.q','blocks.0.mlp.fc','blocks.0.mlp.proj','proj');captured={};hooks=[]
    for name,module in model.named_modules():
        if name in names:
            if args.direction=='forward':
                hooks.append(module.register_forward_pre_hook(lambda mod,inputs,key=name:captured.__setitem__(key,inputs[0].detach())))
            else:
                hooks.append(module.register_full_backward_hook(lambda mod,gi,go,key=name:captured.__setitem__(key,go[0].detach())))
    panel=cache.read(args.state_id,'train')
    if args.direction=='forward':
        with torch.no_grad():engine._forward(edited,panel.inputs[:16].to(device))
    else:
        leaves={n:v.detach().requires_grad_(n in editable) for n,v in edited.items()}
        target=panel.teacher_batch(engine,0,16)
        logits=engine._forward(leaves,panel.inputs[:16].to(device))
        loss=_ChunkedTeacherKL.apply(logits,target.log_probs,engine.chunk_tokens);loss.backward()
        for value in leaves.values():value.grad=None
        del leaves,logits,loss,target
    for hook in hooks:hook.remove()
    if set(captured)!=set(names):raise RuntimeError('missing real forward activation hooks')
    rows=[]
    import wandb
    run=wandb.init(project='chisel',mode='offline',dir=str(args.output),config=dict(kind='compensated_forward_component',production_changed=False))
    with torch.no_grad():
        for name in names:
            x=captured.pop(name);w=engine.parameters[name+'.weight'];b=engine.parameters[name+'.bias']
            ew,eb=edited[name+'.weight'],edited[name+'.bias']
            if args.direction=='dx':
                # Component measures the transposed linear geometry. Record that
                # layout preparation is outside this component timer; full-step
                # adoption must include it explicitly.
                w,ew=w.T.contiguous(),ew.T.contiguous()
                b=torch.zeros(w.shape[0],dtype=w.dtype,device=w.device);eb=b
            r0,c0=F.linear(x,w,b),method(x,w,b)
            if not torch.equal(F.linear(x,w,b),r0):raise RuntimeError('candidate altered strict cuBLAS execution')
            amplitudes={}
            for amplitude in (.1,1.,10.):
                aw,ab=(ew,eb) if amplitude==1. else (w+amplitude*(ew-w),b+amplitude*(eb-b))
                r1,c1=F.linear(x,aw,ab),method(x,aw,ab)
                amplitudes[str(amplitude)]=dict(relative_weight_edit=float((aw-w).norm()/w.norm()),**compare(r0,r1,c0,c1))
                del r1,c1,aw,ab
            del r0,c0
            row=dict(name=name,direction=args.direction,input_shape=list(x.shape),weight_shape=list(w.shape),
                amplitudes=amplitudes,**amplitudes['1.0'],
                ieee=timing(lambda:F.linear(x,ew,eb)),experimental=timing(lambda:method(x,ew,eb)))
            if args.backend=='packed':
                packed_weight=pack_compensated_weight(ew)
                row['weight_pack_bytes']=packed_weight.numel()*packed_weight.element_size()
                row['packed_weight_reuse']=timing(lambda:packed_compensated_linear(x,ew,eb,packed_weight=packed_weight))
                del packed_weight
            rows.append(row);print(json.dumps(row),flush=True);run.log({k:v for k,v in row.items() if isinstance(v,(float,int,str))})
            (args.output/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    summary=dict(backend=args.backend,direction=args.direction,scope='Four real linear layers, frozen repair policy, actual strict-student activations or raw-KL cotangents; component finite responses only; dx weight transpose is outside component timer',
        policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),state_id=args.state_id,rows=rows,
        production_changed=False,whole_model_qualified=False,arithmetic=('Six BF16 residual-expansion products, separate FP32 leading/correction accumulators'+('; Kahan leading accumulator' if args.backend.endswith('_kahan') else '') if args.backend.startswith('bf16x6') else 'Three TF32 products')+' with FP32 storage/output; IEEE thin gradients not tested by this component')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');run.finish()


if __name__=='__main__':main()
