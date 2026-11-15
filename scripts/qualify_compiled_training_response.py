"""Disposed three-step ordinary-GPT response with real saved Adam/Muon moments.

SingleGPU emulates the unchanged512-sequence global batch as eight ordered
virtual ranks (four micro16 perrank), then sums rank gradients in rank order.
Both arms use this same reduction; it is not a certificate of NCCL bit identity
or measured eightGPU/online throughput. The default omits sensors; sensor-leaves
includes the actual saved observer while keeping controller updates omitted.
"""
from pathlib import Path
import argparse,copy,hashlib,json,statistics,sys,time
from contextlib import nullcontext,contextmanager
from dataclasses import fields,is_dataclass
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import torch.nn.functional as F
from baseline.data import FinewebStream,_load_data_shard
from baseline.execution import enable_deterministic_execution,softcap_cross_entropy
from baseline.fp32_execution import enable_fp32_execution
from baseline.model import GPT
from baseline.train import make_optimizers,set_hparams
from rewriter.gpt_precision import fp32_head_inputs
from rewriter.optimizer_shards import merge_muon_shards
from rewriter.function_preservation import _ChunkedTeacherKL
from scripts.qualify_graph_offline import record_qualification_source
from scripts.qualify_training_gemm import comparison


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16<<20),b''):value.update(block)
    return value.hexdigest()


def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def state_tensors(value,prefix=''):
    if isinstance(value,torch.Tensor):return {prefix:value}
    if is_dataclass(value):value={f.name:getattr(value,f.name) for f in fields(value)}
    if isinstance(value,dict):
        return {n:t for k,v in value.items() for n,t in state_tensors(v,prefix+'/'+str(k)).items()}
    if isinstance(value,(list,tuple)):
        return {n:t for k,v in enumerate(value) for n,t in state_tensors(v,prefix+'/'+str(k)).items()}
    return {}


@contextmanager
def strict_leaf_audit(model):
    """Evaluate both weight trajectories with the original eager leaf math."""
    from baseline.model import Linear,Rotary
    changed=[]
    try:
        for module in model.modules():
            if type(module) in (Linear,Rotary) and 'forward' in module.__dict__:
                changed.append((module,module.__dict__['forward']));del module.forward
        yield
    finally:
        for module,forward in changed:module.forward=forward


@torch.no_grad()
def audit(model,inputs,targets,reference=None):
    # Both arms are evaluated with the same strict eager FP32 function. The
    # reference cache stores exact head inputs and its then-current head.
    hidden=fp32_head_inputs(model,inputs)
    raw=F.linear(hidden,model.proj.weight,model.proj.bias)
    ce=float(softcap_cross_entropy(raw,targets))/targets.numel()
    result=dict(mean_ce=ce)
    cached=dict(features=hidden.cpu().clone(),weight=model.proj.weight.detach().cpu().clone(),bias=model.proj.bias.detach().cpu().clone())
    if reference is not None:
        old=F.linear(reference['features'].to(inputs.device),reference['weight'].to(inputs.device),reference['bias'].to(inputs.device))
        old=15*old*(old.square()+225).rsqrt()
        current=15*raw*(raw.square()+225).rsqrt()
        target=torch.empty_like(old,dtype=torch.float64)
        a=old.flatten(0,1);b=current.flatten(0,1);t=target.flatten(0,1)
        sums=torch.zeros(3,dtype=torch.float64,device=inputs.device)
        for start in range(0,len(a),128):
            t[start:start+128].copy_(F.log_softmax(a[start:start+128].double(),dim=-1))
            delta=b[start:start+128].double()-a[start:start+128].double()
            sums[0]+=delta.square().sum();sums[1]+=a[start:start+128].double().square().sum()
            sums[2]=torch.maximum(sums[2],delta.abs().max())
        result.update(reference_to_candidate_kl=float(_ChunkedTeacherKL.apply(current,target,128)),
            absolute_logit_rms=float((sums[0]/old.numel()).sqrt()),relative_logit_error=float((sums[0]/sums[1]).sqrt()),
            max_absolute_logit_error=float(sums[2]))
    return result,cached


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-directory',type=Path,required=True);p.add_argument('--expected-rank0-sha256',required=True)
    p.add_argument('--train-data',required=True);p.add_argument('--val-file',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,choices=(1,3),default=3)
    p.add_argument('--sensor-leaves',action='store_true',help='Actual observer and compiled pure leaves; no whole-model compile or controller updates')
    p.add_argument('--sensor-eager-loss',action='store_true',help='Isolate compiled linear/rotary leaves while retaining native eager training loss')
    args=p.parse_args()
    if args.sensor_eager_loss and not args.sensor_leaves:p.error('sensor-eager-loss requires sensor-leaves')
    args.output.mkdir(parents=True,exist_ok=False);record_qualification_source(args.output,vars(args))
    manifest=json.loads((args.checkpoint_directory/'manifest.json').read_text())
    if manifest['entries'][0]['sha256']!=args.expected_rank0_sha256:raise ValueError('rank0 manifest SHA differs')
    shards=[]
    for rank,entry in enumerate(manifest['entries']):
        path=args.checkpoint_directory/entry['file']
        if entry['rank']!=rank or digest(path)!=entry['sha256']:raise ValueError('retained rank SHA/order differs')
        shards.append(torch.load(path,map_location='cpu',weights_only=False,mmap=True))
    saved=shards[0];config=saved['config'];world=config['world_size']
    if len(shards)!=world or config['batch_tokens']!=524288 or config['seq_len']!=1024:
        raise ValueError('actual full eight-rank production batch required')
    if any(s['step']!=saved['step'] or s['config']!=config for s in shards):raise ValueError('rank checkpoint identity differs')
    merged=merge_muon_shards([s['optimizers'][1] for s in shards])
    torch.set_num_threads(1);enable_deterministic_execution();device=torch.device('cuda:0')
    dist.init_process_group('nccl',init_method='file://'+str((args.output/'process_group').resolve()),rank=0,world_size=1)
    torch.cuda.set_device(device)
    model=GPT(**config['model']).float().to(device);model.load_state_dict(saved['model'])
    enable_fp32_execution(model,compile_loss=False);compiled=torch.compile(model,fullgraph=True,dynamic=False)
    optimizers=make_optimizers(model);parameters=list(model.parameters())
    observer=None
    if args.sensor_leaves:
        from rewriter.state_capture import DistributedStateObserver
        from rewriter.sensor_compile_boundaries import compiled_sensor_leaves
        observer=DistributedStateObserver(model,optimizers,config,sensor_config=saved['controller']['observer']['sensor_config'],start_step=0)
        model._fp32_sensors=observer.collector
    accumulator=[torch.zeros_like(v) for v in parameters]
    validation=_load_data_shard(args.val_file,pin_memory=False)
    validation_inputs=validation[:16*1024].to(device=device,dtype=torch.int32).reshape(16,1024)
    validation_targets=validation[1:16*1024+1].to(device=device,dtype=torch.int64).reshape(16,1024)
    del validation
    def restore():
        model.load_state_dict(saved['model']);optimizers[0].load_state_dict(copy.deepcopy(saved['optimizers'][0]));optimizers[1].load_state_dict(copy.deepcopy(merged))
        model.zero_grad(set_to_none=True)
        if observer is not None:observer.load_state_dict(saved['controller']['observer'])
    def loader():
        value=FinewebStream(args.train_data,config['batch_tokens'],config['seq_len'],rank=0,world_size=1,device=device)
        names={f.name:f for f in value.files};value.files=[names[n] for n in saved['data_cursor']['file_names']]
        cursor=copy.deepcopy(saved['data_cursor']);cursor.update(rank=0,world_size=1);value.load_state_dict(cursor)
        return value
    import wandb
    run=wandb.init(project='chisel',mode='offline',dir=str(args.output),config=dict(kind='ordinary_compiled_saved_muon_response',steps=args.steps,production_changed=False))
    stream=(args.output/'metrics.jsonl').open('w')
    def log(row):
        stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
        run.log({k:v for k,v in row.items() if isinstance(v,(int,float,str,bool))})
    references={};rows=[];errors={};compile_rows=[]
    for backend in ('eager','sensor_leaves' if args.sensor_leaves else 'compiled'):
        restore();data=loader();forward=compiled if backend=='compiled' else model
        context=compiled_sensor_leaves(model,compile_loss=not args.sensor_eager_loss) if backend=='sensor_leaves' else nullcontext()
        with context:
            for update in range(args.steps):
                inputs,targets=next(data)
                set_hparams(optimizers,saved['step']+update,config.get('schedule_steps',config['max_steps']))
                before_weights={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
                torch.cuda.synchronize();setup_start=time.perf_counter()
                if observer is not None:observer.before_step(saved['step']+update,model,optimizers)
                torch.cuda.synchronize();sensor_setup=time.perf_counter()-setup_start
                torch._foreach_zero_(accumulator);loss_total=torch.zeros((),device=device)
                torch.cuda.synchronize();tick=time.perf_counter()
                local=inputs.shape[0]//world
                for rank in range(world):
                    model.zero_grad(set_to_none=True)
                    for start in range(rank*local,(rank+1)*local,16):
                        if observer is not None:observer.before_microbatch(saved['step']+update,torch.arange(start,start+16,device=device),inputs[start:start+16])
                        loss=forward(inputs[start:start+16],targets[start:start+16]);loss_total+=loss.detach();loss.backward()
                    torch._foreach_add_(accumulator,[v.grad for v in parameters])
                for parameter,gradient in zip(parameters,accumulator):parameter.grad=gradient
                torch.cuda.synchronize();fb=time.perf_counter()-tick
                tick=time.perf_counter()
                if observer is not None:observer.after_backward(saved['step']+update,model)
                torch.cuda.synchronize();sensor_backward=time.perf_counter()-tick
                sensor_values={} if observer is None else cpu_copy(state_tensors(observer.sensors))
                grads={n:v.grad.detach().cpu().clone() for n,v in model.named_parameters()}
                if any(not torch.isfinite(v).all() for v in grads.values()):raise FloatingPointError('nonfinite fullglobal gradient')
                tick=time.perf_counter()
                for optimizer in optimizers:optimizer.step()
                snapshot=None if observer is None else observer.finish_snapshot(saved['step']+update)
                model.zero_grad(set_to_none=True);torch.cuda.synchronize();opt=time.perf_counter()-tick
                with strict_leaf_audit(model):
                    audit_result,teacher=audit(model,validation_inputs,validation_targets,None if backend=='eager' else references[update]['teacher'])
                weights={n:v.detach().cpu().clone() for n,v in model.named_parameters()}
                stored_delta={n:v-before_weights[n] for n,v in weights.items()}
                snapshot_values={} if snapshot is None else cpu_copy(state_tensors(snapshot))
                states=cpu_copy([o.state_dict() for o in optimizers])
                row=dict(event='step',backend=backend,update=update+1,completed_steps=saved['step']+update+1,global_tokens=inputs.numel(),
                    train_mean_ce=float(loss_total)/targets.numel(),forward_backward_seconds=fb,optimizer_seconds=opt,compute_seconds=fb+opt+sensor_setup+sensor_backward,
                    sensor_setup_seconds=sensor_setup,sensor_post_backward_seconds=sensor_backward,**audit_result)
                if backend=='eager':references[update]=dict(gradients=grads,weights=weights,states=states,teacher=teacher,metrics=row,stored_delta=stored_delta,sensors=sensor_values,snapshot=snapshot_values)
                else:
                    ref=references[update]
                    errors[str(update+1)]=dict(gradient=comparison(grads,ref['gradients']),weights=comparison(weights,ref['weights']),
                        optimizer_state=comparison(state_tensors(states),state_tensors(ref['states'])),
                        stored_optimizer_delta=comparison(stored_delta,ref['stored_delta']),
                        validation_ce_difference=row['mean_ce']-ref['metrics']['mean_ce'])
                    if observer is not None:
                        errors[str(update+1)]['sensors']=comparison(sensor_values,ref['sensors'])
                        errors[str(update+1)]['next_actor_inputs']=comparison(snapshot_values,ref['snapshot'])
                    row.update(gradient_relative=errors[str(update+1)]['gradient']['relative_error'],
                        weight_relative=errors[str(update+1)]['weights']['relative_error'],
                        optimizer_state_relative=errors[str(update+1)]['optimizer_state']['relative_error'],
                        validation_ce_difference=errors[str(update+1)]['validation_ce_difference'])
                rows.append(row);log(row)
                compile_rows.append(dict(backend=backend,update=update+1,stats=dict(torch._dynamo.utils.counters['stats'])))
                (args.output/'partial.json').write_text(json.dumps(dict(rows=rows,errors=errors),indent=2)+'\n')
    summary=dict(scope=__doc__,source_rank_sha256={e['rank']:e['sha256'] for e in manifest['entries']},
        saved_moments='All72 Muon moment states reconstructed from exact eight owner shards; Adam restored from rank0',
        gradient_reduction='Eight virtual ranks, four micro16 each, rank-order FP32 sum on one GPU; not NCCL bitidentity',
        rows=rows,errors=errors,steps=args.steps,production_changed=False,sensors=args.sensor_leaves,controller_updates=False,sensor_compile_loss=args.sensor_leaves and not args.sensor_eager_loss,
        sensor_scope='Actual saved observer/anchors/settings, all512 global IDs on one GPU; observer callbacks, finalize, momentum and next actor-input features included; no NCCL equivalence claim' if args.sensor_leaves else None,
        validation='First16 complete1024-token sequences from pinned validation shard; same strict eager audit arithmetic for both arms',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),compilation_counters={k:dict(v) for k,v in torch._dynamo.utils.counters.items()},
        compilation_counters_by_step=compile_rows)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');run.finish();stream.close();dist.destroy_process_group()


if __name__=='__main__':main()
