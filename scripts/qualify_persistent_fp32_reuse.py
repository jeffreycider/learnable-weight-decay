"""Checkpoint FP32 plan rebinds, accumulated cotangents, one saved-Adam step.

This is a local integration qualification, not the 200-checkpoint distributed
science update. Every checkpoint receives its own saved primal-dual multiplier.
"""
import argparse,copy,hashlib,json,sys,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from baseline.model import GPT
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.chisel_factory import build_chisel
from rewriter.edit_scope import editable_names,editable_weight_norm_sq,scoped_snapshot
from rewriter.function_preservation import FunctionalPreservation
from rewriter.meta_optimizer import build_meta_adam
from rewriter.persistent_offline import PersistentOfflineTrainer,PersistentActorOptimizer,export_torch_adam
from rewriter.primal_dual import PrimalDual
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from scripts.qualify_compensated_dx import comparison
from scripts.train_multistate_chisel import build_proposal,mean_kl_and_backward


def vector(actor,gradient=False):
    return torch.cat([(p.grad if gradient else p.detach()).flatten() for p in actor.parameters()]).cpu()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('state-bank','teacher-cache','policy','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--states',default='998,999');p.add_argument('--state-weight',type=float,default=1/200)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);enable_deterministic_execution()
    reader=StateBankReader(args.state_bank,verify_once=True,mmap_model=True)
    cache=TeacherCacheReader(args.teacher_cache,verify_once=True,mmap=True)
    states=list(map(int,args.states.split(',')))
    if not set(states)<=set(reader.manifest['split']['train']):raise ValueError('fitting states only')
    saved=torch.load(args.policy,map_location='cuda',weights_only=False)
    actor=build_chisel(saved['metadata']['architecture']).cuda();actor.load_state_dict(saved['controller'])
    actual=copy.deepcopy(actor);initial=vector(actor);forward=torch.compile(actor,fullgraph=True,dynamic=False)
    options=saved['metadata']['options']
    optimizer=build_meta_adam(actor,lr=options['lr'],representation_lr_multiplier=options.get('representation_lr_multiplier',1.))
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    adam_state,rates,betas,eps=export_torch_adam(actor,optimizer);initial_step=int(adam_state['step'])
    ds=saved['primal_dual'];dual=PrimalDual(ds['multipliers'],tolerance=ds['tolerance'],rho=ds['rho'],dual_lr=ds['dual_lr']);dual.load_state_dict(ds)
    model=GPT(**reader.manifest['provenance']['model_config']).cuda().requires_grad_(False)
    engine=FunctionalPreservation(model,gpt_precision='fp32',microbatch_sequences=16,kl_dtype=torch.float64,teacher_cache_device='cuda')
    def load(state):
        record=reader.read(state,model_device='cpu',device='cuda');cache.load_engine(engine,record)
        model.float();model.load_state_dict({**engine.parameters,**engine.buffers})
        grouped=AcceleratedGroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
        editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
        snapshot=scoped_snapshot(record.snapshot,editable_weight_norm_sq(engine.parameters,editable),.01,scope='grouped')
        panel=cache.read(state,'train');teacher=panel.teacher_batch(engine,0,16)
        return record,snapshot,grouped,teacher
    import wandb
    wb=wandb.init(project='chisel',mode='offline',dir=str(args.output),config=dict(states=states,state_weight=args.state_weight,scope=__doc__))
    expected=[]
    for state in states:
        record,snapshot,grouped,teacher=load(state);actor.zero_grad(set_to_none=True)
        edited,regularizer,_,_,credit=build_proposal(forward,snapshot,engine,grouped,radius_fraction=.01)
        result=mean_kl_and_backward(engine,edited,regularizer,teacher.inputs,kl_budget=dual.tolerance,
            state_weight=args.state_weight,actor_credit=credit,teacher_provider=lambda a,b:teacher,
            primal_dual=dual,state_id=state)
        expected.append(dict(state=state,gradient=vector(actor,True),result=result,state_sha256=record.assert_verified()))
        del record,snapshot,grouped,teacher,edited,credit,regularizer
    summed=sum(row['gradient'] for row in expected);offset=0
    for parameter in actor.parameters():
        parameter.grad=summed[offset:offset+parameter.numel()].reshape_as(parameter).cuda();offset+=parameter.numel()
    torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step();optimizer.zero_grad(set_to_none=True)
    reference_update=vector(actor)-initial
    torch.cuda.empty_cache();accumulated={n:torch.zeros_like(v) for n,v in actual.named_parameters()}
    program=None;rows=[]
    for index,state in enumerate(states):
        record,snapshot,grouped,teacher=load(state)
        torch.cuda.synchronize();started=time.perf_counter()
        if program is None:
            program=PersistentOfflineTrainer(actual,model,snapshot,teacher.inputs.cuda(),teacher.log_probs,
                gpt_precision='fp32',thin_backward=True,compensated_backward=True,trust_weight=0.,
                kl_budget=dual.tolerance,kl_multiplier=dual.multipliers[state],augmented_rho=dual.rho,
                state_weight=args.state_weight,gradient_accumulator=accumulated,apply_optimizer=False,
                microbatch_sequences=16,head_chunk_tokens=256)
            pointers=[v.data_ptr() for v in program.plan.storage];stage_pointer=program.plan.stages.data_ptr();kernel=program.plan.kernel
        else:
            program.load_checkpoint(model,snapshot,kl_multiplier=dual.multipliers[state])
            program.load_panel(teacher.inputs.cuda(),teacher.log_probs)
            assert pointers==[v.data_ptr() for v in program.plan.storage]
            assert stage_pointer==program.plan.stages.data_ptr() and kernel is program.plan.kernel
        torch.cuda.synchronize();refresh=time.perf_counter()-started
        started=time.perf_counter();program.step();torch.cuda.synchronize();seconds=time.perf_counter()-started
        observed=torch.cat([program.gradients[n].flatten() for n,_ in actual.named_parameters()]).cpu()
        row=dict(state=state,initial_build=index==0,refresh_seconds=refresh,kernel_seconds=seconds,
            state_sha256=record.assert_verified(),sequence_ids=list(teacher.sequence_ids),sequence_length=teacher.inputs.shape[1],
            reference=expected[index]['result'],literal_kl=float(program.kl),literal_objective=float(program.objective[4]),
            gradient=comparison(observed,expected[index]['gradient']),kl_multiplier=dual.multipliers[state],
            workspace_pointers_unchanged=True,actor_unchanged=torch.equal(vector(actual),initial))
        assert row['actor_unchanged'];rows.append(row);print(json.dumps(row),flush=True)
        wb.log(dict(state=state,refresh_seconds=refresh,kernel_seconds=seconds,gradient_relative_error=row['gradient']['relative_error']))
        del record,snapshot,grouped,teacher,observed
    accumulated_vector=torch.cat([accumulated[n].flatten() for n,_ in actual.named_parameters()]).cpu()
    update=PersistentActorOptimizer(actual,accumulated,optimizer_state=adam_state,parameter_lrs=rates,betas=betas,eps=eps,kernel=program.plan.kernel)
    update.step();torch.cuda.synchronize()
    result=dict(scope=__doc__,states=states,sequences=16,state_weight=args.state_weight,rows=rows,
        accumulated_gradient=comparison(accumulated_vector,summed),adam_update=comparison(vector(actual)-initial,reference_update),
        adam_updates=int(update.optimizer_state()['step'])-initial_step,peak_bytes=torch.cuda.max_memory_allocated(),
        teacher_cache_manifest_sha256=hashlib.sha256((args.teacher_cache/'manifest.json').read_bytes()).hexdigest(),
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path('rewriter/gpt_precision.py'),*Path('rewriter').glob('persistent_*.py'),*Path('rewriter/kernels').glob('persistent*')]},
        runtime_environment={k:v for k,v in os.environ.items() if k.startswith('CHISEL_')},
        policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        limitations=f'{len(states)} local contributions only; no distributed SUM. Refresh excludes source loading and teacher reconstruction. Each checkpoint is one launch; final clipping/Adam is a separate launch.')
    assert result['adam_updates']==1
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);wb.finish()


if __name__=='__main__':main()
