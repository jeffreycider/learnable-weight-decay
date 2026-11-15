"""Additional-compute prototype: learn a finite norm/fidelity edit at one GPT checkpoint.

Uses only previously consumed training data and explicitly synthetic sequences.
Extra GPT evaluations/backwards are counted; this is NOT a Track3 result.
Adam proposals use backtracking on the current training minibatch objective.
Withheld replay inputs never influence updates, backtracking, or stopping.
"""
import argparse
import copy
import glob
import hashlib
import json
from pathlib import Path
import sys
import time
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline.model import GPT
from baseline.train import make_optimizers, source_record
from baseline.execution import enable_deterministic_execution, execution_environment
from rewriter.controller import GlobalRewriter
from rewriter.frozen_snapshot import observe_frozen_checkpoint
from rewriter.function_preservation import FunctionalPreservation
from rewriter.replay_buffer import sample_past_sequences, synthetic_sequences
from rewriter.exact_gauges import baseline_gauge_specs, norm_optimal_gauge_control
from rewriter.algebra import RawAction, Factors
from rewriter.edit_scope import (editable_names, editable_weight_norm_sq, scoped_action,
    scoped_snapshot, keep_readonly_parameters, readonly_hashes, verify_readonly)
from rewriter.grouped_norm import GroupedNormObjective


def chisel_forward_callable(actor, *, compile_chisel=False):
    """Construct once; retain the original actor for Adam and portable checkpoints."""
    return torch.compile(actor, fullgraph=True, dynamic=False) if compile_chisel else actor


def compilation_counters():
    """Diagnostic counters, including Dynamo, AOTAutograd and Inductor namespaces."""
    return {str(name): dict(values) for name, values in torch._dynamo.utils.counters.items() if values}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--train-data',default='/tmp/fineweb10B/fineweb_train_*.bin')
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--buffer-sequences',type=int,default=4096)
    p.add_argument('--calibration-sequences',type=int,default=4)
    p.add_argument('--eval-every',type=int,default=100);p.add_argument('--eval-sequences',type=int,default=64)
    p.add_argument('--final-audit-sequences',type=int,default=1024)
    p.add_argument('--synthetic-audit-sequences',type=int,default=64)
    p.add_argument('--norm-target-fraction',type=float,
                   help='historical squared norm scale; grouped L2 requires .01 so the loss is percentage units')
    p.add_argument('--kl-budget',type=float,default=1e-4)
    p.add_argument('--radius-fraction',type=float,default=.01)
    p.add_argument('--backtracks',type=int,default=10)
    p.add_argument('--optimizer-mode',choices=('adam','backtracking'),default='backtracking')
    p.add_argument('--action-units',choices=('absolute','relative'),default='absolute')
    p.add_argument('--preservation-penalty',choices=('linear','deadband'),default='linear')
    p.add_argument('--editable-scope',choices=('grouped','hidden','all'),default='grouped',
                   help='grouped edits matrices/biases but freezes gains/scalars; hidden/all preserve historical scopes')
    p.add_argument('--norm-objective',choices=('grouped_l2','global_squared'),
                   help='defaults to grouped_l2 for grouped scope; global_squared for historical scopes')
    p.add_argument('--wandb',action='store_true')
    p.add_argument('--compile-chisel',action='store_true',help='compile the actor once; keep eager GPT/objective and canonical checkpoints')
    args=p.parse_args()
    args.norm_objective=args.norm_objective or ('grouped_l2' if args.editable_scope=='grouped' else 'global_squared')
    if (args.editable_scope=='grouped')!=(args.norm_objective=='grouped_l2'):
        p.error('grouped scope requires grouped_l2; historical all/hidden scopes require global_squared')
    if args.norm_target_fraction is None:
        args.norm_target_fraction=.01 if args.norm_objective=='grouped_l2' else 1e-4
    if args.norm_objective=='grouped_l2' and args.norm_target_fraction!=.01:
        p.error('grouped_l2 norm_target_fraction must be .01: the loss uses mean percentage L2 change')
    if min(args.steps,args.lr,args.norm_target_fraction,args.kl_budget,args.radius_fraction)<=0:
        p.error('positive optimization clocks/scales required')
    if min(args.calibration_sequences,args.eval_every,args.eval_sequences,args.final_audit_sequences,
           args.synthetic_audit_sequences)<1 or args.backtracks<0:
        p.error('positive sample counts/evaluation cadence and nonnegative backtracks required')
    torch.set_num_interop_threads(20)
    enable_deterministic_execution()
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    config=state['config'];checkpoint_step=state['step']
    if config['mode']!='baseline' or state['controller'] is not None:
        raise ValueError('prototype requires a fixed baseline checkpoint, not a transferred learned controller')
    if state['data_cursor']['batches']!=checkpoint_step: raise ValueError('invalid checkpoint data clock')
    digest=hashlib.sha256()
    with open(args.checkpoint,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    metadata=dict(vars(args),checkpoint_step=checkpoint_step,checkpoint_sha256=digest.hexdigest(),base_config=config,
        experiment='frozen finite-output preservation, additional-compute prototype',
        benchmark_compliant=False,controller_state_transferred=False,
        chisel_compilation=dict(enabled=args.compile_chisel,fullgraph=True,dynamic=False,mode='default',
            scope='actor forward/backward only; finite GPT and norm/KL objective remain eager'),
        synthetic_training_fraction=.2,optimizer=('Adam' if args.optimizer_mode=='adam' else 'Adam with train-minibatch backtracking'),
        kl_arithmetic='float64 from float32 GPT logits',
        norm_relative_change_definition=('mean_across_functional_groups_of_relative_L2_change'
            if args.norm_objective=='grouped_l2' else 'editable_squared_norm_change_over_reference'),
        norm_loss_units=('mean_percentage_L2_change' if args.norm_objective=='grouped_l2' else 'normalized_squared_norm_change'),
        global_observation_scope='all GPT parameters; editability logged separately',
        source_readout='live',execution=execution_environment())
    (output/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    source_record(metadata,output)
    log_file=(output/'metrics.jsonl').open('w')
    wb=None
    if args.wandb:
        import wandb
        wb=wandb.init(project='learnable-weight-decay',mode='offline',dir=str(output),config=metadata)
    start_time=time.perf_counter()
    def log(event,**values):
        row=dict(event=event,elapsed_seconds=time.perf_counter()-start_time,**values)
        line=json.dumps(row,allow_nan=False);print(line,flush=True);log_file.write(line+'\n');log_file.flush()
        if wb:wb.log(row)
    model=GPT(**config['model']).to(device);model.load_state_dict(state['model'])
    optimizers=make_optimizers(model)
    for opt,saved in zip(optimizers,state['optimizers']):opt.load_state_dict(saved)
    replay=sample_past_sequences(glob.glob(args.train_data),checkpoint_step,
        batch_tokens=config['batch_tokens'],seq_len=config['seq_len'],count=args.buffer_sequences)
    (output/'replay_provenance.json').write_text(json.dumps(dict(metadata=replay.metadata,
        train=replay.train.provenance,audit=replay.audit.provenance),indent=2)+'\n')
    snapshot,_,calibration=observe_frozen_checkpoint(model,optimizers,config,
        replay.train.inputs[:args.calibration_sequences],replay.train.targets[:args.calibration_sequences],
        step=checkpoint_step,radius_fraction=args.radius_fraction)
    del optimizers,state
    model.requires_grad_(False)
    engine=FunctionalPreservation(model,microbatch_sequences=1,teacher_cache_device='cuda',kl_dtype=torch.float64)
    editable=editable_names(engine.parameters,args.editable_scope)
    group_objective=GroupedNormObjective(engine.parameters) if args.norm_objective=='grouped_l2' else None
    editable_norm_sq=(engine.base_weight_norm_sq if args.editable_scope=='all' else
                      editable_weight_norm_sq(engine.parameters,editable))
    snapshot=scoped_snapshot(snapshot,editable_norm_sq,args.radius_fraction,scope=args.editable_scope)
    read_only_baseline_hashes=readonly_hashes(engine.parameters,editable)
    calibration=dict(calibration,editable_scope=args.editable_scope,
        editable_weight_norm=float(editable_norm_sq.sqrt()),total_weight_norm=float(engine.base_weight_norm_sq.sqrt()))
    # Fixed output-unit calibration, independent of controller candidates/data.
    # For IID unit-variance canonical coordinates, expected edit norm squared is
    # r(n+m-r). Thus unit outputs have the tensor's weight norm in expectation.
    # A 1e-3 parameter-RMS floor keeps zero biases/weights editable.
    action_scales={}
    with torch.no_grad():
        for name,base in engine.parameters.items():
            if args.action_units=='absolute':
                action_scales[name]=1.
            else:
                rms=base.float().square().mean().sqrt().clamp_min(1e-3)
                dim=8*(sum(base.shape)-8) if base.ndim==2 else base.numel()
                action_scales[name]=rms*(base.numel()/dim)**.5
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(200001)
        actor=GlobalRewriter(rank=8,trajectory_dim=(2*config['model']['num_layers']+1)*16).to(device)
    actor_forward=chisel_forward_callable(actor,compile_chisel=args.compile_chisel)
    optimizer=torch.optim.Adam(actor.parameters(),lr=args.lr,betas=(.9,.99),eps=1e-8)
    torch.save(dict(snapshot=snapshot,calibration=calibration),output/'observation.pt')
    train_synthetic=synthetic_sequences('empirical_unigram',256,config['seq_len'],config['model']['vocab_size'],past=replay)
    audit_synthetic={kind:synthetic_sequences(kind,args.synthetic_audit_sequences,config['seq_len'],config['model']['vocab_size'],seed=600001+i)
                     for i,kind in enumerate(('uniform','repeated','alternating'))}
    rng=torch.Generator().manual_seed(700001)
    def proposal():
        action=scoped_action(actor_forward(snapshot),editable)
        if args.action_units=='relative':
            action=RawAction({n:Factors(f.P*action_scales[n],f.Q*action_scales[n],f.anchors)
                              for n,f in action.matrices.items()},
                             {n:v*action_scales[n] for n,v in action.vectors.items()})
        norm=action.norm_sq().clamp_min(1e-24).sqrt()
        scale=torch.minimum(norm.new_ones(()),snapshot.radius/(norm+1e-12))
        parameters=keep_readonly_parameters(engine.edited_parameters(action,scale),engine.parameters,editable)
        grouped=group_objective.evaluate(parameters) if group_objective is not None else None
        if grouped is None:
            change,edit_sq=engine.realized_parameter_norm_change(parameters)
            relative_change=change/editable_norm_sq
        else:
            change,edit_sq=grouped.squared_norm_change,grouped.edit_norm_sq
            relative_change=grouped.mean_relative_change
        norm_term=relative_change/args.norm_target_fraction
        trust=.01*edit_sq/snapshot.radius.double().square()
        return parameters,norm_term+trust,dict(norm_relative_change=relative_change,
            editable_squared_norm_relative_change=change/editable_norm_sq,
            total_norm_relative_change=change/engine.base_weight_norm_sq,
            **(grouped.metrics() if grouped is not None else {}),
            realized_edit_norm=edit_sq.sqrt(),raw_action_norm=norm,global_scale=scale,norm_term=norm_term,trust_term=trust)
    def floats(values):return {k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in values.items()}
    def preservation_penalty(kl):
        ratio=kl/args.kl_budget
        # The budget is an allowed finite KL band, not an identity requirement.
        return (ratio-1).clamp_min(0).square() if args.preservation_penalty=='deadband' else ratio
    @torch.no_grad()
    def evaluate_parameters(parameters,label,step,full=False):
        grouped=group_objective.evaluate(parameters) if group_objective is not None else None
        if grouped is None:
            change,edit_sq=engine.realized_parameter_norm_change(parameters)
            relative_change=change/editable_norm_sq
        else:
            change,edit_sq=grouped.squared_norm_change,grouped.edit_norm_sq
            relative_change=grouped.mean_relative_change
        groups={'historical_audit':(replay.audit,'past_training')}
        groups.update({kind:(batch,'synthetic') for kind,batch in audit_synthetic.items()})
        results={}
        for name,(batch,role) in groups.items():
            limit=min(len(batch.inputs),args.final_audit_sequences if full else args.eval_sequences)
            if role=='synthetic' and not full:limit=min(limit,8)
            values=[]
            for i in range(limit):
                teacher=engine.cache_teacher(batch.inputs[i:i+1],data_role=role)[0]
                values.append(float(engine.student_parameter_loss(parameters,teacher)))
            tensor=torch.tensor(values,dtype=torch.float64)
            results[name]=dict(sequences=limit,mean_kl=float(tensor.mean()),
                p95_sequence_mean_kl=float(tensor.quantile(.95)),max_sequence_mean_kl=float(tensor.max()))
        integrity=verify_readonly(parameters,editable,read_only_baseline_hashes) if full else None
        log('evaluation',step=step,label=label,norm_relative_change=float(relative_change),
            norm_objective=args.norm_objective,editable_squared_norm_relative_change=float(change/editable_norm_sq),
            **(grouped.metrics() if grouped is not None else {}),
            total_norm_relative_change=float(change/engine.base_weight_norm_sq),editable_scope=args.editable_scope,
            read_only_integrity=integrity,
            realized_edit_norm=float(edit_sq.sqrt()),distributions=results,compute=engine.compute.as_dict())
        return results
    log('start',checkpoint_step=checkpoint_step,controller_parameters=sum(p.numel() for p in actor.parameters()),
        calibration=calibration,base_weight_norm=float(engine.base_weight_norm_sq.sqrt()),
        editable_weight_norm=float(editable_norm_sq.sqrt()),editable_scope=args.editable_scope,
        editable_parameters=sum(p.numel() for n,p in engine.parameters.items() if n in editable),
        norm_objective=args.norm_objective,
        functional_groups=group_objective.groups if group_objective is not None else None,
        group_reference_l2_norms={n:float(v) for n,v in group_objective.reference_norms.items()} if group_objective is not None else None,
        read_only_parameter_sha256=read_only_baseline_hashes,
        action_scales={n:float(v) for n,v in action_scales.items()},
        replay_sequences=len(replay.train.inputs),withheld_replay_sequences=len(replay.audit.inputs))
    torch.cuda.synchronize()
    first_proposal_start=time.perf_counter()
    with torch.no_grad():
        parameters,_,_=proposal()
        torch.cuda.synchronize()
        log('first_proposal_timing',seconds=time.perf_counter()-first_proposal_start,
            includes_cold_actor_compile=args.compile_chisel,grad_enabled=False,
            compilation_counters=compilation_counters())
        evaluate_parameters(parameters,'identity',0)
    accepted_updates=0
    for step in range(1,args.steps+1):
        torch.cuda.synchronize()
        update_start=time.perf_counter()
        synthetic=(step%5==0);batch=train_synthetic if synthetic else replay.train
        index=int(torch.randint(len(batch.inputs),(1,),generator=rng))
        teacher=engine.cache_teacher(batch.inputs[index:index+1],data_role='synthetic' if synthetic else 'past_training')[0]
        optimizer.zero_grad(set_to_none=True)
        parameters,regularizer,stats=proposal()
        kl=engine.student_parameter_loss(parameters,teacher)
        loss=regularizer+preservation_penalty(kl)
        before=float(loss.detach());loss.backward()
        if step==1:
            torch.cuda.synchronize()
            log('first_training_forward_backward',seconds=time.perf_counter()-update_start,
                includes_cold_actor_compile=args.compile_chisel,
                timing_scope='teacher, actor, finite objective, student GPT, and backward; excludes audit',
                compilation_counters=compilation_counters())
        grad=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True))
        # Release dense candidate/student graphs before testing updated actors.
        del parameters,regularizer,loss,kl,stats
        saved_actor={n:t.detach().clone() for n,t in actor.state_dict().items()}
        saved_optimizer=copy.deepcopy(optimizer.state_dict())
        accepted=False;after=before;trial_lr=args.lr
        for attempt in range(1 if args.optimizer_mode=='adam' else args.backtracks+1):
            if attempt:
                actor.load_state_dict(saved_actor);optimizer.load_state_dict(saved_optimizer)
            trial_lr=args.lr*2**(-attempt)
            for group in optimizer.param_groups:group['lr']=trial_lr
            optimizer.step()
            with torch.no_grad():
                parameters,regularizer,stats=proposal()
                measured_kl=engine.student_parameter_loss(parameters,teacher)
                after=float(regularizer+preservation_penalty(measured_kl))
            log('proposal_trial',step=step,attempt=attempt,lr=trial_lr,objective=after,
                objective_before=before,mean_kl=float(measured_kl),**floats(stats))
            if args.optimizer_mode=='adam' or after<=before:
                accepted=True;accepted_updates+=1;break
        if not accepted:
            actor.load_state_dict(saved_actor);optimizer.load_state_dict(saved_optimizer)
            with torch.no_grad():
                parameters,regularizer,stats=proposal()
                measured_kl=engine.student_parameter_loss(parameters,teacher)
                after=float(regularizer+preservation_penalty(measured_kl))
        torch.cuda.synchronize()
        update_wall_seconds=time.perf_counter()-update_start
        log('train',step=step,objective=after,objective_before=before,accepted=accepted,
            update_wall_seconds=update_wall_seconds,cold_first_update=(step==1),
            update_timing_scope='teacher, actor/objective/student forward-backward, Adam and proposal checks; excludes audit/checkpoint',
            accepted_updates=accepted_updates,backtracks=attempt,lr=trial_lr,gradient_norm=grad,
            mean_kl=float(measured_kl),data_role=teacher.data_role,**floats(stats),
            compute=engine.compute.as_dict(),peak_memory_bytes=torch.cuda.max_memory_allocated())
        if step%args.eval_every==0 or step==args.steps:
            evaluate_parameters(parameters,'learned',step,full=step==args.steps)
            torch.save(dict(controller=actor.state_dict(),optimizer=optimizer.state_dict(),step=step,
                source_checkpoint=args.checkpoint,config=metadata,compute=engine.compute.as_dict()),output/'controller.pt')
        del parameters,regularizer,stats,measured_kl,teacher,saved_actor,saved_optimizer
    # These controls are outside the anchored action family and never train the actor.
    for grid,label in ((None,'continuous_exact_math_gauge'),((.5,1.,2.),'power2_exact_math_gauge')):
        control=norm_optimal_gauge_control(engine.parameters,baseline_gauge_specs(model),scale_grid=grid)
        parameters=dict(engine.parameters,**control.replacements)
        evaluate_parameters(parameters,label,args.steps,full=True)
        (output/(label+'.json')).write_text(json.dumps(control.diagnostics,indent=2)+'\n')
    log('finished',controller_steps=args.steps,accepted_updates=accepted_updates,
        compilation_counters=compilation_counters(),
        compute=engine.compute.as_dict(),calibration=calibration,benchmark_compliant=False)
    if wb:wb.finish()
    log_file.close()


if __name__=='__main__':main()
