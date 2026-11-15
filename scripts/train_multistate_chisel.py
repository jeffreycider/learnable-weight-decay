"""One fresh global Chisel across a same-run baseline state bank and shuffled epochs.

This finite preservation trainer uses extra GPT evaluations on previously consumed
training sequences. It is outside the original Track3 evaluation budget. Audit
states and globally held-out sequence IDs never supply optimizer gradients.
With explicitly enabled convergence training, held-out Chisel checkpoint losses
select learning rates and stopping; official GPT validation is never used.
One optimizer update averages an explicit checkpoint batch;
checkpoint states are streamed sequentially, not all resident on the GPU.
"""
from __future__ import annotations
import argparse,csv,dataclasses,glob,hashlib,json,math,os,time
from pathlib import Path
import sys
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline.model import GPT
from baseline.train import source_record
from baseline.execution import enable_deterministic_execution,execution_environment
from rewriter.state_bank import StateBankReader
from rewriter.chisel_factory import add_chisel_arguments,chisel_architecture,build_chisel,validate_checkpoint_architecture,scale_output_heads
from rewriter.algebra import RawAction,Factors
from rewriter.function_preservation import FunctionalPreservation
from rewriter.offline_systems import CachedBaseNormPreservation, VerifiedStateConstants, select_panel_blocks
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.edit_scope import editable_names,editable_weight_norm_sq,scoped_action,scoped_snapshot,keep_readonly_parameters,readonly_hashes,verify_readonly
from rewriter.replay_buffer import sample_past_sequences,SequenceBatch
from rewriter.fit_schedule import FitSchedule
from rewriter.primal_dual import PrimalDual,feasibility_score
from rewriter.meta_optimizer import build_meta_adam,set_meta_learning_rate,meta_learning_rates


def stable_seed(*parts):
    return int.from_bytes(hashlib.blake2b(':'.join(map(str,parts)).encode(),digest_size=8).digest(),'little')%(2**63-1)


def heldout_sequence(sequence_id,split_seed):
    # A sequence never switches train/audit role across checkpoints or epochs.
    return stable_seed('sequence_partition',split_seed,sequence_id)%5==0


def epoch_batches(state_ids,epoch,seed,checkpoint_batch):
    if checkpoint_batch<1 or epoch<0 or len(state_ids)!=len(set(state_ids)):
        raise ValueError('positive batch, nonnegative epoch, and unique state IDs required')
    generator=torch.Generator().manual_seed(stable_seed('state_order',seed,epoch))
    order=[state_ids[i] for i in torch.randperm(len(state_ids),generator=generator).tolist()]
    return [order[i:i+checkpoint_batch] for i in range(0,len(order),checkpoint_batch)]


def sharded_epoch_batches(state_ids,epoch,seed,checkpoint_batch,world_size):
    """Balanced shuffled epochs with immutable state ownership across ranks.

    State/teacher SHA checks and local filesystem reads are reused by the same
    rank each epoch. Every training state still occurs exactly once per epoch.
    """
    if len(state_ids)%world_size or checkpoint_batch%world_size:
        raise ValueError('fixed state shards require states and checkpoint batch divisible by world size')
    local_batch=checkpoint_batch//world_size;orders=[]
    for rank in range(world_size):
        owned=state_ids[rank::world_size]
        generator=torch.Generator().manual_seed(stable_seed('fixed_state_shard',seed,epoch,rank))
        orders.append([owned[i] for i in torch.randperm(len(owned),generator=generator).tolist()])
    return [[orders[r][i] for i in range(start,min(start+local_batch,len(orders[0]))) for r in range(world_size)]
            for start in range(0,len(orders[0]),local_batch)]


def state_replay(files,completed_steps,training_config,*,pool_sequences,seed,epoch,split_seed):
    available=completed_steps*(training_config['batch_tokens']//training_config['seq_len'])
    count=min(pool_sequences,available)
    if count<2:raise ValueError('past prefix must contain at least two sequences')
    sampled=sample_past_sequences(files,completed_steps,batch_tokens=training_config['batch_tokens'],
        seq_len=training_config['seq_len'],count=count,seed=stable_seed('replay',seed,epoch,completed_steps))
    inputs=torch.cat([sampled.train.inputs,sampled.audit.inputs]);targets=torch.cat([sampled.train.targets,sampled.audit.targets])
    records=sampled.train.provenance+sampled.audit.provenance
    result={}
    for role,is_audit in [('train',False),('audit',True)]:
        selected=[i for i,row in enumerate(records) if heldout_sequence(row['id'],split_seed)==is_audit]
        if not selected:raise ValueError('sampled pool lacks '+role+' sequences; enlarge replay_pool_sequences')
        result[role]=SequenceBatch(inputs[selected],targets[selected],[records[i] for i in selected])
    return result


def select_sequences(batch,count,*,seed):
    if count<1:raise ValueError('positive sequence count required')
    generator=torch.Generator().manual_seed(seed)
    # Use all available unique sequences before drawing any repeated samples.
    indices=torch.randperm(len(batch.inputs),generator=generator)[:count].tolist()
    while len(indices)<count:
        indices.extend(torch.randperm(len(batch.inputs),generator=generator)[:count-len(indices)].tolist())
    return batch.inputs[indices],[batch.provenance[i]['id'] for i in indices]


def globally_heldout_next_sequences(files,state_id,training,count,seed,split_seed):
    """Fresh for this GPT checkpoint and never a Chisel fitting sequence."""
    from rewriter.replay_buffer import next_training_sequences
    available=training['batch_tokens']//training['seq_len']
    batch=next_training_sequences(files,state_id,batch_tokens=training['batch_tokens'],
                                  seq_len=training['seq_len'],count=available,seed=seed)
    selected=[i for i,row in enumerate(batch.provenance) if heldout_sequence(row['id'],split_seed)]
    if len(selected)<count:
        raise ValueError('next ordinary batch has too few globally held-out sequences for the declared audit')
    selected=selected[:count]
    return batch.inputs[selected],[batch.provenance[i]['id'] for i in selected]


def deadband(mean_kl,budget):
    if not math.isfinite(budget) or budget<=0:raise ValueError('positive finite KL budget required')
    excess=max(float(mean_kl)/budget-1.,0.)
    return excess*excess,2*excess/budget


def timed(timings,name,fn,device):
    if timings is None:return fn()
    if device.type=='cuda':torch.cuda.synchronize(device)
    started=time.perf_counter();value=fn()
    if device.type=='cuda':torch.cuda.synchronize(device)
    timings[name]=timings.get(name,0.)+time.perf_counter()-started
    return value


def detached_action_credit(action):
    originals=[];leaves=[]
    def leaf(value):
        result=value.detach().requires_grad_(value.requires_grad)
        originals.append(value);leaves.append(result);return result
    detached=RawAction({n:Factors(leaf(f.P),leaf(f.Q),f.anchors) for n,f in action.matrices.items()},
                       {n:leaf(value) for n,value in action.vectors.items()})
    return detached,(originals,leaves)


def mean_kl_and_backward(engine,parameters,regularizer,inputs,*,kl_budget,state_weight=1.,actor_credit=None,timings=None,teacher_provider=None,diagnostic_parameters=None,primal_dual=None,state_id=None):
    """Exact first-order chain rule for a mean-KL penalty across GPT microbatches.

    GPT backward stops at detached edited-weight leaves. Their accumulated mean
    KL cotangent then traverses the controller graph once. No retained GPT graph,
    no extra counterfactual training evaluation, and no actor update between
    sequence microbatches. This adds one dense editable-gradient bank in memory.
    """
    if len(inputs)<1 or state_weight<=0:raise ValueError('nonempty inputs and positive state weight required')
    leaves={n:value.detach().requires_grad_(value.requires_grad) for n,value in parameters.items()}
    device=next(iter(parameters.values())).device
    total=0.
    for start in range(0,len(inputs),engine.microbatch_sequences):
        selected=inputs[start:start+engine.microbatch_sequences]
        teacher=timed(timings,'teacher_target_total_seconds' if teacher_provider else 'teacher_cache_total_seconds',
                      lambda:teacher_provider(start,start+len(selected)) if teacher_provider else engine.cache_teacher(selected,data_role='past_training')[0],device)
        kl=timed(timings,'student_kl_forward_total_seconds',lambda:engine.student_parameter_loss(leaves,teacher),device)
        fraction=len(selected)/len(inputs)
        value=float(kl.detach())
        if not math.isfinite(value):raise FloatingPointError('nonfinite raw KL')
        total+=value*fraction
        timed(timings,'gpt_kl_backward_seconds',lambda:(kl*fraction).backward(),device)
        del teacher,kl
    if not bool(torch.isfinite(regularizer.detach())):raise FloatingPointError('nonfinite norm/trust objective')
    constraint=primal_dual.terms(total,state_id) if primal_dual is not None else {}
    penalty,coefficient=(constraint["kl_penalty"],constraint["kl_coefficient"]) if constraint else deadband(total,kl_budget)
    outputs=[regularizer];cotangents=[regularizer.new_tensor(state_weight)]
    for name,value in parameters.items():
        if value.requires_grad:
            gradient=leaves[name].grad
            if gradient is None:gradient=torch.zeros_like(value)
            outputs.append(value);cotangents.append(gradient*(coefficient*state_weight))
    diagnostics={}
    if diagnostic_parameters is not None:
        if device.type=='cuda':torch.cuda.synchronize(device)
        diagnostic_started=time.perf_counter()
        if actor_credit is None:raise ValueError('component diagnostics require explicit factor credit')
        original,factor_leaves=actor_credit
        active=[(o,l) for o,l in zip(original,factor_leaves) if l.requires_grad]
        factor_inputs=[x[1] for x in active]
        def component_gradient(component_outputs,component_cotangents):
            factor_grad=torch.autograd.grad(component_outputs,factor_inputs,grad_outputs=component_cotangents,
                                            retain_graph=True,allow_unused=True)
            pairs=[(o,g) for (o,_),g in zip(active,factor_grad) if g is not None]
            grads=torch.autograd.grad([x[0] for x in pairs],diagnostic_parameters,grad_outputs=[x[1] for x in pairs],
                                      retain_graph=True,allow_unused=True)
            return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(diagnostic_parameters,grads)])
        norm_gradient=component_gradient(outputs[:1],cotangents[:1])
        kl_gradient=component_gradient(outputs[1:],cotangents[1:])
        raw_cotangents=[leaves[name].grad if leaves[name].grad is not None else torch.zeros_like(value)
                        for name,value in parameters.items() if value.requires_grad]
        raw_kl_gradient=component_gradient(outputs[1:],raw_cotangents)
        diagnostics=dict(norm_component_gradient_norm=float(norm_gradient.norm()),
                         kl_component_gradient_norm=float(kl_gradient.norm()),
                         raw_kl_component_gradient_norm=float(raw_kl_gradient.norm()),
                         norm_kl_gradient_dot=float(torch.dot(norm_gradient,kl_gradient)),
                         norm_kl_gradient_cosine=float(torch.dot(norm_gradient,kl_gradient)/(norm_gradient.norm()*kl_gradient.norm()).clamp_min(1e-30)))
        if device.type=='cuda':torch.cuda.synchronize(device)
        if timings is not None:timings['component_gradient_diagnostic_seconds']=time.perf_counter()-diagnostic_started
    timed(timings,'norm_edit_backward_seconds' if actor_credit is not None else 'controller_norm_edit_backward_seconds',
          lambda:torch.autograd.backward(outputs,cotangents),device)
    if actor_credit is not None:
        original,leaves=actor_credit
        pairs=[(value,leaf.grad) for value,leaf in zip(original,leaves) if value.requires_grad and leaf.grad is not None]
        if pairs:timed(timings,'actor_backward_seconds',lambda:torch.autograd.backward([x[0] for x in pairs],[x[1] for x in pairs]),device)
    return dict(mean_kl=total,kl_penalty=penalty,objective=float(regularizer.detach())+penalty,**diagnostics,**{k:v for k,v in constraint.items() if k!="kl_penalty"})


@torch.no_grad()
def anchored_norm_ceiling(snapshot,grouped):
    """Ideal continuous-space norm ceiling; ignores KL and radius constraints.

    Cached probes/anchors may be rounded, so clamp tiny infeasible negative
    residuals. This is not a floating-point certificate or a feasible target.
    """
    ceilings={};removable={}
    for group,names in grouped.groups.items():
        reference=grouped.reference_squared[group].double();removed=reference.new_zeros(())
        for name in names:
            if name in snapshot.matrices:
                obs=snapshot.matrices[name];wb,wta=(x.double() for x in obs.weight_probes)
                overlap=obs.anchors.A.double().T@wb
                removed+=wb.square().sum()+wta.square().sum()-overlap.square().sum()
            else:removed+=grouped.parameters[name].double().square().sum()
        removed=removed.clamp(min=0,max=reference)
        value=reference.new_zeros(()) if reference==0 else 100*(1-((reference-removed)/reference).sqrt())
        ceilings[group]=float(value);removable[group]=float(removed)
    return dict(unconstrained_anchored_ceiling_mean_percentage=sum(ceilings.values())/len(ceilings),
                unconstrained_anchored_ceiling_group_percentage=ceilings,
                accessible_squared_weight_norm_by_group=removable,
                ceiling_assumptions='ideal continuous anchored subspace from cached probes; ignores KL/radius and stored-dtype rounding')


@torch.no_grad()
def group_action_scales(grouped,rank):
    """Homogeneous group-relative units, with exact-zero groups frozen, no floor."""
    scales={name:value.new_zeros((),dtype=torch.float32) for name,value in grouped.parameters.items()}
    for group,names in grouped.groups.items():
        count=sum(grouped.parameters[name].numel() for name in names)
        rms=(grouped.reference_norms[group]/math.sqrt(count)).float()
        for name in names:
            value=grouped.parameters[name]
            dimension=rank*(sum(value.shape)-rank) if value.ndim==2 else value.numel()
            scales[name]=rms*math.sqrt(value.numel()/dimension)
    return scales


@torch.no_grad()
def proposed_group_reductions(action,scale,snapshot,grouped):
    """Continuous thin-action norm change, separate from stored-dtype metrics."""
    result={}
    for group,names in grouped.groups.items():
        ref=grouped.reference_squared[group].double();delta=ref.new_zeros(())
        for name in names:
            if name in action.matrices:
                f=action.matrices[name];wb,wta=snapshot.matrices[name].weight_probes
                p,q=f.P.double()*scale.double(),f.Q.double()*scale.double()
                delta+=2*((wb.double()*p).sum()+(wta.double()*q).sum())+p.square().sum()+q.square().sum()
            else:
                edit=action.vectors[name].double()*scale.double()
                delta+=2*(snapshot.vectors[name].weight.double()*edit).sum()+edit.square().sum()
        result[group]=0. if ref==0 else float(-100*delta/(ref.sqrt()*((ref+delta).clamp_min(0).sqrt()+ref.sqrt())))
    return result


@torch.no_grad()
def forward_cast_diagnostics(parameters,grouped,gpt_precision='baseline'):
    """PR323 casts each editable linear/embedding parameter to residual BF16."""
    result={}
    for group,names in grouped.groups.items():
        device=parameters[names[0]].device
        changed=torch.zeros((),device=device,dtype=torch.int64)
        squared=torch.zeros((),device=device,dtype=torch.float64)
        for name in names:
            dtype=torch.float32 if gpt_precision=='fp32' else torch.bfloat16
            before=grouped.parameters[name].to(dtype)
            after=parameters[name].to(dtype)
            changed+=(before!=after).sum()
            squared+=(after.float()-before.float()).double().square().sum()
        result[group]=dict(changed_elements=int(changed),edit_norm=float(squared.sqrt()))
    return result


def build_proposal(actor_forward,snapshot,engine,grouped,*,radius_fraction,timings=None,norm_squared=None,shrinkage_coefficient=1.):
    editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
    norm_sq=editable_weight_norm_sq(engine.parameters,editable) if norm_squared is None else norm_squared
    snapshot=scoped_snapshot(snapshot,norm_sq,radius_fraction,scope='grouped')
    device=next(iter(engine.parameters.values())).device
    action=timed(timings,'actor_forward_seconds',lambda:actor_forward(snapshot),device)
    credit=None
    if torch.is_grad_enabled():action,credit=detached_action_credit(action)
    if device.type=='cuda' and timings is not None:torch.cuda.synchronize(device)
    edit_started=time.perf_counter()
    action=scoped_action(action,editable)
    rank=next(iter(action.matrices.values())).P.shape[1]
    scales=group_action_scales(grouped,rank)
    action=RawAction({n:Factors(f.P*scales.get(n,0.),f.Q*scales.get(n,0.),f.anchors) for n,f in action.matrices.items()},
                     {n:value*scales.get(n,0.) for n,value in action.vectors.items()})
    raw_norm=action.norm_sq().clamp_min(1e-24).sqrt()
    scale=torch.minimum(raw_norm.new_ones(()),snapshot.radius/(raw_norm+1e-12))
    parameters=keep_readonly_parameters(engine.edited_parameters(action,scale),engine.parameters,editable)
    if device.type=='cuda' and timings is not None:torch.cuda.synchronize(device)
    if timings is not None:timings['action_edit_forward_seconds']=time.perf_counter()-edit_started
    result=timed(timings,'norm_forward_seconds',lambda:grouped.evaluate(parameters),device)
    if not math.isfinite(shrinkage_coefficient) or shrinkage_coefficient<0:raise ValueError('nonnegative finite shrinkage coefficient required')
    unweighted_norm_term=100*result.mean_relative_change
    norm_term=shrinkage_coefficient*unweighted_norm_term
    trust=norm_term.new_zeros(())  # removed: edit size is not a loss term
    stats=dict(**result.metrics(),norm_term=float(norm_term.detach()),unweighted_norm_term=float(unweighted_norm_term.detach()),shrinkage_coefficient=shrinkage_coefficient,trust_term=float(trust.detach()),
               raw_action_norm=float(raw_norm.detach()),global_scale=float(scale.detach()),
               hard_radius=float(snapshot.radius),radius_saturated=bool(scale.detach()<1),edit_size_penalty_enabled=False,
               realized_edit_norm=float(result.edit_norm_sq.detach().sqrt()),
               zero_reference_groups=list(grouped.zero_reference_groups),
               frozen_zero_group_parameters=sorted(grouped.frozen_parameter_names))
    stats['proposed_group_l2_percentage_reduction']=proposed_group_reductions(action,scale,snapshot,grouped)
    stats['proposed_edit_norm']=float((raw_norm*scale).detach())
    stats.update(getattr(grouped,'state_diagnostics',{}))
    return parameters,norm_term+trust,stats,editable,credit


@torch.no_grad()
def evaluate_sequences(engine,parameters,inputs,teacher_provider=None,return_ce=False):
    total=0.;n=len(inputs);ce_totals={key:0. for key in ('student_ce','teacher_ce','ce_change')};ce_tokens=0
    rms_squared={key:0. for key in ('logprob_rms_shift','centered_logit_rms_shift')}
    for start in range(0,n,engine.microbatch_sequences):
        selected=inputs[start:start+engine.microbatch_sequences]
        teacher=teacher_provider(start,start+len(selected)) if teacher_provider else engine.cache_teacher(selected,data_role='past_training')[0]
        result=engine.student_parameter_loss(parameters,teacher,return_ce=return_ce)
        if return_ce:
            loss,ce=result;ce_tokens+=ce['ce_tokens']
            for key in ce_totals:ce_totals[key]+=ce[key]*len(selected)/n
            for key in rms_squared:rms_squared[key]+=ce[key]**2*len(selected)/n
        else:loss=result
        total+=float(loss)*len(selected)/n
    return (total,dict(**ce_totals,**{k:v**.5 for k,v in rms_squared.items()},ce_tokens=ce_tokens,ce_scope=ce['ce_scope'])) if return_ce else total


def counters():
    return {str(k):dict(v) for k,v in torch._dynamo.utils.counters.items() if v}


def save_atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def save_calibration_checkpoint(path,value):
    """Publish a complete immutable checkpoint, refusing even identical collisions.

    Stage with the existing fsync/atomic writer, then link without replacement.
    A racing writer cannot overwrite an already published update checkpoint.
    """
    import tempfile
    if path.exists():raise FileExistsError(f'immutable calibration checkpoint exists: {path}')
    with tempfile.TemporaryDirectory(prefix='.calibration-checkpoint-',dir=path.parent) as directory:
        staged=Path(directory)/'checkpoint.pt'
        save_atomic(staged,value)
        os.link(staged,path)


def choose_audit_states(ids,count):
    if not ids:raise ValueError('state bank needs nonempty held-out state split')
    if count==0 or count>=len(ids):return list(ids)
    if count<1:raise ValueError('audit state count must be nonnegative')
    if count==1:return [ids[-1]]
    return [ids[round(i*(len(ids)-1)/(count-1))] for i in range(count)]


_PLOT_METRICS={}


def plot_progress(output):
    """Use standard plotting tools; plotting never influences training decisions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Parse only bytes appended since the previous plot: a full re-parse of the
    # growing metrics file (3.2 s at 232 MB) sat on every rank's critical path.
    from rewriter.progress_metrics import IncrementalMetrics
    cache=_PLOT_METRICS.get(output)
    if cache is None:cache=_PLOT_METRICS[output]=IncrementalMetrics(output/'metrics.jsonl')
    rows=cache.read()
    fig,axes=plt.subplots(4,1,figsize=(9,12),sharex=True)
    for role,label in [('train','training minibatch'),('fixed_training','all fixed training states'),('heldout_sequences','held-out sequences'),('heldout_states','held-out states')]:
        points=[r for r in rows if r.get('event') in ('update','audit_summary') and r.get('role')==role]
        if not points:continue
        x=[r['update'] for r in points]
        axes[0].plot(x,[r['objective'] for r in points],label=label)
        axes[1].plot(x,[r['mean_kl'] for r in points],label=label)
        axes[2].plot(x,[r['kl_penalty'] for r in points],label=label)
        axes[3].plot(x,[r['mean_group_l2_percentage_reduction'] for r in points],label=label)
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    axes[0].axhline(0,color='black',linestyle='--',linewidth=.8);axes[0].set_ylabel('Total objective (identity = 0)')
    axes[1].set_ylabel('Raw KL (nats/token)');axes[2].set_ylabel('KL penalty');axes[3].set_ylabel('Mean group L2 reduction (%)');axes[3].set_xlabel('Controller optimizer updates')
    fig.tight_layout();fig.savefig(output/'learning_curves.tmp.png');plt.close(fig)
    os.replace(output/'learning_curves.tmp.png',output/'learning_curves.png')
    group_rows=[r for r in rows if r.get('event')=='audit_summary' and 'group_l2_percentage_reduction' in r]
    if group_rows:
        groups=list(group_rows[0]['group_l2_percentage_reduction']);fig,axes=plt.subplots(3,5,figsize=(17,9),sharex=True)
        for ax,group in zip(axes.flat,groups):
            for role,label in [('fixed_training','fixed training'),('heldout_sequences','held-out sequences'),('heldout_states','held-out states')]:
                points=[r for r in group_rows if r['role']==role]
                if points:ax.plot([r.get('completed_epochs',r['update']) for r in points],[r['group_l2_percentage_reduction'][group] for r in points],label=label)
            ax.set_title(group);ax.axhline(0,color='black',linewidth=.5);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8);fig.supxlabel('Completed training epochs');fig.supylabel('Mean group L2 reduction (%)')
        fig.tight_layout();fig.savefig(output/'group_curves.tmp.png');plt.close(fig)
        os.replace(output/'group_curves.tmp.png',output/'group_curves.png')
    token_rows=[r for r in rows if r.get('event') in ('update','audit_summary') and 'ranking_penalty' in r]
    if token_rows:
        # Separate changing audit population sizes rather than joining e.g.
        # periodic 32-state and final 800-state panels into one curve.
        populations=sorted({(r['role'],len(r['state_ids']),r['sequence_count']//len(r['state_ids'])) for r in token_rows})
        fig,axes=plt.subplots(len(populations),2,figsize=(14,4*len(populations)),squeeze=False)
        for ax,(role,states,sequences) in zip(axes,populations):
            points=[r for r in token_rows if (r['role'],len(r['state_ids']),r['sequence_count']//len(r['state_ids']))==(role,states,sequences)]
            x=[r['update'] for r in points]
            for key in ('objective','norm_objective','dual_linear_term','tail_quadratic_penalty','ranking_penalty'):
                ax[0].plot(x,[r[key] for r in points],label=key)
            for key in ('token_tail_exceedances_fraction','argmax_changes_fraction','top10_order_changes_fraction','top10_membership_changes_fraction','topk_order_changes_fraction','topk_membership_changes_fraction','strict_topk_changes_fraction'):
                if all(key in r for r in points):ax[1].plot(x,[r[key] for r in points],label=key)
            for a in ax:a.set_title(f'{role}: {states} checkpoints × {sequences} sequences/checkpoint');a.grid(alpha=.2);a.legend(fontsize=7);a.set_xlabel('Chisel Adam updates')
            ax[0].set_ylabel('Signed objective components');ax[1].set_ylabel('Fraction of input token positions')
        fig.tight_layout();fig.savefig(output/'tokenwise_curves.tmp.png');plt.close(fig)
        os.replace(output/'tokenwise_curves.tmp.png',output/'tokenwise_curves.png')


def aggregate_controller_gradients(actor,world_size):
    """SUM local globally-normalized gradients; no GPT DDP or all-reduce here."""
    if world_size==1:return
    parameters=list(actor.parameters())
    used=torch.tensor([p.grad is not None for p in parameters],device=parameters[0].device,dtype=torch.int32)
    dist.all_reduce(used,op=dist.ReduceOp.MAX)
    flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in parameters])
    dist.all_reduce(flat,op=dist.ReduceOp.SUM)
    used_flags=used.tolist()
    offset=0
    for index,parameter in enumerate(parameters):
        gradient=flat[offset:offset+parameter.numel()].view_as(parameter)
        if not used_flags[index]:parameter.grad=None
        elif parameter.grad is None:parameter.grad=gradient.clone()
        else:parameter.grad.copy_(gradient)
        offset+=parameter.numel()


def all_rank_objects(value,world_size):
    if world_size==1:return [value]
    gathered=[None]*world_size;dist.all_gather_object(gathered,value);return gathered


def flat_controller_gradient(parameters):
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten() for p in parameters])


def chisel_gradient_norms(actor):
    heads={id(p) for head in actor.output_heads for p in head.parameters()}
    groups={'residual_blocks':[],'output_heads':[],'input_and_norm':[]}
    for name,p in actor.named_parameters():
        if p.grad is None:continue
        group='output_heads' if id(p) in heads else ('residual_blocks' if name.startswith('blocks.') else 'input_and_norm')
        groups[group].append(p.grad.detach().float().square().sum())
    return {'chisel_'+name+'_gradient_norm':float(torch.stack(values).sum().sqrt()) if values else 0.
            for name,values in groups.items()}


def norm_preservation_diagnostics(stats,grouped):
    """Descriptive consistency/whole-norm metrics; never an extra loss term."""
    reductions=stats['group_l2_percentage_reduction']
    eligible=[g for g in grouped.groups if g not in grouped.zero_reference_groups]
    before=sum(float(v) for v in grouped.reference_squared.values())
    after=sum(float(stats['group_new_l2_norm'][g])**2 for g in grouped.groups)
    return dict(positive_mean_group_reduction=stats['mean_group_l2_percentage_reduction']>0,
        worst_group_l2_percentage_reduction=min((reductions[g] for g in eligible),default=0.),
        nonshrinking_nonzero_groups=[g for g in eligible if reductions[g]<=0],
        zero_reference_groups=list(grouped.zero_reference_groups),
        whole_editable_l2_percentage_reduction=100*(1-math.sqrt(after/before)) if before else 0.)


def norm_consistency_summary(rows):
    return dict(positive_mean_group_reduction_fraction=sum(r['mean_group_l2_percentage_reduction']>0 for r in rows)/len(rows),
        minimum_checkpoint_mean_group_l2_percentage_reduction=min(r['mean_group_l2_percentage_reduction'] for r in rows),
        worst_group_l2_percentage_reduction=min(r['worst_group_l2_percentage_reduction'] for r in rows),
        mean_whole_editable_l2_percentage_reduction=sum(r['whole_editable_l2_percentage_reduction'] for r in rows)/len(rows))


def tokenwise_aggregate(rows):
    """Losses average across checkpoint panels; descriptive rates across tokens."""
    tokens=sum(r['token_count'] for r in rows)
    result={k:sum(r[k] for r in rows)/len(rows) for k in
        ('norm_objective','tail_quadratic_penalty','ranking_penalty')}
    result.update(token_count=tokens,sequence_count=sum(r['sequence_count'] for r in rows),
        maximum_token_kl=max(r['token_kl_quantiles']['max'] for r in rows))
    for name in ('token_tail_exceedances','negative_token_kl_count','argmax_changes','top10_order_changes','top10_membership_changes','topk_order_changes','topk_membership_changes','tie_aware_argmax_changes','strict_order_changes','strict_membership_changes','strict_topk_changes'):
        if not all(name in r for r in rows):continue
        result[name]=sum(r[name] for r in rows)
        result[name+'_fraction']=result[name]/tokens
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    add_chisel_arguments(p)
    p.add_argument('--state-bank',type=Path);p.add_argument('--multibank-index',type=Path,help='explicit pinned cross-seed GPT bank/cache index');p.add_argument('--output',required=True,type=Path)
    p.add_argument('--verify-state-once',action='store_true',help='cache SHA verification while immutable file identity is unchanged; rehash changed files')
    p.add_argument('--mmap-state',action='store_true',help='privately map CPU model weights rather than eagerly copying the entire checkpoint')
    p.add_argument('--teacher-cache',type=Path,help='verified fixed train/audit target panels; no repeated teacher transformer evaluation')
    p.add_argument('--diagnostic-prefix-states',type=int,default=0,help='fixed committed prefix only; diagnostic controller must not initialize final fit')
    p.add_argument('--train-data',required=True);p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--checkpoint-batch',type=int,default=4);p.add_argument('--sequences-per-checkpoint',type=int,default=32)
    p.add_argument('--fixed-state-shards',action='store_true',help='shuffle within balanced immutable rank ownership; reuse integrity checks and local reads')
    p.add_argument('--gpt-microbatch',type=int,default=4);p.add_argument('--replay-pool-sequences',type=int,default=4096)
    p.add_argument('--gpt-precision',choices=('baseline','fp32'),default='baseline')
    p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--kl-budget',type=float,default=.001)
    p.add_argument('--representation-lr-multiplier',type=float,default=1.,help='scale input, normalization and residual-block Adam LR; action heads retain --lr')
    p.add_argument('--objective',choices=('primal-dual','legacy-deadband','tokenwise-v1'),default='primal-dual')
    p.add_argument('--token-loss-backend',choices=('reference','tiled_fp64_v1','tiled_tie_fp64_v1'),default=None,help='explicit tokenwise logit algebra; default reference')
    p.add_argument('--kl-tolerance',type=float,default=1e-8,help='mean KL tolerance; independent of scale in tokenwise-v1')
    p.add_argument('--token-tail-tolerance',type=float)
    p.add_argument('--token-normalization-scale',type=float)
    p.add_argument('--ranking-weight',type=float)
    p.add_argument('--ranking-k',type=int)
    p.add_argument('--ranking-tie-policy',choices=('legacy','exact_teacher_ties'))
    p.add_argument('--ranking-margin-fraction',type=float)
    p.add_argument('--token-dual-update',choices=('unclipped_scaled','clipped_relative'),help='explicit dual ablation; primal loss always uses fixed normalization scale')
    p.add_argument('--calibration-state-ids',type=int,nargs='+',help='explicit fitting-only subset; fixed first cached sequences; diagnostic actors cannot initialize full-bank fits')
    p.add_argument('--calibration-readout-updates',type=int,nargs='+',help='fixed-update fitting readouts; calibration only')
    p.add_argument('--calibration-balance-only',action='store_true',help='read-only initial logit cotangent norms and median gamma; no GPT backward or fitting')
    p.add_argument('--augmented-rho',type=float,default=1.)
    p.add_argument('--dual-lr',type=float,default=.01)
    p.add_argument('--tolerance-evidence',type=Path,help='JSON recording numerical checks and the explicitly chosen tolerance')
    p.add_argument('--primal-init',type=Path,help='explicit same-bank objective transition: load actor, reset Adam, duals, schedule, and epoch cursor')
    p.add_argument('--radius-fraction',type=float,default=.01);p.add_argument('--gradient-clip',type=float,default=1.)
    p.add_argument('--controller-seed',type=int,default=200001);p.add_argument('--data-seed',type=int,default=400001)
    p.add_argument('--split-seed',type=int,default=500001);p.add_argument('--expected-gpt-seed',type=int)
    p.add_argument('--audit-every',type=int,default=50);p.add_argument('--audit-states',type=int,default=8)
    p.add_argument('--audit-sequences',type=int,default=32);p.add_argument('--checkpoint-every',type=int,default=10)
    p.add_argument('--audit-every-epochs',type=int,default=0,help='positive value replaces update-based audits with equal epoch exposure intervals')
    p.add_argument('--full-training-audit',action='store_true',help='evaluate the objective on every fixed training-state panel; diagnostic only, no update selection')
    p.add_argument('--training-audit-states',type=int,default=0,help='predeclared uniformly spaced training-state panels for a quick diagnostic;0 disables')
    p.add_argument('--full-training-audit-final',action='store_true',help='evaluate every training state at the predeclared final epoch, using sampled panels earlier')
    p.add_argument('--gradient-diagnostics',action='store_true',help='measure separate controller component norms and checkpoint-gradient alignment')
    p.add_argument('--track-trained-policy-progress',action='store_true',help='keep the initial policy eligible for best-checkpoint selection, but start LR/stopping progress tracking after the first trained-policy audit')
    p.add_argument('--eager-chisel',action='store_true');p.add_argument('--wandb',action='store_true')
    p.add_argument('--actor-backend',choices=('transformer',),default='transformer',help='direct NanoGPT residual transformer; legacy backends are retired')
    p.add_argument('--norm-backend',choices=('reference','fused'),default='reference')
    p.add_argument('--compile-gpt',action='store_true',help='compile one persistent functional GPT forward; cold teacher/student variants are counted')
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda',help='CPU is for separate tiny correctness tests')
    p.add_argument('--no-plots',action='store_true');p.add_argument('--resume',type=Path)
    p.add_argument('--direction-init',type=Path,help='same-bank learned direction warmup checkpoint; fresh finite-KL Adam state')
    p.add_argument('--direction-head-scale',type=float,default=1.,help='predeclared common output-head scale after direction warmup')
    p.add_argument('--initial-audit-only',action='store_true',help='measure declared initialization without fitting or selecting a checkpoint')
    p.add_argument('--audit-forward-cast',action='store_true',help='count changed BF16-forward parameter values per role on audits')
    p.add_argument('--audit-fresh-next',action='store_true',help='audit only: unused sequences from each checkpoint next ordinary batch; never optimize on them')
    p.add_argument('--convergence',action='store_true',help='use all held-out Chisel checkpoint objectives for LR and stopping; never official GPT validation')
    p.add_argument('--minimum-epochs',type=int,default=30)
    p.add_argument('--plateau-patience',type=int,default=3)
    p.add_argument('--stopping-patience',type=int,default=5)
    p.add_argument('--minimum-lr-reductions',type=int,default=2)
    p.add_argument('--post-reduction-checks',type=int,default=2)
    p.add_argument('--improvement-threshold',type=float,default=.001)
    p.add_argument('--lr-reduction-factor',type=float,default=.5)
    p.add_argument('--minimum-lr',type=float,default=1e-8)
    p.add_argument('--shrinkage-coefficient',type=float,default=1.,help='explicit norm-reward weight; zero enables a preservation-only repair stage without changing shrinkage metrics')
    p.add_argument('--offline-systems-cache',action='store_true',help='opt-in verified state constants and immutable teacher block views; requires teacher cache')
    p.add_argument('--offline-backend',choices=('reference','strict_graph','tokenwise_graph'),default='reference',help='explicit FP32 graph accumulation; audits and global optimizer remain unchanged')
    p.add_argument('--no-component-timing',action='store_true',help='disable synchronous component instrumentation; retain update wall time')
    args=p.parse_args(argv)
    composite=args.multibank_index is not None
    if composite:
        if args.state_bank or args.teacher_cache or args.expected_gpt_seed is not None:p.error('multibank-index replaces state-bank, teacher-cache and expected-gpt-seed')
        if args.diagnostic_prefix_states or args.audit_fresh_next:p.error('composite banks require full child coverage and separate fresh-seed readout')
    elif args.state_bank is None or args.expected_gpt_seed is None:p.error('single-bank fitting requires state-bank and expected-gpt-seed')
    has_cache=bool(args.teacher_cache or composite)
    from rewriter.tokenwise_reference import tokenwise_config_from_options,TokenwiseDual,tokenwise_panel,validate_objective_resume
    try:token_config=tokenwise_config_from_options(args)
    except ValueError as error:p.error(str(error))
    if args.calibration_state_ids is not None:
        if token_config is None or not has_cache or args.convergence or args.diagnostic_prefix_states:
            p.error('calibration requires tokenwise-v1 and teacher cache, without convergence or truncated bank')
        if len(set(args.calibration_state_ids))!=len(args.calibration_state_ids) or args.checkpoint_batch!=len(args.calibration_state_ids):
            p.error('calibration requires unique state IDs and one full accumulated checkpoint batch')
        if args.calibration_readout_updates is None:args.calibration_readout_updates=[0,1,8,32,64,128]
        if min(args.calibration_readout_updates)<0:p.error('nonnegative calibration readout updates required')
        if args.calibration_balance_only:args.initial_audit_only=True
    elif args.calibration_readout_updates is not None or args.calibration_balance_only:
        p.error('calibration options require explicit calibration-state-ids')
    if args.offline_backend in ('strict_graph','tokenwise_graph'):
        if args.offline_backend=='strict_graph':from rewriter.offline_graph_backend import validate_training_options
        else:from rewriter.offline_tokenwise_graph import validate_training_options
        try:validate_training_options(args)
        except ValueError as error:p.error(str(error))
    if not math.isfinite(args.shrinkage_coefficient) or args.shrinkage_coefficient<0:p.error('nonnegative finite shrinkage-coefficient required')
    if args.shrinkage_coefficient!=1 and args.objective not in ('primal-dual','tokenwise-v1'):p.error('shrinkage-coefficient changes require the explicit primal-dual objective')
    if args.offline_systems_cache and not has_cache:p.error('offline-systems-cache requires teacher-cache')
    if sum(x is not None for x in (args.resume,args.direction_init,args.primal_init))>1:p.error('resume, direction-init, and primal-init are mutually exclusive')
    if args.primal_init and args.objective not in ('primal-dual','tokenwise-v1'):p.error('primal-init requires primal-dual objective')
    if args.objective in ('primal-dual','tokenwise-v1') and args.device=='cuda' and not args.tolerance_evidence:p.error('GPU primal-dual fitting requires tolerance-evidence JSON; identity zero alone is not calibration')
    if args.objective in ('primal-dual','tokenwise-v1') and args.actor_backend!='transformer':p.error('legacy persistent kernels do not implement the primal-dual objective')
    if not math.isfinite(args.direction_head_scale) or args.direction_head_scale<0:p.error('nonnegative finite direction-head-scale required')
    if (args.full_training_audit or args.training_audit_states or args.full_training_audit_final) and not has_cache:p.error('fixed-training audit requires immutable target panels')
    if args.training_audit_states<0:p.error('nonnegative training-audit-states required')
    positive=('epochs','checkpoint_batch','sequences_per_checkpoint','gpt_microbatch','replay_pool_sequences',
              'lr','representation_lr_multiplier','kl_budget','kl_tolerance','augmented_rho','dual_lr','radius_fraction','gradient_clip','audit_every','audit_sequences','checkpoint_every')
    if any(not math.isfinite(getattr(args,key)) or getattr(args,key)<=0 for key in positive) or args.audit_states<0 or args.audit_every_epochs<0:
        p.error('positive finite training scales/counts and nonnegative audit_states required')
    if args.replay_pool_sequences<16:p.error('replay pool must contain at least16 candidates')
    world_size=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local_rank=int(os.environ.get('LOCAL_RANK','0'))
    device=torch.device(f'cuda:{local_rank}' if args.device=='cuda' else 'cpu')
    if device.type=='cuda':
        torch.set_num_interop_threads(20);enable_deterministic_execution();torch.cuda.set_device(device)
    else:
        torch.set_num_threads(1)
    from rewriter.gpt_precision import configure_precision,precision_metadata
    configure_precision(args.gpt_precision)
    if world_size>1:dist.init_process_group(backend='nccl' if device.type=='cuda' else 'gloo')
    def synchronize():
        if device.type=='cuda':torch.cuda.synchronize()
    if args.diagnostic_prefix_states and args.diagnostic_prefix_states<5:p.error('diagnostic prefix needs at least5 states including a holdout')
    cache_reader=None
    if composite:
        from rewriter.multibank import MultiStateBankReader,MultiTeacherCacheReader
        reader=MultiStateBankReader(args.multibank_index,verify_once=args.verify_state_once,mmap_model=args.mmap_state)
        cache_reader=MultiTeacherCacheReader(reader,verify_once=args.verify_state_once,mmap=args.mmap_state)
    else:
        reader=StateBankReader(args.state_bank,require_complete=not args.diagnostic_prefix_states,
                              verify_once=args.verify_state_once,mmap_model=args.mmap_state)
        if args.teacher_cache:
            from rewriter.teacher_cache import TeacherCacheReader
            cache_reader=TeacherCacheReader(args.teacher_cache,verify_once=args.verify_state_once,mmap=args.mmap_state)
    provenance=reader.manifest['provenance']
    if cache_reader and cache_reader.manifest.get('execution',{}).get('gpt_precision','baseline')!=args.gpt_precision:
        raise ValueError('teacher cache and requested GPT precision differ')
    limit=args.diagnostic_prefix_states or reader.manifest['expected_steps']
    if len(reader.entries)<limit:p.error('requested diagnostic prefix is not fully committed yet')
    if not composite and provenance['seed']!=args.expected_gpt_seed:p.error('state bank belongs to a different GPT seed; controller transfer is forbidden')
    training=provenance['training_config'];model_config=provenance['model_config']
    probe_rank=provenance['sensor_config'].get('probe_rank',provenance['sensor_config'].get('rank',8))
    architecture=chisel_architecture(args,rank=probe_rank,trajectory_dim=(2*model_config['num_layers']+1)*2*probe_rank)
    train_states=[s for s in reader.manifest['split']['train'] if s<=limit]
    if args.calibration_state_ids is not None:
        if not set(args.calibration_state_ids)<=set(train_states):p.error('calibration IDs must belong to the original fitting split')
        train_states=list(args.calibration_state_ids)
    audit_states=choose_audit_states([s for s in reader.manifest['split']['audit'] if s<=limit],args.audit_states)
    if args.convergence:
        if args.audit_every_epochs<1 or args.minimum_epochs>args.epochs:p.error('convergence needs epoch validation and maximum epochs >= minimum epochs')
        if audit_states!=[s for s in reader.manifest['split']['audit'] if s<=limit]:p.error('convergence must validate every held-out Chisel state')
    sequence_audit_count=args.training_audit_states or (32 if args.convergence else args.audit_states)
    train_audit_states=choose_audit_states(train_states,sequence_audit_count)
    if args.training_audit_states and len(train_audit_states)!=min(args.training_audit_states,len(train_states)):
        raise AssertionError('explicit withheld-sequence checkpoint count was not honored')
    if not train_states:p.error('empty training-state split')
    files=sorted(glob.glob(args.train_data))
    if not files:p.error('no training shards')
    if not args.no_plots:
        import matplotlib  # fail before training if requested plotting dependency is absent
    identity=dict(manifest_sha256=hashlib.sha256((args.multibank_index if composite else args.state_bank/'manifest.json').read_bytes()).hexdigest(),
                  selected_committed_entries_sha256=hashlib.sha256(json.dumps(reader.entries[:limit],sort_keys=True).encode()).hexdigest(),selected_states=limit)
    if composite:identity['composite_teacher_manifest_sha256']=cache_reader.manifest['child_manifest_sha256']
    if args.teacher_cache:identity['teacher_cache_manifest_sha256']=hashlib.sha256((args.teacher_cache/'manifest.json').read_bytes()).hexdigest()
    options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
    source_paths=[Path(__file__),*[Path(__file__).resolve().parents[1]/'rewriter'/name for name in
        ('gpt_precision.py','primal_dual.py','meta_optimizer.py','transformer_chisel.py','chisel_factory.py','fit_schedule.py','controller.py','algebra.py','function_preservation.py','grouped_norm.py','edit_scope.py','state_bank.py','teacher_cache.py','replay_buffer.py','accelerated_grouped_norm.py','accelerated_kl.py','offline_systems.py')],Path(__file__).resolve().parents[1]/'baseline'/'model.py']
    if composite:source_paths.append(Path(__file__).resolve().parents[1]/'rewriter'/'multibank.py')
    if args.offline_backend in ('strict_graph','tokenwise_graph'):
        source_paths.extend(Path(__file__).resolve().parents[1]/'rewriter'/name for name in
            ('offline_graph_backend.py','graph_state_identity.py','graph_offline.py','resident_teacher.py','device_primal_dual.py',
             'factorized_fp32.py','anchored_linear.py','compensated_dx.py','fused_softcap.py','tiled_teacher_kl.py'))
        source_paths.extend(sorted((Path(__file__).resolve().parents[1]/'rewriter'/'kernels').glob('*.py')))
        if args.offline_backend=='tokenwise_graph':
            source_paths.extend(Path(__file__).resolve().parents[1]/'rewriter'/name for name in
                ('offline_tokenwise_graph.py','graph_tokenwise.py','tokenwise_tiled.py','tiled_ranking_loss.py'))
    if token_config is not None:
        source_paths.extend(Path(__file__).resolve().parents[1]/'rewriter'/name for name in
            ('tokenwise_reference.py','token_preservation.py','ranking_loss.py','rank_preservation.py'))
        if token_config.loss_backend in ('tiled_fp64_v1','tiled_tie_fp64_v1'):
            source_paths.extend(Path(__file__).resolve().parents[1]/'rewriter'/name for name in
                ('tokenwise_tiled.py','tiled_teacher_kl.py','tiled_ranking_loss.py'))
        if token_config.loss_backend=='tiled_tie_fp64_v1':
            source_paths.extend(Path(__file__).resolve().parents[1]/'rewriter'/name for name in
                ('tokenwise_tiled_tie.py','tiled_tie_ranking.py'))
    implementation_sha256={str(path.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths if path.exists()}
    metadata=dict(options=options,architecture=architecture,state_bank_identity=identity,state_bank_manifest=reader.manifest,implementation_sha256=implementation_sha256,
        benchmark_compliant=False,controller_pretraining=composite,controller_state_transferred=False,
        same_run_state_replay=not composite,diagnostic_prefix_only=bool(args.diagnostic_prefix_states),selected_state_count=limit,
        checkpoint_batch_execution='sequential gradient accumulation, one Adam update',
        teacher_target_policy='fixed persisted panels; head-only reconstruction if factorized' if cache_reader else 'recompute teacher on epoch-resampled past sequences',
        audit_actor_arithmetic='grad-enabled action then detached evaluation' if cache_reader else 'no-grad action specialization',
        sequence_split='stable global sequence-ID hash; same role across all states and epochs;20percent audit',
        fresh_next_audit_split='globally held-out sequence IDs only; never fitted at any checkpoint',
        validation_feedback='held-out Chisel checkpoint objective for plateau LR and stopping' if args.convergence else 'none',
        official_gpt_validation_feedback=False,
        zero_reference_policy='freeze exact-zero groups, zero contribution in fifteen-group mean',
        action_scale_policy='functional-group RMS without floor, times sqrt(tensor_numel/accessible_dimension); vectors use group RMS',
        controller_compilation='one persistent direct TransformerChisel fullgraph callable; no packed legacy actor; finite grouped objective eager; optional persistent compiled GPT',
        world_size=world_size,distributed_policy='checkpoint minibatches partitioned by rank; globally normalized Chisel gradient SUM before shared Adam',
        actual_next_loss_credit=False,live_gpt_handoff='not performed by this trainer',
        execution=execution_environment() if device.type=='cuda' else dict(device='cpu',purpose='separate correctness tests'),heldout_state_ids=audit_states,heldout_sequence_state_ids=train_audit_states)
    metadata['audit_population_sizes']=dict(periodic_fitting=len(train_states) if args.full_training_audit else min(args.training_audit_states,len(train_states)),withheld_sequences=len(train_audit_states),development_checkpoints=len(audit_states),final_fitting=len(train_states) if args.full_training_audit_final else None)
    metadata['objective_policy']=dict(kind=args.objective,shrinkage_coefficient=args.shrinkage_coefficient,edit_size_penalty=False,hard_radius_separate=True,
        dual_update='projected ascent once after Adam, using globally gathered pre-update fitting-panel mean KL',
        convergence_score='worst normalized violation until all feasible, then negative positive shrinkage; no learned lambdas' if args.objective=='primal-dual' else 'legacy deadband objective')
    if token_config is not None:
        metadata['tokenwise_objective']=dataclasses.asdict(token_config)
        metadata['objective_policy'].update(kl_penalty='mean lambda*(clamp(KL_token,0)-mean_tolerance)/scale + rho/2*mean(relu((clamp(KL_token,0)-tail_tolerance)/scale)^2) + rank_weight*mean(explicit_topk_margin_loss)',
            normalization='one complete panel token denominator; one checkpoint batch weight; fixed scale independent of tolerances',
            convergence_score='1+worst mean-KL tolerance-relative violation when infeasible; otherwise 2/pi*atan(mean fixed tail+rank+signed norm objective), strictly monotone and below1; no learned multipliers')
    if args.calibration_state_ids is not None:
        metadata['calibration']=dict(fitting_state_ids=train_states,sequence_selection='first cached training sequences, fixed across all updates',
            readout_updates=args.calibration_readout_updates,balance_only=args.calibration_balance_only,
            diagnostic_only=True,allowed_to_initialize_full_fit=False,original_bank_split_preserved=True)
        metadata['audit_population_sizes']=dict(fixed_calibration_fitting=len(train_states),development_checkpoints=0)
    metadata['gpt_precision']=precision_metadata(args.gpt_precision)
    metadata['offline_backend']=dict(name=args.offline_backend,
        graph_scope=('strict FP32 forward, thin backward, exact FP64 teacher targets, tokenwise aggregate' if args.offline_backend=='tokenwise_graph' else 'strict FP32 forward, thin backward, tiled FP64 KL; explicit coefficient 0 or 1' if args.offline_backend=='strict_graph' else None),
        global_optimizer_owner='trainer: SUM, clip, Adam, then dual once',
        graph_compute_accounting='actual warmups plus training replays; stream capture itself records without executing')
    metadata['captured_gpt_trajectory_precision']='original PR323 mixed precision; stored checkpoints are not regenerated'
    if args.tolerance_evidence:
        metadata['tolerance_evidence']=json.loads(args.tolerance_evidence.read_text())
        if metadata['tolerance_evidence'].get('selected_tolerance')!=args.kl_tolerance:
            raise ValueError('tolerance evidence does not match selected KL tolerance')
    if rank==0:args.output.mkdir(parents=True,exist_ok=args.resume is not None)
    if world_size>1:dist.barrier()
    model=GPT(**model_config).to(device);model.requires_grad_(False)
    with torch.random.fork_rng(devices=[local_rank] if device.type=='cuda' else []):
        torch.manual_seed(args.controller_seed)
        actor=build_chisel(architecture).to(device)
    if args.direction_init:
        warmup=torch.load(args.direction_init,map_location=device,weights_only=False)
        validate_checkpoint_architecture(warmup,architecture)
        if warmup['metadata']['bank_manifest_sha256']!=identity['manifest_sha256']:
            raise ValueError('direction warmup belongs to another baseline run')
        if warmup['metadata']['bank_provenance']!=provenance:
            raise ValueError('direction warmup provenance differs')
        warm_limit=warmup['metadata']['options']['prefix_states'] or reader.manifest['expected_steps']
        if warm_limit!=limit:raise ValueError('direction warmup state scope differs from finite fit')
        actor.load_state_dict(warmup['controller'])
        scale_output_heads(actor,args.direction_head_scale)
        metadata['same_run_direction_warmup']=dict(path=str(args.direction_init),
            sha256=hashlib.sha256(args.direction_init.read_bytes()).hexdigest(),
            updates=warmup['updates'],head_scale=args.direction_head_scale)
    if args.primal_init:
        initial=torch.load(args.primal_init,map_location=device,weights_only=False)
        if initial['metadata'].get('calibration') is not None and args.calibration_state_ids is None:
            raise ValueError('calibration actors cannot initialize full-bank fitting')
        validate_checkpoint_architecture(initial,architecture)
        if initial['metadata']['state_bank_identity']!=identity:raise ValueError('primal initialization belongs to another bank or diagnostic prefix')
        actor.load_state_dict(initial['controller'])
        metadata['objective_transition']=dict(source=str(args.primal_init),sha256=hashlib.sha256(args.primal_init.read_bytes()).hexdigest(),
            source_updates=initial['updates'],source_implementation=initial['metadata']['implementation_sha256'],
            actor_preserved=True,adam_reset=True,duals_reset=True,schedule_reset=True,data_cursor_reset=True)
    optimizer=build_meta_adam(actor,lr=args.lr,representation_lr_multiplier=args.representation_lr_multiplier)
    dual=PrimalDual(train_states,tolerance=args.kl_tolerance,rho=args.augmented_rho,dual_lr=args.dual_lr) if args.objective=='primal-dual' else None
    if token_config is not None:dual=TokenwiseDual(train_states,config=token_config,dual_lr=args.dual_lr)
    fit_schedule=FitSchedule(minimum_epochs=args.minimum_epochs,plateau_patience=args.plateau_patience,
        stopping_patience=args.stopping_patience,minimum_reductions=args.minimum_lr_reductions,
        post_reduction_checks=args.post_reduction_checks,threshold=args.improvement_threshold,
        factor=args.lr_reduction_factor,minimum_lr=args.minimum_lr) if args.convergence else None
    controller_parameters=list(actor.parameters())
    epoch0=batch0=updates=clipped_updates=0
    if args.resume:
        saved=torch.load(args.resume,map_location=device,weights_only=False)
        validate_checkpoint_architecture(saved,architecture)
        validate_objective_resume(saved['metadata'],token_config)
        old=dict(saved['metadata']);oldopts=dict(old['options']);newopts=dict(options)
        for key in ('output','state_bank','train_data','primal_init','direction_init'):oldopts.pop(key,None);newopts.pop(key,None)
        if args.convergence and oldopts.get('convergence') and newopts['epochs']>=oldopts['epochs']:
            oldopts.pop('epochs');newopts.pop('epochs')
        if args.calibration_state_ids is not None and oldopts.get('calibration_state_ids')==args.calibration_state_ids and newopts['epochs']>=oldopts['epochs']:
            oldopts.pop('epochs');newopts.pop('epochs')
        if oldopts!=newopts or old['state_bank_identity']!=identity or old['implementation_sha256']!=implementation_sha256 or old['world_size']!=world_size:raise ValueError('resume settings or source bank changed')
        if saved.get('completed',False):raise ValueError('controller trial is already complete')
        if 'objective_transition' in old:metadata['objective_transition']=old['objective_transition']
        if 'same_run_direction_warmup' in old:
            metadata['same_run_direction_warmup']=old['same_run_direction_warmup']
        if 'source_migration' in old:metadata['source_migration']=old['source_migration']
        actor.load_state_dict(saved['controller']);optimizer.load_state_dict(saved['optimizer'])
        rng=saved['per_rank_rng'][rank]
        torch.set_rng_state(rng['torch'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state(rng['cuda'].cpu(),device=device)
        epoch0,batch0,updates=saved['next_epoch'],saved['next_batch'],saved['updates']
        clipped_updates=saved.get('clipped_updates',0)
        if dual is not None:
            if saved.get('primal_dual') is None:raise ValueError('primal-dual resume lacks dual state')
            dual.load_state_dict(saved['primal_dual'])
        if args.convergence:
            if saved.get('fit_schedule') is None:raise ValueError('convergence resume lacks schedule state')
            fit_schedule=FitSchedule.from_state_dict(saved['fit_schedule'])
    if world_size>1:
        for parameter in actor.parameters():dist.broadcast(parameter.data,src=0)
    actor_forward=actor if args.eager_chisel else torch.compile(actor,fullgraph=True,dynamic=False)
    if rank==0:
        (args.output/'config.json').write_text(json.dumps(metadata,indent=2)+'\n');source_record(metadata,args.output)
    log_file=(args.output/'metrics.jsonl').open('a' if args.resume else 'w') if rank==0 else None;started=time.perf_counter()
    wb=None
    if args.wandb and rank==0:
        import wandb
        wb=wandb.init(project='chisel',mode='offline',dir=str(args.output),config=metadata)
    compute=dict(saved['per_rank_compute'][rank]) if args.resume else {};seen_zero={};state_diagnostics={};cached_head_seen={'forwards':0,'tokens':0}
    norm_class=GroupedNormObjective
    if args.norm_backend=='fused':
        from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
        norm_class=AcceleratedGroupedNormObjective
    engine_class=CachedBaseNormPreservation if args.offline_systems_cache else FunctionalPreservation
    engine=engine_class(model,microbatch_sequences=args.gpt_microbatch,teacher_cache_device=device,kl_dtype=torch.float64,gpt_precision=args.gpt_precision)
    state_constants=VerifiedStateConstants(engine,norm_class) if args.offline_systems_cache else None
    graph_backend=None;graph_compute_seen={}
    if args.offline_backend in ('strict_graph','tokenwise_graph'):
        if args.offline_backend=='strict_graph':from rewriter.offline_graph_backend import OfflineGraphAccumulator
        else:from rewriter.offline_tokenwise_graph import TokenwiseGraphAccumulator as OfflineGraphAccumulator
        graph_backend=OfflineGraphAccumulator(actor,actor_forward,engine,cache_reader,state_constants,
            radius_fraction=args.radius_fraction,shrinkage_coefficient=args.shrinkage_coefficient)
    if args.compile_gpt:engine.enable_compiled_forward(fullgraph=True,dynamic=False)
    parameter_names=set(dict(model.named_parameters()));buffer_names=set(dict(model.named_buffers()))
    timing_holder=[None];original_forward=engine._forward
    def timed_gpt_forward(parameters,inputs):
        key='teacher_gpt_forward_seconds' if parameters is engine.parameters else 'student_gpt_forward_seconds'
        return timed(timing_holder[0],key,lambda:original_forward(parameters,inputs),device)
    engine._forward=timed_gpt_forward
    def global_compute():
        totals={}
        for local in all_rank_objects(dict(compute),world_size):
            for key,value in local.items():totals[key]=totals.get(key,0)+value
        return totals
    def log(event,**values):
        if rank!=0:return
        row=dict(event=event,elapsed_seconds=time.perf_counter()-started,update=updates,**values)
        line=json.dumps(row,allow_nan=False);print(line,flush=True);log_file.write(line+'\n');log_file.flush()
        if wb:wb.log(row)
    def load_state(state_id):
        record=reader.read(state_id,device=device,model_device='cpu')
        if state_constants:state_constants.load(cache_reader,record)
        elif cache_reader:cache_reader.load_engine(engine,record)
        else:engine.load_teacher_state({n:record.model[n] for n in parameter_names},{n:record.model[n] for n in buffer_names})
        grouped=state_constants.grouped() if state_constants else norm_class(engine.parameters,zero_reference_policy='freeze')
        if state_id not in state_diagnostics:state_diagnostics[state_id]=anchored_norm_ceiling(record.snapshot,grouped)
        grouped.state_diagnostics=state_diagnostics[state_id]
        if seen_zero.get(state_id)!=grouped.zero_reference_groups:
            log('state_policy',state_id=state_id,zero_reference_groups=list(grouped.zero_reference_groups),frozen_zero_group_parameters=sorted(grouped.frozen_parameter_names),chisel_tokens=actor.token_counts(record.snapshot));seen_zero[state_id]=grouped.zero_reference_groups
        return record,engine,grouped
    calibration_panels={}
    def panel_inputs(state_id,role,epoch,count,phase_times,*,return_panel=False):
        if cache_reader:
            panel=timed(phase_times,'teacher_panel_load_seconds',lambda:cache_reader.read(state_id,role=role),device)
            if count==0:count=len(panel.inputs)
            if len(panel.inputs)<count:raise ValueError('cached panel has fewer sequences than declared configuration')
            if args.calibration_state_ids is not None:
                if role!='train':raise ValueError('calibration reads only fixed fitting panels')
                size=panel.metadata['microbatch_sequences']
                if count%size:raise ValueError('calibration count must contain complete cache generation microbatches')
                if len(panel.inputs)>count:panel=panel.select_blocks(list(range(count//size)))
                from rewriter.teacher_cache import tensor_hash
                binding=dict(sequence_ids=list(panel.sequence_ids),input_tensor_sha256=tensor_hash(panel.inputs))
                if state_id in calibration_panels:
                    if binding!=calibration_panels[state_id]:raise ValueError('calibration panel changed')
                else:
                    calibration_panels[state_id]=binding
                    log('calibration_panel',state_id=state_id,**binding)
            elif len(panel.inputs)>count:
                size=panel.metadata['microbatch_sequences']
                if count%size or len(panel.inputs)%count:raise ValueError('subset size must tile the fixed panel in complete generation microbatches')
                chunks=len(panel.inputs)//count
                cycle=max(epoch,0)//chunks
                generator=torch.Generator().manual_seed(stable_seed('cached_panel_blocks',args.data_seed,state_id,cycle))
                blocks=torch.randperm(len(panel.inputs)//size,generator=generator).tolist()
                offset=(max(epoch,0)%chunks)*(count//size)
                panel=(select_panel_blocks(panel,blocks[offset:offset+count//size]) if args.offline_systems_cache
                       else panel.select_blocks(blocks[offset:offset+count//size]))
            if any(heldout_sequence(i,args.split_seed)!=(role=='audit') for i in panel.sequence_ids):raise ValueError('cached panel violates the globally fixed training/audit sequence split')
            if return_panel:return panel
            provider=lambda start,stop:panel.teacher_batch(engine,start,stop)
            return panel.inputs,list(panel.sequence_ids),provider
        replay=state_replay(files,state_id,training,pool_sequences=args.replay_pool_sequences,seed=args.data_seed,epoch=epoch,split_seed=args.split_seed)
        sampling_seed=stable_seed('audit',args.data_seed,state_id) if role=='audit' else stable_seed('train',args.data_seed,epoch,state_id)
        inputs,ids=select_sequences(replay[role],count,seed=sampling_seed)
        return inputs,ids,None
    def collect_graph_compute():
        for key,value in graph_backend.compute.items():
            compute[key]=compute.get(key,0)+value-graph_compute_seen.get(key,0)
            graph_compute_seen[key]=value
    def collect_compute(engine):
        for key,value in engine.compute.as_dict().items():compute[key]=compute.get(key,0)+value
        for kind in ('forwards','tokens'):
            value=getattr(engine,'cached_teacher_head_'+kind,0)
            key='cached_teacher_head_'+kind
            compute[key]=compute.get(key,0)+value-cached_head_seen[kind];cached_head_seen[kind]=value
    def checkpoint(next_epoch,next_batch,completed=False,*,save_epoch=False,best=False,stop_reason=None,retain_calibration=False):
        per_rank_rng=all_rank_objects(dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device).cpu() if device.type=='cuda' else None),world_size)
        per_rank_compute=all_rank_objects(dict(compute),world_size)
        if rank==0:
            state=dict(controller=actor.state_dict(),optimizer=optimizer.state_dict(),
                metadata=metadata,updates=updates,next_epoch=next_epoch,next_batch=next_batch,
                per_rank_compute=per_rank_compute,per_rank_rng=per_rank_rng,completed=completed,clipped_updates=clipped_updates,
                fit_schedule=fit_schedule.state_dict() if fit_schedule else None,stop_reason=stop_reason,
                primal_dual=dual.state_dict() if dual is not None else None)
            if retain_calibration:
                save_calibration_checkpoint(args.output/f'controller_update{updates:04d}.pt',state)
            save_atomic(args.output/'controller_latest.pt',state)
            if save_epoch:save_atomic(args.output/f'controller_epoch_{next_epoch:04d}.pt',state)
            if best:save_atomic(args.output/'controller_best.pt',state)
        if world_size>1:dist.barrier()
    def preserve_failure(stage,context,error):
        failure=dict(stage=stage,rank=rank,updates=updates,exception=repr(error),context=context,metadata=metadata)
        try:
            save_atomic(args.output/f'failure_rank{rank}.pt',dict(**failure,controller=actor.state_dict(),optimizer=optimizer.state_dict()))
        finally:
            (args.output/f'failure_rank{rank}.json').write_text(json.dumps(failure,indent=2,default=str)+'\n')
    audit_epoch=0
    def audit(*,final_training_only=False):
        if graph_backend is not None and graph_backend.graph is not None:
            import gc
            graph_backend.close();gc.collect();torch.cuda.empty_cache()
        validation_objective=None
        roles=[('heldout_sequences',train_audit_states),('heldout_states',audit_states)]
        if args.audit_fresh_next:roles.append(('fresh_next_training',audit_states))
        if final_training_only:roles=[('fixed_training',train_states)]
        elif args.full_training_audit or (args.full_training_audit_final and audit_epoch==args.epochs):roles.insert(0,('fixed_training',train_states))
        elif args.training_audit_states:roles.insert(0,('fixed_training',choose_audit_states(train_states,args.training_audit_states)))
        if args.calibration_state_ids is not None:roles=[('fixed_training',train_states)]
        log('audit_begin',populations={role:len(ids) for role,ids in roles},completed_epochs=audit_epoch)
        for role,ids in roles:
            summaries=[]
            for state_id in ids[rank::world_size]:
                phase_times=None if args.no_component_timing else {};timing_holder[0]=phase_times
                record,engine,grouped=timed(phase_times,'state_load_seconds',lambda:load_state(state_id),device)
                target_role='train' if role=='fixed_training' else 'audit'
                if role=='fresh_next_training':
                    inputs,sequence_ids=globally_heldout_next_sequences(files,state_id,training,args.audit_sequences,
                        stable_seed('fresh_next_audit',args.data_seed,state_id),args.split_seed)
                    provider=lambda start,stop:engine.cache_teacher(inputs[start:stop],data_role='fresh_next_training_audit',sequence_ids=sequence_ids[start:stop])[0]
                else:inputs,sequence_ids,provider=panel_inputs(state_id,target_role,-1,
                    args.sequences_per_checkpoint if args.calibration_state_ids is not None else (0 if target_role=='train' else args.audit_sequences),phase_times)
                with torch.enable_grad() if cache_reader else torch.no_grad():
                    parameters,regularizer,stats,editable,credit=build_proposal(actor_forward,record.snapshot,engine,grouped,radius_fraction=args.radius_fraction,timings=phase_times,
                        norm_squared=state_constants.radius_squared(grouped) if state_constants else None,shrinkage_coefficient=args.shrinkage_coefficient)
                with torch.no_grad():
                    if token_config is None:
                        raw,ce=evaluate_sequences(engine,parameters,inputs,teacher_provider=provider,return_ce=True)
                        constraint=dual.terms(raw,state_id) if dual is not None else {}
                        penalty=constraint['kl_penalty'] if constraint else deadband(raw,args.kl_budget)[0]
                    else:
                        measured=tokenwise_panel(engine,parameters,regularizer,inputs,config=token_config,dual=dual,
                            state_id=state_id,teacher_provider=provider,backward=False,return_ce=True,balance=args.calibration_balance_only)
                        raw=measured.pop('mean_kl');penalty=measured.pop('kl_penalty');measured.pop('objective')
                        constraint=measured;ce={}
                    integrity=verify_readonly(parameters,editable,readonly_hashes(engine.parameters,editable))
                    if args.audit_forward_cast:stats['forward_cast_changes']=forward_cast_diagnostics(parameters,grouped,args.gpt_precision)
                stats.update(norm_preservation_diagnostics(stats,grouped))
                namespace=reader.describe_state(state_id) if composite else {}
                namespace.pop('state_id',None)
                row=dict(state_id=state_id,**namespace,role=role,sequence_ids=sequence_ids,mean_kl=raw,kl_penalty=penalty,
                         objective=float(regularizer.detach())+penalty,**{k:v for k,v in constraint.items() if k!='kl_penalty'},**stats,**ce,read_only_unchanged=integrity['unchanged'],component_seconds=phase_times,initial_zero_edit_floor=updates==0 and not args.direction_init and not args.primal_init)
                summaries.append(row);collect_compute(engine)
                del record,grouped,parameters,regularizer,inputs,credit,provider
            summaries=[row for local in all_rank_objects(summaries,world_size) for row in local]
            if len(summaries)!=len(ids) or sorted(r['state_id'] for r in summaries)!=sorted(ids):raise AssertionError('audit population coverage differs from declared IDs')
            for row in sorted(summaries,key=lambda row:row['state_id']):log('audit_state',**row)
            means={k:sum(r[k] for r in summaries)/len(summaries) for k in ('mean_kl','kl_penalty','objective','mean_group_l2_percentage_reduction','student_ce','teacher_ce','ce_change')}
            for key in ('logprob_rms_shift','centered_logit_rms_shift'):means[key]=(sum(r[key]**2 for r in summaries)/len(summaries))**.5
            means.update(norm_consistency_summary(summaries))
            if composite:
                means['per_seed']={str(seed):dict(checkpoints=len(part),mean_kl=sum(r['mean_kl'] for r in part)/len(part),**norm_consistency_summary(part),**(tokenwise_aggregate(part) if token_config else {})) for seed in reader.readers for part in [[r for r in summaries if r['bank_seed']==seed]] if part}
            means['group_l2_percentage_reduction']={g:sum(r['group_l2_percentage_reduction'][g] for r in summaries)/len(summaries) for g in summaries[0]['group_l2_percentage_reduction']}
            if dual is not None:
                means.update(feasibility_score=feasibility_score(summaries,args.kl_tolerance),
                    feasible_fraction=sum(r['feasible'] for r in summaries)/len(summaries),
                    maximum_kl=max(r['mean_kl'] for r in summaries),maximum_violation=max(r['constraint_violation'] for r in summaries),
                    dual_linear_term=sum(r['dual_linear_term'] for r in summaries)/len(summaries),
                    augmented_penalty=sum(r['augmented_penalty'] for r in summaries)/len(summaries))
            if token_config is not None:
                means.update(tokenwise_aggregate(summaries))
                from rewriter.tokenwise_reference import tokenwise_feasibility_score
                means['feasibility_score']=tokenwise_feasibility_score(summaries,args.kl_tolerance)
            if args.calibration_balance_only:
                import statistics
                ratios=[r['gamma_balance_ratio'] for r in summaries]
                log('ranking_balance_calibration',state_ids=ids,ratios=ratios,gamma_balanced=statistics.median(ratios),
                    rule='median of complete-panel tail/unit-rank logit cotangent L2 ratios; fixed before arms; not actor gradient equivalence',
                    additional_gpt_forwards=0,additional_gpt_backwards=0)
            log('audit_summary',role=role,state_ids=ids,completed_epochs=audit_epoch,**means)
            if role=='heldout_states':validation_objective=means['feasibility_score'] if dual is not None else means['objective']
        return validation_objective
    def schedule_validation(value,epoch):
        decision=fit_schedule.observe(value,epoch,optimizer.param_groups[0]['lr'],
            track_progress=not(args.track_trained_policy_progress and updates==0))
        set_meta_learning_rate(optimizer,decision['lr'])
        log('fit_schedule',completed_epochs=epoch,validation_objective=value,**decision)
        return decision
    log('start',metadata=metadata,controller_parameters=sum(p.numel() for p in actor.parameters()),compilation_counters=counters())
    if not args.resume:
        initial_validation=audit()
        if args.calibration_state_ids is not None and 0 in args.calibration_readout_updates:
            checkpoint(0,0,retain_calibration=True)
        if fit_schedule:
            decision=schedule_validation(initial_validation,0)
            checkpoint(0,0,best=decision['new_best'])
    if args.initial_audit_only:
        checkpoint(0,0,completed=True)
        if wb:wb.finish()
        if log_file:log_file.close()
        if world_size>1:dist.destroy_process_group()
        return
    stop_reason='maximum_epochs';completed_epochs=epoch0
    for epoch in range(epoch0,args.epochs):
        batches=(sharded_epoch_batches(train_states,epoch,args.data_seed,args.checkpoint_batch,world_size)
                 if args.fixed_state_shards else epoch_batches(train_states,epoch,args.data_seed,args.checkpoint_batch))
        for batch_index in range(batch0 if epoch==epoch0 else 0,len(batches)):
            states=batches[batch_index];optimizer.zero_grad(set_to_none=True);rows=[];state_gradients={};update_started=time.perf_counter()
            if graph_backend is not None:graph_backend.begin_update(states[rank::world_size],global_checkpoint_count=len(states))
            for state_id in states[rank::world_size]:
                if graph_backend is not None:
                    sequence_ids=[]
                    try:
                        record=reader.read(state_id,device=device,model_device='cpu')
                        panel=panel_inputs(state_id,'train',epoch,args.sequences_per_checkpoint,None,return_panel=True)
                        sequence_ids=list(panel.sequence_ids)
                        measured,stats=graph_backend.accumulate(record,panel,dual)
                        grouped=graph_backend.graph.grouped
                        if state_id not in state_diagnostics:state_diagnostics[state_id]=anchored_norm_ceiling(record.snapshot,grouped)
                        stats.update(state_diagnostics[state_id])
                        if seen_zero.get(state_id)!=grouped.zero_reference_groups:
                            log('state_policy',state_id=state_id,zero_reference_groups=list(grouped.zero_reference_groups),frozen_zero_group_parameters=sorted(grouped.frozen_parameter_names),chisel_tokens=actor.token_counts(record.snapshot));seen_zero[state_id]=grouped.zero_reference_groups
                    except Exception as error:
                        preserve_failure('graph_state_forward_backward',dict(epoch=epoch,state_id=state_id,sequence_ids=sequence_ids,checkpoint_batch_state_ids=states),error)
                        raise
                    stats.update(norm_preservation_diagnostics(stats,grouped))
                    namespace=reader.describe_state(state_id) if composite else {};namespace.pop('state_id',None)
                    rows.append(dict(role='train',epoch=epoch,state_id=state_id,**namespace,sequence_ids=sequence_ids,component_seconds=None,**measured,**stats))
                    collect_graph_compute();del record,panel,grouped
                    continue
                phase_times=None if args.no_component_timing else {};timing_holder[0]=phase_times
                record,engine,grouped=timed(phase_times,'state_load_seconds',lambda:load_state(state_id),device)
                inputs,sequence_ids,provider=panel_inputs(state_id,'train',epoch,args.sequences_per_checkpoint,phase_times)
                parameters,regularizer,stats,editable,credit=build_proposal(actor_forward,record.snapshot,engine,grouped,radius_fraction=args.radius_fraction,timings=phase_times,
                        norm_squared=state_constants.radius_squared(grouped) if state_constants else None,shrinkage_coefficient=args.shrinkage_coefficient)
                previous_gradient=flat_controller_gradient(controller_parameters) if args.gradient_diagnostics else None
                try:
                    if token_config is None:
                        measured=mean_kl_and_backward(engine,parameters,regularizer,inputs,kl_budget=args.kl_budget,state_weight=1/len(states),actor_credit=credit,timings=phase_times,
                                                     diagnostic_parameters=controller_parameters if args.gradient_diagnostics else None,teacher_provider=provider,primal_dual=dual,state_id=state_id)
                    else:
                        measured=tokenwise_panel(engine,parameters,regularizer,inputs,config=token_config,dual=dual,state_id=state_id,
                            state_weight=1/len(states),actor_credit=credit,teacher_provider=provider)
                except Exception as error:
                    preserve_failure('state_forward_backward',dict(epoch=epoch,state_id=state_id,sequence_ids=sequence_ids,checkpoint_batch_state_ids=states),error)
                    raise
                if args.gradient_diagnostics:state_gradients[state_id]=flat_controller_gradient(controller_parameters)-previous_gradient
                stats.update(norm_preservation_diagnostics(stats,grouped))
                namespace=reader.describe_state(state_id) if composite else {};namespace.pop('state_id',None)
                rows.append(dict(role='train',epoch=epoch,state_id=state_id,**namespace,sequence_ids=sequence_ids,component_seconds=phase_times,**measured,**stats));collect_compute(engine)
                del record,grouped,parameters,regularizer,inputs,credit,provider
            if graph_backend is not None:graph_backend.finish_update()
            aggregate_controller_gradients(actor,world_size)
            family_gradients=chisel_gradient_norms(actor)
            coherence={}
            if args.gradient_diagnostics:
                global_gradient=flat_controller_gradient(controller_parameters)
                for row in rows:
                    local=state_gradients[row['state_id']]
                    row.update(checkpoint_gradient_norm=float(local.norm()),checkpoint_global_gradient_cosine=float(torch.dot(local,global_gradient)/(local.norm()*global_gradient.norm()).clamp_min(1e-30)))
                sum_squared=sum((g.square().sum() for g in state_gradients.values()),global_gradient.new_zeros(()))
                if world_size>1:dist.all_reduce(sum_squared,op=dist.ReduceOp.SUM)
                coherence=dict(checkpoint_gradient_coherence=float(global_gradient.square().sum()/(len(states)*sum_squared).clamp_min(1e-30)))
            rows=[row for local in all_rank_objects(rows,world_size) for row in local]
            for row in sorted(rows,key=lambda row:row['state_id']):log('train_state',**row)
            synchronize();adam_started=time.perf_counter()
            try:
                grad=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),args.gradient_clip,error_if_nonfinite=True))
            except Exception as error:
                preserve_failure('controller_gradient',dict(epoch=epoch,checkpoint_batch_state_ids=states,local_measurements=rows),error)
                raise
            clipped_updates+=int(grad>args.gradient_clip)
            optimizer.step();updates+=1
            if dual is not None:
                dual.step([(row['state_id'],row['mean_kl']) for row in rows],expected_ids=states)
                log('dual_update',state_ids=states,multipliers={i:dual.multipliers[i] for i in states},
                    dual_updates=dual.updates,measurement_at='pre_primal_update',applied_after='Adam')
            synchronize();adam_seconds=time.perf_counter()-adam_started
            synchronize()
            means={k:sum(row[k] for row in rows)/len(rows) for k in ('mean_kl','kl_penalty','objective','mean_group_l2_percentage_reduction')}
            if dual is not None:
                means.update(dual_linear_term=sum(r['dual_linear_term'] for r in rows)/len(rows),
                    augmented_penalty=sum(r['augmented_penalty'] for r in rows)/len(rows),
                    feasible_fraction=sum(r['feasible'] for r in rows)/len(rows),maximum_kl=max(r['mean_kl'] for r in rows),
                    maximum_violation=max(r['constraint_violation'] for r in rows),
                    dual_multiplier_mean=sum(dual.multipliers.values())/len(dual.multipliers),dual_multiplier_max=max(dual.multipliers.values()),
                    radius_saturation_fraction=sum(r['radius_saturated'] for r in rows)/len(rows))
            if token_config is not None:means.update(tokenwise_aggregate(rows))
            log('update',role='train',epoch=epoch,state_ids=states,gradient_norm=grad,gradient_clipped=grad>args.gradient_clip,gradient_clip_fraction=clipped_updates/updates,adam_and_clip_seconds=adam_seconds,wall_seconds=time.perf_counter()-update_started,
                learning_rate=optimizer.param_groups[0]['lr'],
                parameter_family_learning_rates=meta_learning_rates(optimizer),
                rank0_peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                rank0_peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0,
                **means,**coherence,**family_gradients,measurement_at='before_optimizer_update',compute=global_compute(),compilation_counters=counters())
            if args.calibration_state_ids is not None:
                if updates in args.calibration_readout_updates:audit()
            elif not args.audit_every_epochs and updates%args.audit_every==0:audit()
            next_epoch,next_batch=(epoch+1,0) if batch_index+1==len(batches) else (epoch,batch_index+1)
            calibration_milestone=args.calibration_state_ids is not None and updates in args.calibration_readout_updates
            if (updates%args.checkpoint_every==0 or next_epoch==args.epochs or calibration_milestone) and (not fit_schedule or next_batch!=0):
                checkpoint(next_epoch,next_batch,retain_calibration=args.calibration_state_ids is not None)
        log('epoch_finished',epoch=epoch,training_state_count=len(train_states),training_state_ids=train_states,checkpoint_batches=len(batches))
        audit_epoch=epoch+1
        completed_epochs=audit_epoch;decision=None
        if args.calibration_state_ids is None and args.audit_every_epochs and audit_epoch%args.audit_every_epochs==0:
            validation=audit()
            if fit_schedule:decision=schedule_validation(validation,audit_epoch)
        if fit_schedule:
            checkpoint(audit_epoch,0,save_epoch=audit_epoch%5==0,best=bool(decision and decision['new_best']))
        batch0=0
        if not args.no_plots and rank==0:plot_progress(args.output)
        if decision and decision['stop']:
            stop_reason='validation_plateau'
            if args.full_training_audit_final and audit_epoch!=args.epochs:audit(final_training_only=True)
            break
    if ((args.calibration_state_ids is not None and updates not in args.calibration_readout_updates) or
        (args.calibration_state_ids is None and ((args.audit_every_epochs and completed_epochs%args.audit_every_epochs) or (not args.audit_every_epochs and updates%args.audit_every)))):
        validation=audit()
        if fit_schedule:
            decision=schedule_validation(validation,completed_epochs)
            checkpoint(completed_epochs,0,best=decision['new_best'])
    if not args.no_plots and rank==0:plot_progress(args.output)
    checkpoint(completed_epochs,0,completed=(not fit_schedule and args.calibration_state_ids is None) or stop_reason=='validation_plateau',stop_reason=stop_reason)
    log('finished',epochs=completed_epochs,maximum_epochs=args.epochs,stop_reason=stop_reason,
        compute=global_compute(),compilation_counters=counters(),controller_live_handoff=False)
    if graph_backend is not None:graph_backend.close()
    if log_file:log_file.close()
    if wb:wb.finish()
    if world_size>1:dist.destroy_process_group()


if __name__=='__main__':main()
