"""One-checkpoint supervised capacity diagnostic; never the main Chisel objective.

Fit nominal physical post-group-RMS/post-radius edits BEFORE FP32 storage to a
fixed analytic target. All other editable outputs target zero. No GPT execution
occurs here. Stored-weight readouts do not establish finite logit preservation.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.adaptive_action import AdaptiveAction,AdaptiveFactors
from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from rewriter.adaptive_edit_target import edit_target_terms
from rewriter.adaptive_gauge_control import orthonormal_product
from rewriter.adaptive_preservation import group_rms_scales
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.edit_scope import editable_names,editable_weight_norm_sq,scoped_snapshot
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.prefix_cache import unpack_record
from scripts.fixed_frame_target import FixedFrameTarget


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def physical_action(raw,grouped,snapshot):
    """Identical RMS and global guard to adaptive_preservation; no dense cast."""
    editable=editable_names(grouped.parameters,'grouped')-grouped.frozen_parameter_names
    scales=group_rms_scales(grouped,dtype=torch.float64)
    # Readonly outputs do not enter radius or loss. Their input tokens remain.
    action=AdaptiveAction({n:f.scaled(scales[n]) for n,f in raw.matrices.items() if n in editable},
                         {n:v*scales[n] for n,v in raw.vectors.items() if n in editable})
    raw_norm=action.norm_sq().clamp_min(1e-24).sqrt()
    guard=torch.minimum(raw_norm.new_ones(()),snapshot.radius/(raw_norm+1e-12))
    return action.scaled(guard),raw_norm,guard


@torch.no_grad()
def make_target(parameters,snapshot,selection,editable):
    """Known binary edits in computational orientation; frozen FP64 thin factors."""
    matrices={}
    for name in snapshot.matrices:
        if name not in editable:continue
        p=parameters[name];n,m=(p.shape[1],p.shape[0]) if name=='embed.weight' else p.shape
        matrices[name]=AdaptiveFactors(torch.zeros(n,1,dtype=torch.float64),
            torch.zeros(1,1,dtype=torch.float64),torch.zeros(m,1,dtype=torch.float64))
    vectors={n:torch.zeros_like(parameters[n],dtype=torch.float64) for n in snapshot.vectors if n in editable}
    used=set()
    for row in selection:
        if row['rank_limit']!=8:raise ValueError('rank8 target required')
        blocks={}
        for c in row['selected']:blocks.setdefault(c['first_weight'],[]).append(c)
        for first,choices in blocks.items():
            second=choices[0]['second_weight'];bias=choices[0]['first_bias']
            if len(choices)>8 or used&{first,second,bias}:raise ValueError('overlap/rank target mismatch')
            if not {first,second,bias}.issubset(editable):raise ValueError('target writes a frozen parameter')
            used.update((first,second,bias))
            ids=torch.tensor([c['channel'] for c in choices])
            if len(set(ids.tolist()))!=len(ids):raise ValueError('repeated target channel')
            d=2.**torch.tensor([c['exponent'] for c in choices],dtype=torch.float64)
            w=parameters[first].double();z=parameters[second].double()
            e=torch.zeros(len(w),len(ids),dtype=torch.float64);e[ids,torch.arange(len(ids))]=1
            matrices[first]=orthonormal_product(e*(d-1),w[ids].T,rank=8)
            matrices[second]=orthonormal_product(z[:,ids]*(d.pow(-row['inverse_power'])-1),e,rank=8)
            vectors[bias][ids]=parameters[bias].double()[ids]*(d-1)
    if not used:raise ValueError('nonzero teaching edit required')
    return AdaptiveAction(matrices,vectors),used


def target_loss(predicted,target,target_squared):
    terms=edit_target_terms(predicted,target)
    return sum(terms.values())/target_squared,terms


def relative_target_error(terms, grouped):
    """Group reference squared norms are frozen; zero-reference groups excluded."""
    zero=next(iter(terms.values())).new_zeros(())
    return sum((sum((terms[n] for n in names if n in terms),zero)/grouped.reference_squared[g]
                for g,names in grouped.groups.items() if float(grouped.reference_squared[g])>0),zero)


def normalized_target_objective(terms,target_squared,relative_squared,grouped,objective='global'):
    """Diagnostic fidelity only; does not alter the grouped norm reward."""
    global_error=sum(terms.values())/target_squared
    if objective=='global':return global_error
    relative_error=relative_target_error(terms,grouped)/relative_squared
    if objective=='group-relative':return relative_error
    if objective=='balanced':return .5*(global_error+relative_error)
    raise ValueError('unknown physical target objective: '+objective)


def error_metrics(terms,target_squared,relative_squared,grouped):
    zero=target_squared.new_zeros(())
    global_error=sum(terms.values())/target_squared
    relative_error=relative_target_error(terms,grouped)/relative_squared
    return dict(normalized_edit_error=float(global_error.detach()),
        relative_normalized_edit_error=float(relative_error.detach()),
        balanced_normalized_edit_error=float((.5*(global_error+relative_error)).detach()),
        matrix_error_squared=float(sum((v for n,v in terms.items() if n.endswith('.weight')),zero).detach()),
        bias_error_squared=float(sum((v for n,v in terms.items() if n.endswith('.bias')),zero).detach()),
        group_relative_error_squared={g:float((sum((terms[n] for n in ns if n in terms),zero)/grouped.reference_squared[g]).detach())
            if float(grouped.reference_squared[g])>0 else 0. for g,ns in grouped.groups.items()})


@torch.no_grad()
def stored_readout(predicted,target,parameters,grouped=None,all_parameters=None):
    rows={};squared=0.;changed=0;deltas={"predicted":{},"teacher":{}}
    for name,f in predicted.matrices.items():
        a=f.dense();b=target.matrices[name].dense()
        if name=='embed.weight':a=a.T;b=b.T
        base=parameters[name];actual=(base.double()+a).float();wanted=(base.double()+b).float()
        error=actual.double()-wanted.double();ss=float(error.square().sum());count=int((actual!=wanted).sum())
        rows[name]=dict(error_squared=ss,different_elements=count,max_abs=float(error.abs().max()))
        for role,value in [('predicted',actual),('teacher',wanted)]:
            deltas[role][name]=float(((value.double()-base.double())*(value.double()+base.double())).sum())
        squared+=ss;changed+=count
    for name,value in predicted.vectors.items():
        base=parameters[name];actual=(base.double()+value).float();wanted=(base.double()+target.vectors[name]).float()
        error=actual.double()-wanted.double();ss=float(error.square().sum());count=int((actual!=wanted).sum())
        rows[name]=dict(error_squared=ss,different_elements=count,max_abs=float(error.abs().max()))
        for role,value in [('predicted',actual),('teacher',wanted)]:
            deltas[role][name]=float(((value.double()-base.double())*(value.double()+base.double())).sum())
        squared+=ss;changed+=count
    norms={}
    if grouped is not None:
        all_parameters=parameters if all_parameters is None else all_parameters
        def reduction(base,delta):
            return 0. if base==0 and delta==0 else -100*delta/(math.sqrt(base)*(math.sqrt(max(0.,base+delta))+math.sqrt(base)))
        total=sum(float(p.double().square().sum()) for p in all_parameters.values())
        matrix_total=sum(float(p.double().square().sum()) for p in all_parameters.values() if p.ndim==2)
        for role,changes in deltas.items():
            groups={g:reduction(float(grouped.reference_squared[g]),sum(changes.get(n,0.) for n in ns)) for g,ns in grouped.groups.items()}
            matrix_groups=[g for g in groups if g.endswith('.weight')]
            norms[role]=dict(group_l2_percentage_reduction=groups,
                mean_group_l2_percentage_reduction=sum(groups.values())/len(groups),
                matrix_group_mean_l2_percentage_reduction=sum(groups[g] for g in matrix_groups)/len(matrix_groups),
                matrix_contribution_to_all_group_mean=sum(groups[g] for g in matrix_groups)/len(groups),
                whole_parameter_frobenius_percentage_reduction=reduction(total,sum(changes.values())),
                matrix_frobenius_percentage_reduction=reduction(matrix_total,sum(v for n,v in changes.items() if parameters[n].ndim==2)))
    return dict(stored_norm_metrics=norms,norm_sign='positive means reduction; individual groups may grow',stored_weight_error_squared=squared,stored_different_elements=changed,by_parameter=rows,
                gpt_forwards=0,gpt_backwards=0,preservation_qualified=False)


@torch.no_grad()
def write_span_capture(private,target):
    """Best unconstrained cores in CURRENT spans; not a learned action/readout.

    Q columns are orthonormal for n>=L; when n<L, Q rows are orthonormal
    and QQ.T is identity. Thus the same contraction measures projection
    energy in either case. This ignores shared-head constraints and the cap.
    """
    captured={}
    for i,name in enumerate(private.matrix_names):
        if name not in target.matrices:continue
        f=target.matrices[name]
        u=private.matrix_rows[i].orthonormal_matrix().double()
        v=private.matrix_columns[i].orthonormal_matrix().double()
        core=(u.T@f.U)@f.C@(f.V.T@v)
        captured[name]=float(core.square().sum())
    for i,name in enumerate(private.vector_names):
        if name not in target.vectors:continue
        q=private.vector_frames[i].orthonormal_matrix().double()
        captured[name]=float((q.T@target.vectors[name]).square().sum())
    total=float(target.norm_sq());fraction=sum(captured.values())/total
    if not -1e-12<=fraction<=1+1e-10:raise ValueError('invalid orthonormal target projection')
    return dict(captured_target_energy_fraction=fraction,
        decoder_independent_normalized_error_floor=1-fraction,
        captured_squared_by_parameter=captured,
        scope='fixed current spans, independent unconstrained cores; no action applied')


def save(path,payload):
    tmp=path.with_suffix(path.suffix+'.incomplete')
    with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def resume_adam(path,metadata,actor,private,optimizer,private_optimizer):
    """Restore a qualified same-objective Adam trajectory, including its RNG."""
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('diagnostic_control'):raise ValueError('analytic control checkpoint is not an optimizer resume')
    old=state['metadata']
    for key in ('state_sha256','selection_sha256','finite_evidence_sha256','architecture',
                'private_identity','target_norm_squared','relative_target_squared','group_reference_squared','radius'):
        if old[key]!=metadata[key]:raise ValueError('resume identity mismatch: '+key)
    for key in ('optimizer','objective','head_lr','representation_lr','private_lr','seed','private_seed'):
        if old['args'][key]!=metadata['args'][key]:raise ValueError('resume configuration mismatch: '+key)
    for name,digest in metadata['source_sha256'].items():
        if name!='scripts/fit_cpu_edit_teacher.py' and old['source_sha256'].get(name)!=digest:
            raise ValueError('resume dependency mismatch: '+name)
    step=state['step']
    if metadata['args']['optimizer']!='adam' or state.get('lbfgs_optimizer') is not None:
        raise ValueError('resume supports Adam only')
    evaluate_only=metadata['args'].get('evaluate_only',False)
    valid_step=(step==metadata['args']['steps']) if evaluate_only else (0<=step<metadata['args']['steps']<=step+1000)
    if not isinstance(step,int) or not valid_step:
        raise ValueError('resume requires endpoint beyond saved update count, or explicit equal-step evaluation')
    actor.load_state_dict(state['actor']);private.load_state_dict(state['private'])
    optimizer.load_state_dict(state['optimizer']);private_optimizer.load_state_dict(state['private_optimizer'])
    torch.set_rng_state(state['rng'])
    return step


def fork_private_rate(optimizer,rate):
    """Explicit post-restore rate fork; retain every Adam moment and counter."""
    if not math.isfinite(rate) or rate<=0:raise ValueError('positive finite fork rate required')
    previous=[group['lr'] for group in optimizer.param_groups]
    for group in optimizer.param_groups:group['lr']=rate
    return previous


def reset_objective_fork(optimizer,private_optimizer,head_rate):
    """Reset both Adam histories for an explicitly different objective."""
    if not math.isfinite(head_rate) or head_rate<=0:raise ValueError('positive finite fork head rate required')
    heads=[g for g in optimizer.param_groups if g.get('role')=='heads']
    if len(heads)!=1:raise ValueError('exactly one head parameter group required')
    previous=heads[0]['lr']
    optimizer.state.clear();private_optimizer.state.clear()
    heads[0]['lr']=head_rate
    return previous


def seed_analytic_control(path,metadata,actor,private):
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if not state.get('diagnostic_control') or state['metadata'].get('diagnostic_control') not in ('conditional_shared_head_global_lstsq','conditional_shared_head_balanced_lstsq'):
        raise ValueError('explicit fitted conditional-head analytic-control checkpoint required')
    for key in ('state_sha256','selection_sha256','finite_evidence_sha256','architecture','private_identity',
                'target_norm_squared','relative_target_squared','group_reference_squared','radius'):
        if state['metadata'][key]!=metadata[key]:raise ValueError('analytic seed identity mismatch: '+key)
    actor.load_state_dict(state['actor']);private.load_state_dict(state['private']);private.requires_grad_(False)
    torch.set_rng_state(state['rng'])
    return dict(path=str(path),sha256=sha(path),parent_step=state['step'],
        parent_control=state['metadata'].get('diagnostic_control'),optimizer_states='fresh; none inherited',private_frozen=True)


def exact_state_equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact_state_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact_state_equal(x,y) for x,y in zip(a,b))
    return a==b


class ClosureBudgetExceeded(RuntimeError):
    pass


def bounded_lbfgs(parameters, closure, *, max_iter=20, closure_budget=40, wolfe_bracket_tolerance=1e-9):
    """One deterministic mixed-precision L-BFGS diagnostic, with hard budget.

    PyTorch concatenates FP32/FP64 gradients in FP64 and casts trial parameter
    increments back to each parameter dtype. No gradient clipping: Wolfe uses
    the actual objective derivative. Its line search can exceed max_eval; the
    explicit cap restores the INITIAL state and empty optimizer if exceeded,
    never publishing an unaccepted trial as an optimized endpoint.
    """
    parameters=list(parameters)
    initial=[p.detach().clone() for p in parameters]
    optimizer=torch.optim.LBFGS(parameters,lr=1.,max_iter=max_iter,max_eval=min(20,closure_budget),
        tolerance_grad=1e-12,tolerance_change=1e-15,history_size=10,line_search_fn='strong_wolfe')
    calls=0;gradient_dtypes=set()
    def wrapped():
        nonlocal calls
        if calls>=closure_budget:raise ClosureBudgetExceeded()
        optimizer.zero_grad(set_to_none=True)
        loss=closure(calls)
        calls+=1
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite L-BFGS closure')
        loss.backward()
        for p in parameters:
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():raise FloatingPointError('nonfinite L-BFGS gradient')
                p.grad=p.grad.contiguous()
                gradient_dtypes.add(str(p.grad.dtype))
        return loss
    aborted=False
    # Isolated diagnostic process only: torch exposes no public bracket-tolerance
    # argument. Scope and restore this internal override; default remains exact.
    import importlib
    from functools import partial
    from unittest.mock import patch
    module=importlib.import_module('torch.optim.lbfgs')
    original=module._strong_wolfe
    try:
        with patch.object(module,'_strong_wolfe',partial(original,tolerance_change=wolfe_bracket_tolerance)):
            optimizer.step(wrapped)
    except ClosureBudgetExceeded:
        aborted=True
        with torch.no_grad():
            for p,value in zip(parameters,initial):p.copy_(value)
        optimizer.state.clear()
    except BaseException:
        with torch.no_grad():
            for p,value in zip(parameters,initial):p.copy_(value)
        raise
    return optimizer,dict(closure_evaluations=calls,hard_closure_budget=closure_budget,
        budget_aborted_restored_initial=aborted,gradient_dtypes=sorted(gradient_dtypes),
        flat_gradient_dtype=str(optimizer._gather_flat_grad().dtype),
        clipping=False,history_size=10,line_search='strong_wolfe',wolfe_bracket_tolerance=wolfe_bracket_tolerance)


def validate_seed_flags(args):
    if args.seed_analytic_control is not None:
        if args.resume is not None or args.evaluate_only or args.fork_objective is not None or args.fork_private_lr is not None or not args.freeze_private:
            raise ValueError('analytic seed requires isolated frozen-private path')
        if args.optimizer=='adam' and not args.global_grad_clip:raise ValueError('seeded Adam requires combined global clipping')
        if args.optimizer=='lbfgs' and args.global_grad_clip:raise ValueError('L-BFGS requires clipping disabled')
    elif args.freeze_private or args.global_grad_clip:raise ValueError('frozen-private/global clipping is scoped to explicit analytic seed')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('state','selection','finite-evidence','output'):parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--resume',type=Path,help='Adam checkpoint; --steps is total completed updates')
    parser.add_argument('--evaluate-only',action='store_true',help='Re-evaluate saved --steps endpoint without optimizer updates; requires --resume')
    parser.add_argument('--fork-private-lr',type=float,help='Explicit post-resume private rate override, retaining moments')
    parser.add_argument('--fork-objective',choices=['global','group-relative','balanced'])
    parser.add_argument('--fork-head-lr',type=float)
    parser.add_argument('--reset-optimizers',action='store_true')
    parser.add_argument('--seed-analytic-control',type=Path)
    parser.add_argument('--freeze-private',action='store_true')
    parser.add_argument('--global-grad-clip',action='store_true')
    parser.add_argument('--threads',type=int,choices=[1,2],default=2)
    parser.add_argument('--objective',choices=['global','group-relative','balanced'],default='global')
    parser.add_argument('--optimizer',choices=['adam','lbfgs'],default='adam')
    parser.add_argument('--wolfe-bracket-tolerance',type=float,default=1e-9)
    parser.add_argument('--closure-budget',type=int,default=40)
    parser.add_argument('--phase',choices=['smoke','full'],default='smoke')
    parser.add_argument('--head-lr',type=float,default=1e-4)
    parser.add_argument('--representation-lr',type=float,default=1e-4)
    parser.add_argument('--private-lr',type=float,default=1e-3)
    parser.add_argument('--seed',type=int,default=910001)
    parser.add_argument('--private-seed',type=int,default=920001)
    args=parser.parse_args()
    if not 1<=args.steps or (args.resume is None and args.steps>1000) or any(not math.isfinite(v) or v<=0 for v in (args.head_lr,args.representation_lr,args.private_lr)):
        raise ValueError('bounded positive diagnostic settings required')
    if not math.isfinite(args.wolfe_bracket_tolerance) or args.wolfe_bracket_tolerance<=0:raise ValueError('positive finite bracket tolerance required')
    if args.optimizer=='lbfgs' and (args.steps>20 or not 1<=args.closure_budget<=40):raise ValueError('L-BFGS diagnostic capped at20 iterations/40 closures')
    if args.fork_objective is not None:
        if args.resume is None or not args.reset_optimizers or args.fork_head_lr is None or args.fork_private_lr is not None or args.fork_objective==args.objective:
            raise ValueError('objective fork requires resume, changed objective, explicit head rate and both optimizer resets')
        if not math.isfinite(args.fork_head_lr) or args.fork_head_lr<=0:raise ValueError('invalid fork head rate')
    elif args.reset_optimizers or args.fork_head_lr is not None:
        raise ValueError('optimizer reset/head rate override requires explicit objective fork')
    if args.evaluate_only and (args.resume is None or args.fork_private_lr is not None or args.fork_objective is not None):
        raise ValueError('evaluation-only requires resume without a rate fork')
    if args.fork_private_lr is not None and (args.resume is None or not math.isfinite(args.fork_private_lr) or args.fork_private_lr<=0):
        raise ValueError('positive fork rate requires resume')
    validate_seed_flags(args)
    args.output.mkdir(parents=True,exist_ok=False)
    # Reference norms/target construction use the original two-thread recipe.
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    started=time.perf_counter()
    evidence=json.loads(args.finite_evidence.read_text());selection=json.loads(args.selection.read_text())
    if sha(args.state)!=evidence['state_sha256'] or sha(args.selection)!=evidence['selection_sha256']:
        raise ValueError('teaching evidence identity mismatch')
    if not evidence['fp64_cast_stored_weights_equal_direct'] or evidence['rows']['direct_binary']['different_logits']!=0 or evidence['rows']['direct_binary']['strict_top100_changes']!=0:
        raise ValueError('teaching target lacks declared finite CPU qualification')
    data=torch.load(args.state,map_location='cpu',weights_only=False,mmap=True)
    snapshot=unpack_record(data['observation'],device='cpu').snapshot
    names=set(snapshot.matrices)|set(snapshot.vectors)
    parameters={n:v.detach().float() for n,v in data['model'].items() if n in names}
    if any(r['state_id']!=data['completed_steps'] or r['state_sha256']!=evidence['state_sha256'] for r in selection['rows']):
        raise ValueError('target belongs to different state')
    grouped=GroupedNormObjective(parameters,zero_reference_policy='freeze')
    editable=editable_names(parameters,'grouped')-grouped.frozen_parameter_names
    norm_squared=editable_weight_norm_sq(parameters,editable)
    snapshot=scoped_snapshot(snapshot,norm_squared,.01,scope='grouped')
    target,selected_names=make_target(parameters,snapshot,selection['rows'],editable)
    target_squared=target.norm_sq().detach()
    relative_squared=relative_target_error(edit_target_terms(target.scaled(0.),target),grouped).detach()
    if not 0<float(target_squared)<=float(snapshot.radius)**2:
        raise ValueError('target is zero or outside unchanged .01 global radius')
    torch.manual_seed(args.seed)
    actor=AdaptiveCoordinateChisel(rank=8,trajectory_dim=400,model_dim=256,num_layers=2,latent_tokens=8,
        fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
    private=actor.make_private(snapshot,state_key=('cpu-teaching',0,data['completed_steps'],evidence['state_sha256']),seed=args.private_seed)
    heads=list(actor.output_heads.parameters()) if isinstance(actor.output_heads,torch.nn.Module) else [p for h in actor.output_heads for p in h.parameters()]
    ids={id(p) for p in heads};representation=[p for p in actor.parameters() if id(p) not in ids]
    optimizer=torch.optim.Adam([dict(params=heads,lr=args.head_lr,role='heads'),
        dict(params=representation,lr=args.representation_lr,role='representation')],betas=(.9,.99),eps=1e-8)
    private_optimizer=torch.optim.Adam(private.parameters(),lr=args.private_lr,betas=(.9,.99),eps=1e-12)
    root=Path(__file__).resolve().parents[1]
    dependencies=['scripts/fit_cpu_edit_teacher.py','rewriter/adaptive_coordinate_chisel.py','rewriter/coordinate_chisel.py',
        'rewriter/adaptive_edit_target.py','rewriter/adaptive_action.py','rewriter/adaptive_preservation.py',
        'rewriter/adaptive_sensing.py','rewriter/adaptive_gauge_control.py','rewriter/grouped_norm.py',
        'rewriter/edit_scope.py','rewriter/prefix_cache.py','baseline/model.py']
    if args.freeze_private:dependencies.append('scripts/fixed_frame_target.py')
    metadata=dict(schema='cpu_edit_teacher_fit_v1',scope=__doc__,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_sha256={n:sha(root/n) for n in dependencies},state_sha256=evidence['state_sha256'],
        selection_sha256=sha(args.selection),finite_evidence_sha256=sha(args.finite_evidence),
        architecture=actor.configuration(),private_identity=private.get_extra_state(),
        target_norm_squared=float(target_squared),radius=float(snapshot.radius),radius_fraction=.01,
        loss=args.objective+' normalized nominal physical target MSE; identity=1; before FP32 weight storage',
        balanced_definition='0.5*(global normalized physical target MSE + group-relative normalized physical target MSE)',
        relative_target_squared=float(relative_squared),group_reference_squared={g:float(v) for g,v in grouped.reference_squared.items()},
        trainable=('shared actor and private probes; parameters, optimizer moments and RNG restored from resumed_from' if args.resume is not None else 'fresh shared actor and fresh private probes, no reused optimizer'),betas=[.9,.99],
        active_optimizer=args.optimizer,optimizer_note='Adam per-role clipping; optional L-BFGS has no clipping and shared FP64 flat gradient, retains original parameter dtypes',
        clip=1.,shared_epsilon=1e-8,private_epsilon=1e-12,reference_construction_threads=2,threads=args.threads,torch_version=torch.__version__,
        gpt_forwards=0,gpt_backwards=0,query_access=False)
    start_step=0
    if args.resume is not None:
        start_step=resume_adam(args.resume,metadata,actor,private,optimizer,private_optimizer)
        metadata['resumed_from']=dict(path=str(args.resume),sha256=sha(args.resume),step=start_step)
        if args.fork_private_lr is not None:
            previous=fork_private_rate(private_optimizer,args.fork_private_lr)
            metadata['rate_fork']=dict(parent=metadata['resumed_from'],previous_private_rates=previous,
                new_private_rate=args.fork_private_lr,optimizer_moments_preserved=True,
                parent_threads=torch.load(args.resume,map_location='cpu',weights_only=False,mmap=True)['metadata']['threads'])
            args.private_lr=args.fork_private_lr
            metadata['args']['private_lr']=args.private_lr
        if args.fork_objective is not None:
            previous=reset_objective_fork(optimizer,private_optimizer,args.fork_head_lr)
            metadata['objective_fork']=dict(parent=metadata['resumed_from'],previous_objective=args.objective,
                new_objective=args.fork_objective,previous_head_rate=previous,new_head_rate=args.fork_head_lr,
                both_adam_states_reset=True,all_actor_private_parameters_trainable=True)
            args.objective=args.fork_objective;args.head_lr=args.fork_head_lr
            metadata['args'].update(objective=args.objective,head_lr=args.head_lr)
            metadata['loss']=args.objective+' normalized nominal physical target MSE; identity=1; before FP32 weight storage'
            metadata['trainable']='shared actor/private restored from parent; both Adam states reset explicitly for objective fork'
    frozen_private_state=None
    if args.seed_analytic_control is not None:
        metadata['analytic_seed']=seed_analytic_control(args.seed_analytic_control,metadata,actor,private)
        metadata['diagnostic_control']='analytic_seeded_frozen_private_shared_fit'
        metadata['trainable']='shared body and heads only; analytic seed, fresh optimizer state, all private parameters frozen'
        metadata['optimizer_note']=('single combined shared gradient clipping norm1' if args.optimizer=='adam' else 'fresh L-BFGS, no clipping; Adam states empty/unused')+'; private optimizer remains empty/unused'
        frozen_private_state=copy.deepcopy(private.state_dict())
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    torch.set_num_threads(args.threads)
    hidden_stats={};head_inputs={}
    name_to_group={n:g for g,ns in grouped.groups.items() for n in ns}
    def head_input(kind,names,width):
        def hook(module,args):
            head_inputs[kind]=args[0].detach()
            values=head_inputs[kind].reshape(len(names),width,-1)
            accum={}
            for name,x in zip(names,values):
                g=name_to_group.get(name,'readonly')
                row=accum.setdefault(g,[0.,0,0.])
                row[0]+=float(x.double().square().sum());row[1]+=x.numel();row[2]=max(row[2],float(x.abs().max()))
            hidden_stats[kind]={g:dict(rms=math.sqrt(v[0]/v[1]),max_abs=v[2],elements=v[1]) for g,v in accum.items()}
        return hook
    actor.matrix_output.register_forward_pre_hook(head_input('matrix',tuple(snapshot.matrices),16))
    actor.vector_output.register_forward_pre_hook(head_input('vector',tuple(snapshot.vectors),8))
    cached_cores=None;cached_target=None
    if args.freeze_private:
        with torch.no_grad():
            cached_cores={n:v.detach() for n,v in query_weight_cores(parameters,private).items()}
            example,_,_=physical_action(actor(snapshot,private,weight_cores=cached_cores),grouped,snapshot)
            cached_target=FixedFrameTarget(example,target)
    def evaluate(use_cache=True,use_target_cache=True):
        cores=cached_cores if cached_cores is not None and use_cache else query_weight_cores(parameters,private)
        raw=actor(snapshot,private,weight_cores=cores)
        predicted,norm,guard=physical_action(raw,grouped,snapshot)
        terms=cached_target(predicted) if cached_target is not None and use_target_cache else edit_target_terms(predicted,target)
        loss=normalized_target_objective(terms,target_squared,relative_squared,grouped,args.objective)
        return predicted,loss,terms,norm,guard
    if args.freeze_private:
        benchmark=[];reference=None
        for use_cache in (True,False,False,True):
            optimizer.zero_grad(set_to_none=True);t=time.perf_counter()
            _,probe_loss,_,_,_=evaluate(use_cache=use_cache,use_target_cache=False);probe_loss.backward()
            elapsed=time.perf_counter()-t
            gradients={n:p.grad.detach().clone() for n,p in actor.named_parameters() if p.grad is not None}
            if reference is None:reference=gradients
            else:
                if gradients.keys()!=reference.keys():raise ValueError('cache gradient inventory mismatch')
                for n in gradients:torch.testing.assert_close(gradients[n],reference[n],rtol=0,atol=0)
            benchmark.append(dict(cached=use_cache,forward_backward_seconds=elapsed,objective_error=float(probe_loss.detach())))
        optimizer.zero_grad(set_to_none=True)
        if not exact_state_equal(private.state_dict(),frozen_private_state):raise ValueError('private state changed during cache check')
        (args.output/'cache_gradient_benchmark.json').write_text(json.dumps(dict(rows=benchmark,gradients_bitwise_equal=True,optimizer_steps=0,
            timing_scope='forward/backward including sensing with uncached target in both modes; excludes optimizer step; identical initial actor',cache_scope='fixed private probes and fixed GPT weights'),indent=2)+'\n')
        del reference,gradients
        target_benchmark=[];reference=None;reference_loss=None
        for use_target_cache in (False,True,True,False):
            optimizer.zero_grad(set_to_none=True);t=time.perf_counter()
            _,probe_loss,_,_,_=evaluate(use_cache=True,use_target_cache=use_target_cache);probe_loss.backward()
            elapsed=time.perf_counter()-t
            gradients={n:p.grad.detach().clone() for n,p in actor.named_parameters() if p.grad is not None}
            if reference is None:reference=gradients;reference_loss=probe_loss.detach().clone()
            torch.testing.assert_close(probe_loss.detach(),reference_loss,rtol=1e-10,atol=1e-10)
            if gradients.keys()!=reference.keys():raise ValueError('target-cache gradient inventory mismatch')
            errors={}
            for n,g in gradients.items():
                atol,rtol=(1e-7,1e-5) if g.dtype==torch.float32 else (1e-9,1e-9)
                torch.testing.assert_close(g,reference[n],atol=atol,rtol=rtol)
                diff=g.double()-reference[n].double();key=str(g.dtype)
                stats=errors.setdefault(key,dict(max_abs=0.,difference_squared=0.,reference_squared=0.,bitwise_equal=True))
                stats['max_abs']=max(stats['max_abs'],float(diff.abs().max()))
                stats['difference_squared']+=float(diff.square().sum());stats['reference_squared']+=float(reference[n].double().square().sum())
                stats['bitwise_equal']=stats['bitwise_equal'] and torch.equal(g,reference[n])
            for stats in errors.values():stats['relative_l2']=math.sqrt(stats['difference_squared']/max(stats['reference_squared'],1e-300))
            target_benchmark.append(dict(target_cached=use_target_cache,weight_cores_cached=True,forward_backward_seconds=elapsed,
                objective_error=float(probe_loss.detach()),loss_abs_error=float((probe_loss.detach()-reference_loss).abs()),gradient_errors=errors))
        optimizer.zero_grad(set_to_none=True)
        if not exact_state_equal(private.state_dict(),frozen_private_state):raise ValueError('private state changed during target cache check')
        (args.output/'target_cache_gradient_benchmark.json').write_text(json.dumps(dict(rows=target_benchmark,optimizer_steps=0,
            timing_scope='forward/backward with fixed weight-core cache in both modes; excludes optimizer step',
            tolerances={'FP32_body':dict(rtol=1e-5,atol=1e-7),'FP64_heads':dict(rtol=1e-9,atol=1e-9)},
            cache_scope='fixed U/V and frozen targets; joint-QR target constants'),indent=2)+'\n')
        del reference,gradients
    lbfgs_optimizer=None
    def checkpoint(step,filename=None):
        save(args.output/(filename or f'checkpoint_{step:04d}.pt'),dict(metadata=metadata,step=step,diagnostic_control=args.seed_analytic_control is not None,actor=actor.state_dict(),
            private=private.state_dict(),optimizer=optimizer.state_dict(),private_optimizer=private_optimizer.state_dict(),
            lbfgs_optimizer=None if lbfgs_optimizer is None else lbfgs_optimizer.state_dict(),
            rng=torch.get_rng_state(),elapsed_seconds=time.perf_counter()-started))
    checkpoint(start_step)
    if args.optimizer=='lbfgs':
        initial_capture=write_span_capture(private,target)
        with (args.output/'metrics.jsonl').open('a',buffering=1) as stream:
            def closure(index):
                t=time.perf_counter()
                predicted,loss,terms,norm,guard=evaluate()
                row=dict(closure=index,event='line_search_trial_not_accepted_endpoint',objective_error=float(loss.detach()),
                    **error_metrics(terms,target_squared,relative_squared,grouped),
                    raw_action_norm=float(norm.detach()),guard=float(guard.detach()),radius=float(snapshot.radius),
                    last_hidden_features=dict(hidden_stats),
                    selected_error_squared=float(sum(v for n,v in terms.items() if n in selected_names).detach()),
                    zero_target_error_squared=float(sum(v for n,v in terms.items() if n not in selected_names).detach()),
                    forward_seconds=time.perf_counter()-t,gpt_forwards=0,gpt_backwards=0)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                print(json.dumps(row),flush=True)
                return loss
            lbfgs_optimizer,result=bounded_lbfgs([p for p in (*actor.parameters(),*private.parameters()) if p.requires_grad],closure,
                max_iter=args.steps,closure_budget=args.closure_budget,wolfe_bracket_tolerance=args.wolfe_bracket_tolerance)
        with torch.no_grad():predicted,loss,terms,norm,guard=evaluate()
        readout=stored_readout(predicted,target,parameters,grouped,
            {n:v for n,v in data['model'].items() if not n.endswith('.angular_freq')})
        result.update(final_normalized_edit_error=error_metrics(terms,target_squared,relative_squared,grouped)['normalized_edit_error'],
            final_objective_error=float(loss),final_errors=error_metrics(terms,target_squared,relative_squared,grouped),initial_span_capture=initial_capture,
            final_selected_error_squared=float(sum(v for n,v in terms.items() if n in selected_names)),
            final_zero_target_error_squared=float(sum(v for n,v in terms.items() if n not in selected_names)),
            final_raw_action_norm=float(norm),final_guard=float(guard),
            final_span_capture=write_span_capture(private,target),elapsed_seconds=time.perf_counter()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            gpt_forwards=0,gpt_backwards=0,query_access=False,learned_preservation_claim=False)
        if frozen_private_state is not None and not exact_state_equal(private.state_dict(),frozen_private_state):raise ValueError('private state changed in frozen L-BFGS phase')
        result['private_frozen_exactly_unchanged']=args.freeze_private
        result['analytic_seed']=args.seed_analytic_control is not None
        checkpoint(int(lbfgs_optimizer.state.get(next(iter(actor.parameters())),{}).get('n_iter',0)),'checkpoint_final.pt')
        (args.output/'stored_readout.json').write_text(json.dumps(readout,indent=2)+'\n')
        (args.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
        return
    with (args.output/'metrics.jsonl').open('a',buffering=1) as stream:
        for step in range(start_step,args.steps+1):
            t=time.perf_counter()
            optimizer.zero_grad(set_to_none=True);private_optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(step<args.steps):predicted,loss,terms,norm,guard=evaluate()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite supervised diagnostic')
            row=dict(step=step,event='pre_update' if step<args.steps else 'final',objective_error=float(loss.detach()),
                    **error_metrics(terms,target_squared,relative_squared,grouped),
                raw_action_norm=float(norm.detach()),guard=float(guard.detach()),radius=float(snapshot.radius),
                last_hidden_features=dict(hidden_stats),
                predicted_physical_norm=float(predicted.norm_sq().detach().sqrt()),
                selected_error_squared=float(sum(v for n,v in terms.items() if n in selected_names).detach()),
                zero_target_error_squared=float(sum(v for n,v in terms.items() if n not in selected_names).detach()),
                group_error_squared={g:float(sum((terms[n] for n in ns if n in terms),loss.new_zeros(())).detach()) for g,ns in grouped.groups.items()},
                gpt_forwards=0,gpt_backwards=0)
            if step in (start_step,args.steps):row['target_span_capture']=write_span_capture(private,target)
            if step==0 and args.seed_analytic_control is None and abs(row['normalized_edit_error']-1)>1e-10:raise ValueError('fresh zero-head identity loss is not one')
            if step<args.steps:
                loss.backward()
                if args.global_grad_clip:
                    row['preclip']={'global_shared':float(torch.nn.utils.clip_grad_norm_(list(actor.parameters()),1.,error_if_nonfinite=True))}
                else:
                    row['preclip']={role:float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
                        for role,params in [('heads',heads),('representation',representation),('private',tuple(private.parameters()))]}
                before=[p.detach().clone() for p in heads]
                output_before={kind:torch.nn.functional.linear(head_inputs[kind],head.weight.detach(),head.bias.detach()) for kind,head in [('matrix',actor.matrix_output),('vector',actor.vector_output)]}
                optimizer.step()
                if not args.freeze_private:private_optimizer.step()
                with torch.no_grad():
                    row['head_output_step_at_fixed_hidden']={}
                    for kind,head in [('matrix',actor.matrix_output),('vector',actor.vector_output)]:
                        delta=head(head_inputs[kind])-output_before[kind]
                        row['head_output_step_at_fixed_hidden'][kind]=dict(l2=float(delta.norm()),rms=float(delta.square().mean().sqrt()),max_abs=float(delta.abs().max()),units='raw head outputs before dimension/group/guard scaling')
                row['head_parameter_step_norm']=float(sum((p.detach()-old).square().sum() for p,old in zip(heads,before)).sqrt())
            row['compute_seconds']=time.perf_counter()-t
            row['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            stream.write(json.dumps(row,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
            if step<args.steps and (step+1)%100==0:checkpoint(step+1)
        readout=stored_readout(predicted,target,parameters,grouped,
            {n:v for n,v in data['model'].items() if not n.endswith('.angular_freq')})
    if args.freeze_private and not exact_state_equal(private.state_dict(),frozen_private_state):raise ValueError('frozen private state mutated')
    checkpoint(args.steps)
    (args.output/'stored_readout.json').write_text(json.dumps(readout,indent=2)+'\n')
    (args.output/'complete.json').write_text(json.dumps(dict(steps=args.steps,final_normalized_edit_error=row['normalized_edit_error'],
        final_relative_normalized_edit_error=row['relative_normalized_edit_error'],final_balanced_normalized_edit_error=row['balanced_normalized_edit_error'],final_objective_error=row['objective_error'],
        private_frozen_exactly_unchanged=args.freeze_private,analytic_seed=args.seed_analytic_control is not None,
        elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        gpt_forwards=0,gpt_backwards=0,query_access=False,learned_preservation_claim=False),indent=2)+'\n')


if __name__=='__main__':main()
