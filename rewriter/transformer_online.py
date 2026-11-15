"""Rank-zero lifecycle for the modern grouped online actor.

Caller owns sensor aggregation, base optimizer, broadcasts, and GPT execution.
Call consume after SUM-reducing ordinary token-sum gradients, before the base
optimizer mutates them; call commit after that optimizer. No second all-reduce.
Checkpoint only at these boundaries. The trainer supplies sensor snapshots and owns distributed collectives.
"""
import copy
import math
import time

import torch

from rewriter.integration import _checkpoint_tensor_copy
from rewriter.lag_one import LagOneHypergradient
from rewriter.transformer_action import prepare_transformer_action


class TransformerOnlineLifecycle:
    def __init__(self, actor, optimizer, *, global_token_count, final_action_step,
                 radius_fraction=.01, embedding_names=('embed.weight',), gradient_clip=1.,
                 policy_provenance=None, actor_forward=None, actor_execution=None, norm_auxiliary=None,
                 action_budget=None):
        if type(global_token_count) is not int or global_token_count < 1:
            raise ValueError('positive global token count required')
        if type(final_action_step) is not int or final_action_step < 0:
            raise ValueError('nonnegative terminal training index required')
        if not math.isfinite(gradient_clip) or gradient_clip <= 0:
            raise ValueError('positive finite clipping norm required')
        actual = [p for g in optimizer.param_groups for p in g['params']]
        if len(actual) != len(set(map(id, actual))) or set(map(id, actual)) != set(map(id, actor.parameters())):
            raise ValueError('optimizer must own every actor parameter exactly once')
        self.actor, self.optimizer = actor, optimizer
        if actor_forward is not None and not actor_execution:
            raise ValueError('injected actor_forward requires a pinned execution policy')
        self.actor_forward = actor if actor_forward is None else actor_forward
        self.actor_execution = copy.deepcopy(actor_execution or {'backend': 'eager'})
        self.config = dict(global_token_count=global_token_count, final_action_step=final_action_step,
                           radius_fraction=radius_fraction, embedding_names=tuple(embedding_names), gradient_clip=gradient_clip,
                           policy_provenance=copy.deepcopy(policy_provenance or {}), actor_execution=self.actor_execution,
                           optimizer_groups=[{k: copy.deepcopy(v) for k,v in g.items() if k != 'params'} for g in optimizer.param_groups])
        from .online_norm import validate_norm_auxiliary
        self.norm_auxiliary = validate_norm_auxiliary(norm_auxiliary)
        if self.norm_auxiliary is not None:
            if not isinstance(optimizer, torch.optim.Adam):
                raise ValueError('alternating norm auxiliary requires shared Adam')
            if any(g.get('role') not in (None,'action_heads','representation') for g in optimizer.param_groups):
                raise ValueError('unknown Adam group role for norm auxiliary learning rate')
            self.config['norm_auxiliary'] = copy.deepcopy(self.norm_auxiliary)
        if action_budget is not None:
            if (not isinstance(action_budget,dict) or set(action_budget)!={'mode','fraction'} or
                    action_budget['mode']!='base_motion' or isinstance(action_budget['fraction'],bool) or
                    not isinstance(action_budget['fraction'],(int,float)) or
                    not math.isfinite(action_budget['fraction']) or action_budget['fraction']<=0):
                raise ValueError('action_budget requires base_motion mode and positive finite fraction')
            self.config['action_budget']=copy.deepcopy(action_budget)
        self.current_action_budget=None
        self.norm_updates = 0
        self.lag = LagOneHypergradient()
        self.action = None
        self.action_step = None
        self.last_consumed_step = None
        self.meta_updates = 0

    @staticmethod
    def _require_authority():
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            raise RuntimeError('only rank zero owns actor actions and delayed credit')

    def bind_action_budget(self, base_update_norm, *, action_step):
        """Freeze actual ALL-parameter displacement once, after the base step."""
        self._require_authority()
        if 'action_budget' not in self.config:
            return
        if self.action is not None or self.lag.pending:
            raise RuntimeError('consume pending credit before binding action budget')
        if type(action_step) is not int or not 0<=action_step<=self.config['final_action_step']:
            raise ValueError('action budget clock outside training interval')
        if self.current_action_budget is not None and action_step<=self.current_action_budget['action_step']:
            raise ValueError('action budget already bound or clock regressed')
        from .action_budget import base_motion_radius
        # displacement_norms returns the FP64 norm of ACTUAL stored differences.
        device=next(self.actor.parameters()).device
        if isinstance(base_update_norm,torch.Tensor) and base_update_norm.requires_grad:
            raise ValueError('base displacement must be detached')
        norm=torch.as_tensor(base_update_norm,dtype=torch.float64,device=device).detach().clone()
        fraction=self.config['action_budget']['fraction']
        radius=base_motion_radius(norm,fraction=fraction)
        self.current_action_budget=dict(action_step=action_step,base_update_norm=norm,radius=radius,
                                        fraction=fraction,scope='all_parameters')

    def _check_action_budget(self, action_step, prepared=None):
        from .transformer_action import copy_action_budget
        budget=self.current_action_budget
        enabled='action_budget' in self.config
        if enabled and (budget is None or budget['action_step']!=action_step or
                        budget['fraction']!=self.config['action_budget']['fraction']):
            raise ValueError('current action budget was not bound to this base step')
        if not enabled and budget is not None:
            raise ValueError('unexpected action budget in disabled lifecycle')
        if prepared is not None:
            actual=prepared.action_budget
            if (actual is None)!=(budget is None):
                raise ValueError('prepared action budget differs from lifecycle')
            if budget is not None:
                copy_action_budget(actual,device=prepared.scale.device)
                if any(not torch.equal(actual[k],budget[k]) if isinstance(budget[k],torch.Tensor)
                       else actual[k]!=budget[k] for k in budget):
                    raise ValueError('prepared action budget differs from lifecycle')

    def action_budget_metrics(self):
        if 'action_budget' not in self.config:
            return {}
        b=self.current_action_budget
        result=dict(chisel_action_budget_mode='base_motion',chisel_action_budget_scope='all_parameters',
                    chisel_action_budget_fraction=b['fraction'],chisel_action_budget_radius=float(b['radius']),
                    chisel_action_budget_base_update_norm=float(b['base_update_norm']),
                    chisel_action_budget_action_step=b['action_step'])
        if self.action is not None:
            with torch.no_grad():
                squared=self.action.action.norm_sq().clamp_min(0)
                result['chisel_action_budget_proposed_edit_norm']=float(
                    self.action.scale*squared.sqrt())
                result['chisel_action_budget_raw_edit_norm']=float(squared.sqrt())
                raw=squared.clamp_min(1e-24).sqrt()
                static=(torch.minimum(raw.new_ones(()),self.action.snapshot.radius/(raw+1e-12))
                        if self.action.editable else raw.new_ones(()))
                result['chisel_action_budget_active']=bool(self.action.scale<static)
                result['chisel_action_budget_final_scale']=float(self.action.scale)
        return result

    def consume(self, gradients, *, gradient_step):
        self._require_authority()
        if self.action is None:
            raise RuntimeError('no committed action awaiting next-gradient credit')
        projected = self.action.project_next_gradients(gradients,
            global_token_count=self.config['global_token_count'], loss_reduction='token_sum')
        result = self.lag.consume(projected, gradient_step=gradient_step)
        self.optimizer.zero_grad(set_to_none=True)
        result.assign_gradients(dict(self.actor.named_parameters()))
        norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config['gradient_clip'], error_if_nonfinite=True)
        before={name:p.detach().clone() for name,p in self.actor.named_parameters()}
        self.optimizer.step()
        step_metrics=actor_displacement_metrics(self.actor,before)
        self.meta_updates += 1
        self.action = None
        self.action_step = None
        self.last_consumed_step = gradient_step
        return dict(action_step=result.action_step, gradient_step=gradient_step,
                    task_vjp_scalar=result.contraction_value, task_gradient_norm=float(norm),
                    task_gradient_clipped=bool(norm>self.config['gradient_clip']),
                    task_gradient_clip_threshold=self.config['gradient_clip'], meta_updates=self.meta_updates,
                    chisel_actor_learning_rates={g.get('role',str(i)):g['lr'] for i,g in enumerate(self.optimizer.param_groups)},
                    **step_metrics,
                    **({'chisel_task_consumed_action_step':result.action_step,
                        'chisel_norm_auxiliary_updates':self.norm_updates,
                        'chisel_actor_optimizer_step_calls':self.meta_updates+self.norm_updates} if self.norm_auxiliary is not None else {}))

    def _norm_step(self, snapshot, base):
        """One intentional second shared-Adam tick, before the FINAL action graph."""
        from .online_norm import continuous_group_norm_objective
        from .transformer_action import build_action_reference_cache
        if self.lag.pending or self.action is not None:
            raise RuntimeError('cannot update norm auxiliary while task credit is pending')
        started=time.perf_counter()
        reference_cache=build_action_reference_cache(base)
        candidate=prepare_transformer_action(self.actor,snapshot,base,
            radius_fraction=self.config['radius_fraction'],embedding_names=self.config['embedding_names'],
            actor_forward=self.actor_forward,reference_cache=reference_cache,action_budget=self.current_action_budget)
        grouped=reference_cache.grouped
        objective,groups=continuous_group_norm_objective(candidate,grouped)
        self.optimizer.zero_grad(set_to_none=True)
        weighted=self.norm_auxiliary['coefficient']*objective
        if weighted.requires_grad:
            gradients=torch.autograd.grad(weighted,tuple(self.actor.parameters()),allow_unused=True)
        else:
            gradients=(None,)*len(tuple(self.actor.parameters()))
        if any(g is not None and not bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError('nonfinite norm gradient; no auxiliary Adam step performed')
        for parameter,gradient in zip(self.actor.parameters(),gradients):
            parameter.grad=None if gradient is None else gradient.detach()
        norm=torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.norm_auxiliary['gradient_clip'],error_if_nonfinite=True)
        metrics=dict(chisel_norm_auxiliary_group_reference_scans=1,
            chisel_norm_auxiliary_global_radius_scans=int(bool(reference_cache.editable)),
            chisel_norm_auxiliary_fractional_change=float(objective.detach()),
            chisel_norm_auxiliary_predicted_shrink_percentage=-100*float(objective.detach()),
            chisel_norm_auxiliary_group_predicted_shrink_percentage={k:-100*float(v.detach()) for k,v in groups.items()},
            chisel_norm_auxiliary_weighted_gradient_norm=float(norm),
            chisel_norm_auxiliary_unweighted_gradient_norm=float(norm)/self.norm_auxiliary['coefficient'],
            chisel_norm_auxiliary_gradient_clipped=bool(norm>self.norm_auxiliary['gradient_clip']),
            chisel_norm_auxiliary_scope='continuous cached finite-edit proposal BEFORE auxiliary Adam; not final stored-weight shrinkage')
        # No reference to this graph survives the actor mutation. The final
        # executable action must be recomputed at the updated actor parameters.
        del candidate,objective,groups,weighted,gradients
        before={name:p.detach().clone() for name,p in self.actor.named_parameters()}
        old_lrs=[g['lr'] for g in self.optimizer.param_groups]
        try:
            for group in self.optimizer.param_groups:
                group['lr']=self.norm_auxiliary['learning_rate']*(self.norm_auxiliary['representation_lr_multiplier'] if group.get('role')=='representation' else 1.)
            metrics['chisel_norm_auxiliary_learning_rates']=[g['lr'] for g in self.optimizer.param_groups]
            self.optimizer.step()
        finally:
            for group,lr in zip(self.optimizer.param_groups,old_lrs):group['lr']=lr
        self.norm_updates+=1
        step=actor_displacement_metrics(self.actor,before)
        metrics.update(chisel_norm_auxiliary_actor_step_norm=step['chisel_actor_step_norm'],
                       chisel_norm_auxiliary_updates=self.norm_updates,
                       chisel_actor_optimizer_step_calls=self.meta_updates+self.norm_updates,
                       chisel_norm_auxiliary_actor_forward_calls=1,
                       chisel_norm_auxiliary_actor_backward_calls=int(bool(any(p.grad is not None for p in self.actor.parameters()))),
                       chisel_norm_auxiliary_wall_seconds=time.perf_counter()-started)
        return metrics,reference_cache

    def commit(self, snapshot, live_parameters, *, action_step):
        self._require_authority()
        if self.action is not None or self.lag.pending:
            raise RuntimeError('consume pending credit before another commit')
        if type(action_step) is not int or not 0 <= action_step <= self.config['final_action_step']:
            raise ValueError('action outside declared training interval')
        if self.last_consumed_step is not None and action_step != self.last_consumed_step:
            raise ValueError('commit must follow the current ordinary backward and base optimizer')
        if action_step == self.config['final_action_step']:
            return dict(terminal_identity=True, meta_updates=self.meta_updates)
        self._check_action_budget(action_step)
        # Own detached copies: live weights mutate on commit and the next base step.
        base = {n: p.detach().clone() for n, p in live_parameters.items()}
        auxiliary_metrics={}
        reference_cache=None
        if self.norm_auxiliary is not None:
            auxiliary_metrics,reference_cache=self._norm_step(snapshot,base)
            auxiliary_metrics['chisel_norm_auxiliary_action_step']=action_step
        prepared = prepare_transformer_action(self.actor, snapshot, base,
            radius_fraction=self.config['radius_fraction'], embedding_names=self.config['embedding_names'],
            actor_forward=self.actor_forward,reference_cache=reference_cache,action_budget=self.current_action_budget)
        proposal = prepared.materialize()
        if self.norm_auxiliary is not None:
            from .online_norm import continuous_group_norm_objective
            with torch.no_grad():
                reference_cache.validate(base)
                final_objective,_=continuous_group_norm_objective(prepared,reference_cache.grouped)
                auxiliary_metrics['chisel_norm_auxiliary_final_fractional_change']=float(final_objective)
                auxiliary_metrics['chisel_norm_auxiliary_final_predicted_shrink_percentage']=-100*float(final_objective)
        if any(not bool(torch.isfinite(value).all()) for value in proposal.values()):
            raise FloatingPointError('nonfinite proposal; live weights unchanged')
        # All preconditions verified before any parameter copy.
        with torch.no_grad():
            for name in prepared.editable:
                live_parameters[name].copy_(proposal[name])
        prepared.record_executed(self.lag, action_step=action_step, actor=self.actor)
        self.action, self.action_step = prepared, action_step
        return dict(terminal_identity=False, action_step=action_step,
                    committed_edit_norm=float(sum(((proposal[n]-base[n]).float().square().sum() for n in prepared.editable), prepared.scale.new_zeros(())).sqrt().detach()),
                    zero_reference_groups=prepared.zero_reference_groups, meta_updates=self.meta_updates, **auxiliary_metrics)

    def commit_prepared(self, prepared, live_parameters, *, action_step):
        """Commit an already evaluated FINAL replay action without rerunning actor."""
        self._require_authority()
        if self.action is not None or self.lag.pending:
            raise RuntimeError('consume pending task action before final commit')
        if not 0 <= action_step < self.config['final_action_step']:
            raise ValueError('prepared action outside nonterminal interval')
        if self.last_consumed_step is not None and self.last_consumed_step != action_step:
            raise ValueError('prepared action clock differs')
        self._check_action_budget(action_step,prepared)
        proposal = prepared.materialize()
        if set(proposal) != set(live_parameters):raise ValueError('prepared inventory differs')
        for name, value in proposal.items():
            live=live_parameters[name]
            if not torch.equal(live.detach(),prepared.parameters[name]):
                raise RuntimeError('live post-base changed after replay')
            if value.shape!=live.shape or value.dtype!=live.dtype or not bool(torch.isfinite(value).all()):
                raise FloatingPointError('invalid final replay proposal')
        with torch.no_grad():
            for name in prepared.editable:live_parameters[name].copy_(proposal[name])
        prepared.record_executed(self.lag, action_step=action_step, actor=self.actor)
        self.action,self.action_step=prepared,action_step
        return dict(terminal_identity=False,action_step=action_step,meta_updates=self.meta_updates,
                    committed_edit_norm=float(sum(((proposal[n]-prepared.parameters[n]).float().square().sum() for n in prepared.editable),prepared.scale.new_zeros(())).sqrt().detach()),
                    zero_reference_groups=prepared.zero_reference_groups)

    def state_dict(self):
        pending = None
        if self.action is not None:
            self.action._check_versions()
            pending = dict(action_step=self.action_step,
                           snapshot=_checkpoint_tensor_copy(self.action.snapshot),
                           base=_checkpoint_tensor_copy(self.action.parameters),
                           executed=_checkpoint_tensor_copy(self.action.materialize()))
            if 'action_budget' in self.config:
                pending['action_budget']=_checkpoint_tensor_copy(self.action.action_budget)
        return dict(schema=3 if 'action_budget' in self.config else 2 if self.norm_auxiliary is not None else 1, config=copy.deepcopy(self.config), actor=_checkpoint_tensor_copy(self.actor.state_dict()),
                    optimizer=copy.deepcopy(self.optimizer.state_dict()), pending=pending,
                    meta_updates=self.meta_updates, last_consumed_step=self.last_consumed_step,
                    **({'norm_updates':self.norm_updates} if self.norm_auxiliary is not None else {}))

    def load_state_dict(self, saved, live_parameters):
        self._require_authority()
        if self.action is not None or self.lag.pending:
            raise RuntimeError('cannot restore over a pending action')
        if saved['schema'] != (3 if 'action_budget' in self.config else 2 if self.norm_auxiliary is not None else 1) or saved['config'] != self.config:
            raise ValueError('online lifecycle resume configuration changed')
        if self.norm_auxiliary is not None:
            ticks=saved.get('norm_updates')
            if type(ticks) is not int or ticks<0 or ticks!=saved['meta_updates']+int(saved['pending'] is not None):
                raise ValueError('alternating norm/task checkpoint clocks differ')
        device = next(self.actor.parameters()).device
        pending = saved['pending']
        if pending is not None:
            if set(live_parameters) != set(pending['executed']):
                raise ValueError('live checkpoint parameter inventory differs')
            for n, p in live_parameters.items():
                expected = pending['executed'][n].to(device=p.device)
                if p.dtype != expected.dtype or p.shape != expected.shape or not torch.equal(p, expected):
                    raise ValueError('live checkpoint differs from the executed proposal: '+n)
        # Snapshot reconstruction must not perturb the task RNG stream.
        devices = [device.index or 0] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            self.actor.load_state_dict(saved['actor'])
            self.optimizer.load_state_dict(saved['optimizer'])
            # A restored no-pending boundary owns no action-time radius, even
            # when this instance previously advanced to a later action clock.
            self.current_action_budget=None
            if pending is not None:
                if 'action_budget' in self.config:
                    from .transformer_action import copy_action_budget
                    self.current_action_budget=copy_action_budget(pending.get('action_budget'),device=device)
                    self._check_action_budget(pending['action_step'])
                prepared = prepare_transformer_action(self.actor,
                    _checkpoint_tensor_copy(pending['snapshot'],device),
                    _checkpoint_tensor_copy(pending['base'],device),
                    radius_fraction=self.config['radius_fraction'], embedding_names=self.config['embedding_names'], actor_forward=self.actor_forward,
                    action_budget=self.current_action_budget)
                for n, p in prepared.materialize().items():
                    if not torch.equal(p, pending['executed'][n].to(device)):
                        raise ValueError('reconstructed action differs from executed proposal: '+n)
                prepared.record_executed(self.lag, action_step=pending['action_step'], actor=self.actor)
                self.action, self.action_step = prepared, pending['action_step']
        self.meta_updates = saved['meta_updates']
        self.last_consumed_step = saved['last_consumed_step']
        if self.norm_auxiliary is not None:self.norm_updates=saved['norm_updates']


@torch.no_grad()
def actor_displacement_metrics(actor,before):
    """Measured Adam parameter displacement; gradient norm is not a substitute."""
    heads={id(p) for head in getattr(actor,'output_heads',()) for p in head.parameters()}
    device=next(actor.parameters()).device
    deltas={k:torch.zeros((),device=device,dtype=torch.float64) for k in ('action_heads','residual_blocks','input_and_norm')}
    references={k:torch.zeros_like(v) for k,v in deltas.items()}
    for name,p in actor.named_parameters():
        family='action_heads' if id(p) in heads else ('residual_blocks' if name.startswith('blocks.') else 'input_and_norm')
        old=before[name].reshape(-1);new=p.reshape(-1)
        for start in range(0,p.numel(),65536):
            a=old[start:start+65536].double();b=new[start:start+65536].double()
            deltas[family].add_((b-a).square().sum());references[family].add_(a.square().sum())
    norms={k:float(v.sqrt()) for k,v in deltas.items()}
    return dict(chisel_actor_step_norm=float(sum(deltas.values()).sqrt()),
                chisel_actor_step_norm_by_family=norms,
                chisel_actor_relative_step_norm_by_family={k:(norms[k]/float(v.sqrt()) if float(v)>0 else None) for k,v in references.items()})


@torch.no_grad()
def displacement_norms(parameters, before):
    """Actual stored-weight displacement, globally and in the fifteen groups."""
    from .grouped_norm import functional_groups
    groups=functional_groups(parameters)
    group_for={name:group for group,names in groups.items() for name in names}
    device=next(iter(parameters.values())).device
    sums={group:torch.zeros((),device=device,dtype=torch.float64) for group in groups}
    total=torch.zeros((),device=device,dtype=torch.float64)
    for name,p in parameters.items():
        square=torch.zeros_like(total)
        for start in range(0,len(p),512):
            difference=p[start:start+512].double()-before[name][start:start+512].double()
            square.add_(difference.square().sum())
        total.add_(square)
        if name in group_for:sums[group_for[name]].add_(square)
    return dict(all_parameters=float(total.sqrt()),grouped=float(sum(sums.values()).sqrt()),
                groups={name:float(value.sqrt()) for name,value in sums.items()})


def edit_base_ratios(edit, base):
    # A zero base displacement makes the ratio undefined; never hide it with
    # an invented floor or silently use it as a new edit cap.
    ratio=lambda a,b:a/b if b>0 else None
    return dict(chisel_base_displacement_norm_all_parameters=base['all_parameters'],
                chisel_base_displacement_norm_grouped=base['grouped'],
                chisel_realized_edit_norm_grouped=edit['grouped'],
                chisel_edit_to_base_norm_ratio_all_parameters=ratio(edit['all_parameters'],base['all_parameters']),
                chisel_edit_to_base_norm_ratio_grouped=ratio(edit['grouped'],base['grouped']),
                chisel_base_displacement_norm_by_group=base['groups'],
                chisel_realized_edit_norm_by_group=edit['groups'],
                chisel_edit_to_base_norm_ratio_by_group={g:ratio(edit['groups'][g],b) for g,b in base['groups'].items()})


@torch.no_grad()
def checkpoint_group_norms(parameters):
    """Absolute norms of actual live weights, for paired checkpoint readouts."""
    from .grouped_norm import functional_groups
    from .edit_scope import editable_weight_norm_sq
    started=time.perf_counter()
    groups=functional_groups(parameters)
    return dict(group_l2_norm={g:float(editable_weight_norm_sq(parameters,names).sqrt()) for g,names in groups.items()},
                group_norm_scope='combined L2 of actual stored weights in each of15 functional groups; gains/scalars excluded',
                group_norm_wall_seconds=time.perf_counter()-started)


@torch.no_grad()
def realized_group_norm_metrics(parameters, before):
    """Stored-weight norm change relative to the current post-base proposal.

    Cache this action's base norms once, using the same stable FP64 norm-change
    calculation and equal fifteen-group weighting as the offline objective.
    This is an observation, not an extra training objective or a new edit cap.
    """
    from .grouped_norm import GroupedNormObjective
    started = time.perf_counter()
    objective = GroupedNormObjective(before, zero_reference_policy='freeze')
    result = objective.evaluate(parameters).metrics()
    return dict(chisel_action_mean_group_l2_percentage_reduction=result['mean_group_l2_percentage_reduction'],
                chisel_action_group_l2_percentage_reduction=result['group_l2_percentage_reduction'],
                chisel_action_group_new_l2_norm=result['group_new_l2_norm'],
                chisel_action_group_base_l2_norm={g:float(n) for g,n in objective.reference_norms.items()},
                chisel_action_zero_reference_groups=list(objective.zero_reference_groups),
                chisel_group_norm_metric_scope='realized stored-weight edit relative to this step post-base, pre-edit weights; equal 15 groups; positive means shrinkage; frozen zero-reference groups contribute zero',
                chisel_group_norm_metric_wall_seconds=time.perf_counter()-started)


def authoritative_call(function, *, device):
    """Run rank-zero Python work and propagate failures before later collectives."""
    dist = torch.distributed
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    result = None
    error = None
    if rank == 0:
        try:
            result = function()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    if distributed:
        flag = torch.tensor(int(error is not None), device=device, dtype=torch.int32)
        dist.broadcast(flag, src=0)
        if bool(flag):
            message = [error]
            dist.broadcast_object_list(message, src=0, device=device)
            raise RuntimeError('rank-zero Chisel operation failed: ' + message[0])
    elif error is not None:
        raise RuntimeError('rank-zero Chisel operation failed: ' + error)
    return result


def collective_local_call(function, *, device):
    """Propagate a local validation failure on ANY rank, before root collectives.

    The supplied function itself must contain no distributed collectives.
    """
    dist = torch.distributed
    result = None;error = None
    try:
        result = function()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    if not dist.is_initialized():
        if error is not None:raise RuntimeError('Chisel local operation failed: '+error)
        return result
    culprit = torch.tensor(dist.get_rank() if error is not None else -1,device=device,dtype=torch.int32)
    dist.all_reduce(culprit,op=dist.ReduceOp.MAX)
    if int(culprit)>=0:
        message=[error];dist.broadcast_object_list(message,src=int(culprit),device=device)
        raise RuntimeError('Chisel local operation failed: '+message[0])
    return result


@torch.no_grad()
def assert_replicated_before_edit(model, *, device):
    """Exact raw-byte fingerprints before root edits can overwrite divergence.

    This copies model bytes to CPU and is an occasional diagnostic, not a cheap
    device checksum. Its wall time is reported separately and remains included
    in measured whole-iteration time.
    """
    from baseline.fingerprint import model_fingerprint
    dist=torch.distributed
    started=time.perf_counter()
    local=collective_local_call(lambda:model_fingerprint(model),device=device)
    hashes=[local['sha256']]
    if dist.is_initialized():
        hashes=[None]*dist.get_world_size();dist.all_gather_object(hashes,local['sha256'])
    if len(set(hashes))!=1:
        raise RuntimeError('model replicas differ before Chisel edit/broadcast: '+str(hashes))
    return dict(chisel_pre_edit_replica_sha256=hashes[0],chisel_pre_edit_replica_count=len(hashes),
                chisel_replica_check_wall_seconds=time.perf_counter()-started)


class TransformerChiselAdapter:
    """Distributed observer plus a rank-zero actor; no auxiliary GPT evaluations.

    `shadow` collects identical ordinary-pass observations but commits no edits.
    `learned` consumes only the next ordinary task gradient. Neither mode adds
    validation-dependent learning or claims that future live edits preserve KL.
    Initial and resumed checkpoints use the same baseline checkpoint machinery.
    """
    def __init__(self, model, optimizers, config, *, mode, actor=None, actor_optimizer=None):
        from .state_capture import DistributedStateObserver
        self.config = config
        self.c = copy.deepcopy(config['transformer_chisel'])
        if mode not in ('shadow', 'learned'):
            raise ValueError('modern Chisel requires shadow or learned mode')
        self.mode = mode
        self.replica_check_every=self.c.get('replica_check_every',64)
        if type(self.replica_check_every) is not int or self.replica_check_every<1:
            raise ValueError('positive integer replica check cadence required')
        self.group_norm_metrics_every=self.c.get('group_norm_metrics_every',64)
        if type(self.group_norm_metrics_every) is not int or self.group_norm_metrics_every<1:
            raise ValueError('positive integer group norm metric cadence required')
        self.observer = DistributedStateObserver(model, optimizers, config,
            sensor_config=self.c['sensor_config'], start_step=self.c['prefix_steps'])
        self.collector = self.observer.collector
        # Constructor allocates buffers; it must not observe the initial validation.
        self.collector.active = False
        self.params, self.device = self.observer.params, self.observer.device
        self.rank_id, self.distributed = self.observer.rank_id, self.observer.distributed
        self.lifecycle = None
        self.metrics = {}
        def initialize():
            nonlocal actor, actor_optimizer
            from .chisel_factory import build_chisel
            from .meta_optimizer import build_meta_adam
            devices = [self.device.index or 0] if self.device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                if actor is None:
                    torch.manual_seed(self.c['actor_seed'])
                    actor = build_chisel(self.c['architecture']).to(self.device)
                if actor_optimizer is None:
                    actor_optimizer = build_meta_adam(actor, lr=self.c['task_lr'],
                        representation_lr_multiplier=self.c['representation_lr_multiplier'])
            execution = self.c['actor_execution']
            if execution == {'backend': 'eager'}:
                forward = actor
            elif execution == {'backend': 'torch.compile', 'fullgraph': True, 'dynamic': False}:
                forward = torch.compile(actor, fullgraph=True, dynamic=False)
            else:
                raise ValueError('unsupported or unpinned actor execution policy')
            return TransformerOnlineLifecycle(actor, actor_optimizer,
                global_token_count=config['batch_tokens'], final_action_step=config['max_steps']-1,
                radius_fraction=self.c['radius_fraction'], embedding_names=sorted(self.observer.embedding_names),
                gradient_clip=self.c['gradient_clip'], policy_provenance=self.c['policy_provenance'],
                actor_forward=forward, actor_execution=execution, norm_auxiliary=self.c.get('norm_auxiliary'),
                action_budget=self.c.get('action_budget'))
        self.lifecycle = authoritative_call(initialize, device=self.device)
        from .online_preservation import validate_replay_maintenance, ReplayMaintenance
        replay_settings=validate_replay_maintenance(self.c.get('replay_maintenance'))
        if replay_settings is not None and (self.c.get('norm_auxiliary') or {}).get('mode','off')!='off':
            raise ValueError('norm auxiliary and replay maintenance are mutually exclusive')
        self.replay=None
        if replay_settings is not None and mode=='learned':
            self.replay=collective_local_call(lambda:ReplayMaintenance(model,replay_settings,
                batch_tokens=config['batch_tokens'],seq_len=config['seq_len'],
                embedding_names=self.observer.embedding_names),device=self.device)

    def before_step(self, t, model, optimizers):
        self.metrics = {}
        self.observer.before_step(t, model, optimizers)

    def before_microbatch(self, t, sequence_ids, inputs):
        self.observer.before_microbatch(t, sequence_ids, inputs)

    def after_backward(self, t, model):
        self.observer.after_backward(t, model)
        def consume():
            if self.mode == 'learned' and self.lifecycle.action is not None:
                self.metrics.update(self.lifecycle.consume({n:p.grad for n,p in self.params.items()}, gradient_step=t))
        authoritative_call(consume, device=self.device)

    def after_step(self, t, model, optimizers):
        if self.replay is not None:
            return self._after_step_replay(t,model,optimizers)
        # Check the replicated base proposal BEFORE rank zero intentionally
        # diverges by committing its edit. Post-edit pre-broadcast equality is
        # not expected and would be the wrong invariant.
        if self.distributed and ((t-self.c['prefix_steps'])%self.replica_check_every==0 or t==self.config['max_steps']-1):
            self.metrics.update(assert_replicated_before_edit(model,device=self.device))
        # The sharded momentum projection has collective validity checks.
        momentum = self.observer.momentum_projections()
        def act():
            snapshot, _, error = self.observer.snapshot_from_momentum(t, momentum)
            self.metrics['chisel_allones_relative_error'] = error
            self.metrics.update(chisel_train_sequence_probes=self.collector.nt,
                                chisel_audit_sequence_probes=self.collector.na,
                                chisel_sequence_gradient_banks=self.collector.K)
            base_norms = displacement_norms(self.params, self.observer.before)
            if self.mode == 'learned':
                self.lifecycle.bind_action_budget(base_norms['all_parameters'],action_step=t)
                self.metrics.update(self.lifecycle.commit(snapshot, self.params, action_step=t))
                self.metrics.update(self.lifecycle.action_budget_metrics())
            else:
                self.metrics.update(shadow_identity=True, meta_updates=0)
            if self.mode == 'learned' and self.lifecycle.action is not None:
                edit_norms = displacement_norms(self.params, self.lifecycle.action.parameters)
            else:
                edit_norms = dict(all_parameters=0.,grouped=0.,groups={g:0. for g in base_norms['groups']})
            self.metrics.update(edit_base_ratios(edit_norms,base_norms))
            self.metrics.update(chisel_group_norm_metric_measured=False,
                                chisel_group_norm_metric_every=self.group_norm_metrics_every)
            if (self.mode == 'learned' and self.lifecycle.action is not None and
                    (t-self.c['prefix_steps']) % self.group_norm_metrics_every == 0):
                self.metrics.update(realized_group_norm_metrics(self.params, self.lifecycle.action.parameters))
                self.metrics['chisel_group_norm_metric_measured'] = True
                if self.lifecycle.norm_auxiliary is not None:
                    self.metrics['chisel_norm_auxiliary_final_prediction_gap_percentage_points']=(
                        self.metrics['chisel_norm_auxiliary_final_predicted_shrink_percentage']-
                        self.metrics['chisel_action_mean_group_l2_percentage_reduction'])
            self.metrics.update(chisel_extra_gpt_forwards=0, chisel_extra_gpt_backwards=0,
                                chisel_online_auxiliary='none; ordinary-next-gradient credit only')
        authoritative_call(act, device=self.device)
        # Readonly tensors were never edited. Broadcast every tensor so rank
        # equality also covers frozen zero-reference groups and normalization.
        if self.distributed and self.mode == 'learned':
            for value in self.params.values():
                torch.distributed.broadcast(value.detach(), src=0)
        self.observer.release_step(t)

    def _after_step_replay(self,t,model,optimizers):
        # Explicit all-rank phase: never enter replay collectives inside a
        # rank-zero authoritative_call closure.
        if self.distributed and ((t-self.c['prefix_steps'])%self.replica_check_every==0 or t==self.config['max_steps']-1):
            self.metrics.update(assert_replicated_before_edit(model,device=self.device))
        momentum=self.observer.momentum_projections()
        def observation():
            snapshot,_,error=self.observer.snapshot_from_momentum(t,momentum)
            self.metrics.update(chisel_allones_relative_error=error,
                chisel_train_sequence_probes=self.collector.nt,chisel_audit_sequence_probes=self.collector.na,
                chisel_sequence_gradient_banks=self.collector.K)
            base_norms=displacement_norms(self.params,self.observer.before)
            self.lifecycle.bind_action_budget(base_norms['all_parameters'],action_step=t)
            return snapshot,base_norms
        observed=authoritative_call(observation,device=self.device)
        snapshot=observed[0] if self.rank_id==0 else None
        if t==self.config['max_steps']-1:
            def terminal():
                self.metrics.update(self.lifecycle.commit(snapshot,self.params,action_step=t),
                    chisel_extra_gpt_forwards=0,chisel_extra_gpt_backwards=0,
                    chisel_replay_updates=self.replay.updates,
                    chisel_actor_optimizer_step_calls=self.lifecycle.meta_updates+self.replay.updates)
                self.metrics.update(self.lifecycle.action_budget_metrics())
            authoritative_call(terminal,device=self.device)
        else:
            prepared,replay_metrics=self.replay.run(self.lifecycle,snapshot,self.params,self.collector,action_step=t)
            def commit():
                self.metrics.update(self.lifecycle.commit_prepared(prepared,self.params,action_step=t),**replay_metrics)
                self.metrics.update(self.lifecycle.action_budget_metrics())
                edits=displacement_norms(self.params,prepared.parameters)
                self.metrics.update(edit_base_ratios(edits,observed[1]),**realized_group_norm_metrics(self.params,prepared.parameters),
                    chisel_group_norm_metric_measured=True,chisel_group_norm_metric_every=1)
            authoritative_call(commit,device=self.device)
        if self.distributed:
            for value in self.params.values():torch.distributed.broadcast(value.detach(),src=0)
        self.observer.release_step(t)

    def state_dict(self):
        learner = authoritative_call(lambda: self.lifecycle.state_dict(), device=self.device)
        return dict(schema=2 if self.replay is not None else 1, mode=self.mode, settings=copy.deepcopy(self.c), rank=self.rank_id,
                    observer=self.observer.state_dict(), learner=learner,
                    **({'replay':self.replay.state_dict()} if self.replay is not None else {}))

    def load_state_dict(self, saved):
        # Validate local rank/config on every process before entering root work.
        valid = (saved.get('schema') == (1 if saved.get('bootstrap') or self.replay is None else 2) and saved.get('mode') == self.mode and
                 saved.get('settings') == self.c and saved.get('rank') == self.rank_id)
        flags = torch.tensor(int(valid), device=self.device, dtype=torch.int32)
        if self.distributed:torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MIN)
        if not bool(flags):raise ValueError('modern adapter rank or configuration mismatch')
        if saved.get('bootstrap'):
            if saved.get('completed_steps') != self.c['prefix_steps']:
                raise ValueError('bootstrap does not match the declared prefix')
            def initialize_policy():
                from baseline.chisel_transition import load_policy_checkpoint
                loaded = load_policy_checkpoint(self.c, device=self.device, require_qualification=self.mode=='learned')
                self.lifecycle.actor.load_state_dict(loaded['controller'])
            authoritative_call(initialize_policy, device=self.device)
            return
        collective_local_call(lambda: self.observer.load_state_dict(saved['observer']), device=self.device)
        authoritative_call(lambda: self.lifecycle.load_state_dict(saved['learner'], self.params), device=self.device)
        if self.replay is not None:
            collective_local_call(lambda:self.replay.load_state_dict(saved['replay'],self.lifecycle),device=self.device)

    def close(self):
        self.observer.close()
