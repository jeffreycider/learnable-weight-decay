"""Explicit adaptive/grouped lag-one binding for unwrapped FP32 GPT parameters.

Caller order: ordinary backward and global gradient reduction; consume OLD
executed action; base Muon/AdamW step; commit NEW action. No GPT replay/backward
occurs here. Lookahead/pre-forward parameter wrappers are deliberately rejected:
their actual next-forward map would need an explicit additional derivative.

Fresh U.T W V sensing differentiates private frames with W frozen at action time.
The graph retains dense detached post-base weights and sensor snapshots through
the next ordinary gradient. This is not a low-memory/distributed trainer adapter.

Caller gradient inventory includes every stored parameter, including read-only
1D gains/scalars: explicitly replace missing ordinary .grad values with zeros.
Do this BEFORE the base optimizer mutates parameters or gradient buffers.

Example explicit optimizer mapping (rates are choices, not online qualification)::

    binding = AdaptiveOnlineBinding.from_fitted(shared, private, snapshot, run_id='run')
    heads = [p for n,p in binding.shared.named_parameters()
             if n.startswith(('matrix_output.', 'vector_output.'))]
    body = [p for n,p in binding.shared.named_parameters()
            if not n.startswith(('matrix_output.', 'vector_output.'))]
    shared_adam = torch.optim.Adam([
        dict(params=heads, lr=1e-9, role='heads'),
        dict(params=body, lr=1e-7, role='body')], betas=(.9,.99), eps=1e-8)
    private_adam = torch.optim.Adam(binding.private.parameters(), lr=1e-4,
                                    betas=(.9,.99), eps=1e-12)
    # private_adam owns both raw QR frames AND the local trajectory_input map.
    online = AdaptiveOnlineLifecycle(binding, shared_adam, private_adam,
        global_token_count=batch_tokens, final_action_step=max_steps-1,
        radius_fraction=.01, shared_clip=1., private_clip=1.,
        parameter_view='unwrapped_stored_fp32')
    # After the globally SUMmed ordinary backward, before the base optimizer:
    task_grads = {n: p.grad if p.grad is not None else torch.zeros_like(p)
                  for n,p in live_parameters.items()}
    if online.pending is not None:
        online.consume(task_grads, gradient_step=step)
    # Caller executes the ordinary Muon/AdamW base update, then refreshes sensors.
    online.commit(snapshot, live_parameters, action_step=step, base_step_completed=step)

shared_clip clips heads and body TOGETHER; private_clip is separate. This explicit
policy is distinct from offline per-role clipping; no offline policy is imported.

"""
import copy
import math
from types import SimpleNamespace
import torch
from torch import nn
from .adaptive_coordinate_chisel import AdaptiveCoordinateChisel,AdaptiveCheckpointProbes
from .grouped_coordinate_chisel import GroupedCoordinateChisel
from .adaptive_sensing import query_weight_cores
from .adaptive_preservation import build_adaptive_proposal
from .adaptive_credit import AdaptiveCreditRecord
from .grouped_norm import GroupedNormObjective
from .lag_one import LagOneHypergradient
from .transformer_action import _copy_observation,_tensors
from .integration import _checkpoint_tensor_copy


class AdaptiveOnlineBinding(nn.Module):
    schema='adaptive_online_binding_v1'

    def __init__(self,shared,private,*,run_id,source_private_key):
        super().__init__()
        if type(shared) not in (AdaptiveCoordinateChisel,GroupedCoordinateChisel) or type(private) is not AdaptiveCheckpointProbes:
            raise TypeError('explicit adaptive/grouped shared actor and adaptive private probes required')
        if not isinstance(run_id,str) or not run_id or private.state_key!=('online',run_id):raise ValueError('private online run identity differs')
        if not shared.fresh_weight_core or shared.factor_precision!='fp64' or shared.residual_precision!='fp32':raise ValueError('online path requires fresh weight cores, FP64 factors and FP32 residuals')
        self.shared=shared;self.private=private;self.run_id=run_id;self.source_private_key=tuple(source_private_key)

    @classmethod
    def from_fitted(cls,shared,private,snapshot,*,run_id):
        if type(shared) not in (AdaptiveCoordinateChisel,GroupedCoordinateChisel) or type(private) is not AdaptiveCheckpointProbes:raise TypeError('unsupported adaptive binding source')
        private.validate_layout(snapshot)
        owner=copy.deepcopy(shared);live=owner.make_private(snapshot,state_key=('online',run_id),seed=0)
        source_identity=copy.deepcopy(live.get_extra_state());source_identity['state_key']=private.state_key
        if private.get_extra_state()!=source_identity:raise ValueError('source private architecture differs')
        state=copy.deepcopy(private.state_dict());state['_extra_state']=live.get_extra_state();live.load_state_dict(state,strict=True)
        # Explicit new online ownership; source requires_grad flags/weights stay unchanged.
        owner.requires_grad_(True).eval();live.requires_grad_(True).eval()
        return cls(owner,live,run_id=run_id,source_private_key=private.state_key)

    def forward(self,snapshot,parameters,*,embedding_names):
        cores=query_weight_cores(parameters,self.private,embedding_names=embedding_names)
        return self.shared(snapshot,self.private,weight_cores=cores)

    def get_extra_state(self):
        return dict(schema=self.schema,run_id=self.run_id,source_private_key=self.source_private_key,
            architecture=self.shared.configuration(),private_metadata=self.private.get_extra_state(),optimizer_import='none; caller-owned separate optimizers')

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('adaptive online identity or architecture differs')

    def load_state_dict(self,state_dict,strict=True,assign=False):
        self.set_extra_state(state_dict.get('_extra_state'))
        if state_dict.get('private._extra_state')!=self.private.get_extra_state():raise ValueError('private online identity differs')
        return super().load_state_dict(state_dict,strict=strict,assign=assign)


def _optimizer_settings(optimizer):
    return [{k:copy.deepcopy(v) for k,v in g.items() if k!='params'} for g in optimizer.param_groups]


def _ownership(optimizer,module):
    actual=[p for g in optimizer.param_groups for p in g['params']];expected=list(module.parameters())
    if len(actual)!=len(set(map(id,actual))) or set(map(id,actual))!=set(map(id,expected)) or not all(p.requires_grad for p in expected):raise ValueError('optimizer must own every trainable parameter of its shared/private module exactly once')


class AdaptiveOnlineLifecycle:
    """One pending UCV action; two optimizers and one guarded lag-one owner.

    commit requires explicit base_step_completed confirmation. Live parameter
    versions additionally reject a base mutation before consume. The caller must
    not feed AMP-scaled/DDP-averaged buffers as globally SUMmed token gradients.
    Clipping and parameter-view policy are required explicitly, never inherited
    from an offline training run. Only the direct next-task VJP is learned here;
    this module does not invent an online norm/preservation auxiliary objective.
    """
    def __init__(self,binding,shared_optimizer,private_optimizer,*,global_token_count,
                 final_action_step,radius_fraction,shared_clip,private_clip,
                 parameter_view,embedding_names=('embed.weight',)):
        if type(binding) is not AdaptiveOnlineBinding:raise TypeError('explicit adaptive binding required')
        if parameter_view!='unwrapped_stored_fp32':raise ValueError('pre-forward/lookahead wrappers unsupported; require unwrapped_stored_fp32')
        if type(global_token_count) is not int or global_token_count<1 or type(final_action_step) is not int or final_action_step<0:raise ValueError('invalid online token/terminal clocks')
        if any(not math.isfinite(v) or v<=0 for v in (radius_fraction,shared_clip,private_clip)):raise ValueError('positive finite radius and explicit clipping required')
        _ownership(shared_optimizer,binding.shared);_ownership(private_optimizer,binding.private)
        if set(map(id,binding.shared.parameters()))&set(map(id,binding.private.parameters())):raise ValueError('shared/private parameter ownership overlaps')
        self.binding=binding;self.shared_optimizer=shared_optimizer;self.private_optimizer=private_optimizer
        self.config=dict(global_token_count=global_token_count,final_action_step=final_action_step,radius_fraction=radius_fraction,
            shared_clip=shared_clip,private_clip=private_clip,parameter_view=parameter_view,embedding_names=tuple(embedding_names),
            binding=binding.get_extra_state(),execution='eager_eval',shared_optimizer_type=type(shared_optimizer).__module__+'.'+type(shared_optimizer).__qualname__,
            private_optimizer_type=type(private_optimizer).__module__+'.'+type(private_optimizer).__qualname__,
            shared_optimizer=_optimizer_settings(shared_optimizer),private_optimizer=_optimizer_settings(private_optimizer))
        self.lag=LagOneHypergradient();self.pending=None;self.last_consumed_step=None;self.last_committed_step=None;self.meta_updates=0;self.terminal=False

    @staticmethod
    def _authority():
        if torch.distributed.is_initialized() and torch.distributed.get_rank()!=0:raise RuntimeError('rank zero owns adaptive online lifecycle')

    def _check_optimizers(self):
        if self.binding.shared.training or self.binding.private.training:raise ValueError('adaptive online execution requires explicit eager eval mode')
        if self.binding.get_extra_state()!=self.config['binding']:raise ValueError('online binding identity changed')
        _ownership(self.shared_optimizer,self.binding.shared);_ownership(self.private_optimizer,self.binding.private)
        if _optimizer_settings(self.shared_optimizer)!=self.config['shared_optimizer'] or _optimizer_settings(self.private_optimizer)!=self.config['private_optimizer']:raise ValueError('online optimizer policy changed')

    def _prepare(self,snapshot,base):
        frozen=_copy_observation(snapshot);base={n:p.detach().clone() for n,p in base.items()}
        if not base or any(p.dtype!=torch.float32 for p in base.values()):raise ValueError('unwrapped FP32 stored parameters required')
        embeddings=self.config['embedding_names']
        if not set(embeddings).issubset(base):raise ValueError('embedding orientation inventory differs')
        engine=SimpleNamespace(parameters=base,transpose_storage={n:n in embeddings for n in base})
        grouped=GroupedNormObjective(base,zero_reference_policy='freeze');captured=[]
        # No detached factor leaves: credit must reach the actual shared AND private graph.
        with torch.enable_grad():
            proposal,_,stats,editable,_=build_adaptive_proposal(
                lambda snap:self.binding(snap,base,embedding_names=embeddings),frozen,engine,grouped,
                radius_fraction=self.config['radius_fraction'],shrinkage_coefficient=1.,detach_credit=False,
                execution_capture=lambda action,scale:captured.append((action,scale)),materialization_grad=False)
        action,scale=captured[0]
        # Captured factors already include group RMS and dimension normalization;
        # scale is exactly the offline radius guard, not a second normalization.
        guarded=list(self.binding.parameters())+_tensors(frozen)+list(base.values())
        guarded+=_tensors(dict(matrices={n:dict(U=f.U,C=f.C,V=f.V) for n,f in action.matrices.items()},vectors=action.vectors,scale=scale))
        versions=[(p,p._version) for p in guarded]
        return dict(snapshot=frozen,base=base,proposal=proposal,action=action,scale=scale,editable=editable,stats=stats,versions=versions)

    @staticmethod
    def _check_versions(prepared,*,live=True):
        guards=prepared['versions']+(prepared.get('live_versions',[]) if live else [])
        if any(p._version!=version for p,version in guards):raise RuntimeError('action-time actor/private/snapshot/base or committed live weights mutated before credit')

    def _record(self,prepared,live,step):
        prepared['record']=AdaptiveCreditRecord(self.lag,prepared['action'],prepared['scale'],action_step=step,
            hyperparameters=dict(self.binding.named_parameters()),guarded_observations=(_tensors(prepared['snapshot']),prepared['base']))
        prepared['action_step']=step;prepared['live_versions']=[(p,p._version) for p in live.values()]
        # Proposal storage already belongs to this action, never the live GPT.
        # Read-only entries may share immutable base storage; its version is
        # guarded above. Transfer ownership instead of another model-sized copy.
        proposal=prepared.pop('proposal')
        if any(p.requires_grad or p.grad_fn is not None for p in proposal.values()):
            raise RuntimeError('online materialized proposal must not retain a dense autograd graph')
        prepared['executed']={n:p.detach() for n,p in proposal.items()}
        self.pending=prepared

    def commit(self,snapshot,live_parameters,*,action_step,base_step_completed):
        self._authority();self._check_optimizers()
        if self.pending is not None or self.lag.pending:raise RuntimeError('consume pending action before commit')
        if self.terminal:raise RuntimeError('terminal identity already committed')
        if type(action_step) is not int or not 0<=action_step<=self.config['final_action_step'] or base_step_completed!=action_step:raise ValueError('commit requires explicit matching completed base optimizer step')
        if self.last_consumed_step is not None and action_step!=self.last_consumed_step:raise ValueError('commit must follow current next-gradient consume and base step')
        if self.last_committed_step is not None and action_step<=self.last_committed_step:raise ValueError('action clock regressed')
        if action_step==self.config['final_action_step']:
            self.terminal=True;self.last_committed_step=action_step
            return dict(terminal_identity=True,meta_updates=self.meta_updates,gpt_forwards=0,gpt_backwards=0)
        prepared=self._prepare(snapshot,live_parameters);proposal=prepared['proposal']
        if set(proposal)!=set(live_parameters) or any(p.shape!=live_parameters[n].shape or p.dtype!=live_parameters[n].dtype or not torch.isfinite(p).all() for n,p in proposal.items()):raise ValueError('invalid adaptive proposal; live weights unchanged')
        self._check_versions(prepared,live=False)
        with torch.no_grad():
            for n in prepared['editable']:live_parameters[n].copy_(proposal[n])
        self._record(prepared,live_parameters,action_step);self.last_committed_step=action_step
        return dict(prepared['stats'],terminal_identity=False,action_step=action_step,meta_updates=self.meta_updates,gpt_forwards=0,gpt_backwards=0)

    def rebind_after_broadcast(self,live_parameters):
        """Accept only version changes that leave every executed stored value exact.

        Called immediately after the collective committed-weight broadcast. All
        actor/private/snapshot/base guards remain enforced; no arbitrary mutation
        may be blessed by replacing the live version counters.
        """
        self._authority();self._check_optimizers()
        if self.pending is None:return
        self._check_versions(self.pending,live=False)
        expected=self.pending['executed']
        if set(live_parameters)!=set(expected) or any(p.dtype!=expected[n].dtype or p.shape!=expected[n].shape or
                p.device!=expected[n].device or not torch.equal(p.detach(),expected[n]) for n,p in live_parameters.items()):
            raise ValueError('post-broadcast live values differ from executed action')
        self.pending['live_versions']=[(p,p._version) for p in live_parameters.values()]

    def consume(self,gradients,*,gradient_step):
        self._authority();self._check_optimizers()
        if self.pending is None:raise RuntimeError('no pending adaptive action')
        self._check_versions(self.pending)
        projected=self.pending['record'].project(gradients,global_token_count=self.config['global_token_count'],
            loss_reduction='global_token_sum',transpose_storage=self.config['embedding_names'])
        result=self.lag.consume(projected,gradient_step=gradient_step)
        self.shared_optimizer.zero_grad(set_to_none=True);self.private_optimizer.zero_grad(set_to_none=True)
        result.assign_gradients(dict(self.binding.named_parameters()))
        shared_norm=torch.nn.utils.clip_grad_norm_(self.binding.shared.parameters(),self.config['shared_clip'],error_if_nonfinite=True)
        private_norm=torch.nn.utils.clip_grad_norm_(self.binding.private.parameters(),self.config['private_clip'],error_if_nonfinite=True)
        self.shared_optimizer.step();self.private_optimizer.step()
        self.pending=None;self.last_consumed_step=gradient_step;self.meta_updates+=1
        return dict(action_step=result.action_step,gradient_step=gradient_step,contraction=result.contraction_value,
            shared_preclip_norm=float(shared_norm),private_preclip_norm=float(private_norm),meta_updates=self.meta_updates,gpt_forwards=0,gpt_backwards=0)

    def state_dict(self):
        self._check_optimizers();pending=None
        if self.pending is not None:
            self._check_versions(self.pending)
            pending={k:_checkpoint_tensor_copy(self.pending[k]) for k in ('snapshot','base','executed','action_step')}
        return dict(schema='adaptive_online_lifecycle_v1',config=copy.deepcopy(self.config),binding=_checkpoint_tensor_copy(self.binding.state_dict()),
            shared_optimizer=copy.deepcopy(self.shared_optimizer.state_dict()),private_optimizer=copy.deepcopy(self.private_optimizer.state_dict()),pending=pending,
            last_consumed_step=self.last_consumed_step,last_committed_step=self.last_committed_step,meta_updates=self.meta_updates,terminal=self.terminal)

    def load_state_dict(self,saved,live_parameters):
        self._authority();self._check_optimizers()
        if self.pending is not None or self.lag.pending:raise RuntimeError('cannot restore over pending adaptive action')
        if saved['schema']!='adaptive_online_lifecycle_v1' or saved['config']!=self.config:raise ValueError('adaptive online resume configuration differs')
        for key,opt in (('shared_optimizer',self.shared_optimizer),('private_optimizer',self.private_optimizer)):
            if saved[key]['param_groups']!=opt.state_dict()['param_groups']:raise ValueError('saved optimizer settings differ')
        if type(saved['meta_updates']) is not int or saved['meta_updates']<0 or type(saved['terminal']) is not bool:raise ValueError('invalid saved lifecycle clocks')
        for key in ('last_consumed_step','last_committed_step'):
            if saved[key] is not None and (type(saved[key]) is not int or not 0<=saved[key]<=self.config['final_action_step']):raise ValueError('invalid saved action clock')
        if any(p.dtype!=torch.float32 for p in live_parameters.values()):raise ValueError('resume requires FP32 live storage')
        pending=saved['pending']
        if saved['terminal'] and saved['last_committed_step']!=self.config['final_action_step']:raise ValueError('invalid terminal action clock')
        if pending is not None:
            if saved['terminal'] or pending['action_step']!=saved['last_committed_step'] or not 0<=pending['action_step']<self.config['final_action_step']:raise ValueError('invalid pending action clock')
            if set(live_parameters)!=set(pending['executed']) or any(p.dtype!=pending['executed'][n].dtype or p.shape!=pending['executed'][n].shape or not torch.equal(p,pending['executed'][n].to(device=p.device)) for n,p in live_parameters.items()):raise ValueError('live weights differ from saved executed edit')
        device=next(self.binding.parameters()).device
        devices=[device.index or 0] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            self.binding.load_state_dict(saved['binding'],strict=True)
            self.shared_optimizer.load_state_dict(saved['shared_optimizer']);self.private_optimizer.load_state_dict(saved['private_optimizer'])
            if pending is not None:
                prepared=self._prepare(_checkpoint_tensor_copy(pending['snapshot'],device),_checkpoint_tensor_copy(pending['base'],device))
                if any(not torch.equal(value,pending['executed'][n].to(value)) for n,value in prepared['proposal'].items()):raise ValueError('reconstructed adaptive proposal differs from executed values')
                self._record(prepared,live_parameters,pending['action_step'])
        self.last_consumed_step=saved['last_consumed_step'];self.last_committed_step=saved['last_committed_step'];self.meta_updates=saved['meta_updates'];self.terminal=saved['terminal']
