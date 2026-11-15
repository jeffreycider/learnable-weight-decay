"""Experimental eager CUDA graphs for one coordinate actor forward and VJP.

Only the actor is captured. Student models, loss, clipping and both Adam owners
stay outside. One static private slot/layout is rebound between complete visits.
The editable output subset is captured so unused read-only decoders keep None
rather than fabricated zero gradients and therefore retain their Adam clocks.
"""
from dataclasses import replace
import copy
import torch
from .algebra import RawAction,Factors
from .coordinate_chisel import observation_layout
from .graph_offline import clone_static_snapshot


def actor_trainability_signature(actor):
    """Named mask only: parameter values may change without recapture.

    Joint heads-only and private-only frozen captures are distinct programs even
    when they share the same Parameter objects and observation layout.
    """
    return tuple((name, bool(parameter.requires_grad))
                 for name, parameter in actor.named_parameters())


def actor_observation_tensors(snapshot):
    values={}
    for name,obs in snapshot.matrices.items():
        for field,value in [('rows',obs.rows),('columns',obs.columns),('A',obs.anchors.A),('B',obs.anchors.B)]:values[f'matrix:{name}:{field}']=value
    for name,obs in snapshot.vectors.items():values[f'vector:{name}']=obs.coordinates
    values['trajectories']=snapshot.trajectories
    if snapshot.trajectory_mask is not None:values['trajectory_mask']=snapshot.trajectory_mask
    return values


class EditableCoordinateOutputs(torch.nn.Module):
    """Private capture wrapper; original actor/private ownership is unchanged."""
    def __init__(self,actor,private,snapshot,editable):
        super().__init__();self.actor=actor;self.private=private;self.snapshot=snapshot
        names=set(snapshot.matrices)|set(snapshot.vectors)
        if not editable or not set(editable)<=names:raise ValueError('nonempty valid editable output set required')
        self.editable=frozenset(editable)
        self.matrix_names=tuple(n for n in snapshot.matrices if n in editable)
        self.vector_names=tuple(n for n in snapshot.vectors if n in editable)

    def forward(self,dummy):
        action=self.actor(self.snapshot,self.private)
        return tuple(t for name in self.matrix_names for t in (action.matrices[name].P,action.matrices[name].Q))+tuple(action.vectors[n] for n in self.vector_names)

    def action(self,outputs):
        if len(outputs)!=2*len(self.matrix_names)+len(self.vector_names):raise ValueError('captured output layout changed')
        values=iter(outputs);matrices={};vectors={}
        for name,obs in self.snapshot.matrices.items():
            p,q=(next(values),next(values)) if name in self.editable else (torch.zeros_like(obs.anchors.A),torch.zeros_like(obs.anchors.B))
            matrices[name]=Factors(p,q,obs.anchors)
        for name,obs in self.snapshot.vectors.items():vectors[name]=next(values) if name in self.editable else torch.zeros_like(obs.weight)
        return RawAction(matrices,vectors)


class CoordinateActorGraph:
    def _make_wrapper(self,actor,private,snapshot,editable):
        return EditableCoordinateOutputs(actor,private,snapshot,editable)

    def __init__(self,actor,slot,snapshot,editable,*,warmups=2):
        if next(actor.parameters()).device.type!='cuda':raise ValueError('coordinate actor graph requires CUDA')
        if slot.shared is not actor:raise ValueError('slot shared owner differs')
        slot.require_bound(slot.state_key)
        if any(p.grad is not None for p in (*slot.shared_parameters,*slot.private_parameters)):raise ValueError('capture before accumulating gradients')
        static=clone_static_snapshot(snapshot)
        if snapshot.trajectory_mask is not None:static=replace(static,trajectory_mask=snapshot.trajectory_mask.detach().clone())
        self.slot=slot;self.snapshot=static;self.layout=observation_layout(snapshot);self.editable=frozenset(editable)
        self.shared_trainability=actor_trainability_signature(actor)
        self.private_trainability=actor_trainability_signature(slot.private)
        self.capture_mask_signature=(self.shared_trainability,self.private_trainability)
        self._actor_configuration=copy.deepcopy(actor.configuration())
        self._actor_tokens=dict(actor.token_counts(snapshot))
        self._parameters=tuple((id(p),p.data_ptr(),p.requires_grad) for p in (*actor.parameters(),*slot.private.parameters()))
        self._inputs=actor_observation_tensors(static);self._input_layout={n:(v.shape,v.stride(),v.dtype,v.device) for n,v in actor_observation_tensors(snapshot).items()}
        actor.prepare_static_indices(static)
        # The actor cache may be rebound by another capture of this same layout.
        # CUDA graphs retain raw pointers, not Python ownership of these external
        # constants. Keep this capture's exact index allocations alive.
        self._index_owners=actor._static_indices(static,next(actor.parameters()).device)
        self.wrapper=self._make_wrapper(actor,slot.private,static,editable)
        self.dummy=torch.zeros((),device=next(actor.parameters()).device)
        self._wrapper_training=self.wrapper.training
        self._module_training=tuple((name,module.training) for name,module in self.wrapper.named_modules())
        self.call=torch.cuda.make_graphed_callables(self.wrapper,(self.dummy,),num_warmup_iters=warmups,allow_unused_input=True)
        self.pending=False;self.replays=0;self._scope_key=None;self._generation=0

    def bind_scope(self,private,editable):
        if self.pending:raise RuntimeError('previous actor VJP is pending')
        self._check_architecture(self.snapshot)
        self._check_trainability()
        if private is not self.slot.private or frozenset(editable)!=self.editable:raise ValueError('captured editable mask/private container changed')
        self.slot.require_bound(private.state_key);self._scope_key=private.state_key

    def _check_trainability(self):
        current=(actor_trainability_signature(self.slot.shared),
                 actor_trainability_signature(self.slot.private))
        if current!=self.capture_mask_signature:
            raise RuntimeError('captured trainability mask changed; use a separately captured program')

    def _check_architecture(self,snapshot):
        # Encoder- and decoder-observation stages can have identical parameter
        # addresses/shapes while executing different token sequences. A Python
        # attribute mutation must never silently reuse the old captured graph.
        actor=self.slot.shared
        if hasattr(self,'_wrapper_training') and self.wrapper.training!=self._wrapper_training:
            raise RuntimeError('captured wrapper training mode changed')
        if hasattr(self,'_module_training') and tuple((name,module.training) for name,module in self.wrapper.named_modules())!=self._module_training:
            raise RuntimeError('captured actor/private submodule training mode changed')
        if actor.configuration()!=self._actor_configuration or actor.token_counts(snapshot)!=self._actor_tokens:
            raise RuntimeError('captured actor architecture/token layout changed; create a new capture')

    def _backward_started(self,gradient,generation=None):
        if generation is not None and (not self.pending or generation!=self._generation):
            raise RuntimeError('stale actor graph output or repeated backward')
        if generation is not None:
            current=tuple((p._version,p.data_ptr(),p.requires_grad) for p in (*self.slot.shared_parameters,*self.slot.private_parameters))
            if current!=self._pending_versions:raise RuntimeError('actor/private parameters changed before pending VJP')
        if generation is None or not self._backward_scheduled:
            self._backward_scheduled=True
            torch.autograd.Variable._execution_engine.queue_callback(self._backward_complete)
        return gradient

    def _backward_complete(self):
        # AccumulateGrad may adopt a returned graph gradient when .grad is None.
        # That tensor aliases the replay buffer, which the next state overwrites.
        # Own the accumulated shared values before permitting another replay.
        # Private gradients are consumed/cleared per visit and are not SUMed.
        for parameter in self.slot.shared_parameters:
            if parameter.grad is not None:parameter.grad=parameter.grad.clone()
        self.pending=False

    def __call__(self,snapshot,private):
        if self.pending:raise RuntimeError('finish the previous actor VJP before replay or rebinding')
        self._check_architecture(snapshot)
        self._check_trainability()
        if any(p.grad is not None for p in self.slot.private_parameters):
            raise RuntimeError('clear private gradients before actor graph replay')
        if private is not self.slot.private:raise ValueError('private parameter container changed')
        self.slot.require_bound(private.state_key)
        if self._scope_key!=private.state_key:raise RuntimeError('bind the current editable scope before each replay')
        self._scope_key=None
        if observation_layout(snapshot)!=self.layout:raise ValueError('observation layout changed')
        signature=tuple((id(p),p.data_ptr(),p.requires_grad) for p in (*self.slot.shared.parameters(),*private.parameters()))
        if signature!=self._parameters:raise RuntimeError('actor/private parameter storage or gradient ownership changed')
        values=actor_observation_tensors(snapshot)
        if values.keys()!=self._inputs.keys():raise ValueError('observation mask schema changed')
        for name,value in values.items():
            if (value.shape,value.stride(),value.dtype,value.device)!=self._input_layout[name]:raise ValueError('observation tensor layout changed: '+name)
        with torch.no_grad():
            for name,value in values.items():self._inputs[name].copy_(value)
        outputs=self.call(self.dummy);self.replays+=1
        if torch.is_grad_enabled() and any(v.requires_grad for v in outputs):
            self.pending=True;self._generation+=1;generation=self._generation
            self._pending_versions=tuple((p._version,p.data_ptr(),p.requires_grad) for p in (*self.slot.shared_parameters,*self.slot.private_parameters))
            self._backward_scheduled=False
            for output in outputs:
                if output.requires_grad:output.register_hook(lambda gradient:self._backward_started(gradient,generation))
        return self.wrapper.action(outputs)


class PrivateActorDispatch:
    """Default-off private-only captures; caller owns freeze/Adam/dual lifecycle.

    Construct at a clean boundary before any gradient accumulation. Shared
    requires_grad flags are restored after capture. During each use the caller
    must freeze shared parameters, as in the ordinary private-inner loop.
    """
    def __init__(self,actor,slots,templates,*,backend='eager_actor_graph'):
        from .edit_scope import editable_names
        parameters=tuple(actor.parameters());flags=tuple(p.requires_grad for p in parameters)
        if any(p.grad is not None for p in parameters) or any(p.grad is not None for slot in slots.values() for p in slot.private_parameters):
            raise ValueError('private captures require a clean gradient boundary')
        self.actor=actor;self.graphs={};self.replays=0;self.eager_fallbacks=0
        try:
            actor.requires_grad_(False)
            for domain,slot in slots.items():
                context=templates[domain]
                editable=editable_names(context['engine'].parameters,'grouped')-context['grouped'].frozen_parameter_names
                if backend=='adaptive_actor_graph':
                    from .adaptive_actor_graph import capture_adaptive_actor
                    self.graphs[domain]=capture_adaptive_actor(actor,slot,context,editable,warmups=2)
                elif backend=='eager_actor_graph':
                    self.graphs[domain]=CoordinateActorGraph(actor,slot,context['snapshot'],editable,warmups=2)
                else:raise ValueError('unknown private actor graph backend')
                slot.private.zero_grad(set_to_none=True)
        finally:
            for parameter,flag in zip(parameters,flags):parameter.requires_grad_(flag)

    def for_visit(self,domain,context,private):
        from .edit_scope import editable_names
        if any(p.requires_grad for p in self.actor.parameters()):
            raise RuntimeError('freeze shared parameters before private-only graph replay')
        graph=self.graphs[domain]
        editable=editable_names(context['engine'].parameters,'grouped')-context['grouped'].frozen_parameter_names
        if frozenset(editable)!=graph.editable:
            self.eager_fallbacks+=1
            return self.actor
        if hasattr(graph,'bind_context'):graph.bind_context(context)
        graph.bind_scope(private,editable);self.replays+=1
        return graph
