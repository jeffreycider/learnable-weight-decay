"""Explicit adaptive query+actor capture, reusing coordinate graph ownership.

Dense student/loss/norm/Adam are outside. Live private QR participates in every
fresh-core query inside the captured autograd program. Base weights are immutable
private copies, rebound only between completed VJPs. This module is experimental.
"""
import torch
from .coordinate_actor_graph import CoordinateActorGraph,EditableCoordinateOutputs,actor_observation_tensors
from .adaptive_action import AdaptiveAction,AdaptiveFactors
from .adaptive_sensing import query_weight_cores


def sensed_parameters(engine):
    if hasattr(engine,'_all_parameters'):
        return {n:p.flatten(1) if p.ndim==4 else p for n,p in engine._all_parameters.items()}
    return engine.parameters


class EditableAdaptiveOutputs(EditableCoordinateOutputs):
    def __init__(self,actor,private,snapshot,editable,weights,embedding_names):
        super().__init__(actor,private,snapshot,editable)
        if actor.configuration().get('action_family')!='adaptive_ucv_v1' or not actor.fresh_weight_core:
            raise ValueError('explicit adaptive fresh-core actor required')
        self.weights=weights;self.embedding_names=tuple(embedding_names)

    def forward(self,dummy):
        cores=query_weight_cores(self.weights,self.private,embedding_names=self.embedding_names)
        action=self.actor(self.snapshot,self.private,weight_cores=cores)
        return tuple(t for n in self.matrix_names for t in (action.matrices[n].U,action.matrices[n].C,action.matrices[n].V))+tuple(action.vectors[n] for n in self.vector_names)

    def action(self,outputs):
        if len(outputs)!=3*len(self.matrix_names)+len(self.vector_names):raise ValueError('adaptive captured output layout changed')
        values=iter(outputs);matrices={};vectors={};L=self.actor.latent_tokens
        sample=self.private.matrix_rows[0].raw
        for name,obs in self.snapshot.matrices.items():
            if name in self.editable:u,c,v=next(values),next(values),next(values)
            else:u,c,v=(sample.new_zeros((obs.rows.shape[0],L)),sample.new_zeros((L,L)),sample.new_zeros((obs.columns.shape[0],L)))
            matrices[name]=AdaptiveFactors(u,c,v)
        for name,obs in self.snapshot.vectors.items():vectors[name]=next(values) if name in self.editable else sample.new_zeros(obs.weight.shape)
        return AdaptiveAction(matrices,vectors)


class AdaptiveActorGraph(CoordinateActorGraph):
    # FixedContexts must not issue a second eager query. The actual query remains
    # live inside this callable, and callers account its replays explicitly.
    fresh_weight_core=False
    captures_fresh_weight_core=True

    def __init__(self,actor,slot,snapshot,editable,parameters,*,embedding_names=(),warmups=2):
        self.embedding_names=tuple(embedding_names)
        if set(n for n,p in parameters.items() if p.ndim==2)!=set(snapshot.matrices):
            raise ValueError('fresh query dense matrix inventory differs')
        self.weights={n:p.detach().clone(memory_format=torch.preserve_format) for n,p in parameters.items() if p.ndim==2}
        self.weight_layout={n:(p.shape,p.stride(),p.dtype,p.device) for n,p in self.weights.items()}
        if any((p.shape,p.stride(),p.dtype,p.device)!=self.weight_layout[n] for n,p in parameters.items() if p.ndim==2):
            raise ValueError('query weight clone cannot preserve source layout')
        self._weight_versions=tuple((n,p._version,p.data_ptr()) for n,p in self.weights.items())
        self._weights_bound=True
        super().__init__(actor,slot,snapshot,editable,warmups=warmups)

    def _make_wrapper(self,actor,private,snapshot,editable):
        return EditableAdaptiveOutputs(actor,private,snapshot,editable,self.weights,self.embedding_names)

    def configuration(self):return self.slot.shared.configuration()

    def fresh_query_counts(self):
        return dict(fresh_weight_core_calls=1,fresh_weight_core_matrix_queries=len(self.weights),
                    fresh_weight_core_weight_elements_scanned=sum(p.numel() for p in self.weights.values()))

    def bind_context(self,context):
        engine=context['engine'];parameters=sensed_parameters(engine)
        embeddings=tuple(n for n,t in engine.transpose_storage.items() if t)
        key=(context['local_id'],getattr(engine,'teacher_id',None),embeddings,
             tuple((n,p.data_ptr(),p._version,p.shape,p.stride(),p.dtype,p.device)
                   for n,p in parameters.items() if p.ndim==2))
        if key!=getattr(self,'_source_key',None):
            self.bind_weights(parameters,embedding_names=embeddings);self._source_key=key
        else:self._check_weights()
        self._bound_observations=self._observation_signature(context['snapshot']) if 'snapshot' in context else None
        self._bound_context_key=self.slot.state_key if hasattr(self,'slot') else None

    @staticmethod
    def _observation_signature(snapshot):
        return tuple((n,p.data_ptr(),p._version,p.shape,p.stride(),p.dtype,p.device) for n,p in actor_observation_tensors(snapshot).items())

    def bind_scope(self,private,editable):
        if getattr(self,'_bound_context_key',None) is not None and private.state_key!=self._bound_context_key:
            raise RuntimeError('checkpoint/private identity changed after binding dense weights')
        return super().bind_scope(private,editable)

    def _check_weights(self):
        if hasattr(self,'_source_values') and tuple((n,p._version,p.data_ptr()) for n,p in self._source_values.items())!=self._source_versions:
            raise RuntimeError('source checkpoint changed; bind current context before replay')
        if not self._weights_bound or tuple((n,p._version,p.data_ptr()) for n,p in self.weights.items())!=self._weight_versions:
            raise RuntimeError('captured base weights changed outside verified rebind')

    def bind_weights(self,parameters,*,embedding_names):
        if self.pending:raise RuntimeError('cannot rebind weights while adaptive VJP is pending')
        self._weights_bound=False
        values={n:p for n,p in parameters.items() if p.ndim==2}
        if tuple(embedding_names)!=self.embedding_names or values.keys()!=self.weights.keys():raise ValueError('query orientation/inventory changed')
        for n,p in values.items():
            if (p.shape,p.stride(),p.dtype,p.device)!=self.weight_layout[n]:raise ValueError('query weight layout changed: '+n)
        with torch.no_grad():
            for n,p in values.items():self.weights[n].copy_(p)
        self._weight_versions=tuple((n,p._version,p.data_ptr()) for n,p in self.weights.items());self._weights_bound=True
        self._source_values=values
        self._source_versions=tuple((n,p._version,p.data_ptr()) for n,p in values.items())

    def _backward_started(self,gradient,generation=None):
        self._check_weights()
        return super()._backward_started(gradient,generation)

    def __call__(self,snapshot,private,*,weight_cores=None):
        if weight_cores is not None:raise ValueError('capture owns live fresh cores; external cores forbidden')
        self._check_weights()
        if getattr(self,'_bound_observations',None) is not None and self._observation_signature(snapshot)!=self._bound_observations:
            raise RuntimeError('snapshot changed after binding dense checkpoint weights')
        return super().__call__(snapshot,private)


def capture_adaptive_actor(actor,slot,context,editable,*,warmups=2):
    engine=context['engine']
    graph=AdaptiveActorGraph(actor,slot,context['snapshot'],editable,sensed_parameters(engine),
        embedding_names=tuple(n for n,t in engine.transpose_storage.items() if t),warmups=warmups)
    graph.bind_context(context)
    return graph
