"""One-shot experimental direct-factor credit using the unchanged tokenwise panel.

The deferred scalar has the norm's VALUE but the executed panel loss's factor
VJP. It is not a standalone differentiable norm function. This explicit boundary
injects previously accumulated functional-loss credit exactly once, together
with normal norm and vector credit. No higher-order gradients are supported.
"""
import torch
from torch.autograd.function import once_differentiable
from .experimental_factor_forwards import factor_forwards
from .tokenwise_reference import tokenwise_panel


class _DeferredPanelCredit(torch.autograd.Function):
    @staticmethod
    def forward(ctx,regularizer,owner,*originals):
        ctx.owner=owner
        ctx.save_for_backward(*originals)
        return regularizer.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent):
        owner=ctx.owner
        if owner.phase!='running' or owner.injected:raise RuntimeError('deferred panel credit is one-shot and only valid during execution')
        _=ctx.saved_tensors  # Enforce autograd's original-tensor mutation checks.
        owner._check_versions()
        owner.injected=True
        credit=[None if leaf.grad is None else leaf.grad.detach()*cotangent for leaf in owner.credit_leaves]
        return (cotangent,None,*credit)


class FactorPanelCapture:
    """Pass as build_adaptive_proposal(execution_capture=capture), then run once.

run() mirrors tokenwise_panel's arguments. Keep the returned proposal parameters,
regularizer and actor_credit unchanged. Bias/vector leaves use the ordinary panel
path; only matrices lose their dense functional cotangents. Shared scales/factors
are deduplicated by identity. prepare_factor_panel optionally supplies streamed
norm credit and restores routing/vector credit after detached materialization.

This is eager first-order GPT FP32 research code, not a production backend.
Failures consume the capture; partial backward must never be retried. Neither
captured tensors nor their source actor/private parameters may be mutated between
capture and completion. Do not use a capture across optimizer steps or panels.
"""
    def __init__(self,*,chunk_tokens=128):
        if type(chunk_tokens) is not int or chunk_tokens<1:raise ValueError('positive integer chunk_tokens required')
        self.chunk_tokens=chunk_tokens;self.phase='empty';self.injected=False
        self.action=None;self.scale=None;self.norm_measurement=None;self.detached_editable=None;self.originals=[];self.credit_leaves=[];self.versions=[]

    def __call__(self,action,scale):
        if self.phase!='empty':raise RuntimeError('capture is one-shot')
        self.action=action;self.scale=scale;self.phase='captured'
        self.versions=[(t,t._version) for f in action.matrices.values() for t in (f.U,f.C,f.V)]+[(t,t._version) for t in action.vectors.values()]+[(scale,scale._version)]

    def capture_detached_materialization(self,editable):
        if self.phase!='captured' or self.detached_editable is not None:raise RuntimeError('detached materialization capture is one-shot')
        self.detached_editable=frozenset(editable)

    def capture_norm_measurement(self,parameters,grouped,result):
        if self.phase!='captured' or self.norm_measurement is not None:raise RuntimeError('norm measurement capture is one-shot')
        from .experimental_factor_norm import MeasuredGroupedNorm
        self.norm_measurement=MeasuredGroupedNorm(parameters,grouped,result)

    def _check_versions(self):
        if any(t._version!=version for t,version in self.versions):raise RuntimeError('captured factor mutated')

    def run(self,engine,parameters,regularizer,inputs,**kwargs):
        if self.phase!='captured':raise RuntimeError('capture must be fresh and populated exactly once')
        self.phase='running'
        try:
            self._check_versions()
            if self.detached_editable is not None and not getattr(self,'_streamed_norm_prepared',False):raise RuntimeError('detached materialization requires prepare_factor_panel before run')
            if engine.gpt_precision!='fp32':raise ValueError('experimental panel requires FP32 GPT')
            if not kwargs.get('backward',True):
                return tokenwise_panel(engine,parameters,regularizer,inputs,**kwargs)
            if not torch.is_grad_enabled():raise ValueError('backward panel requires enabled autograd')
            matrix_names={n for n,p in parameters.items() if p.ndim==2}
            if matrix_names!=set(self.action.matrices):raise ValueError('captured matrix inventory differs')
            deduplicated={}
            def leaf(t):
                if id(t) not in deduplicated:
                    result=t.detach().clone().requires_grad_(t.requires_grad)
                    deduplicated[id(t)]=result
                    if t.requires_grad:self.originals.append(t);self.credit_leaves.append(result)
                return deduplicated[id(t)]
            factors={}
            for name in sorted(matrix_names):
                if parameters[name].requires_grad:
                    f=self.action.matrices[name]
                    factors[name]=tuple(leaf(t) for t in (f.U,f.C,f.V,self.scale))
            wrapped=_DeferredPanelCredit.apply(regularizer,self,*self.originals)
            values={n:p.detach() if n in matrix_names else p for n,p in parameters.items()}
            with factor_forwards(engine.view.gpt,factors,chunk_tokens=self.chunk_tokens):
                result=tokenwise_panel(engine,values,wrapped,inputs,**kwargs)
            if not self.injected:raise RuntimeError('panel did not execute deferred credit')
            return result
        finally:
            self.phase='consumed';self.action=None;self.scale=None;self.norm_measurement=None;self.detached_editable=None
            self.originals.clear();self.credit_leaves.clear();self.versions.clear()
