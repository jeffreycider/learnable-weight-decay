"""First-order streamed VJP for the exact realized grouped norm value.

No dense replacement gradient is constructed. The stored-dtype cast of each
norm cotangent happens BEFORE scale multiplication and thin contraction, matching
_RealizedParameterNorm's cast STE. Chunk reductions can reorder roundoff.
"""
import weakref
import torch
from torch.autograd.function import once_differentiable


def _value_identity(t):
    return (t.data_ptr(),tuple(t.shape),tuple(t.stride()),t.dtype,t.device,t._version)


def _reference_identity(grouped):
    tensors={**{'base:'+n:t for n,t in grouped.parameters.items()},
             **{'squared:'+n:t for n,t in grouped.reference_squared.items()},
             **{'norm:'+n:t for n,t in grouped.reference_norms.items()}}
    return (tuple((g,tuple(names)) for g,names in grouped.groups.items()),
            grouped.zero_reference_groups,grouped.chunk_rows,
            {n:(id(t),_value_identity(t)) for n,t in tensors.items()})


class MeasuredGroupedNorm:
    """One-use scalar-only measurement; weak identities never retain edit graphs."""
    def __init__(self,parameters,grouped,result):
        self.used=False;self.grouped=weakref.ref(grouped)
        self.reference=_reference_identity(grouped)
        self.values={n:(weakref.ref(t),_value_identity(t)) for n,t in parameters.items()}
        self.mean=result.mean_relative_change.detach().clone()
        self.norms={n:t.detach().clone() for n,t in result.new_norms.items()}
        self.scalars={n:(id(t),_value_identity(t)) for n,t in {'mean':self.mean,**{'norm:'+n:t for n,t in self.norms.items()}}.items()}

    def consume(self,values,grouped):
        if self.used:raise RuntimeError('norm measurement already consumed')
        self.used=True
        if self.scalars!={n:(id(t),_value_identity(t)) for n,t in {'mean':self.mean,**{'norm:'+n:t for n,t in self.norms.items()}}.items()}:raise RuntimeError('stale norm scalar measurement')
        if self.grouped() is not grouped or self.reference!=_reference_identity(grouped):raise RuntimeError('stale norm reference identity')
        if values.keys()!=self.values.keys():raise RuntimeError('stale norm replacement inventory')
        for name,t in values.items():
            original,identity=self.values[name]
            if original() is None or _value_identity(original())!=identity or _value_identity(t)!=identity:raise RuntimeError('stale norm replacement identity/version')
        mean=self.mean.clone();norms={n:t.clone() for n,t in self.norms.items()}
        self.mean=None;self.norms.clear();self.values.clear();self.scalars.clear();self.reference=None
        return mean,norms


class _ThinNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,values,grouped,layout,transpose,chunk_rows,shrinkage,measurement,*factors):
        # Function.forward runs without autograd: retain only stored values and
        # small factors, not a dense FP64 edit/materialization graph.
        if measurement is None:
            result=grouped.evaluate(values);mean=result.mean_relative_change;new_norms=result.new_norms
        else:mean,new_norms=measurement.consume(values,grouped)
        ctx.values=dict(values);ctx.grouped=grouped;ctx.layout=layout;ctx.transpose=transpose
        ctx.chunk_rows=chunk_rows;ctx.shrinkage=shrinkage;ctx.new_norms=new_norms
        ctx.value_versions={n:t._version for n,t in values.items()}
        ctx.save_for_backward(*factors)
        return shrinkage*(100*mean)

    @staticmethod
    @once_differentiable
    def backward(ctx,cotangent):
        if any(t._version!=ctx.value_versions[n] for n,t in ctx.values.items()):raise RuntimeError('rounded norm replacement mutated')
        grouped=ctx.grouped
        if any(t._version!=grouped._reference_versions[n] for n,t in grouped.parameters.items()):raise RuntimeError('norm reference mutated')
        factors=ctx.saved_tensors
        grads=[torch.zeros_like(t) if t.requires_grad else None for t in factors]
        def add(index,value):
            if grads[index] is not None:grads[index].add_(value)
        scale_index=ctx.layout['scale'];scale=factors[scale_index]
        # Match multiplication/division order of shrinkage*(100*mean(...)).
        group_credit=(cotangent*ctx.shrinkage)*100/len(grouped.groups)
        for group,names in grouped.groups.items():
            if group in grouped.zero_reference_groups:continue
            old=grouped.reference_norms[group];new=ctx.new_norms[group]
            derivative=torch.where(new>0,1/(2*old*new.clamp_min(torch.finfo(new.dtype).tiny)),torch.zeros_like(new))
            norm_credit=group_credit*derivative
            for name in names:
                stored=ctx.values[name];indices=ctx.layout[name]
                if stored.ndim==1:
                    vi=indices[0];vector=factors[vi]
                    for start in range(0,len(stored),ctx.chunk_rows):
                        sl=slice(start,start+ctx.chunk_rows)
                        g=(2*norm_credit*stored[sl].double()).to(stored.dtype).to(vector.dtype)
                        if grads[vi] is not None:grads[vi][sl].add_(g*scale)
                        add(scale_index,(g*vector[sl]).sum().reshape_as(scale))
                    continue
                ui,ci,vi=indices;u,c,v=(factors[i] for i in indices)
                uc=u@c  # Thin, never a full dense edit.
                transposed=ctx.transpose.get(name,False)
                for start in range(0,len(stored),ctx.chunk_rows):
                    sl=slice(start,start+ctx.chunk_rows)
                    g=(2*norm_credit*stored[sl].double()).to(stored.dtype).to(u.dtype)
                    if transposed:
                        g=g.T;vc=v[sl];scaled=g*scale;gv=scaled@vc
                        add(ui,gv@c.T);add(ci,u.T@gv)
                        if grads[vi] is not None:grads[vi][sl].add_(scaled.T@uc)
                        delta=uc@vc.T
                    else:
                        uc_slice=uc[sl];scaled=g*scale;gv=scaled@v
                        if grads[ui] is not None:grads[ui][sl].add_(gv@c.T)
                        add(ci,u[sl].T@gv);add(vi,scaled.T@uc_slice)
                        delta=uc_slice@v.T
                    add(scale_index,(g*delta).sum().reshape_as(scale))
        return (None,None,None,None,None,None,None,*grads)


def streamed_grouped_norm(parameters,grouped,action,scale,*,transpose_storage=None,chunk_rows=128,shrinkage_coefficient=1.,measurement=None):
    """Return (detached stored values, exact scalar with thin factor VJP).

Use the GROUP-RMS-SCALED action and shared scale from execution_capture. All
factor/vector/scale inputs must be FP64: this adapter models the explicit FP64
sum-then-stored-cast proposal only. Shared tensor identities are deduplicated.

To release the dense proposal graph, discard the old proposal regularizer and
parameter mapping after this call. Returned values are detached, including
vectors: a functional-loss backend must arrange separate vector credit (and
select active matrices using captured action, not value.requires_grad). Use prepare_factor_panel below to adapt this mapping to FactorPanelCapture.
This standalone adapter does not modify production or experimental panel code.
"""
    if type(chunk_rows) is not int or chunk_rows<1:raise ValueError('positive integer row chunk required')
    if not torch.isfinite(torch.tensor(shrinkage_coefficient)) or shrinkage_coefficient<=0:raise ValueError('positive finite shrinkage required')
    if scale.numel()!=1:raise ValueError('scalar shared scale required')
    if parameters.keys()!=grouped.parameters.keys():raise ValueError('replacement inventory mismatch')
    unique=[];identities={}
    def index(t):
        if t.dtype!=torch.float64:raise ValueError('FP64 factors, vectors and scale required')
        if id(t) not in identities:identities[id(t)]=len(unique);unique.append(t)
        return identities[id(t)]
    layout={'scale':index(scale)}
    for group,names in grouped.groups.items():
        if group in grouped.zero_reference_groups:continue
        for name in names:
            if parameters[name].ndim==2:
                f=action.matrices[name];layout[name]=tuple(index(t) for t in (f.U,f.C,f.V))
            else:layout[name]=(index(action.vectors[name]),)
    values={n:t.detach() for n,t in parameters.items()}
    return values,_ThinNorm.apply(values,grouped,layout,transpose_storage or {},chunk_rows,shrinkage_coefficient,measurement,*unique)


def prepare_factor_panel(parameters,grouped,capture,*,transpose_storage=None,chunk_rows=128,shrinkage_coefficient=1.):
    """Optional adapter for existing FactorPanelCapture; default path is unchanged.

After build_adaptive_proposal(..., execution_capture=capture), assign
    parameters, regularizer = prepare_factor_panel(parameters, grouped, capture,
        transpose_storage=engine.transpose_storage)
then call capture.run with unchanged actor_credit and other panel arguments.
The assignment must discard BOTH the old parameters and old regularizer before
run. Matrix routing flags are carried by detached leaf values (no dense edit
graph); original vector parameter graphs remain for functional vector credit.
The new regularizer supplies all matrix AND vector norm credit exactly once.
The optional builder hook supplies a one-use measured norm, avoiding a second
grouped.evaluate scan. Standalone calls without that measurement recompute.
Dense stored values and their materialization forward remain required. With
materialization_grad=False the builder never creates the dense edit graph; this
adapter restores matrix routing and tiny vector graphs from the captured action.
The default preparation still discards the originally constructed dense graph.
"""
    if capture.phase!='captured' or getattr(capture,'_streamed_norm_prepared',False):raise RuntimeError('fresh captured proposal required for norm preparation')
    try:
        capture._check_versions()
        values,norm=streamed_grouped_norm(parameters,grouped,capture.action,capture.scale,
            transpose_storage=transpose_storage,chunk_rows=chunk_rows,shrinkage_coefficient=shrinkage_coefficient,
            measurement=getattr(capture,'norm_measurement',None))
        if capture.detached_editable is None:
            prepared={n:values[n].requires_grad_(p.requires_grad) if p.ndim==2 else p for n,p in parameters.items()}
        else:
            # Dense values have no trainability flags/graphs in this mode.
            # Restore only matrix routing and the small functional vector graph;
            # norm credit still uses the original authoritative stored values.
            prepared={}
            for n,p in parameters.items():
                if n not in capture.detached_editable:
                    prepared[n]=values[n]
                elif p.ndim==2:
                    f=capture.action.matrices[n]
                    active=any(t.requires_grad for t in (f.U,f.C,f.V,capture.scale))
                    prepared[n]=values[n].requires_grad_(active)
                else:
                    prepared[n]=(grouped.parameters[n].double()+capture.scale*capture.action.vectors[n]).to(p.dtype)
    except Exception:
        capture.phase='consumed';capture.action=None;capture.scale=None;capture.detached_editable=None
        capture.originals.clear();capture.credit_leaves.clear();capture.versions.clear()
        raise
    finally:
        capture.norm_measurement=None
    capture._streamed_norm_prepared=True
    return prepared,norm
