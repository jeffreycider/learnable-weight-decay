"""Scoped eager-only instance wrappers for the isolated direct-factor prototype.

Not a tokenwise backend. Not thread-safe, compile-safe, or reentrant on overlapping
modules. Dense norm regularization and delayed actor-credit accumulation remain
the caller's responsibility. No parameters, buffers or hooks are replaced.
"""
from contextlib import contextmanager
import torch
from baseline.model import Linear
from .experimental_factor_linear import factor_linear,factor_embedding,_validate


@contextmanager
def factor_forwards(model,factors,*,chunk_tokens=128):
    """Route specified weight names to (U,C,V,scale) factor VJPs.

Module weight values are read DURING forward, after functional_call installs its
replacements. They are detached only for the functional loss: their separate
realized-norm graph may still be used outside this context. Bias is read from the
live module and retains its supplied graph, including scale/guard credit. Thus
callers must keep replacement biases connected (or accumulate bias-leaf credit).
Embedding factors use computational [dimension,vocabulary] orientation.

All replacements/factors must describe the SAME composed action; this wrapper
does not recompose weights to validate that expensive invariant. Unlisted
modules retain their ordinary behavior. Nested/overlapping contexts are rejected;
exit, including exceptions, restores exact original instance forward ownership.
"""
    modules=dict(model.named_modules());plan=[]
    for name,values in factors.items():
        if not name.endswith('.weight') and name!='weight':raise ValueError('factor key must name a weight')
        path=name[:-7] if name!='weight' else ''
        module=modules.get(path)
        if module is None or type(module) not in (Linear,torch.nn.Linear,torch.nn.Embedding):
            raise ValueError('only baseline Linear, plain Linear and plain Embedding are supported')
        if len(values)!=4:raise ValueError('expected U,C,V,scale')
        if isinstance(module,torch.nn.Embedding) and (module.padding_idx is not None or module.max_norm is not None or module.scale_grad_by_freq or module.sparse):
            raise ValueError('unsupported embedding options')
        u,c,v,scale=values
        weight=module.weight.detach().T if isinstance(module,torch.nn.Embedding) else module.weight.detach()
        _validate(weight,u,c,v,scale,chunk_tokens)
        plan.append((module,values))
    guarded=[model]+[m for m,_ in plan if m is not model]
    if any(hasattr(m,'_factor_forward_active') for m in guarded):raise ValueError('nested or overlapping factor contexts are unsupported')
    saved=[]
    try:
        for module in guarded:module._factor_forward_active=True
        for module,values in plan:
            saved.append((module,'forward' in module.__dict__,module.__dict__.get('forward')))
            def forward(x,_module=module,_values=values):
                u,c,v,scale=_values
                if isinstance(_module,torch.nn.Embedding):
                    return factor_embedding(x,_module.weight.detach(),u,c,v,scale,chunk_tokens=chunk_tokens)
                return factor_linear(x,_module.weight.detach().type_as(x),u,c,v,scale,_module.bias,chunk_tokens=chunk_tokens)
            module.forward=forward
        yield
    finally:
        for module,owned,old in reversed(saved):
            if owned:module.forward=old
            else:del module.forward
        for module in guarded:del module._factor_forward_active
