"""Generic direct lag-one hypergradients through optimizer/controller algebra.

One executed update is pending at a time. The next ordinary task backward supplies
cotangents; this module never evaluates a task model or differentiates through an
earlier task gradient. This is direct one-step credit, not a trajectory gradient.

`from_outputs` accepts differentiable committed weights or a composed actual-next-
forward map, including optimizer wrappers. `from_contraction` supports thin-factor
VJPs without materializing differentiable dense edits. Its callback must contain
only optimizer/controller algebra; arbitrary Python callbacks cannot be audited
for model evaluation by this utility.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Callable
import torch


def frozen_copy(value):
    """Detach AND copy observation/cotangent trees; no aliases to live buffers."""
    if isinstance(value,torch.Tensor):return value.detach().clone()
    if isinstance(value,Mapping):return {k:frozen_copy(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(frozen_copy(v) for v in value)
    if isinstance(value,list):return [frozen_copy(v) for v in value]
    raise TypeError('tensor trees may contain only tensors, mappings, tuples, and lists')


def tensor_leaves(value):
    if isinstance(value,torch.Tensor):return [value]
    if isinstance(value,Mapping):return [t for v in value.values() for t in tensor_leaves(v)]
    if isinstance(value,(list,tuple)):return [t for v in value for t in tensor_leaves(v)]
    raise TypeError('expected tensor tree')


@dataclass(frozen=True)
class _Version:
    name: str
    tensor: torch.Tensor
    version: int

    def check(self):
        if self.tensor._version!=self.version:
            raise RuntimeError(f'{self.name} mutated between executed update and delayed credit')


@dataclass(frozen=True)
class HypergradientResult:
    action_step: int
    gradient_step: int
    contraction_value: float
    gradients: dict[str,torch.Tensor | None]
    _parameters: tuple[_Version,...]

    def assign_gradients(self,parameters):
        """Set only the named hyperparameter grads; unused grads remain None.

        The caller owns clipping and optimizer.step. This does not advance Adam
        or mutate hyperparameters, and may not be applied after another update.
        """
        if set(parameters)!=set(self.gradients):raise ValueError('hyperparameter names differ')
        for guard in self._parameters:
            guard.check()
            if parameters[guard.name] is not guard.tensor:raise ValueError('hyperparameter identity differs')
        for name,gradient in self.gradients.items():
            parameters[name].grad=None if gradient is None else gradient.clone()


class LagOneHypergradient:
    """Single pending action graph with explicit action/feedback clocks."""
    def __init__(self):
        self._pending=None

    @property
    def pending(self):return self._pending is not None

    def from_contraction(self,*,action_step,hyperparameters,contraction:Callable,guarded_tensors=()):
        """Record an EXECUTED action, not an unexecuted/warmup proposal.

        `contraction(detached_next_gradients)` returns the scalar <g_next,q(phi)>.
        q is the actual next-forward parameter representation. Include all guard,
        cast, and wrapper derivatives intended by the differentiable convention.
        Build the action graph from `frozen_copy` observations and optionally pass
        those tensors as `guarded_tensors` to catch later accidental mutation.
        """
        if self.pending:raise RuntimeError('consume the one pending update before recording another')
        if not isinstance(action_step,int) or action_step<0:raise ValueError('nonnegative action step required')
        parameters=dict(hyperparameters)
        if not parameters or any(not isinstance(n,str) or not n for n in parameters):raise ValueError('named hyperparameters required')
        if len({id(p) for p in parameters.values()})!=len(parameters):raise ValueError('name each shared hyperparameter exactly once')
        if any(not isinstance(p,torch.Tensor) or not p.requires_grad for p in parameters.values()):raise ValueError('hyperparameters must require gradients')
        guards=tuple(_Version(n,p,p._version) for n,p in parameters.items())
        observations=tuple(_Version('frozen observation '+str(i),p,p._version) for i,p in enumerate(tensor_leaves(guarded_tensors)))
        self._pending=(action_step,parameters,contraction,guards,observations)

    def from_outputs(self,*,action_step,hyperparameters,outputs,compose_forward=None,guarded_tensors=()):
        """Record named optimizer outputs and optionally their wrapper composition.

        compose_forward is evaluated ONCE at action time and must perform only
        optimizer algebra. It can contain learnable wrapper coefficients, which
        must be listed among hyperparameters. The resulting names/shapes must
        match the next ordinary task gradient's parameter representation.
        """
        if self.pending:raise RuntimeError('consume the one pending update before recording another')
        outputs=dict(outputs)
        forward=dict(compose_forward(outputs) if compose_forward else outputs)
        if not forward or any(not isinstance(v,torch.Tensor) for v in forward.values()):raise ValueError('named tensor outputs required')
        def contraction(cotangents):
            if not isinstance(cotangents,Mapping) or set(cotangents)!=set(forward):raise ValueError('next-gradient names differ from forward outputs')
            terms=[]
            for name,value in forward.items():
                gradient=cotangents[name]
                if not isinstance(gradient,torch.Tensor) or gradient.shape!=value.shape:raise ValueError('next-gradient shape mismatch for '+name)
                if gradient.device!=value.device:raise ValueError('next-gradient device mismatch for '+name)
                terms.append((value*gradient).sum())
            return sum(terms)
        self.from_contraction(action_step=action_step,hyperparameters=hyperparameters,contraction=contraction,
                              guarded_tensors=(guarded_tensors,forward))

    def consume(self,next_gradients,*,gradient_step):
        """Return detached named hypergradients; no task forward/backward is run."""
        if not self.pending:raise RuntimeError('no executed update is awaiting credit')
        action_step,parameters,contraction,guards,observations=self._pending
        if gradient_step!=action_step+1:raise ValueError('lag-one credit requires the next ordinary training step')
        for guard in (*guards,*observations):guard.check()
        # Calling autograd on the auxiliary scalar cannot traverse the GPT graph.
        cotangents=frozen_copy(next_gradients)
        scalar=contraction(cotangents)
        if not isinstance(scalar,torch.Tensor) or scalar.numel()!=1:raise ValueError('contraction must return a scalar tensor')
        if not bool(torch.isfinite(scalar.detach())):raise FloatingPointError('nonfinite delayed-credit contraction')
        if scalar.requires_grad:
            gradients=torch.autograd.grad(scalar,tuple(parameters.values()),allow_unused=True)
        else:gradients=(None,)*len(parameters)
        self._pending=None
        if any(g is not None and not bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError('nonfinite lag-one hypergradient; no optimizer update was performed')
        result=HypergradientResult(action_step,gradient_step,float(scalar.detach()),
            {name:None if gradient is None else gradient.detach().clone() for name,gradient in zip(parameters,gradients)},guards)
        return result
