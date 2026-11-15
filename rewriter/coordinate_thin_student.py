"""Experimental strict dense-forward GPT with thin action cotangents.

No compiler or precision override. The existing tokenwise panel owns teacher
validation, all loss/metric arithmetic, microbatch accounting and the final actor
VJP. Only dense GPT weight gradients are replaced by anchored contractions.
This changes FP32 backward reduction order and needs finite-update qualification.
"""
import torch
from torch.autograd.function import once_differentiable
from .algebra import RawAction, Factors
from .factorized_fp32 import factorized_logits
from .tokenwise_reference import tokenwise_panel


class ProposalCapture:
    """Read-through engine adapter for one existing build_proposal invocation."""
    def __init__(self, engine):
        if engine.gpt_precision!='fp32':
            raise ValueError('strict FP32 GPT required')
        self.engine=engine
        self.action=self.scale=None

    def __getattr__(self,name):
        return getattr(self.engine,name)

    def edited_parameters(self,action,scale):
        if self.action is not None:
            raise RuntimeError('one proposal per capture')
        self.action,self.scale=action,scale
        return self.engine.edited_parameters(action,scale)


class _ThinCredit(torch.autograd.Function):
    """Apply accumulated student cotangents alongside the unchanged norm VJP.

    The panel runs microbatch backwards into independent thin leaves first.
    Its one final norm backward supplies the checkpoint weight exactly once.
    No model graph is retained across microbatches and no dense dW is built.
    """
    @staticmethod
    def forward(ctx,regularizer,leaves,*originals):
        ctx.leaves=leaves
        return regularizer.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx,weight):
        return (weight,None,*[None if leaf.grad is None else leaf.grad*weight for leaf in ctx.leaves])


class _ThinForward:
    def __init__(self,engine,action,scale,realized):
        self.engine,self.action,self.scale,self.realized=engine,action,scale,realized

    def __getattr__(self,name):
        return getattr(self.engine,name)

    def _forward(self,parameters,inputs):
        return factorized_logits(self.engine.view.gpt,self.engine.parameters,self.engine.buffers,
            inputs,self.action,self.scale,realized_parameters=self.realized,
            linear_backend='ieee',dx_backend='ieee',softcap_backend='native',activation_backend='native')


def thin_tokenwise_panel(capture,parameters,regularizer,inputs,**kwargs):
    """Same public result as tokenwise_panel, explicit experimental call only.

    Pass a ProposalCapture to the unchanged build_proposal, then pass that
    capture here with its returned parameters/regularizer/actor_credit. Neither
    the engine nor the production defaults are mutated.
    """
    if capture.action is None or capture.scale is None:
        raise ValueError('capture the actual stored proposal before student execution')
    if capture.engine.kl_dtype!=torch.float64:
        raise ValueError('exact FP64 teacher target required')
    backward=kwargs.get('backward',True);originals=[];leaves=[]
    def leaf(value):
        result=value.detach().requires_grad_(value.requires_grad and backward)
        originals.append(value);leaves.append(result)
        return result
    action=RawAction({n:Factors(leaf(f.P),leaf(f.Q),f.anchors) for n,f in capture.action.matrices.items()},
                     {n:leaf(v) for n,v in capture.action.vectors.items()})
    scale=leaf(capture.scale)
    realized={n:value.detach() for n,value in parameters.items()}
    if backward:
        regularizer=_ThinCredit.apply(regularizer,leaves,*originals)
    engine=_ThinForward(capture.engine,action,scale,realized)
    # Detached stored weights deliberately expose no dense parameter leaves to
    # tokenwise_panel. Norm and student thin credit meet in _ThinCredit above.
    return tokenwise_panel(engine,realized,regularizer,inputs,**kwargs)
