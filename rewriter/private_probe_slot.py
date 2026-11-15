"""One stable private-probe device container, rebound between complete visits.

This module does not own an optimizer, distributed collective or checkpoint
store. The caller verifies the bank record, restores that state's private Adam,
and clears consumed private gradients before staging the next state. Shared
weighted gradients may remain accumulated across visits. Compiled calls receive
``slot.private``; identity checks happen outside the compiled function.
"""
from __future__ import annotations
import copy
import math
from collections.abc import Mapping
import torch
from .coordinate_chisel import CheckpointProbes


def _identity(key):
    if not isinstance(key,tuple) or not key or any(not isinstance(v,(str,int)) for v in key):
        raise ValueError('explicit immutable private checkpoint identity tuple required')
    return key


class PrivateProbeSlot:
    def __init__(self,shared,private):
        if not isinstance(private,CheckpointProbes):
            raise TypeError('CheckpointProbes container required')
        self.shared=shared;self.private=private
        self.shared_parameters=tuple(shared.parameters())
        self.private_parameters=tuple(private.parameters())
        if not self.shared_parameters or not self.private_parameters:
            raise ValueError('nonempty shared and private parameter sets required')
        shared_ids={id(p) for p in self.shared_parameters}
        shared_storage={p.untyped_storage().data_ptr() for p in self.shared_parameters}
        if any(id(p) in shared_ids or p.untyped_storage().data_ptr() in shared_storage for p in self.private_parameters):
            raise ValueError('shared/private parameter storage must be disjoint')
        self._tensors=dict(private.named_parameters())|dict(private.named_buffers())
        self._schema=copy.deepcopy(private.get_extra_state());self._schema.pop('state_key')
        self._signature=self.storage_signature()
        self._bound_key=_identity(private.state_key)
        self._unweighted=False

    @property
    def state_key(self):return self._bound_key

    def storage_signature(self):
        return tuple((name,id(value),value.data_ptr(),tuple(value.shape),value.dtype,value.device)
                     for name,value in (dict(self.private.named_parameters())|dict(self.private.named_buffers())).items())

    def require_bound(self,state_key):
        """Host-side identity/address guard; never put this in the compiled call."""
        if _identity(state_key)!=self._bound_key or self.private.state_key!=self._bound_key:
            raise ValueError('private slot checkpoint identity mismatch or incomplete staging')
        if self.storage_signature()!=self._signature:
            raise RuntimeError('private slot tensor addresses/layout changed')
        if tuple(map(id,self.shared.parameters()))!=tuple(map(id,self.shared_parameters)):
            raise RuntimeError('shared parameter ownership changed')
        return self.private

    @torch.no_grad()
    def stage(self,state,*,state_key):
        """Validate the entire source before any copy; preserve all tensor addresses.

        The explicit route key must equal the source's own serialized key. A
        different current slot key is expected during reuse and is changed only
        after successful copies. Private optimizer restoration remains separate.
        """
        key=_identity(state_key)
        self.require_bound(self._bound_key)
        if any(p.grad is not None for p in self.private_parameters):
            raise RuntimeError('consume and clear private gradients before staging another visit')
        if not isinstance(state,Mapping) or set(state)!=set(self._tensors)|{'_extra_state'}:
            raise ValueError('private state tensor names/extra state differ')
        extra=state['_extra_state']
        if not isinstance(extra,Mapping) or extra.get('state_key')!=key:
            raise ValueError('private source checkpoint identity differs from requested route')
        schema=copy.deepcopy(dict(extra));schema.pop('state_key')
        if schema!=self._schema:
            raise ValueError('private source architecture/layout/schema differs')
        for name,target in self._tensors.items():
            value=state[name]
            if not isinstance(value,torch.Tensor) or value.shape!=target.shape or value.dtype!=target.dtype:
                raise ValueError(f'private tensor shape/dtype differs: {name}')
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'nonfinite private tensor: {name}')
        self._bound_key=None  # A device-copy exception leaves the slot unusable.
        for name,target in self._tensors.items():target.copy_(state[name].detach())
        self.private.state_key=key;self._bound_key=key;self._unweighted=False
        self.require_bound(key)
        return self.private

    @torch.no_grad()
    def unweight_gradients(self,state_weight):
        """Once per visit: private Adam sees its own loss, shared grads stay weighted."""
        self.require_bound(self._bound_key)
        if not math.isfinite(state_weight) or state_weight<=0:
            raise ValueError('positive finite state weight required')
        if self._unweighted:
            raise RuntimeError('private gradients already unweighted for this visit')
        gradients=[p.grad for p in self.private_parameters if p.grad is not None]
        if not gradients:
            raise RuntimeError('private backward must finish before unweighting')
        if not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError('nonfinite private gradients')
        torch._foreach_div_(gradients,float(state_weight))
        self._unweighted=True
        return gradients

    def export_state(self):
        """Independent CPU payload for this bound state; optimizer is not included."""
        self.require_bound(self._bound_key)
        return {**{name:value.detach().to(device='cpu',copy=True) for name,value in self._tensors.items()},
                '_extra_state':copy.deepcopy(self.private.get_extra_state())}
