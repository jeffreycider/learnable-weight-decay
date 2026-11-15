"""Equal-weight relative L2 norms of functional parameter roles across layers.

PR323 has fifteen groups: embedding, head weight/bias, and weight/bias groups
for six hidden linear roles. Each group's reference is its combined L2 norm,
not a mean of tensor norms. No scalar/gain parameters enter these groups.
"""
from dataclasses import dataclass
import re

import torch

from .edit_scope import editable_names,editable_weight_norm_sq
from .function_preservation import _RealizedParameterNorm


GROUP_NAMES = ('embedding.weight','head.weight','head.bias',
    *tuple(f'{role}.{kind}' for role in ('attention.q','attention.k','attention.v','attention.out','mlp.in','mlp.out')
           for kind in ('weight','bias')))


def functional_groups(parameters):
    """Return ordered group -> tuple of names; fail on unrecognized editable roles."""
    groups={name:[] for name in GROUP_NAMES}
    direct={'embed.weight':'embedding.weight','proj.weight':'head.weight','proj.bias':'head.bias'}
    roles={'attn.q':'attention.q','attn.k':'attention.k','attn.v':'attention.v',
           'attn.proj':'attention.out','mlp.fc':'mlp.in','mlp.proj':'mlp.out'}
    editable=editable_names(parameters,'grouped')
    for name in parameters:
        if name not in editable:continue
        group=direct.get(name)
        if group is None:
            match=re.fullmatch(r'blocks\.\d+\.(attn\.(?:q|k|v|proj)|mlp\.(?:fc|proj))\.(weight|bias)',name)
            if match is None:raise ValueError('unrecognized functional parameter role: '+name)
            group=roles[match[1]]+'.'+match[2]
        groups[group].append(name)
    missing=[name for name,values in groups.items() if not values]
    if missing:raise ValueError('missing PR323 functional groups: '+', '.join(missing))
    return {name:tuple(values) for name,values in groups.items()}


class _RelativeL2(torch.autograd.Function):
    """Stable relative L2 change with zero subgradient when the new group is zero."""
    @staticmethod
    def forward(ctx,delta,reference_squared):
        new_squared=reference_squared+delta
        if bool(new_squared < -1e-10*reference_squared):
            raise ValueError('negative realized group squared norm beyond numerical tolerance')
        new_norm=new_squared.clamp_min(0).sqrt()
        old_norm=reference_squared.sqrt()
        ctx.save_for_backward(old_norm,new_norm)
        # Rationalization preserves tiny changes that subtraction of norms loses.
        effective_delta=torch.where(new_squared>=0,delta,-reference_squared)
        return effective_delta/(old_norm*(new_norm+old_norm))

    @staticmethod
    def backward(ctx,cotangent):
        old_norm,new_norm=ctx.saved_tensors
        derivative=torch.where(new_norm>0,1/(2*old_norm*new_norm.clamp_min(torch.finfo(new_norm.dtype).tiny)),
                               torch.zeros_like(new_norm))
        return cotangent*derivative,None


@dataclass
class GroupedNormResult:
    mean_relative_change: torch.Tensor
    relative_changes: dict[str,torch.Tensor]
    new_norms: dict[str,torch.Tensor]
    squared_norm_change: torch.Tensor
    edit_norm_sq: torch.Tensor

    def metrics(self):
        return dict(mean_group_l2_percentage_reduction=-100*float(self.mean_relative_change.detach()),
            group_l2_percentage_reduction={name:-100*float(value.detach()) for name,value in self.relative_changes.items()},
            group_new_l2_norm={name:float(value.detach()) for name,value in self.new_norms.items()})


class GroupedNormObjective:
    """Frozen references; actual stored-dtype edits; standard cast-STE derivatives."""
    def __init__(self,parameters,groups=None,*,chunk_rows=512,zero_reference_policy="error"):
        self.parameters=dict(parameters)
        self._reference_versions={name:value._version for name,value in parameters.items()}
        self.groups=functional_groups(parameters) if groups is None else {n:tuple(v) for n,v in groups.items()}
        if not self.groups or any(not names for names in self.groups.values()):
            raise ValueError('nonempty groups required')
        flattened=[name for names in self.groups.values() for name in names]
        if len(flattened)!=len(set(flattened)) or set(flattened)!=set(editable_names(parameters,'grouped')):
            raise ValueError('functional groups must cover every editable matrix/bias exactly once')
        if chunk_rows<1:raise ValueError('positive norm accumulation chunk required')
        self.chunk_rows=chunk_rows
        self.reference_squared={name:editable_weight_norm_sq(parameters,names,chunk_rows=chunk_rows)
                                for name,names in self.groups.items()}
        if zero_reference_policy not in ('error','freeze'):
            raise ValueError('zero_reference_policy must be error or freeze')
        invalid=[name for name,value in self.reference_squared.items() if not torch.isfinite(value) or value<0]
        if invalid:raise ValueError('nonfinite or negative group reference norm: '+', '.join(invalid))
        self.zero_reference_groups=tuple(name for name,value in self.reference_squared.items() if value==0)
        if self.zero_reference_groups and zero_reference_policy=='error':
            raise ValueError('percentage norm reduction requires positive finite reference norms; undefined groups: '+', '.join(self.zero_reference_groups))
        self.frozen_parameter_names=frozenset(n for group in self.zero_reference_groups for n in self.groups[group])
        self.reference_norms={name:value.sqrt() for name,value in self.reference_squared.items()}

    def evaluate(self,replacements):
        if any(value._version!=self._reference_versions[name] for name,value in self.parameters.items()):
            raise RuntimeError('frozen grouped norm reference was modified in place')
        if replacements.keys()!=self.parameters.keys():raise ValueError('replacement mapping differs from frozen reference')
        changes={};new_norms={};deltas=[];edits=[]
        for group,names in self.groups.items():
            delta=self.reference_squared[group].new_zeros(())
            squared_edit=delta.clone()
            for name in names:
                value,base=replacements[name],self.parameters[name]
                if value.shape!=base.shape or value.dtype!=base.dtype or value.device!=base.device:
                    raise ValueError('replacement shape/dtype/device mismatch: '+name)
                change,edit=_RealizedParameterNorm.apply(value,base,self.chunk_rows)
                delta=delta+change;squared_edit=squared_edit+edit
            if group in self.zero_reference_groups:
                # No invented percentage denominator: retain this group's zero
                # contribution in the same equal-weight mean and require no edit.
                if any(not torch.equal(replacements[name],self.parameters[name]) for name in names):
                    raise ValueError('zero-reference group must remain frozen: '+group)
                changes[group]=delta*0
            else:
                changes[group]=_RelativeL2.apply(delta,self.reference_squared[group])
            new_norms[group]=(self.reference_squared[group]+delta).clamp_min(0).sqrt().detach()
            deltas.append(delta);edits.append(squared_edit)
        return GroupedNormResult(torch.stack(list(changes.values())).mean(),changes,new_norms,
                                 torch.stack(deltas).sum(),torch.stack(edits).sum())
