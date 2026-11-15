"""Chisel write scope, independent of its globally complete observations.

The PR323 lookup table and entire output head remain visible to the controller.
In hidden scope their edits are exactly zero and their norms are excluded from
the objective/radius reference. Historical configurations without a scope mean
``all`` when replayed; new experiment frontends select ``hidden`` explicitly.
"""
from dataclasses import replace
import hashlib
import re

import torch

from .algebra import Factors, RawAction


READ_ONLY_NAMES = frozenset(('embed.weight', 'proj.weight', 'proj.bias'))


def editable_names(parameters, scope):
    if scope not in ('all', 'hidden', 'grouped'):
        raise ValueError('editable scope must be all, hidden, or grouped')
    if scope=='grouped':
        return frozenset(name for name,value in parameters.items()
            if value.numel()>1
            and not any(re.fullmatch(r'(?:rms_?norm|layer_?norm|norm)\d*',part,re.I)
                        for part in name.split('.')[:-1])
            and (value.ndim==2 or (value.ndim==1 and name.rsplit('.',1)[-1]=='bias')))
    if scope == 'hidden' and not READ_ONLY_NAMES.issubset(parameters):
        raise ValueError('hidden scope requires explicit PR323 embedding/head names')
    return frozenset(parameters) if scope == 'all' else frozenset(parameters)-READ_ONLY_NAMES


def scoped_action(action, names):
    """Mask before any action norm/guard; preserve all globally observed tokens."""
    if not set(names).issubset(set(action.matrices)|set(action.vectors)):
        raise ValueError('editable parameter missing from actor action')
    return RawAction({n:f if n in names else Factors(torch.zeros_like(f.P),torch.zeros_like(f.Q),f.anchors)
                      for n,f in action.matrices.items()},
                     {n:v if n in names else torch.zeros_like(v) for n,v in action.vectors.items()})


@torch.no_grad()
def editable_weight_norm_sq(parameters, names, *, chunk_rows=512):
    if not names or not set(names).issubset(parameters):
        raise ValueError('nonempty valid editable parameter set required')
    value = next(iter(parameters.values())).new_zeros((),dtype=torch.float64)
    # Preserve model parameter order, never iterate a hash-dependent set.
    for name,parameter in parameters.items():
        if name in names:
            for start in range(0,len(parameter),chunk_rows):
                value.add_(parameter[start:start+chunk_rows].double().square().sum())
    return value


def scoped_snapshot(snapshot, norm_squared, radius_fraction, *, scope):
    if scope == 'all':
        return snapshot  # Historical FP32 accumulation/radius stays identical.
    if scope not in ('hidden','grouped') or radius_fraction<=0 or not torch.isfinite(norm_squared) or norm_squared<=0:
        raise ValueError('scoped editing requires a positive finite reference norm/radius')
    norm = norm_squared.sqrt().to(snapshot.weight_norm)
    return replace(snapshot,weight_norm=norm,radius=norm*radius_fraction)


def keep_readonly_parameters(replacements, base, names):
    """Preserve exact stored bits, including signed zero; avoid even adding +0."""
    if replacements.keys()!=base.keys():
        raise ValueError('replacement parameter mapping differs from baseline')
    return {n:value if n in names else base[n] for n,value in replacements.items()}


@torch.no_grad()
def readonly_hashes(parameters, names):
    result = {}
    for name,value in parameters.items():
        if name not in names:
            digest = hashlib.sha256()
            rows=value.reshape(1) if value.ndim==0 else value
            for start in range(0,len(rows),512):
                data=rows[start:start+512].detach().contiguous().view(torch.uint8).cpu().numpy()
                digest.update(data.tobytes())
            result[name]=digest.hexdigest()
    return result


def verify_readonly(parameters, names, expected):
    actual=readonly_hashes(parameters,names)
    if actual!=expected:
        raise RuntimeError('read-only parameter stored bits changed')
    return dict(unchanged=True,parameter_sha256=actual)
