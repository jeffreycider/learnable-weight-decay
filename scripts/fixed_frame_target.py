"""Cached target-distance algebra for the frozen-probe diagnostic phase only.

The joint-span QR is the same as matrix_edit_distance. Cache its constants when
U/V cannot change; evaluate only small cores thereafter. No orthogonality-based
norm subtraction, dense edits, or gradient approximation is needed. This is not
valid for trainable probes and intentionally rejects them.
"""
import torch
from rewriter.adaptive_action import AdaptiveAction


class FixedFrameTarget:
    @torch.no_grad()
    def __init__(self, example: AdaptiveAction, target: AdaptiveAction):
        if example.matrices.keys()!=target.matrices.keys() or example.vectors.keys()!=target.vectors.keys():
            raise ValueError('target inventory mismatch')
        self.matrices={}
        for name, f in example.matrices.items():
            t=target.matrices[name]
            if f.U.requires_grad or f.V.requires_grad:
                raise ValueError('fixed-frame target requires frozen probes')
            if any(x.requires_grad for x in (t.U,t.C,t.V)):
                raise ValueError('target must be frozen')
            if (f.U.shape[0]!=t.U.shape[0] or f.V.shape[0]!=t.V.shape[0]
                    or f.U.dtype!=t.U.dtype or f.U.device!=t.U.device):
                raise ValueError('target and prediction dimensions/dtype/device differ')
            left=torch.linalg.qr(torch.cat((f.U,t.U),dim=1),mode='reduced').Q
            right=torch.linalg.qr(torch.cat((f.V,t.V),dim=1),mode='reduced').Q
            self.matrices[name]=(f.U.clone(),f.V.clone(),left.T@f.U,right.T@f.V,
                (left.T@t.U)@t.C@(right.T@t.V).T)
        if any(v.requires_grad for v in target.vectors.values()):
            raise ValueError('target must be frozen')
        self.vectors={name:v.detach().clone() for name,v in target.vectors.items()}

    def __call__(self, action: AdaptiveAction):
        if action.matrices.keys()!=self.matrices.keys() or action.vectors.keys()!=self.vectors.keys():
            raise ValueError('cached target inventory mismatch')
        terms={}
        for name, f in action.matrices.items():
            u,v,left,right,target=self.matrices[name]
            if f.U.requires_grad or f.V.requires_grad:
                raise ValueError('fixed-frame target requires frozen probes')
            if (f.U.dtype!=u.dtype or f.U.device!=u.device or not torch.equal(f.U,u)
                    or not torch.equal(f.V,v)):
                raise ValueError('cached probe frames changed')
            terms[name]=(left@f.C@right.T-target).square().sum()
        for name,value in action.vectors.items():
            target=self.vectors[name]
            if value.shape!=target.shape or value.dtype!=target.dtype or value.device!=target.device:
                raise ValueError('cached vector target mismatch')
            terms[name]=(value-target).square().sum()
        return terms
