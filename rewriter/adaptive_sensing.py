"""Fresh low-bandwidth weight queries for explicitly adaptive write frames.

This scans detached dense W and computes U.T W V with live private QR frames.
The shared transformer receives only L-by-L cores. Compute is O(n*m*L), not
O(n+m); it is matrix algebra, with no GPT evaluation or new G/M measurement.
Source weights are read-only and receive no gradient. Their storage must stay
unchanged until the private-frame VJP completes (detach is not a dense copy).
Embedding names are an explicit storage-orientation convention; CNN engines
already provide flattened matrices and simply have no matching embedding name.
"""
from collections.abc import Mapping

import torch

from .adaptive_coordinate_chisel import AdaptiveCheckpointProbes


def query_weight_cores(parameters, private, *, embedding_names=('embed.weight',)):
    """Return {matrix_name: U.T @ detached(oriented W) @ V}, preserving QR VJP.

    ``parameters`` may also contain vectors/scalars, which are not scanned.
    Every matrix must match the private layout exactly. No sensor feature or
    original fixed-anchor contraction is used to approximate this new query.
    """
    if not isinstance(private, AdaptiveCheckpointProbes):
        raise TypeError('adaptive private probes required for weight-core queries')
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError('nonempty named parameter mapping required')
    if any(not isinstance(p, torch.Tensor) for p in parameters.values()):
        raise ValueError('named parameter values must be tensors')
    names = {n for n, p in parameters.items() if p.ndim == 2}
    if names != set(private.matrix_names):
        raise ValueError('dense weight matrix inventory differs from private frames')
    embeddings = set(embedding_names)
    cores = {}
    for i, name in enumerate(private.matrix_names):
        stored = parameters[name]
        if not stored.is_floating_point():
            raise ValueError('floating-point dense weights required')
        weight = stored.detach().T if name in embeddings else stored.detach()
        u = private.matrix_rows[i].orthonormal_matrix()
        v = private.matrix_columns[i].orthonormal_matrix()
        if weight.shape != (u.shape[0], v.shape[0]):
            raise ValueError('oriented dense weight shape differs from private frames: '+name)
        if weight.device != u.device or v.device != u.device:
            raise ValueError('weight/core frame device mismatch')
        weight = weight.to(dtype=u.dtype)
        # Keep the intermediate on the smaller physical axis; both paths are
        # exact real-arithmetic contractions, with ordinary GEMM rounding.
        cores[name] = ((u.T @ weight) @ v if weight.shape[1] <= weight.shape[0]
                       else u.T @ (weight @ v))
    return cores
