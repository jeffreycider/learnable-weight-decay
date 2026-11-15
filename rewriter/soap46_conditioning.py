"""Optional full-rank outer-residual coordinates, selected explicitly.

Stored Z acts in parameter coordinates: X_left = Z_left Q_left and
X_right = Z_right Q_right. At each fixed orthogonal SOAP basis this spans exactly
the existing additive-X action family. Across basis refreshes it has different
persistence and coordinatewise Adam behavior; it is an explicit new experiment.
All Q/V observations remain frozen baseline optimizer state.
"""
from .soap46_map import committed_weight


def parameter_frame_weight(record, z_left, z_right):
    if record.variant != 'outer-only':
        raise ValueError('parameter-frame conditioning is defined here for outer-only B')
    if record.q_left is None:
        return committed_weight(record, z_left, z_right)
    return committed_weight(record, z_left @ record.q_left, z_right @ record.q_right)
