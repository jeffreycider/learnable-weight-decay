"""Research-only compensated dx; never selected by the production accumulator.

Student forward, teacher targets, and thin-factor contractions remain IEEE FP32
or the existing FP64 objective arithmetic. Only activation-gradient GEMMs for
the head and MLP input projections use the existing three-product TF32 backend.
Every replay reconstructs the decomposition from its realized edited weights.
"""
import os
import torch
from .graph_tokenwise import TokenwiseThinGradientGraph


def require_compensated_dx_environment():
    if os.environ.get('NVIDIA_TF32_OVERRIDE') == '0':
        raise ValueError('NVIDIA_TF32_OVERRIDE=0 disables the requested compensated tensor-core backend; keep default Torch IEEE settings but remove the global override')


class CompensatedDxTokenwiseGraph(TokenwiseThinGradientGraph):
    def __init__(self, *args, **kwargs):
        require_compensated_dx_environment()
        if kwargs.get('teacher_representation', 'log_probs') != 'log_probs':
            raise ValueError('qualify compensated dx independently of teacher representation')
        super().__init__(*args, **kwargs)
        # NativeThinGradientGraph wraps eager/capture in this explicit context.
        # Use that stable interface, also present in the frozen production base.
        self.dx_backend = 'cublas3_cached'

    @torch.no_grad()
    def _student_options(self, realized):
        from .kernels.compensated_pack_triton import split
        weights = {
            name: split(value.detach().T.contiguous())
            for name, value in realized.items()
            if name.endswith('.weight') and name != 'embed.weight'
            and value.ndim == 2 and value.shape[0] >= 3072
        }
        return {'dx_split_weights': weights}
