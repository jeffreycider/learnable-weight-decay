"""Execution-only loss compilation; GPT architecture and sensors remain unchanged.

Compiling the whole instrumented GPT allows hook graph breaks to exhaust the
recompile cache and fall back to a very large eager loss. Compile the pure loss
instead: no hooks, mutable model state, or additional forward/backward calls.
"""
from types import MethodType
import torch
import torch.nn.functional as F


def softcap_cross_entropy(logits, targets):
    """The pinned GPT loss equations, including float conversion and SUM units."""
    logits = logits.float()
    logits = 15 * logits * (logits.square() + 15**2).rsqrt()
    return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction='sum')


def _forward_with_compiled_loss(self, inputs, targets):
    # Same operations/order as baseline.model.GPT.forward. Only the final pure
    # loss is compiled, so activation/gradient hooks execute outside its graph.
    x = self.norm1(self.embed(inputs))
    for block in self.blocks:
        x = block(x)
    logits = self.proj(self.norm2(x))
    return self._compiled_loss(logits, targets)


def enable_compiled_loss(model):
    """Install before running GPT; no warmup batch or optimizer state is touched.

    Fullgraph deliberately fails rather than silently executing the large loss
    eagerly. Compilation can specialize for ordinary training versus no-grad
    validation, neither of which creates an extra GPT evaluation.
    """
    if hasattr(model, '_compiled_loss'):
        raise ValueError('compiled loss is already installed')
    model._compiled_loss = torch.compile(softcap_cross_entropy, dynamic=False, fullgraph=True)
    model.forward = MethodType(_forward_with_compiled_loss, model)
    return model


def enable_deterministic_execution():
    """Apply the muoff torch2.11 policy before CUDA initialization.

    Ported unchanged in vendor/muoff_determinism_policy.py. Its recorded evidence
    is fresh/fresh byte equality at tested boundaries through update30 on the
    audited 8xA100 configuration, not a universal guarantee for new workloads.
    PYTHONHASHSEED must be set by the launcher; setting it after startup cannot
    change this interpreter's hash seed.
    """
    from .vendor import muoff_determinism_policy
    muoff_determinism_policy.apply()
    return muoff_determinism_policy.verify(require_launch_env=True)


def execution_environment():
    """Record deterministic, SDPA, precision, and launch choices for every run."""
    import os
    import torch._inductor.config as inductor
    return dict(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        inductor_deterministic=inductor.deterministic,
        inductor_max_autotune=inductor.max_autotune,
        inductor_max_autotune_gemm=inductor.max_autotune_gemm,
        inductor_coordinate_descent_tuning=inductor.coordinate_descent_tuning,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        sdpa=dict(flash=torch.backends.cuda.flash_sdp_enabled(),
                  cudnn=torch.backends.cuda.cudnn_sdp_enabled(),
                  memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                  math=torch.backends.cuda.math_sdp_enabled()),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        launch_env={name: os.environ.get(name) for name in (
            'CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED', 'TORCHINDUCTOR_DETERMINISTIC',
            'NCCL_ALGO', 'NCCL_PROTO', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NVIDIA_TF32_OVERRIDE')})
