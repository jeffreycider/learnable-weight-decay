"""Explicit experimental shared-core actor with private adaptive write spans.

This is NOT the existing anchored rank-2r actor. The cached observations still
refer to their ORIGINAL fixed sensor anchors. Private Q.T mixes their neuron
coordinates, never reconstructing new raw G/M probes. Each matrix edit is
U C V.T with rank <= min(n,m,L). Same-frame reading makes the write directions
observable through available fields, not identifiable from arbitrary spectra.

Core readout: row-write head gives R[i,j]; column-write head gives S[j,i].
C0=(R+S.T)/sqrt(2), using one shared Linear(d,L). This is variance-preserving
only for uncorrelated equal-variance head entries; no independence is assumed
for trained/correlated tokens. Both write contexts contribute differentiably.

Scaling: C=sqrt(n*m/(k_n*k_m))*C0 where k_n=min(n,L), k_m=min(m,L).
For an iid unit-variance C0 independent of frames, E||U C V.T||²=n*m
since ||U||F²=k_n and ||V||F²=k_m. Vectors keep sqrt(n/k_n) Q c,
giving E||edit||²=n under the same convention. A finite-preservation adapter
must apply functional-group RMS ONCE and then its global radius. Do NOT also
apply the anchored sqrt(n*m/[r*(n+m-r)]) multiplier: it double-normalizes.
The legacy anchored convention targets n*m after that latter multiplier;
this is a comparable nominal RMS convention, not matched trained action norms.
More exactly, for independent unit-variance legacy P/Q latent coefficients,
raw expected squared norm is r*(n+m-||A.T D_row||F²), where the scaled row
decoder has ||D_row||F²=n. The commonly used r*(n+m-r) denominator replaces
that overlap by its isotropic-frame expectation r; learned frame alignment
can change it. Neither convention guarantees a particular realized norm.

The Block architecture, attention, precision, lifts and global causal ordering
are inherited unchanged. Initialization keeps zero heads and nonzero private
frames: head gradients exist first, frame rotation gradients follow nonzero C.

Optional fresh_weight_core adds live U.T W V sensing. Its normalized rows and
columns, plus differentiable log Frobenius scale, enter the existing matrix
read/write tokens through one shared Linear(L+1,d). No tokens or Blocks are
added. The encoder helper owns the detached dense-weight scan; forward accepts
only the small cores. Old G/M banks remain fixed and are never retargeted.

Optional factor_precision='fp64' uses FP64 private raw frames/QR, both matrix
and vector output heads, fresh-weight contractions and small-core normalization.
The initialized raw frames are the same seeded unit-scale FP32 values promoted
to FP64; initialization is not a separate random experiment. Shared lifts,
trajectory map, residual Blocks and CUDA BF16 activation policy are unchanged.
Read features cross an explicit FP32 boundary before the backbone; final hidden
features are promoted for FP64 heads. No whole-actor .double()/.float() migration
is supported. The finite adapter uses FP64 group RMS, guard, dense edit and base
addition, then one cast into the original GPT/CNN storage dtype. FP64 factors do
not imply exact function preservation or qualified GPU performance. This is a
distinct checkpoint architecture, with no silent promotion of old learned state.

Optional residual_precision='fp32' disables the CUDA BF16 residual cast and
autocast for this actor, and selects math SDPA while reusing the exact NanoGPT
Block. Parameters remain FP32 except explicitly FP64 factor heads/frames.
The default 'bf16' preserves legacy configuration bytes and execution. This
independent shared execution identity does not change private layout or seeds.
No CUDA performance or fitting benefit is implied; new captures need separate
qualification. The pinned Torch 2.11 CUDA Block RMS default is opmath 2**-23,
which must be rechecked for a new runtime rather than inferred from input dtype.
"""
import math
from collections.abc import Mapping

import torch
from torch import nn

from .adaptive_action import AdaptiveAction, AdaptiveFactors
from .coordinate_chisel import CheckpointProbes, CoordinateChisel, TiedOrthonormalFrame


ARCHITECTURE = 'checkpoint_coordinate_adaptive_ucv_v1'
WEIGHT_CORE_ARCHITECTURE = 'checkpoint_coordinate_adaptive_ucv_weightcore_v1'
FP64_ARCHITECTURE = 'checkpoint_coordinate_adaptive_ucv_fp64_v1'
FP64_WEIGHT_CORE_ARCHITECTURE = 'checkpoint_coordinate_adaptive_ucv_weightcore_fp64_v1'
FACTOR_PRECISIONS = ('fp32', 'fp64')


def _validate_precision(value):
    if value not in FACTOR_PRECISIONS:
        raise ValueError('factor_precision must be fp32 or fp64')


def _check_state_dtypes(module, state):
    # torch.load_state_dict otherwise silently casts a forged/mixed payload.
    for name, parameter in module.named_parameters():
        if name in state and state[name].dtype != parameter.dtype:
            raise ValueError('adaptive checkpoint tensor precision mismatch: '+name)


WEIGHT_CORE_NORM_FLOOR = 1e-12  # Frobenius weight-core units, not loss units.


def normalized_weight_core(core):
    """Smooth, differentiable normalization including its magnitude channel.

    sqrt(||core||F² + floor²) has finite derivatives at an exactly zero W.
    The floor has weight-core units; it is unrelated to KL or radius floors.
    """
    scale = (core.square().sum() + WEIGHT_CORE_NORM_FLOOR**2).sqrt()
    return core / scale, scale.log()


class FP64AdaptiveFrame(TiedOrthonormalFrame):
    """FP64 QR/read contraction, explicit small-feature FP32 backbone boundary."""
    def __init__(self, frame):
        nn.Module.__init__(self)
        self.raw = nn.Parameter(frame.raw.detach().double())

    def encode_with_constants(self, features, dynamic_indices, *, source='encoder'):
        pooled = self.encode(features[:, dynamic_indices], source=source)
        result = features[:1].expand(self.raw.shape[1], -1).clone()
        result[:, dynamic_indices] = pooled.to(result.dtype)
        return result


class AdaptiveCheckpointProbes(CheckpointProbes):
    def __init__(self, snapshot, *, model_dim, latent_tokens=8, seed=0, state_key,
                 fresh_weight_core=False, factor_precision='fp32'):
        if type(fresh_weight_core) is not bool:
            raise ValueError('fresh_weight_core must be an explicit boolean')
        _validate_precision(factor_precision)
        self.factor_precision = factor_precision
        self.fresh_weight_core = fresh_weight_core
        super().__init__(snapshot, model_dim=model_dim, latent_tokens=latent_tokens,
                         seed=seed, state_key=state_key, frame_mode='tied_orthonormal')
        if factor_precision == 'fp64':
            for frames in (self.matrix_rows, self.matrix_columns, self.vector_frames):
                for i, frame in enumerate(frames):
                    frames[i] = FP64AdaptiveFrame(frame)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.set_extra_state(state_dict.get('_extra_state'))
        _check_state_dtypes(self, state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def get_extra_state(self):
        state = super().get_extra_state()
        state.update(schema=ARCHITECTURE, action_family='adaptive_ucv_v1',
                     matrix_decoder='U_C_V_transpose',
                     original_fixed_anchors='observation_metadata_only')
        if self.fresh_weight_core:
            state.update(schema=WEIGHT_CORE_ARCHITECTURE, fresh_weight_core=True)
        if self.factor_precision == 'fp64':
            state.update(schema=FP64_WEIGHT_CORE_ARCHITECTURE if self.fresh_weight_core else FP64_ARCHITECTURE,
                         factor_precision='fp64', vector_precision='fp64',
                         raw_initialization='seeded_fp32_unit_frames_promoted_to_fp64')
        return state


class AdaptiveCoordinateChisel(CoordinateChisel):
    architecture = ARCHITECTURE

    def __init__(self, rank=8, trajectory_dim=400, model_dim=512, num_layers=4, latent_tokens=8,
                 fresh_weight_core=False, factor_precision='fp32', residual_precision='bf16'):
        if type(fresh_weight_core) is not bool:
            raise ValueError('fresh_weight_core must be an explicit boolean')
        _validate_precision(factor_precision)
        if residual_precision not in ('bf16', 'fp32'):
            raise ValueError('residual_precision must be bf16 or fp32')
        self.residual_precision = residual_precision
        self.factor_precision = factor_precision
        # rank remains SENSOR rank; latent_tokens is the NEW write rank limit.
        super().__init__(rank=rank, trajectory_dim=trajectory_dim, model_dim=model_dim,
                         num_layers=num_layers, latent_tokens=latent_tokens,
                         frame_mode='tied_orthonormal')
        self.architecture = ARCHITECTURE
        self.fresh_weight_core = fresh_weight_core
        self.matrix_output = nn.Linear(model_dim, latent_tokens)
        nn.init.zeros_(self.matrix_output.weight)
        nn.init.zeros_(self.matrix_output.bias)
        if fresh_weight_core:
            self.architecture = WEIGHT_CORE_ARCHITECTURE
            self.weight_core_input = nn.Linear(latent_tokens+1, model_dim)
        if factor_precision == 'fp64':
            self.architecture = FP64_WEIGHT_CORE_ARCHITECTURE if fresh_weight_core else FP64_ARCHITECTURE
            self.matrix_output.double()
            self.vector_output.double()
        if residual_precision == 'fp32':
            self.architecture += '_residual_fp32_v1'

    def make_private(self, snapshot, *, state_key, seed=0):
        return AdaptiveCheckpointProbes(snapshot, model_dim=self.model_dim,
                                        latent_tokens=self.latent_tokens, seed=seed,
                                        state_key=state_key, fresh_weight_core=self.fresh_weight_core,
                                        factor_precision=self.factor_precision).to(next(self.parameters()).device)

    def configuration(self):
        config = super().configuration()
        config.update(name=ARCHITECTURE, action_family='adaptive_ucv_v1',
                      private='orthonormal_read_write_frames_and_trajectory_linear',
                      fixed_action_anchors=False, sensor_rank=self.rank,
                      write_rank_limit=self.latent_tokens,
                      sensor_anchor_policy='original_fixed_cached_banks_no_retargeted_G_or_M',
                      core_readout='(row_head+column_head_transpose)/sqrt(2)',
                      matrix_core_scale='sqrt(n*m/(min(n,L)*min(m,L)))',
                      external_group_scale='group_RMS_only_no_anchored_dimension_multiplier',
                      decoder_scale='matrix_core_scale; vectors_sqrt(n/min(n,L))')
        if self.fresh_weight_core:
            config.update(name=WEIGHT_CORE_ARCHITECTURE, fresh_weight_core=True,
                          weight_core_query='U.T_detached_oriented_W_V; live_private_QR',
                          weight_core_input='shared_Linear_Lplus1_to_d_on_normalized_rows_and_columns_plus_logscale',
                          weight_core_normalization='sqrt(Frobenius_squared+floor_squared)',
                          weight_core_norm_floor=WEIGHT_CORE_NORM_FLOOR,
                          weight_core_destinations='existing_matrix_read_and_write_tokens',
                          fresh_gradient_or_momentum_queries=False)
        if self.factor_precision == 'fp64':
            config.update(name=self.architecture, factor_precision='fp64', vector_precision='fp64',
                          shared_backbone_precision='unchanged_fp32_parameters_bf16_cuda_activations',
                          read_boundary='normalize_fresh_core_in_fp64_then_cast_small_features_to_backbone',
                          stored_parameter_rule='fp64_base_plus_scaled_edit_then_one_storage_cast')
        if self.residual_precision == 'fp32':
            config.update(name=self.architecture, residual_precision='fp32',
                          cuda_residual_dtype='float32',
                          shared_backbone_precision='fp32_parameters_and_activations',
                          actor_attention_backend='math', actor_autocast=False,
                          actor_tf32=False, actor_residual_cast='no_bf16_cast',
                          actor_rms_epsilon_policy='unchanged_Block_default_pinned_torch211_opmath_2^-23')
        return config

    def get_extra_state(self):
        return self.configuration()

    def set_extra_state(self, state):
        if state != self.configuration():
            raise ValueError('adaptive shared actor architecture mismatch')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.set_extra_state(state_dict.get('_extra_state'))
        _check_state_dtypes(self, state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _weight_core_tokens(self, snapshot, private, x, read_count, matrix_count, weight_cores):
        if private.fresh_weight_core != self.fresh_weight_core:
            raise ValueError('private weight-core observation identity differs')
        if not self.fresh_weight_core:
            if weight_cores is not None:
                raise ValueError('coreless actor cannot silently consume fresh weight cores')
            return x
        if not isinstance(weight_cores, Mapping) or set(weight_cores) != set(snapshot.matrices):
            raise ValueError('complete fresh weight-core mapping required')
        features = []
        for name in snapshot.matrices:
            core = weight_cores[name]
            if (not isinstance(core, torch.Tensor) or core.shape != (self.latent_tokens, self.latent_tokens)
                    or core.dtype not in (torch.float32, torch.float64) or core.device != x.device):
                raise ValueError('fresh weight-core shape/dtype/device mismatch: '+name)
            if self.factor_precision == 'fp64' and core.dtype != torch.float64:
                raise ValueError('FP64 fresh weight-core query required: '+name)
            normalized, logscale = normalized_weight_core(core if self.factor_precision == 'fp64' else core.to(x.dtype))
            magnitude = logscale.expand(self.latent_tokens, 1)
            features.extend((torch.cat((normalized, magnitude), dim=-1),
                             torch.cat((normalized.T, magnitude), dim=-1)))
        lifted = self.weight_core_input(torch.cat(features).to(x.dtype))[None]
        # Read prefix already contains every matrix before any write token.
        # Functional concatenation avoids mutating tensors retained by autograd.
        return torch.cat((x[:,:matrix_count]+lifted,
                          x[:,matrix_count:read_count],
                          x[:,read_count:read_count+matrix_count]+lifted,
                          x[:,read_count+matrix_count:]), dim=1)

    def forward(self, snapshot, private, *, weight_cores=None):
        if self.residual_precision == 'fp32':
            # Preserve the exact Block equations/modules. This explicit mode
            # changes arithmetic and SDPA execution, not the private schema.
            # Never silently mutate process-wide precision settings.
            device = next(self.parameters()).device
            if device.type == 'cuda' and torch.backends.cuda.matmul.allow_tf32:
                raise ValueError('FP32 residual mode requires TF32 disabled before execution')
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with torch.autocast(device_type=device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
                return self._forward_impl(snapshot, private, weight_cores=weight_cores)
        return self._forward_impl(snapshot, private, weight_cores=weight_cores)

    def _forward_impl(self, snapshot, private, *, weight_cores=None):
        if not isinstance(private, AdaptiveCheckpointProbes):
            raise ValueError('adaptive action requires explicitly adaptive private identity')
        if private.factor_precision != self.factor_precision:
            raise ValueError('private factor precision identity differs')
        dtype = torch.float64 if self.factor_precision == 'fp64' else torch.float32
        frames = (*private.matrix_rows, *private.matrix_columns, *private.vector_frames)
        if any(f.raw.dtype != dtype for f in frames) or any(p.dtype != dtype for h in self.output_heads for p in h.parameters()):
            raise ValueError('adaptive factor precision changed outside configured mode')
        if self.factor_precision == 'fp64' or self.residual_precision == 'fp32':
            if any(p.dtype != torch.float32 for n,p in self.named_parameters()
                   if not n.startswith(('matrix_output.', 'vector_output.'))) or private.trajectory_input.weight.dtype != torch.float32:
                raise ValueError('explicit precision mode requires FP32 shared backbone/trajectory')
        x, read_count, matrix_count = self._inputs(snapshot, private)
        x = self._weight_core_tokens(snapshot, private, x, read_count, matrix_count, weight_cores)
        if x.is_cuda and self.residual_precision == 'bf16':
            x = x.bfloat16()
        x = self.norm1(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm2(x[:, read_count:]).float()[0]
        latent = self._matrix_readout(snapshot, x[:matrix_count].to(dtype))
        matrices = {}
        for i, (name, observation) in enumerate(snapshot.matrices.items()):
            u = private.matrix_rows[i].orthonormal_matrix()
            v = private.matrix_columns[i].orthonormal_matrix()
            n, m = observation.rows.shape[0], observation.columns.shape[0]
            scale = math.sqrt(n * m / (min(n, self.latent_tokens) * min(m, self.latent_tokens)))
            core = (latent[2*i] + latent[2*i+1].T) * (scale / math.sqrt(2))
            matrices[name] = AdaptiveFactors(u, core, v)
        vectors = {}
        if snapshot.vectors:
            latent_vectors = self._vector_readout(snapshot, x[matrix_count:].to(dtype))
            vectors = {name: private.vector_frames[i].decode(latent_vectors[i]).flatten()
                       for i, name in enumerate(snapshot.vectors)}
        return AdaptiveAction(matrices, vectors)

    def _matrix_readout(self, snapshot, hidden):
        return self.matrix_output(hidden).reshape(-1, self.latent_tokens, self.latent_tokens)

    def _vector_readout(self, snapshot, hidden):
        return self.vector_output(hidden).reshape(-1, self.latent_tokens, 1)
