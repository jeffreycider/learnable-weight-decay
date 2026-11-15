"""Optional functional-group output linears after the unchanged global Chisel trunk.

All matrices/vectors remain in the global observations and attention sequence.
Only the final Linear differs by functional role, shared across layers and
checkpoints. No per-checkpoint head parameters or automatic checkpoint conversion.
Private frame layout/identity remains the adaptive frame schema by design.
"""
import re

import torch
from torch import nn

from .adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from .grouped_norm import GROUP_NAMES


GROUPED_ARCHITECTURE = 'checkpoint_coordinate_grouped_output_ucv_v1'
MATRIX_GROUPS = tuple(g for g in GROUP_NAMES if g.endswith('.weight')) + (
    'conv.entry.weight', 'conv.hidden.weight')
VECTOR_GROUPS = tuple(g for g in GROUP_NAMES if g.endswith('.bias')) + ('whiten.bias',)
ROUTING_VERSION = 'gpt_functional_roles_airbench_roles_readonly_v1'


def head_key(group):
    return group.replace('.', '__')


def functional_head_group(name, kind, coordinates):
    """Return an explicit functional role, or None for read-only observations.

    kind is matrix/vector; coordinates is the physical element count. Unknown
    editable matrix/bias roles raise; vector gains, normalization and scalar
    observations, plus fixed CNN whitening filters, are read-only.
    """
    if kind not in ('matrix', 'vector'):
        raise ValueError('matrix/vector observation kind required')
    if coordinates <= 1 or any(re.fullmatch(r'(?:rms_?norm|layer_?norm|norm)\d*',part,re.I)
                               for part in name.split('.')[:-1]):
        return None
    if kind == 'matrix':
        if name == 'whiten.weight':
            return None
        if name == 'embed.weight':
            return 'embedding.weight'
        if name in ('proj.weight', 'head.weight'):
            return 'head.weight'
        if name.endswith('.conv1.weight'):
            return 'conv.entry.weight'
        if name.endswith('.conv2.weight'):
            return 'conv.hidden.weight'
    else:
        if name.rsplit('.',1)[-1] != 'bias':
            return None
        if name == 'proj.bias':
            return 'head.bias'
        if name == 'whiten.bias':
            return 'whiten.bias'
    match = re.fullmatch(r'blocks\.\d+\.(attn\.(?:q|k|v|proj)|mlp\.(?:fc|proj))\.(weight|bias)', name)
    if match and match[2] == ('weight' if kind == 'matrix' else 'bias'):
        role = {'attn.q':'attention.q','attn.k':'attention.k','attn.v':'attention.v',
                'attn.proj':'attention.out','mlp.fc':'mlp.in','mlp.proj':'mlp.out'}[match[1]]
        return role + '.' + match[2]
    raise ValueError('unknown editable functional output role: '+name)


class GroupedCoordinateChisel(AdaptiveCoordinateChisel):
    """Globally coupled adaptive actor with one typed Linear per functional group.

    output_head(group) returns the Linear for explicit coefficient import.
    Serialized keys are matrix_output.<role with dots replaced by __>.weight
    (and bias), or vector_output.<role>.weight/bias. Extra state identifies all
    routes and precision modes; legacy shared-head checkpoints are rejected.
    """
    def __init__(self, rank=8, trajectory_dim=400, model_dim=512, num_layers=4,
                 latent_tokens=8, fresh_weight_core=False, factor_precision='fp32',
                 residual_precision='bf16'):
        super().__init__(rank=rank,trajectory_dim=trajectory_dim,model_dim=model_dim,
            num_layers=num_layers,latent_tokens=latent_tokens,fresh_weight_core=fresh_weight_core,
            factor_precision=factor_precision,residual_precision=residual_precision)
        # Retain the exact legacy trunk draw sequence. Head constructors get a
        # separate CPU RNG scope; all new heads are then exactly zero.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            self.matrix_output = nn.ModuleDict({head_key(g):nn.Linear(model_dim,latent_tokens) for g in MATRIX_GROUPS})
            self.vector_output = nn.ModuleDict({head_key(g):nn.Linear(model_dim,1) for g in VECTOR_GROUPS})
        for bank in self.output_heads:
            for head in bank.values():
                nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
            if factor_precision == 'fp64':
                bank.double()
        self.architecture = GROUPED_ARCHITECTURE + '__' + self.architecture

    def configuration(self):
        config = super().configuration()
        config.update(name=self.architecture,output_head_family='functional_group_linears_v1',
            matrix_output_groups=list(MATRIX_GROUPS),vector_output_groups=list(VECTOR_GROUPS),
            output_routing=ROUTING_VERSION,output_group_key_encoding='dots_to_double_underscore',
            output_group_sharing='across_layers_and_checkpoints; head.weight_shared_across_domains',
            output_group_initialization='zero; separate_cpu_rng_scope',
            readonly_outputs='zero; norms_vector_gains_scalars_and_CNN_whitening_weight_stay_observed',
            global_backbone='unchanged_single_global_attention_sequence',
            core_readout='functional_group_Linear(row)+same_group_Linear(column).T; divided_by_sqrt2')
        return config

    def output_head(self, group):
        if group in MATRIX_GROUPS:
            return self.matrix_output[head_key(group)]
        if group in VECTOR_GROUPS:
            return self.vector_output[head_key(group)]
        raise ValueError('unknown functional output head: '+group)

    def _matrix_readout(self, snapshot, hidden):
        L = self.latent_tokens
        pairs = hidden.reshape(len(snapshot.matrices),2*L,self.model_dim)
        values = []
        for i,(name,obs) in enumerate(snapshot.matrices.items()):
            group = functional_head_group(name,'matrix',obs.rows.shape[0]*obs.columns.shape[0])
            values.append(pairs.new_zeros((2,L,L)) if group is None
                          else self.output_head(group)(pairs[i]).reshape(2,L,L))
        return torch.cat(values,dim=0)

    def _vector_readout(self, snapshot, hidden):
        L = self.latent_tokens
        tokens = hidden.reshape(len(snapshot.vectors),L,self.model_dim)
        values = []
        for i,(name,obs) in enumerate(snapshot.vectors.items()):
            group = functional_head_group(name,'vector',obs.weight.numel())
            values.append(tokens.new_zeros((L,1)) if group is None
                          else self.output_head(group)(tokens[i]))
        return torch.stack(values,dim=0)
