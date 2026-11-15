"""Checkpoint-private neuron-axis linear coordinates around a shared backbone.

Encoders pool the EXISTING coordinate feature banks, not new raw G/M probes.
Fixed A/B action anchors remain unchanged. Private modules are explicit arguments,
never children of the shared actor. Production may copy per-state private weights
into one static device container per layout; state identity is bound outside forward.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from baseline.model import Block,RMSNorm,initialize_parameters_like_record
from .algebra import Factors,RawAction

ARCHITECTURE='checkpoint_coordinate_ed_nanogpt_v1'
WRITE_FRAME_ARCHITECTURE='checkpoint_coordinate_ed_writeframe_v2'
TIED_FRAME_ARCHITECTURE='checkpoint_coordinate_tied_q_nanogpt_v1'
FRAME_MODES=('independent','tied_orthonormal')


def observation_layout(snapshot):
    return (tuple((n,tuple(v.rows.shape),tuple(v.columns.shape),tuple(v.anchors.A.shape),tuple(v.anchors.B.shape))
                  for n,v in snapshot.matrices.items()),
            tuple((n,tuple(v.coordinates.shape)) for n,v in snapshot.vectors.items()),
            int(snapshot.trajectories.shape[-1]))


class CoordinateFrame(nn.Module):
    """Learned neuron-axis E,D; stable unit-column maps without an action penalty."""
    def __init__(self,coordinates,latent_tokens,*,generator):
        super().__init__()
        if min(coordinates,latent_tokens)<1:raise ValueError('positive coordinate dimensions required')
        value=torch.randn(coordinates,latent_tokens,generator=generator,dtype=torch.float32)
        value=F.normalize(value,dim=0)
        self.encoder=nn.Parameter(value)
        self.decoder=nn.Parameter(value.clone())

    def encode(self,features,*,source='encoder'):
        if source not in ('encoder','decoder'):raise ValueError('unknown coordinate observation frame')
        return F.normalize(getattr(self,source),dim=0,eps=1e-12).T@features.float()

    def encode_with_constants(self,features,dynamic_indices,*,source='encoder'):
        """Pool neuron signals; metadata/log magnitudes bypass either frame."""
        pooled=self.encode(features[:,dynamic_indices],source=source)
        result=features[:1].expand(self.encoder.shape[1],-1).clone()
        result[:,dynamic_indices]=pooled
        return result

    def decode(self,latent):
        return math.sqrt(self.decoder.shape[0]/self.decoder.shape[1])*(F.normalize(self.decoder,dim=0,eps=1e-12)@latent.float())


class TiedOrthonormalFrame(CoordinateFrame):
    """One unconstrained private parameter, differentiable signed thin QR.

    n>=L: Q.T Q=I and decode=sqrt(n/L) Q C, matching the old total
    output-variance convention. n<L: QR(raw.T).T gives Q Q.T=I;
    its L columns form a Parseval tight frame (not unit columns), and decode
    is Q C with scale one. Both cases give decoder Frobenius norm sqrt(n).
    QR requires full rank; no epsilon perturbation or hidden rank repair is
    applied. Adam acts on raw coordinates, not a transported manifold state.
    Raw columns (rows for n<L) start at unit norm to avoid an implicit sqrt(n)
    reduction of angular learning rate compared with the independent frames.
    Transposed QR has an ordered row chart: it is not row-permutation equivariant.
    """
    def __init__(self,coordinates,latent_tokens,*,generator):
        nn.Module.__init__(self)
        if min(coordinates,latent_tokens)<1:raise ValueError('positive coordinate dimensions required')
        value=torch.randn(coordinates,latent_tokens,generator=generator,dtype=torch.float32)
        self.raw=nn.Parameter(F.normalize(value,dim=0 if coordinates>=latent_tokens else 1))

    def orthonormal_matrix(self):
        raw=self.raw if self.raw.dtype==torch.float64 else self.raw.float()
        tall=raw if raw.shape[0]>=raw.shape[1] else raw.T
        q,r=torch.linalg.qr(tall,mode='reduced')
        sign=torch.where(r.diagonal()<0,-torch.ones_like(r.diagonal()),torch.ones_like(r.diagonal()))
        q=q*sign
        return q if raw.shape[0]>=raw.shape[1] else q.T

    @property
    def encoder(self):return self.orthonormal_matrix()

    @property
    def decoder(self):return self.orthonormal_matrix()

    def encode(self,features,*,source='encoder'):
        if source not in ('encoder','decoder'):raise ValueError('unknown coordinate observation frame')
        q=self.orthonormal_matrix()
        return q.T@features.to(q.dtype)

    def encode_with_constants(self,features,dynamic_indices,*,source='encoder'):
        pooled=self.encode(features[:,dynamic_indices],source=source)
        result=features[:1].expand(self.raw.shape[1],-1).clone()
        result[:,dynamic_indices]=pooled
        return result

    def decode(self,latent):
        q=self.orthonormal_matrix()
        return math.sqrt(q.shape[0]/min(q.shape))*(q@latent.to(q.dtype))


class CheckpointProbes(nn.Module):
    """One checkpoint's private encoders/decoders and trajectory map.

    Optimizer ownership is external: one private optimizer state per state_key,
    and no distributed SUM over these parameters. Extra state rejects loading a
    different checkpoint identity accidentally. Static staging explicitly copies
    tensor payloads and binds the identity separately; it must not bypass checks.
    """
    def __init__(self,snapshot,*,model_dim,latent_tokens=8,seed=0,state_key,frame_mode='independent'):
        super().__init__()
        if frame_mode not in FRAME_MODES:raise ValueError('unknown private frame mode')
        self.frame_mode=frame_mode
        if not isinstance(state_key,tuple) or not state_key or any(not isinstance(v,(str,int)) for v in state_key):
            raise ValueError('explicit immutable checkpoint identity tuple required')
        self.state_key=state_key;self.layout=observation_layout(snapshot)
        self.model_dim=int(model_dim);self.latent_tokens=int(latent_tokens)
        self.matrix_names=tuple(snapshot.matrices);self.vector_names=tuple(snapshot.vectors)
        if not self.matrix_names:raise ValueError('matrix observations required')
        self.rank=next(iter(snapshot.matrices.values())).anchors.A.shape[1]
        self.trajectory_dim=snapshot.trajectories.shape[-1]
        if self.trajectory_dim%(2*self.rank):raise ValueError('aligned activation/adjoint trajectory required')
        generator=torch.Generator(device='cpu').manual_seed(seed)
        frame=CoordinateFrame if frame_mode=='independent' else TiedOrthonormalFrame
        self.matrix_rows=nn.ModuleList([frame(v.rows.shape[0],latent_tokens,generator=generator) for v in snapshot.matrices.values()])
        self.matrix_columns=nn.ModuleList([frame(v.columns.shape[0],latent_tokens,generator=generator) for v in snapshot.matrices.values()])
        self.vector_frames=nn.ModuleList([frame(v.coordinates.shape[0],latent_tokens,generator=generator) for v in snapshot.vectors.values()])
        # Do not consume the global actor/data RNG when adding a new checkpoint.
        with torch.random.fork_rng(devices=[]):
            # torch.manual_seed also seeds every CUDA generator; only the CPU
            # default generator is forked here and used by this CPU Linear.
            torch.random.default_generator.manual_seed(seed+104729)
            self.trajectory_input=nn.Linear(self.trajectory_dim+2+self.trajectory_dim//(2*self.rank),model_dim)

    def validate_layout(self,snapshot):
        if observation_layout(snapshot)!=self.layout:raise ValueError('private probes belong to a different observation layout')

    def get_extra_state(self):
        result=dict(schema=ARCHITECTURE,state_key=self.state_key,layout=self.layout,model_dim=self.model_dim,
                    latent_tokens=self.latent_tokens,rank=self.rank,column_normalization='differentiable_unit_l2',decoder_scale='sqrt_coordinates_over_latent_tokens')
        if self.frame_mode!='independent':
            result.update(schema=TIED_FRAME_ARCHITECTURE,frame_mode=self.frame_mode,
                          column_normalization='signed_thin_qr_or_transposed_qr_parseval',
                          decoder_scale='sqrt_coordinates_over_min_coordinates_latent_tokens',
                          raw_initialization='unit_columns_tall_unit_rows_wide')
        return result

    def load_state_dict(self,state_dict,strict=True,assign=False):
        # Reject identity/layout before any child tensors can be copied.
        self.set_extra_state(state_dict.get('_extra_state'))
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('private probe checkpoint identity/layout mismatch')


class CoordinateChisel(nn.Module):
    architecture=ARCHITECTURE
    def __init__(self,rank=8,trajectory_dim=400,model_dim=512,num_layers=4,latent_tokens=8,
                 write_observation_frame='encoder',frame_mode='independent'):
        super().__init__()
        if write_observation_frame not in ('encoder','decoder'):
            raise ValueError('write observation frame must be encoder or decoder')
        self.write_observation_frame=write_observation_frame
        if frame_mode not in FRAME_MODES:raise ValueError('unknown private frame mode')
        self.frame_mode=frame_mode
        self.architecture=ARCHITECTURE if write_observation_frame=='encoder' else WRITE_FRAME_ARCHITECTURE
        if frame_mode!='independent':self.architecture=TIED_FRAME_ARCHITECTURE
        if model_dim<128 or model_dim%128 or min(rank,num_layers,latent_tokens)<1:
            raise ValueError('positive dimensions and unchanged128-wide attention heads required')
        if trajectory_dim%(2*rank):raise ValueError('aligned trajectory width required')
        self.rank=rank;self.trajectory_dim=trajectory_dim;self.model_dim=model_dim
        self.num_layers=num_layers;self.latent_tokens=latent_tokens
        self.matrix_input=nn.Linear(5*rank+15,model_dim)
        self.vector_input=nn.Linear(18,model_dim)
        self.token_type=nn.Embedding(7,model_dim)
        self.norm1=RMSNorm(model_dim);self.blocks=nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.norm2=RMSNorm(model_dim)
        self.matrix_output=nn.Linear(model_dim,rank);self.vector_output=nn.Linear(model_dim,1)
        initialize_parameters_like_record(self.blocks);nn.init.normal_(self.token_type.weight,std=.02)
        for head in self.output_heads:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    @property
    def output_heads(self):return self.matrix_output,self.vector_output

    def make_private(self,snapshot,*,state_key,seed=0):
        return CheckpointProbes(snapshot,model_dim=self.model_dim,latent_tokens=self.latent_tokens,seed=seed,state_key=state_key,frame_mode=self.frame_mode).to(next(self.parameters()).device)

    def configuration(self):
        result=dict(name=self.architecture,rank=self.rank,trajectory_dim=self.trajectory_dim,model_dim=self.model_dim,
            num_layers=self.num_layers,latent_tokens=self.latent_tokens,private='neuron_axis_E_D_and_trajectory_linear',
            fixed_action_anchors=True,column_normalization='differentiable_unit_l2',decoder_scale='sqrt_coordinates_over_latent_tokens',cuda_residual_dtype='bfloat16',parameter_dtype='float32')
        # Preserve the original default metadata exactly; the alternative must
        # be explicitly declared when loading or constructing an experiment.
        if self.write_observation_frame!='encoder':result['write_observation_frame']=self.write_observation_frame
        if self.frame_mode!='independent':
            result.update(frame_mode=self.frame_mode,private='neuron_axis_tied_Q_and_trajectory_linear',
                column_normalization='signed_thin_qr_or_transposed_qr_parseval',
                decoder_scale='sqrt_coordinates_over_min_coordinates_latent_tokens',
                raw_initialization='unit_columns_tall_unit_rows_wide')
        return result

    def _trajectory_features(self,snapshot):
        value=snapshot.trajectories.reshape(snapshot.trajectories.shape[0],-1,2,self.rank).float()
        boundaries=self.trajectory_dim//(2*self.rank);mask=snapshot.trajectory_mask
        if mask is None:mask=value.new_ones(boundaries)
        if tuple(mask.shape)!=(boundaries,):raise ValueError('trajectory mask layout differs')
        mask=mask.to(value);masked=torch.where(mask[None,:,None,None].bool(),value,0.)
        denominator=(mask.sum()*value.shape[0]*self.rank).clamp_min(1)
        scales=(masked.square().sum((0,1,3))/denominator).sqrt().clamp_min(1e-12).detach()
        normalized=(masked/scales[None,None,:,None]).flatten(1)
        return torch.cat((normalized,scales.log().expand(len(value),-1),mask.expand(len(value),-1)),dim=-1)

    def prepare_static_indices(self,snapshot):
        """Opt-in constant staging for eager CUDA capture; no checkpoint fields.

        Call outside capture for every accepted observation layout. Default
        execution retains the original construction/indexing operations.
        """
        device=next(self.parameters()).device
        if any(v.rows.device!=device or v.columns.device!=device for v in snapshot.matrices.values()):
            raise ValueError('static index snapshot and actor device differ')
        layout=observation_layout(snapshot)
        cache=getattr(self,'_static_index_cache',None)
        if cache is None:cache={};self._static_index_cache=cache
        kinds=[kind for _ in snapshot.matrices for kind in (0,1)]
        types=torch.tensor(kinds,device=device).repeat_interleave(self.latent_tokens)
        vectors=torch.tensor([0,2,4,6],device=device,dtype=torch.long)
        cache[layout]=(types,vectors,types._version,vectors._version)

    def _static_indices(self,snapshot,device):
        cache=getattr(self,'_static_index_cache',None)
        if cache is None:return None
        layout=observation_layout(snapshot)
        if layout not in cache:raise ValueError('observation layout has no prepared static indices')
        types,vectors,tv,vv=cache[layout]
        if (types.device!=device or vectors.device!=device or types.dtype!=torch.long or vectors.dtype!=torch.long
                or types.shape!=(2*len(snapshot.matrices)*self.latent_tokens,) or vectors.shape!=(4,)
                or types._version!=tv or vectors._version!=vv):
            raise ValueError('prepared index device/layout/storage values changed')
        return types,vectors

    def _inputs(self,snapshot,private):
        private.validate_layout(snapshot)
        if private.frame_mode!=self.frame_mode:raise ValueError('private frame mode differs from shared actor configuration')
        if (private.model_dim,private.latent_tokens,private.rank,private.trajectory_dim)!=(self.model_dim,self.latent_tokens,self.rank,self.trajectory_dim):
            raise ValueError('private probe architecture differs from shared backbone')
        indices=self._static_indices(snapshot,next(self.parameters()).device)
        matrix=[];kinds=[]
        for i,obs in enumerate(snapshot.matrices.values()):
            matrix.extend((private.matrix_rows[i].encode_with_constants(obs.rows,slice(0,5*self.rank+1)),private.matrix_columns[i].encode_with_constants(obs.columns,slice(0,5*self.rank+1))))
            kinds.extend((0,1))
        values=self.matrix_input(torch.cat(matrix));type_ids=(torch.tensor(kinds,device=values.device).repeat_interleave(self.latent_tokens) if indices is None else indices[0])
        write_values=values
        if self.write_observation_frame=='decoder':
            # Coefficients are decoded through D, so expose the local fields
            # projected through D as well. E-only tokens cannot reveal a change
            # in an independently fitted write frame. This does NOT reveal D's
            # full Gram matrix or components invisible to the cached fields.
            write_matrix=[]
            for i,obs in enumerate(snapshot.matrices.values()):
                write_matrix.extend((private.matrix_rows[i].encode_with_constants(obs.rows,slice(0,5*self.rank+1),source='decoder'),
                                     private.matrix_columns[i].encode_with_constants(obs.columns,slice(0,5*self.rank+1),source='decoder')))
            write_values=self.matrix_input(torch.cat(write_matrix))
        reads=[values+self.token_type(type_ids)];writes=[write_values+self.token_type(type_ids+4)];matrix_count=len(values)
        if snapshot.vectors:
            vectors=self.vector_input(torch.cat([private.vector_frames[i].encode_with_constants(obs.coordinates,[0,2,4,6] if indices is None else indices[1]) for i,obs in enumerate(snapshot.vectors.values())]))
            write_vectors=vectors
            if self.write_observation_frame=='decoder':
                write_vectors=self.vector_input(torch.cat([private.vector_frames[i].encode_with_constants(obs.coordinates,[0,2,4,6] if indices is None else indices[1],source='decoder') for i,obs in enumerate(snapshot.vectors.values())]))
            reads.append(vectors+self.token_type.weight[2]);writes.append(write_vectors+self.token_type.weight[6])
        trajectory=private.trajectory_input(self._trajectory_features(snapshot))
        reads.append(trajectory+self.token_type.weight[3])
        if self.write_observation_frame=='decoder':
            # Causal attention: putting D fields only in each final write token
            # would hide later write frames from earlier decisions. Make ALL
            # write-frame observations available before ANY final write.
            reads.extend(writes)
        read=torch.cat(reads);write=torch.cat(writes)
        return torch.cat((read,write))[None],len(read),matrix_count

    def token_counts(self,snapshot):
        write=(2*len(snapshot.matrices)+len(snapshot.vectors))*self.latent_tokens
        read=write+len(snapshot.trajectories)
        if self.write_observation_frame=='decoder':read+=write
        return dict(observation_tokens=read,write_tokens=write,total_tokens=read+write)

    def forward(self,snapshot,private):
        x,read_count,matrix_count=self._inputs(snapshot,private)
        if x.is_cuda:x=x.bfloat16()
        x=self.norm1(x)
        for block in self.blocks:x=block(x)
        x=self.norm2(x[:,read_count:]).float()[0]
        latent=self.matrix_output(x[:matrix_count]).reshape(-1,self.latent_tokens,self.rank)
        matrices={}
        for i,(name,obs) in enumerate(snapshot.matrices.items()):
            p=private.matrix_rows[i].decode(latent[2*i]);q=private.matrix_columns[i].decode(latent[2*i+1])
            matrices[name]=Factors.canonical(p,q,obs.anchors)
        vectors={}
        if snapshot.vectors:
            latent_vectors=self.vector_output(x[matrix_count:]).reshape(-1,self.latent_tokens,1)
            vectors={name:private.vector_frames[i].decode(latent_vectors[i]).flatten() for i,name in enumerate(snapshot.vectors)}
        return RawAction(matrices,vectors)
