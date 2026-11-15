"""Native-actor thin-GPT gradient graph for explicit offline accumulation.

The graph computes one complete checkpoint panel and applies its mean-KL
coefficient once. It never clips gradients or runs Adam: those belong after
all checkpoint contributions and any distributed SUM. Teacher targets can be
resident FP64 probabilities or reconstructed inside the graph from exact frozen
head inputs. State/cache validation and loading remain outside capture.
"""
from __future__ import annotations
import math
from dataclasses import replace
import torch


class PinnedAddressTables:
    """Own captured HtoD pointer-table sources until the graph is destroyed.

    Addresses describe graph-private allocations; never reuse these tables for
    a different capture or mutate their pinned host contents after capture.
    """
    def __init__(self):self.host_tables=[]

    def __call__(self,values,device):
        host=torch.tensor([value.data_ptr() for value in values],dtype=torch.int64,pin_memory=True)
        self.host_tables.append(host)
        return host.to(device=device,non_blocking=True)

from .algebra import Factors,RawAction
from .compensated_dx import experimental_dx_backend
from .device_primal_dual import DevicePrimalDualInputs,ordered_panel_mean
from .edit_scope import editable_names,scoped_action,keep_readonly_parameters
from .factorized_fp32 import factorized_logits
from .function_preservation import _ChunkedTeacherKL


def snapshot_tensor_items(snapshot):
    result={}
    for name,obs in snapshot.matrices.items():
        for key,value in [('rows',obs.rows),('columns',obs.columns),('A',obs.anchors.A),('B',obs.anchors.B),
                          ('weight_probe0',obs.weight_probes[0]),('weight_probe1',obs.weight_probes[1])]:
            result['matrix:'+name+':'+key]=value
    for name,obs in snapshot.vectors.items():
        result['vector:'+name+':coordinates']=obs.coordinates;result['vector:'+name+':weight']=obs.weight
    for name in ('trajectories','weight_norm','radius'):result[name]=getattr(snapshot,name)
    return result


def clone_static_snapshot(snapshot):
    # Preserve dense strides (notably column-major QR anchors) so the compiled
    # actor keeps the reference GEMM layouts. torch.clone only densifies
    # overlapping/broadcast views, giving each refill independently owned bytes.
    def clone(value):return value.detach().clone(memory_format=torch.preserve_format)
    matrices={name:replace(obs,rows=clone(obs.rows),columns=clone(obs.columns),
        anchors=replace(obs.anchors,A=clone(obs.anchors.A),B=clone(obs.anchors.B)),
        weight_probes=tuple(clone(v) for v in obs.weight_probes)) for name,obs in snapshot.matrices.items()}
    vectors={name:replace(obs,coordinates=clone(obs.coordinates),weight=clone(obs.weight)) for name,obs in snapshot.vectors.items()}
    return replace(snapshot,matrices=matrices,vectors=vectors,trajectories=clone(snapshot.trajectories),
                   weight_norm=clone(snapshot.weight_norm),radius=clone(snapshot.radius))


class NativeThinGradientGraph:
    def __init__(self,actor,actor_forward,snapshot,engine,grouped,inputs,teacher_log_probs,
                 dual,state_id,*,state_weight=1/200,raw_kl_only=False,dx_backend='ieee',shrinkage_coefficient=1.,teacher_features=None,fused_softcap=False,dynamic_zero_groups=False,kl_backend='chunked',linear_backend='ieee',activation_backend='native',retain_metrics=False):
        if inputs.device.type!='cuda' or inputs.ndim!=2:
            raise ValueError('resident CUDA sequence matrix required')
        if (teacher_log_probs is None)==(teacher_features is None):
            raise ValueError('provide exactly one teacher representation')
        if kl_backend not in ('chunked','tiled'):
            raise ValueError('unknown teacher KL backend')
        if kl_backend=='tiled' and teacher_features is None:
            raise ValueError('tiled KL requires exact teacher head features')
        self.kl_backend=kl_backend
        if linear_backend not in ('ieee','tf32x3','packed3'):raise ValueError('unknown linear forward backend')
        self.linear_backend=linear_backend
        if activation_backend not in ('native','fused'):raise ValueError('unknown activation backend')
        self.activation_backend=activation_backend
        self.retain_metrics=retain_metrics;self.metric_tensors=None;self.metric_action=None
        target=teacher_log_probs if teacher_features is None else teacher_features
        if target.shape[:2]!=inputs.shape or target.device!=inputs.device or state_weight<=0:
            raise ValueError('matching teacher panel/device and positive state weight required')
        if teacher_log_probs is not None and target.dtype!=torch.float64:
            raise ValueError('resident teacher probabilities must be exact FP64')
        if teacher_features is not None and (target.dtype!=torch.float32 or engine.gpt_precision!='fp32'):
            raise ValueError('exact FP32 head features required for FP32 teacher reconstruction')
        if dx_backend not in ('ieee','tf32x3','cublas3_large','cublas3_cached'):raise ValueError('unknown dx backend')
        if not math.isfinite(shrinkage_coefficient) or shrinkage_coefficient<0:raise ValueError('nonnegative finite shrinkage coefficient required')
        if len(inputs)%engine.microbatch_sequences:raise ValueError('complete fixed microbatches required')
        self.actor=actor;self.forward=actor_forward;self.snapshot=clone_static_snapshot(snapshot);self.engine=engine;self.grouped=grouped
        self.inputs=inputs;self.teacher=teacher_log_probs;self.teacher_features=teacher_features;self.state_weight=state_weight
        self.raw_kl_only=raw_kl_only;self.dx_backend=dx_backend;self.graph=None;self.fused_softcap=fused_softcap
        self.shrinkage_coefficient=shrinkage_coefficient;self.state_id=state_id;self.binding_valid=True
        self.model_layout={n:(v.data_ptr(),v.shape,v.stride(),v.dtype,v.device) for n,v in {**engine.parameters,**engine.buffers}.items()}
        self.base_versions={n:(v.data_ptr(),v._version) for n,v in {**engine.parameters,**engine.buffers}.items()}
        self.parameters=tuple(actor.parameters());self.structural_editable=editable_names(engine.parameters,'grouped')
        self.editable=self.structural_editable-grouped.frozen_parameter_names
        self.dynamic_zero_groups=dynamic_zero_groups
        self.edit_masks={n:torch.tensor(n in self.editable,dtype=torch.bool,device=inputs.device) for n in self.structural_editable} if dynamic_zero_groups else {}
        self.dual_inputs=DevicePrimalDualInputs(dual,device=inputs.device);self.dual_inputs.bind(dual,state_id)
        self._dual=dual;self._bound_dual_value=dual.multipliers[state_id]
        self._teacher_versions=(inputs._version,target._version)
        self._snapshot_versions={n:v._version for n,v in snapshot_tensor_items(self.snapshot).items()}
        self._actor_addresses=tuple(p.data_ptr() for p in self.parameters)
        # Construction-only constants; no reference reductions or Python scalar
        # extraction are hidden inside the captured/timed contribution.
        from scripts.train_multistate_chisel import group_action_scales
        self.scales=group_action_scales(grouped,actor.rank)
        self.norm_weight=torch.tensor(state_weight,dtype=torch.float64,device=inputs.device)
        self.output=None;self.gradient_buffers=None;self.address_tables=None

    @torch.no_grad()
    def _teacher_slice(self,start,stop):
        if self.teacher_features is None:return self.teacher[start:stop]
        # Same head shape, softcap order and128-token FP64 log-softmax chunks as
        # TeacherPanel.teacher_batch. Provenance validation runs before capture.
        from .teacher_cache import head_logits
        if self.fused_softcap:
            from .fused_softcap import fused_softcap
            raw=torch.nn.functional.linear(self.teacher_features[start:stop],self.engine.parameters['proj.weight'],self.engine.parameters['proj.bias'])
            logits=fused_softcap(raw)
        else:logits=head_logits(self.engine,self.teacher_features[start:stop])
        if self.kl_backend=='tiled':return logits
        probabilities=torch.empty(logits.shape,dtype=torch.float64,device=self.inputs.device)
        flat=logits.flatten(0,1);out=probabilities.flatten(0,1)
        for token in range(0,len(flat),self.engine.chunk_tokens):
            out[token:token+self.engine.chunk_tokens].copy_(torch.nn.functional.log_softmax(
                flat[token:token+self.engine.chunk_tokens].double(),dim=-1))
        return probabilities

    def _body(self):
        action=scoped_action(self.forward(self.snapshot),self.structural_editable if self.dynamic_zero_groups else self.editable)
        if self.dynamic_zero_groups:
            def masked(name,value):
                return torch.where(self.edit_masks[name],value,torch.zeros_like(value)) if name in self.edit_masks else value
            action=RawAction({n:Factors(masked(n,f.P),masked(n,f.Q),f.anchors) for n,f in action.matrices.items()},
                             {n:masked(n,v) for n,v in action.vectors.items()})
        action=RawAction({n:Factors(f.P*self.scales[n],f.Q*self.scales[n],f.anchors)
                          for n,f in action.matrices.items()},
                         {n:v*self.scales[n] for n,v in action.vectors.items()})
        raw_norm=action.norm_sq().clamp_min(1e-24).sqrt()
        scale=torch.minimum(raw_norm.new_ones(()),self.snapshot.radius/(raw_norm+1e-12))
        realized=keep_readonly_parameters(self.engine.edited_parameters(action,scale),self.engine.parameters,
            self.structural_editable if self.dynamic_zero_groups else self.editable)
        if self.dynamic_zero_groups:
            # Select the original stored bits, including signed zero, for every
            # currently frozen group. False branches also carry exactly zero VJP.
            realized={n:torch.where(self.edit_masks[n],v,self.engine.parameters[n]) if n in self.edit_masks else v
                      for n,v in realized.items()}
        norm_result=self.grouped.evaluate(realized,edit_masks=self.edit_masks if self.dynamic_zero_groups else None)
        unweighted=100*norm_result.mean_relative_change
        regularizer=self.shrinkage_coefficient*unweighted
        if self.raw_kl_only:regularizer=regularizer*0
        originals=[];leaves=[]
        def leaf(value):
            detached=value.detach().requires_grad_(value.requires_grad)
            originals.append(value);leaves.append(detached);return detached
        measured=RawAction({n:Factors(leaf(f.P),leaf(f.Q),f.anchors) for n,f in action.matrices.items()},
                           {n:leaf(v) for n,v in action.vectors.items()})
        scale_leaf=leaf(scale);losses=[];sizes=[]
        dx_split_weights=None
        if self.dx_backend=='cublas3_cached':
            # Recompute from this replay's finite edited weights ONCE. No
            # learned/cross-checkpoint cache; preprocessing stays in step cost.
            from .kernels.compensated_pack_triton import split
            with torch.no_grad():
                dx_split_weights={name:split(value.detach().T.contiguous())
                    for name,value in realized.items() if name.endswith('.weight')
                    and name!='embed.weight' and value.ndim==2 and value.shape[0]>=3072}
        for start in range(0,len(self.inputs),self.engine.microbatch_sequences):
            stop=min(start+self.engine.microbatch_sequences,len(self.inputs))
            teacher=self._teacher_slice(start,stop)
            logits=factorized_logits(self.engine.view.gpt,self.engine.parameters,self.engine.buffers,
                self.inputs[start:stop],measured,scale_leaf,realized_parameters=realized,softcap_backend="fused" if self.fused_softcap else "native",linear_backend=self.linear_backend,activation_backend=self.activation_backend,dx_split_weights=dx_split_weights)
            if self.kl_backend=='tiled':
                from .tiled_teacher_kl import tiled_teacher_kl
                # Both operands already have the separately qualified exact
                # softcap. Preserve its autograd derivative; only KL reductions
                # change summation order, with FP64 accumulation throughout.
                loss=tiled_teacher_kl(logits,teacher)
            else:loss=_ChunkedTeacherKL.apply(logits,teacher,self.engine.chunk_tokens)
            losses.append(loss.detach());sizes.append(stop-start)
            (loss*((stop-start)/len(self.inputs))).backward()
            del logits,loss,teacher
        mean_kl=ordered_panel_mean(losses,sizes)
        terms=self.dual_inputs.terms(mean_kl)
        coefficient=torch.ones_like(mean_kl) if self.raw_kl_only else terms['kl_coefficient']
        penalty=mean_kl if self.raw_kl_only else terms['kl_penalty']
        # Production forms coefficient*state_weight in FP64 before the scalar
        # converts to the FP32 cotangent dtype. Do not split it into two FP32
        # gradient multiplications.
        weighted_coefficient=coefficient*self.state_weight
        outputs=[regularizer];cotangents=[self.norm_weight]
        for original,detached in zip(originals,leaves):
            if original.requires_grad and detached.grad is not None:
                outputs.append(original)
                cotangents.append(detached.grad*weighted_coefficient.to(detached.grad.dtype))
        torch.autograd.backward(outputs,cotangents)
        if self.retain_metrics:
            # Retain existing graph allocations, with no extra model pass.
            # Host scalar extraction and optional thin-action diagnostics occur
            # only after replay, outside capture, in the training adapter.
            self.metric_tensors=dict(new_norms={n:v.detach() for n,v in norm_result.new_norms.items()},
                                     edit_norm_sq=norm_result.edit_norm_sq.detach())
            self.metric_action=RawAction({n:Factors(f.P.detach(),f.Q.detach(),f.anchors) for n,f in action.matrices.items()},
                                        {n:v.detach() for n,v in action.vectors.items()})
        return dict(mean_kl=mean_kl,regularizer=regularizer.detach(),unweighted_norm_term=unweighted.detach(),objective=regularizer.detach()+penalty,
                    finite=terms['finite'] & torch.isfinite(regularizer.detach()),
                    kl_coefficient=coefficient,group_relative_change=torch.stack(tuple(norm_result.relative_changes.values())).detach(),
                    raw_action_norm=raw_norm.detach(),global_scale=scale.detach())

    def storage_signature(self):
        values={**{'model:'+n:v for n,v in {**self.engine.parameters,**self.engine.buffers}.items()},
                **{'actor:'+str(i):v for i,v in enumerate(self.parameters)},
                **{'observation:'+n:v for n,v in snapshot_tensor_items(self.snapshot).items()},
                **{'scale:'+n:v for n,v in self.scales.items()},
                **{'mask:'+n:v for n,v in self.edit_masks.items()},'inputs':self.inputs}
        values['teacher']=self.teacher if self.teacher_features is None else self.teacher_features
        for name in ('base_addresses','sizes','bf16','tensor_ids','offsets','starts','ends','group_ids','reference_vector'):
            values['norm:'+name]=getattr(self.grouped,name)
        for name in ('multiplier','tolerance','rho'):values['dual:'+name]=getattr(self.dual_inputs,name)
        if self.gradient_buffers is not None:values.update({'gradient:'+str(i):v for i,v in enumerate(self.gradient_buffers)})
        return tuple((n,v.data_ptr(),tuple(v.shape),tuple(v.stride()),str(v.dtype)) for n,v in values.items())

    def eager(self):
        if self.graph is not None:raise RuntimeError('use a separate instance for eager and captured paths')
        self.actor.zero_grad(set_to_none=True)
        with experimental_dx_backend(self.dx_backend):self.output=self._body()
        return self.output

    def capture(self,*,warmups=2):
        if self.graph is not None or warmups<1:raise ValueError('capture once after at least one warmup')
        current=torch.cuda.current_stream();warm=torch.cuda.Stream();warm.wait_stream(current)
        with torch.cuda.stream(warm):
            for _ in range(warmups):self.eager()
        current.wait_stream(warm);torch.cuda.synchronize()
        self.actor.zero_grad(set_to_none=True)
        self.graph=torch.cuda.CUDAGraph()
        self.address_tables=PinnedAddressTables()
        if hasattr(self.grouped,'address_table'):raise RuntimeError('norm metadata already has a capture owner')
        self.grouped.address_table=self.address_tables
        try:
            with experimental_dx_backend(self.dx_backend),torch.cuda.graph(self.graph):self.output=self._body()
        finally:
            del self.grouped.address_table
        self.gradient_buffers=tuple(p.grad for p in self.parameters)
        if any(g is None for g in self.gradient_buffers):raise RuntimeError('captured actor gradient coverage differs')
        return self

    def replay(self):
        if self.graph is None:raise RuntimeError('capture before replay')
        if not self.binding_valid or any((v.data_ptr(),v._version)!=self.base_versions[n] for n,v in {**self.engine.parameters,**self.engine.buffers}.items()):
            raise RuntimeError('GPT checkpoint changed outside verified graph binding')
        target=self.teacher if self.teacher_features is None else self.teacher_features
        if (self.inputs._version,target._version)!=self._teacher_versions:
            raise RuntimeError('teacher panel changed outside verified graph binding')
        if any(v._version!=self._snapshot_versions[n] for n,v in snapshot_tensor_items(self.snapshot).items()):
            raise RuntimeError('captured observation changed outside verified graph binding')
        if tuple(p.data_ptr() for p in self.parameters)!=self._actor_addresses:
            raise RuntimeError('actor parameter storage changed after capture')
        if self._dual.updates!=self.dual_inputs.bound_update or self._dual.multipliers[self.state_id]!=self._bound_dual_value:
            raise RuntimeError('dual state changed without explicit bind_dual')
        # Optimizer.zero_grad(set_to_none=True) outside this class may clear the
        # Python aliases, but the captured buffers must retain their addresses.
        for p,g in zip(self.parameters,self.gradient_buffers):p.grad=g
        self.graph.replay();return self.output

    @torch.no_grad()
    def accumulate_into(self,buffers):
        """Add one already-weighted contribution to distinct external buffers.

        The caller performs distributed SUM, clipping and Adam only after all
        checkpoint contributions. replay itself overwrites its private grads.
        """
        if len(buffers)!=len(self.parameters):raise ValueError("one accumulation buffer per actor parameter")
        for buffer,parameter in zip(buffers,self.parameters):
            if buffer.shape!=parameter.shape or buffer.dtype!=parameter.dtype or buffer.device!=parameter.device:
                raise ValueError("accumulation buffer metadata mismatch")
        self.replay()
        if any(buffer.data_ptr()==gradient.data_ptr() for buffer,gradient in zip(buffers,self.gradient_buffers)):
            raise ValueError("accumulation buffers must not alias overwritten graph gradients")
        torch._foreach_add_(list(buffers),list(self.gradient_buffers))
        return self.output

    def check_finite(self):
        if self.output is None or not bool(self.output['finite']):raise FloatingPointError('nonfinite panel loss')

    @torch.no_grad()
    def bind_dual(self,dual,state_id):
        if state_id!=self.state_id:raise ValueError('fixed-state graph cannot bind another checkpoint')
        self.dual_inputs.bind(dual,state_id)
        self._dual=dual;self._bound_dual_value=dual.multipliers[state_id]

    def close(self):
        """Release the private pool before another full eager backward."""
        self.actor.zero_grad(set_to_none=True)
        self.gradient_buffers=None;self.output=None
        self.metric_tensors=None;self.metric_action=None
        if self.graph is not None:self.graph.reset()
        self.graph=None;self.address_tables=None

    @torch.no_grad()
    def load_checkpoint(self,cache_reader,record,panel,dual,*,state_constants=None,radius_fraction=.01):
        """Refill the same graph addresses from a SHA-verified checkpoint/panel.

        No learned representation is cached. All frozen weights, buffers,
        observations/anchors, norm references, radius, action scales, input IDs,
        exact teacher features and checkpoint-specific duals are refreshed.
        A changed zero-reference mask is rejected unless dynamic masks are enabled; a failed bind
        invalidates replay until a successful load. No optimizer/accumulator
        state is touched. Call on the replay stream between completed steps.
        """
        # Even pre-copy SHA/schema validation failures revoke replay. Leaving
        # the previous checkpoint valid would hide a failed requested rebind.
        self.binding_valid=False
        from .edit_scope import editable_weight_norm_sq,scoped_snapshot
        from .teacher_cache import validate_precision,_head_signature
        from .resident_teacher import _feature_blocks,_validate_panel
        from scripts.train_multistate_chisel import group_action_scales
        if self.teacher_features is None:raise ValueError('checkpoint reuse requires exact teacher-feature mode')
        from .graph_state_identity import validate_graph_panel_identity
        state_id,sha=validate_graph_panel_identity(record,panel)
        validate_precision(panel.metadata,self.engine)
        if panel.metadata['microbatch_sequences']!=self.engine.microbatch_sequences:
            raise ValueError('teacher generation microbatch shape changed')
        blocks=list(_feature_blocks(panel))
        if not blocks or panel.inputs.shape!=self.inputs.shape or (len(panel.inputs),*blocks[0].shape[1:])!=self.teacher_features.shape:
            raise ValueError('fixed graph panel shape changed; select matching complete blocks')
        if any(b.shape!=blocks[0].shape or b.dtype!=self.teacher_features.dtype for b in blocks):
            raise ValueError('teacher feature block schema changed')
        incoming=snapshot_tensor_items(record.snapshot);owned=snapshot_tensor_items(self.snapshot)
        if incoming.keys()!=owned.keys() or any(v.shape!=owned[n].shape or v.dtype!=owned[n].dtype for n,v in incoming.items()):
            raise ValueError('checkpoint observation layout changed')
        if state_id not in dual.multipliers:raise ValueError('fitting checkpoint multiplier required')
        self.binding_valid=False
        if state_constants is None:
            cache_reader.load_engine(self.engine,record)
            grouped=type(self.grouped)(self.engine.parameters,zero_reference_policy='freeze')
        else:
            if state_constants.engine is not self.engine:raise ValueError('state constants belong to another engine')
            state_constants.load(cache_reader,record);grouped=state_constants.grouped()
        layout={n:(v.data_ptr(),v.shape,v.stride(),v.dtype,v.device) for n,v in {**self.engine.parameters,**self.engine.buffers}.items()}
        if layout!=self.model_layout:raise ValueError('frozen model storage/layout changed')
        editable=editable_names(self.engine.parameters,'grouped')-grouped.frozen_parameter_names
        if grouped.groups!=self.grouped.groups or (not self.dynamic_zero_groups and editable!=self.editable):
            raise ValueError('checkpoint editable/zero-reference group structure changed')
        if self.dynamic_zero_groups:
            for name,mask in self.edit_masks.items():mask.fill_(name in editable)
            self.editable=editable
            self.grouped.frozen_parameter_names=grouped.frozen_parameter_names
            self.grouped.zero_reference_groups=grouped.zero_reference_groups
        norm_sq=(editable_weight_norm_sq(self.engine.parameters,editable) if state_constants is None
                 else state_constants.radius_squared(grouped))
        scoped=scoped_snapshot(record.snapshot,norm_sq,radius_fraction,scope='grouped')
        for name,value in snapshot_tensor_items(scoped).items():owned[name].copy_(value)
        # The captured kernels hold these original reference-vector addresses.
        self.grouped.reference_vector.copy_(grouped.reference_vector)
        for name,value in grouped.reference_squared.items():self.grouped.reference_squared[name].copy_(value)
        for name,value in grouped.reference_norms.items():self.grouped.reference_norms[name].copy_(value)
        self.grouped._reference_versions={n:v._version for n,v in self.engine.parameters.items()}
        for name,value in group_action_scales(grouped,self.actor.rank).items():self.scales[name].copy_(value)
        _validate_panel(panel,self.engine)
        self.inputs.copy_(panel.inputs)
        size=self.engine.microbatch_sequences
        for index,block in enumerate(blocks):self.teacher_features[index*size:(index+1)*size].copy_(block)
        if getattr(self.engine,'_teacher_cache_binding',None)!=(self.engine.teacher_id,sha):
            raise ValueError('engine lacks matching verified teacher binding')
        if getattr(self.engine,'_teacher_cache_head_signature',None)!=_head_signature(self.engine):
            raise ValueError('verified teacher head changed during bind')
        self.dual_inputs.bind(dual,state_id);self.state_id=state_id
        self._dual=dual;self._bound_dual_value=dual.multipliers[state_id]
        self._teacher_versions=(self.inputs._version,self.teacher_features._version)
        self._snapshot_versions={n:v._version for n,v in snapshot_tensor_items(self.snapshot).items()}
        self.base_versions={n:(v.data_ptr(),v._version) for n,v in {**self.engine.parameters,**self.engine.buffers}.items()}
        self.binding_valid=True
        return dict(state_id=state_id,state_sha256=sha,sequence_ids=tuple(panel.sequence_ids))
