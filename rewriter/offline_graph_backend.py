"""Explicit offline accumulation adapter for the qualified strict graph path.

The adapter owns only local, globally normalized gradient accumulation. The
trainer retains checkpoint/panel selection, global SUM, clipping, saved Adam,
dual ascent, audits and checkpointing. No optimizer step occurs here.
"""
from __future__ import annotations

import math
import time
import torch

from .edit_scope import scoped_snapshot
from .graph_offline import NativeThinGradientGraph
from .resident_teacher import ResidentTeacherFeatures
from .graph_state_identity import training_state_id, validate_graph_panel_identity


def validate_training_options(args):
    """Fail before CUDA/data setup when the explicit backend is out of scope."""
    required=dict(gpt_precision='fp32',norm_backend='fused',objective='primal-dual',
                  actor_backend='transformer',device='cuda',
                  sequences_per_checkpoint=64,gpt_microbatch=16,checkpoint_batch=200)
    for key,value in required.items():
        if getattr(args,key)!=value:
            raise ValueError(f'strict_graph currently requires {key}={value}')
    if args.shrinkage_coefficient not in (0.,1.):
        raise ValueError('strict_graph requires shrinkage_coefficient 0 or 1')
    if not (getattr(args,'teacher_cache',None) or getattr(args,'multibank_index',None)) or not args.offline_systems_cache:
        raise ValueError('strict_graph requires teacher-cache and offline-systems-cache')
    if args.eager_chisel or args.compile_gpt or args.gradient_diagnostics:
        raise ValueError('strict_graph requires compiled Chisel, eager GPT and no component-gradient diagnostics')
    if not args.no_component_timing:
        raise ValueError('strict_graph requires no-component-timing; complete update wall time remains logged')


class OfflineGraphAccumulator:
    """Reusable strict FP32 GPT graph, qualified for preservation repair and unit shrinkage reward.

    begin_update(local_ids, global_count) -> accumulate(record, panel)* ->
    finish_update(). The caller then performs global SUM/clip/Adam/dual once.
    close() releases the private pool before ordinary eager backward/audits.
    Changed panel shapes or global batch sizes explicitly cause recapture.
    """
    def __init__(self, actor, actor_forward, engine, cache_reader, state_constants,
                 *, radius_fraction, shrinkage_coefficient=0., capture_warmups=2):
        if engine.gpt_precision != 'fp32' or engine.kl_dtype != torch.float64:
            raise ValueError('offline graph requires FP32 GPT and FP64 KL')
        if engine.device.type != 'cuda':
            raise ValueError('offline graph requires CUDA')
        if state_constants.engine is not engine:
            raise ValueError('state constants must belong to this GPT engine')
        if shrinkage_coefficient not in (0.,1.):
            raise ValueError('production graph requires shrinkage_coefficient 0 or 1')
        if not math.isfinite(radius_fraction) or radius_fraction <= 0 or capture_warmups < 1:
            raise ValueError('positive radius and capture warmup count required')
        self.actor=actor;self.forward=actor_forward;self.engine=engine
        self.cache=cache_reader;self.constants=state_constants
        self.radius_fraction=radius_fraction;self.capture_warmups=capture_warmups
        self.shrinkage_coefficient=float(shrinkage_coefficient)
        self.parameters=tuple(actor.parameters())
        self.buffers=[torch.zeros_like(p) for p in self.parameters]
        self.graph=None;self._schema=None;self._expected=None;self._seen=set()
        self._failed=False;self.capture_count=0;self.capture_seconds=0.
        self.compute={}

    def begin_update(self, local_state_ids, *, global_checkpoint_count):
        ids=tuple(local_state_ids)
        if self._expected is not None:
            raise RuntimeError('finish or abort the active graph accumulation first')
        if (len(ids)!=len(set(ids)) or type(global_checkpoint_count) is not int
                or global_checkpoint_count < max(1,len(ids))):
            raise ValueError('unique local IDs and actual positive global checkpoint count required')
        self._expected=set(ids);self._seen=set();self._failed=False
        self.global_count=global_checkpoint_count
        torch._foreach_zero_(self.buffers)
        self.actor.zero_grad(set_to_none=True)

    def _count_execution(self, inputs, executions=1, *, setup=False):
        micros=len(inputs)//self.engine.microbatch_sequences
        values=dict(student_forwards=micros,student_backwards=micros,
                    student_tokens=inputs.numel(),student_backward_tokens=inputs.numel(),
                    cached_teacher_head_forwards=micros,cached_teacher_head_tokens=inputs.numel(),
                    chisel_forwards=1,chisel_backwards=1)
        for key,value in values.items():
            self.compute[key]=self.compute.get(key,0)+executions*value
            detail=('graph_setup_' if setup else 'graph_training_')+key
            self.compute[detail]=self.compute.get(detail,0)+executions*value

    def _capture(self, record, panel, dual):
        self._release_graph()
        state, _ = validate_graph_panel_identity(record, panel)
        self.constants.load(self.cache,record);grouped=self.constants.grouped()
        snapshot=scoped_snapshot(record.snapshot,self.constants.radius_squared(grouped),
                                 self.radius_fraction,scope='grouped')
        resident=ResidentTeacherFeatures(panel,self.engine)
        graph=NativeThinGradientGraph(self.actor,self.forward,snapshot,self.engine,grouped,
            resident.inputs,None,dual,state,state_weight=1/self.global_count,
            teacher_features=resident.features,shrinkage_coefficient=self.shrinkage_coefficient,
            fused_softcap=True,dynamic_zero_groups=True,kl_backend='tiled',retain_metrics=True)
        self.graph=graph
        started=time.perf_counter()
        graph.capture(warmups=self.capture_warmups)
        torch.cuda.synchronize(self.engine.device)
        self.capture_seconds+=time.perf_counter()-started;self.capture_count+=1
        # CUDA stream capture records work without executing it. Count only
        # the actual eager warmups; every subsequent replay is counted below.
        self._count_execution(resident.inputs,self.capture_warmups,setup=True)
        self._signature=graph.storage_signature()

    def accumulate(self, record, panel, dual):
        state=training_state_id(record)
        if self._expected is None or self._failed:
            raise RuntimeError('begin a valid update before accumulating')
        if state not in self._expected or state in self._seen:
            raise ValueError('checkpoint must occur exactly once in the declared local batch')
        if state not in dual.multipliers:
            raise ValueError('only fitting checkpoints may accumulate gradients')
        schema=(tuple(panel.inputs.shape),self.global_count)
        try:
            if self.graph is None or schema!=self._schema:
                self._capture(record,panel,dual);self._schema=schema
            else:
                self.graph.load_checkpoint(self.cache,record,panel,dual,state_constants=self.constants,
                                           radius_fraction=self.radius_fraction)
                if self.graph.storage_signature()!=self._signature:
                    raise RuntimeError('graph storage addresses changed during checkpoint reuse')
            output=self.graph.accumulate_into(self.buffers)
            self.graph.check_finite()
            self._count_execution(self.graph.inputs)
            measured,stats=self._metrics(output,dual,state)
        except Exception:
            self._failed=True
            raise
        self._seen.add(state)
        return measured,stats

    def _metrics(self, output, dual, state):
        from scripts.train_multistate_chisel import proposed_group_reductions
        graph=self.graph;grouped=graph.grouped
        kl=float(output['mean_kl']);terms=dual.terms(kl,state)
        measured=dict(mean_kl=kl,objective=float(output['objective']),**terms)
        groups=tuple(grouped.groups)
        changes=output['group_relative_change'].tolist()
        raw=float(output['raw_action_norm']);scale=float(output['global_scale'])
        stats=dict(mean_group_l2_percentage_reduction=-float(output['unweighted_norm_term']),
            group_l2_percentage_reduction={n:-100*v for n,v in zip(groups,changes)},
            group_new_l2_norm={n:float(v) for n,v in graph.metric_tensors['new_norms'].items()},
            norm_term=float(output['regularizer']),unweighted_norm_term=float(output['unweighted_norm_term']),
            shrinkage_coefficient=self.shrinkage_coefficient,trust_term=0.,raw_action_norm=raw,global_scale=scale,
            hard_radius=float(graph.snapshot.radius),radius_saturated=scale<1.,
            edit_size_penalty_enabled=False,realized_edit_norm=float(graph.metric_tensors['edit_norm_sq'].sqrt()),
            zero_reference_groups=list(grouped.zero_reference_groups),
            frozen_zero_group_parameters=sorted(grouped.frozen_parameter_names),
            proposed_edit_norm=float((output['raw_action_norm']*output['global_scale']).detach()),
            proposed_group_l2_percentage_reduction=proposed_group_reductions(
                graph.metric_action,output['global_scale'],graph.snapshot,grouped),
            offline_backend='strict_graph',graph_capture_count=self.capture_count,
            graph_capture_seconds=self.capture_seconds)
        return measured,stats

    def finish_update(self):
        if self._expected is None or self._failed or self._seen!=self._expected:
            raise RuntimeError('cannot publish an incomplete or failed local accumulation')
        if not bool(torch.stack([torch.isfinite(g).all() for g in self.buffers]).all()):
            self._failed=True
            raise FloatingPointError('nonfinite accumulated graph gradient')
        for parameter,gradient in zip(self.parameters,self.buffers):parameter.grad=gradient
        self._expected=None
        return self.buffers

    def _release_graph(self):
        if self.graph is not None:self.graph.close()
        self.graph=None;self._schema=None

    def close(self):
        """Abort any incomplete batch and release capture before eager evaluation."""
        self._release_graph();self._expected=None;self._seen=set();self._failed=False
        self.actor.zero_grad(set_to_none=True)
