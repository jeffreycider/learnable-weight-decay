"""Explicit adapter for unwrapped PR323 Muon/AdamW, single authoritative GPU.

The baseline backward uses token-SUM CE. Controller task signals divide by the
GLOBAL token count; sequence hooks divide only by sequence length. Neither
conversion changes .grad passed to the original optimizers.

Checkpoints preserve exact trajectory projections and global token positions.
Optional ``rewriter.checkpoint_train_measurements`` (default False) also exports
the latest nonterminal decision's frozen snapshot and four train-probe banks,
separately from pending-credit audit data. At PR323 r=8 the train banks require
4 * (8 * 268032 + 153216) FP32 values = 36,759,552 bytes (35.06 MiB).
When pending credit shares the snapshot, serialization reuses that object;
warmup/shadow/terminal exports also retain a full snapshot. Each diagnostic
record reports actual unique tensor-storage bytes, including that distinction.
No measurements are acquired when saving, and none of this evidence feeds back
to the actor or to optimization.
"""
import math
import warnings
from dataclasses import fields, is_dataclass
import torch
from torch import nn
from .algebra import Anchors, Measurements
from .controller import (GlobalRewriter, ControllerSnapshot, MatrixObservation,
                         VectorObservation, coordinate_features, vector_features)
from .online import OnlineRewriter, PendingCredit
from .sensors import SensorCollector
from .edit_scope import editable_names


def _checkpoint_tensor_copy(value, device='cpu'):
    """Independent detached storage for observations exported to a checkpoint."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, copy=True)
    if is_dataclass(value):
        return type(value)(**{f.name: _checkpoint_tensor_copy(getattr(value, f.name), device) for f in fields(value)})
    if isinstance(value, dict):
        return {key: _checkpoint_tensor_copy(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_checkpoint_tensor_copy(item, device) for item in value)
    if isinstance(value, list):
        return [_checkpoint_tensor_copy(item, device) for item in value]
    return value


def _checkpoint_tensor_bytes(value):
    """Count actual unique backing storages, including any padding or views."""
    storages = {}
    def visit(item):
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            storages[(str(item.device), storage.data_ptr())] = storage.nbytes()
        elif is_dataclass(item):
            for f in fields(item):
                visit(getattr(item, f.name))
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
    visit(value)
    return sum(storages.values())


class RewriterAdapter:
    def __init__(self, model, optimizers, config, mode='learned'):
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise ValueError('distributed controller synchronization not yet certified')
        self.config, self.c, self.mode = config, config.get('rewriter', {}), mode
        self.params = dict(model.named_parameters())  # shared parameters enumerated once
        self.editable_scope=self.c.get('editable_scope','all')
        self.editable=editable_names(self.params,self.editable_scope)
        self.device = next(model.parameters()).device
        self.rank = self.c.get('probe_rank', 8)
        self.embedding_names = {name + '.weight' for name, m in model.named_modules()
                                if isinstance(m, nn.Embedding)}
        self.matrix_names = [n for n, p in self.params.items() if p.ndim == 2]
        self.vector_names = [n for n, p in self.params.items() if p.ndim == 1]
        if any(p.ndim not in (1, 2) for p in self.params.values()):
            raise ValueError('only matrix and vector parameters supported')
        self.anchor_rng = torch.Generator().manual_seed(self.c.get('anchor_seed', 100001))
        self.anchors = self.make_anchors()
        # fork_rng restores both model CPU and device RNG; anchor/sign RNG separate.
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.c.get('controller_seed', 200001))
            self.controller = GlobalRewriter(rank=self.rank,
                trajectory_dim=(2 * config['model']['num_layers'] + 1) * 2 * self.rank,
                coordinate_width=self.c.get('coordinate_width', 64),
                global_width=self.c.get('global_width', 128),
                pool_queries=self.c.get('pool_queries', 4), latents=self.c.get('global_latents', 64),
                blocks=self.c.get('global_blocks', 3), heads=self.c.get('attention_heads', 4),
                ff_width=self.c.get('ff_width', 512)).to(self.device)
        self.online = OnlineRewriter(self.controller, self.c)
        self.collector = SensorCollector(model, {n:(a.A,a.B) for n,a in self.anchors.items()},
            num_sequences=config['batch_tokens']//config['seq_len'], sequence_length=config['seq_len'],
            signs_seed=self.c.get('signs_seed', 300001), trajectory_rank=self.rank,
            trajectory_positions=self.c.get('trajectory_positions', 64))
        self.optimizer_for = {id(p):(o,g) for o in optimizers for g in o.param_groups for p in g['params']}
        self.metrics = {}
        # Optional evidence for an offline diagnostic solver, never an actor
        # input or additional task evaluation. Kept only until superseded.
        self._diagnostic_training = None
        from .edit_diagnostics import EditDiagnostics
        forward_dtype = model.embed.weight.dtype if hasattr(model, 'embed') else next(model.parameters()).dtype
        self.diagnostics = EditDiagnostics(self.params, self.device, forward_dtype=forward_dtype) if self.c.get('diagnostics', False) else None

    def oriented(self, name, tensor):
        return tensor.T if name in self.embedding_names else tensor

    def make_anchors(self):
        return {n: Anchors.random(*self.oriented(n, self.params[n]).shape, self.rank,
                    generator=self.anchor_rng, device=self.device) for n in self.matrix_names}

    def before_step(self, t, model, optimizers):
        period = self.c.get('anchor_refresh_period', 32)
        if t and period and t % period == 0:
            # Seeded-random refresh v1; no retroactive current-gradient sensing.
            self.anchors = self.make_anchors()
        self.collector.reset({n:(a.A,a.B) for n,a in self.anchors.items()}, step=t)

    def before_microbatch(self, t, sequence_ids, inputs):
        self.collector.set_microbatch(sequence_ids, inputs)

    @torch.no_grad()
    def _project_gradients(self, anchors):
        denom = self.config['batch_tokens']
        matrix = {}
        for n, a in anchors.items():
            g = self.oriented(n, self.params[n].grad).float()
            matrix[n] = (g @ a.B / denom, g.T @ a.A / denom)
        vector = {n: self.params[n].grad.float().clone() / denom for n in self.vector_names}
        return matrix, vector

    def after_backward(self, t, model):
        self.sensors = self.collector.finalize()
        self.gradient_banks, self.vector_gradients = self._project_gradients(self.anchors)
        if self.online.pending is not None:
            old = {n:f.anchors for n,f in self.online.pending.action.matrices.items()}
            old_m, old_v = self._project_gradients(old)
            self.online.consume_credit(old_m, old_v)
        self.credit_metrics = dict(self.online.metrics)
        self.raw_gradient_norms = {n: float(p.grad.float().norm() / self.config['batch_tokens']) for n,p in self.params.items()}
        # Dense copies are optimizer-envelope bookkeeping, never controller inputs.
        # Needed to measure complete displacement including decay and BF16 rounding.
        self.before = {n:p.detach().clone() for n,p in self.params.items()}

    def _metadata(self, name, index, t, activity_scale=0.):
        p = self.params[name]
        opt, group = self.optimizer_for[id(p)]
        rows, cols = (self.oriented(name, p).shape if p.ndim == 2 else (p.numel(), 1))
        types = ['embed', '.attn.q.', '.attn.k.', '.attn.v.', '.attn.proj.', '.mlp.fc.', '.mlp.proj.', 'gains', 'bias', 'proj.weight']
        type_id = next((i for i, key in enumerate(types) if key in name), len(types))
        return [index / max(1, len(self.params)-1), math.log1p(rows), math.log1p(cols),
                t / self.config.get('schedule_steps', self.config['max_steps']),
                float(group['lr']), float(group.get('mu', group.get('betas', (0.,))[0])),
                type_id / len(types), 0., math.log(max(self.raw_gradient_norms[name], 1e-12)), activity_scale]

    @torch.no_grad()
    def _snapshot(self, t):
        matrix, vector = {}, {}
        train_m, audit_m, train_v, audit_v = {}, {}, {}, {}
        weight_sq, update_sq = torch.zeros((), device=self.device), torch.zeros((), device=self.device)
        total_weight_sq,total_update_sq=torch.zeros((),device=self.device),torch.zeros((),device=self.device)
        allones_error, allones_scale = torch.zeros((), device=self.device), torch.zeros((), device=self.device)
        family_errors = {}
        if self.diagnostics:
            self.diagnostics.begin_reversal()
        for index, (name, parameter) in enumerate(self.params.items()):
            w = self.oriented(name, parameter.detach()).float()
            before = self.oriented(name, self.before[name]).float()
            u = w - before
            if self.diagnostics:
                self.diagnostics.observe_base_update(name, u)
            wnorm_sq = w.square().sum()
            unorm_sq=u.square().sum()
            total_weight_sq+=wnorm_sq
            total_update_sq+=unorm_sq
            if name in self.editable:
                weight_sq += wnorm_sq
                update_sq += unorm_sq
            opt, group = self.optimizer_for[id(parameter)]
            state = opt.state[parameter]
            momentum = state.get('momentum', state.get('exp_avg'))
            if momentum is None:
                raise RuntimeError(f'no parameter-coordinate momentum for {name}')
            m = self.oriented(name, momentum).float() / self.config['batch_tokens']
            meta = self._metadata(name, index, t)
            meta[7] = float(wnorm_sq.sqrt().clamp_min(1e-12).log())
            family = 'embedding' if 'embed' in name else ('head' if name.startswith('proj.') else ('hidden_matrices' if parameter.ndim==2 else 'vectors'))
            if family not in family_errors:
                family_errors[family] = [torch.zeros((),device=self.device),torch.zeros((),device=self.device)]
            if parameter.ndim == 2:
                a, s = self.anchors[name], self.sensors.matrix[name]
                wp = ((w @ a.B).clone(), (w.T @ a.A).clone())
                gp = self.gradient_banks[name]
                mp, up = (m @ a.B, m.T @ a.A), (u @ a.B, u.T @ a.A)
                rows = coordinate_features([wp[0],gp[0],mp[0],up[0]], a.A, s.output_rms, meta)
                cols = coordinate_features([wp[1],gp[1],mp[1],up[1]], a.B, s.input_rms, meta)
                matrix[name] = MatrixObservation(a, rows, cols, wp)
                train_m[name], audit_m[name] = s.train, s.audit
                for measured, expected in zip(s.ones, gp):
                    expected = expected * math.sqrt(self.config['batch_tokens']/self.config['seq_len'])
                    allones_error += (measured - expected).square().sum()
                    allones_scale += expected.square().sum()
                    family_errors[family][0] += (measured-expected).square().sum()
                    family_errors[family][1] += expected.square().sum()
            else:
                if name not in self.sensors.vectors:
                    raise RuntimeError(f'cannot edit unmeasured vector {name}')
                s = self.sensors.vectors[name]
                vector[name] = VectorObservation(vector_features([w,self.vector_gradients[name],m,u],meta), w.clone())
                train_v[name], audit_v[name] = s.train, s.audit
                expected = self.vector_gradients[name] * math.sqrt(self.config['batch_tokens']/self.config['seq_len'])
                family_errors[family][0] += (s.ones-expected).square().sum()
                family_errors[family][1] += expected.square().sum()
        if self.diagnostics:
            self.metrics.update(self.diagnostics.finish_reversal())
        # 1e-8 parameter norm is only a positive numerical radius floor; semantic
        # loss tolerance is separate. At zero base motion this remains negligible.
        radius = (self.c.get('radius_fraction', .25) * update_sq.sqrt()).clamp_min(
                    self.c.get('radius_floor', 1e-8))
        self.metrics.update(base_update_norm=float(update_sq.sqrt()), weight_norm=float(weight_sq.sqrt()),
            total_weight_norm=float(total_weight_sq.sqrt()),total_base_update_norm=float(total_update_sq.sqrt()),
            editable_scope=self.editable_scope,
            radius=float(radius), allones_relative_error=float((allones_error/allones_scale.clamp_min(1e-24)).sqrt()),
            controller_parameters=sum(p.numel() for p in self.controller.parameters()),
            matrix_count=len(matrix), vector_count=len(vector),
            actor_snapshot_elements=sum(x.rows.numel()+x.columns.numel()+x.anchors.A.numel()+x.anchors.B.numel()
                +sum(y.numel() for y in x.weight_probes) for x in matrix.values())
                +sum(x.coordinates.numel()+x.weight.numel() for x in vector.values())
                +self.sensors.trajectories.numel())
        self.metrics.update({f'allones_relative_error_{key}':float((error/scale.clamp_min(1e-24)).sqrt()) for key,(error,scale) in family_errors.items()})
        return (ControllerSnapshot(matrix, vector, self.sensors.trajectories.detach().clone(),
                                   weight_sq.sqrt(), radius),
                Measurements(train_m,train_v), Measurements(audit_m,audit_v))

    def after_step(self, t, model, optimizers):
        self.metrics = {k:v for k,v in self.credit_metrics.items() if k.startswith('task_')}
        if t == self.config['max_steps'] - 1:
            # Consume prior credit in after_backward, then perform base update only.
            self.metrics.update(terminal_identity=True, meta_updates=self.online.updates)
            if self.diagnostics:
                self.diagnostics.begin_reversal()
                for name, parameter in self.params.items():
                    # Terminal bypasses _snapshot but still observes the final base motion.
                    after = self.oriented(name, parameter.detach())
                    before = self.oriented(name, self.before[name])
                    update = after.float() - before.float()
                    self.diagnostics.observe_base_update(name, update)
                self.metrics.update(self.diagnostics.finish_reversal())
                self.metrics.update(self.diagnostics.identity_metrics())
            del self.before, self.sensors, self.gradient_banks, self.vector_gradients
            return
        snapshot, train, audit = self._snapshot(t)
        if self.c.get('checkpoint_train_measurements', False):
            # Snapshot and finalized train banks already have independent,
            # detached storage. All subsequent uses are read-only. The actual
            # checkpoint export below makes independent CPU copies.
            self._diagnostic_training = dict(step=t, snapshot=snapshot, measurements=train,
                loss_units='sequence mean CE in nats/token',
                num_sequences=self.collector.N, sequence_length=self.collector.T)
        del self.before, self.sensors, self.gradient_banks, self.vector_gradients
        pending = self.online.propose(snapshot, train, audit, t, self.config['max_steps'], shadow=self.mode=='shadow')
        self.metrics.update(self.online.metrics)
        if pending is not None:
            self._commit(pending)
        elif self.diagnostics:
            self.metrics.update(self.diagnostics.identity_metrics())

    @torch.no_grad()
    def _commit(self, pending):
        realized_sq, rounding_sq = torch.zeros((), device=self.device), torch.zeros((), device=self.device)
        kappa = pending.kappa.detach()
        readonly_versions={n:p._version for n,p in self.params.items() if n not in self.editable}
        if self.diagnostics:
            self.diagnostics.begin_commit()
        by_family = {}
        def add_family(name, proposed_sq, actual_sq, error_sq):
            family = 'embedding' if 'embed' in name else ('head' if name.startswith('proj.') else ('hidden_matrices' if self.params[name].ndim==2 else 'vectors'))
            if family not in by_family:
                by_family[family] = [torch.zeros((),device=self.device) for _ in range(3)]
            for i,value in enumerate((proposed_sq,actual_sq,error_sq)):
                by_family[family][i] += value
        for name, f in pending.action.matrices.items():
            if name not in self.editable:
                continue  # Preserve exact stored bits; do not add even +0.
            target = self.oriented(name, self.params[name])
            if self.diagnostics:
                self.diagnostics.prepare_parameter(name, target)
            # Chunk output coordinates: no vocabulary-sized differentiable graph.
            for start in range(0, target.shape[0], 512):
                end = min(start+512, target.shape[0])
                delta = kappa * (f.P[start:end] @ f.anchors.B.T + f.anchors.A[start:end] @ f.Q.T)
                before = target[start:end].float().clone()
                target[start:end].copy_((before + delta).to(target.dtype))
                actual = target[start:end].float() - before
                if self.diagnostics:
                    self.diagnostics.record_chunk(name, start, before, target[start:end], actual)
                realized_sq += actual.square().sum()
                rounding_sq += (actual-delta).square().sum()
                add_family(name,delta.square().sum(),actual.square().sum(),(actual-delta).square().sum())
        for name, delta in pending.action.vectors.items():
            if name not in self.editable:
                continue
            target = self.params[name]
            before = target.float().clone()
            desired = kappa * delta
            target.copy_((before+desired).to(target.dtype))
            actual = target.float()-before
            if self.diagnostics:
                self.diagnostics.prepare_parameter(name, target)
                self.diagnostics.record_chunk(name, 0, before, target, actual)
            realized_sq += actual.square().sum()
            rounding_sq += (actual-desired).square().sum()
            add_family(name,desired.square().sum(),actual.square().sum(),(actual-desired).square().sum())
        if self.diagnostics:
            self.metrics.update(self.diagnostics.finish_commit())
        if any(self.params[n]._version!=version for n,version in readonly_versions.items()):
            raise RuntimeError('controller commit modified a read-only parameter')
        self.metrics['read_only_parameters_unchanged']=True
        self.metrics.update(realized_edit_norm=float(realized_sq.sqrt()), casting_error_norm=float(rounding_sq.sqrt()))
        for family, values in by_family.items():
            for label,value in zip(('proposed_edit_norm','realized_edit_norm','casting_error_norm'),values):
                self.metrics[f'{label}_{family}'] = float(value.sqrt())

    def state_dict(self):
        pending = self.online.pending
        # Save detached small replay inputs, never the retained autograd graph.
        pending_state = None if pending is None else dict(
            snapshot=_checkpoint_tensor_copy(pending.snapshot),
            audit=_checkpoint_tensor_copy(pending.audit), warmup=pending.warmup)
        diagnostic = None
        if self.c.get('checkpoint_train_measurements', False) and self._diagnostic_training is not None:
            record = self._diagnostic_training
            shared = pending is not None and record['snapshot'] is pending.snapshot
            snapshot = pending_state['snapshot'] if shared else _checkpoint_tensor_copy(record['snapshot'])
            measurements = _checkpoint_tensor_copy(record['measurements'])
            diagnostic = dict(step=record['step'], snapshot=snapshot, measurements=measurements,
                loss_units=record['loss_units'], num_sequences=record['num_sequences'],
                sequence_length=record['sequence_length'], snapshot_shared_with_pending=shared,
                measurements_tensor_bytes=_checkpoint_tensor_bytes(measurements),
                snapshot_tensor_bytes=_checkpoint_tensor_bytes(snapshot))
            diagnostic['additional_checkpoint_tensor_bytes'] = diagnostic['measurements_tensor_bytes'] + (
                0 if shared else diagnostic['snapshot_tensor_bytes'])
        # Save the actual QR basis and chosen positions: seeds alone do not
        # guarantee identical QR signs/RNG algorithms on another software host.
        trajectory_sensors = dict(omega=_checkpoint_tensor_copy(self.collector.omega),
            positions=_checkpoint_tensor_copy(self.collector.positions),
            num_sequences=self.collector.N, sequence_length=self.collector.T,
            signs_seed=self.collector.seed, train_probes=self.collector.nt, audit_probes=self.collector.na)
        return dict(controller=self.controller.state_dict(), optimizer=self.online.optimizer.state_dict(),
            editable_scope=self.editable_scope,
            updates=self.online.updates, aux_evaluations=self.online.aux_evaluations,
            anchor_rng=self.anchor_rng.get_state(), anchors=self.anchors,
            diagnostics=self.diagnostics.state_dict() if self.diagnostics else None,
            trajectory_sensors=trajectory_sensors, diagnostic_training=diagnostic,
            pending=pending_state)

    def load_state_dict(self, state):
        from .algebra import guard
        if state.get('editable_scope','all')!=self.editable_scope:
            raise ValueError('checkpoint editable scope differs from this run')
        self.controller.load_state_dict(state['controller'])
        self.online.optimizer.load_state_dict(state['optimizer'])
        self.online.updates=state['updates']; self.online.aux_evaluations=state['aux_evaluations']
        self.anchor_rng.set_state(state['anchor_rng'].cpu())
        self.anchors=_checkpoint_tensor_copy(state['anchors'], self.device)
        sensor_state = state.get('trajectory_sensors')
        if sensor_state is None:
            warnings.warn('legacy checkpoint omits exact trajectory projection/positions; '
                          'reconstructing from seeds may differ across software hosts', RuntimeWarning)
        else:
            expected = dict(num_sequences=self.collector.N, sequence_length=self.collector.T,
                            signs_seed=self.collector.seed, train_probes=self.collector.nt,
                            audit_probes=self.collector.na)
            if any(sensor_state.get(key) != value for key, value in expected.items()):
                raise ValueError('checkpoint trajectory sensor configuration differs from this run')
            omega, positions = sensor_state['omega'], sensor_state['positions']
            if (omega is None) != (self.collector.omega is None):
                raise ValueError('checkpoint trajectory projection presence differs from this model')
            if omega is not None:
                if omega.shape != self.collector.omega.shape or not bool(torch.isfinite(omega).all()):
                    raise ValueError('invalid checkpoint trajectory projection')
                self.collector.omega = omega.detach().to(device=self.device, dtype=torch.float32, copy=True)
            if (positions.shape != self.collector.positions.shape or positions.dtype != torch.int64
                    or positions.unique().numel() != positions.numel()
                    or bool(((positions < 0) | (positions >= self.collector.N * self.collector.T)).any())):
                raise ValueError('invalid checkpoint globally sampled token positions')
            self.collector.positions = positions.detach().to(device='cpu', copy=True)
        # Diagnostics are restored only for later export. They never enter the
        # controller, auxiliary loss, audit guard, or delayed-credit path.
        self._diagnostic_training = (_checkpoint_tensor_copy(state.get('diagnostic_training'))
                                    if self.c.get('checkpoint_train_measurements', False) else None)
        if self.diagnostics and state.get('diagnostics') is not None:
            self.diagnostics.load_state_dict(state['diagnostics'])
        elif state.get('diagnostics') is not None:
            raise ValueError('checkpoint diagnostics enabled but current configuration disables them')
        saved = _checkpoint_tensor_copy(state['pending'], self.device)
        self.online.pending = None
        if saved is not None:
            if self._diagnostic_training is not None and self._diagnostic_training.get('snapshot_shared_with_pending', False):
                self._diagnostic_training = dict(self._diagnostic_training, snapshot=saved['snapshot'])
            action = self.online.action(saved['snapshot'])
            kappa = guard(action,saved['audit'],saved['snapshot'].radius,self.c.get('loss_budget',.001),saved['warmup'])
            self.online.pending=PendingCredit(action,kappa,self.online._versions(),saved['snapshot'],saved['audit'],saved['warmup'])
