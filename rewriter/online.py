"""Alternating auxiliary updates and direct credit from the next ordinary gradient."""
from dataclasses import dataclass
import torch
from .algebra import auxiliary_objective, guard
from .edit_scope import editable_names, scoped_action


@dataclass
class PendingCredit:
    action: object
    kappa: torch.Tensor
    controller_versions: tuple
    snapshot: object
    audit: object
    warmup: float


class OnlineRewriter:
    def __init__(self, controller, config=None):
        c = config or {}
        if c.get('editable_scope')=='grouped' or c.get('norm_objective')=='grouped_l2':
            raise NotImplementedError('online grouped-L2 objective needs cached per-group reference norms; use the frozen grouped producer')
        self.controller = controller
        self.optimizer = torch.optim.Adam(controller.parameters(), lr=c.get('auxiliary_lr', 2e-5),
                                          betas=(.9, .99), eps=1e-8, weight_decay=0.)
        self.config = c
        self.pending = None
        self.updates = 0
        self.aux_evaluations = 0
        self.metrics = {}

    def _versions(self):
        return tuple(p._version for p in self.controller.parameters())

    def action(self, snapshot):
        # Missing scope preserves historical checkpoints. New Chisel run
        # configurations explicitly select hidden; observations stay complete.
        names=editable_names(dict(snapshot.weights,**snapshot.vector_weights),
                             self.config.get('editable_scope','all'))
        return scoped_action(self.controller(snapshot),names)

    def _step(self, loss, lr, kind):
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.controller.parameters(),
                                     self.config.get('gradient_clip_norm', 1.), error_if_nonfinite=True)
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self.optimizer.step()
        self.updates += 1
        self.metrics[kind + '_gradient_norm'] = float(grad_norm.detach())
        self.metrics['task_vjp_scalar' if kind == 'task' else kind + '_loss'] = float(loss.detach())
        self.metrics['meta_updates'] = self.updates

    def consume_credit(self, matrix_cotangents, vector_cotangents):
        if self.pending is None:
            return
        pending = self.pending
        if self._versions() != pending.controller_versions:
            raise RuntimeError('controller mutated between action and delayed credit')
        self.pending = None
        if self.config.get('task_credit', True):
            scalar = pending.kappa * pending.action.contraction(matrix_cotangents, vector_cotangents)
            self._step(scalar, self.config.get('task_lr', 1e-4), 'task')

    def propose(self, snapshot, train, audit, t, max_steps, *, shadow=False):
        if self.pending is not None:
            raise RuntimeError('must consume previous credit before any controller update')
        if not 0 <= t < max_steps:
            raise ValueError('iteration must lie inside the declared run')
        self.metrics = {'meta_updates': self.updates}
        if t == max_steps - 1:
            self.metrics['terminal_identity'] = True
            return None
        for _ in range(self.config.get('auxiliary_steps', 8)):
            action = self.action(snapshot)
            loss = auxiliary_objective(action, snapshot.weights, snapshot.vector_weights, train,
                    weight_norm=snapshot.weight_norm, radius=snapshot.radius,
                    loss_budget=self.config.get('loss_budget', .001),
                    beta=self.config.get('beta', 1.), gamma=self.config.get('gamma', .01))
            self._step(loss, self.config.get('auxiliary_lr', 2e-5), 'auxiliary')
            self.aux_evaluations += 1
        warmup_steps = self.config.get('warmup_steps', 64)
        ramp_steps = self.config.get('ramp_steps', 192)
        warmup = min(1., max(0., (t - warmup_steps + 1) / max(1, ramp_steps)))
        if t < warmup_steps or shadow:
            self.metrics.update(kappa=0., auxiliary_evaluations=self.aux_evaluations)
            return None
        action = self.action(snapshot)
        # Audit is accessed exactly once here, after all current auxiliary updates.
        kappa, audit_value = guard(action, audit, snapshot.radius, self.config.get('loss_budget', .001), warmup, return_measurements=True)
        self.pending = PendingCredit(action, kappa, self._versions(), snapshot, audit, warmup)
        with torch.no_grad():
            self.metrics.update(kappa=float(kappa), action_norm=float(action.norm_sq().sqrt()),
                committed_norm=float(kappa * action.norm_sq().sqrt()),
                norm_change=float(kappa * 2 * action.contraction(snapshot.weights, snapshot.vector_weights)
                                  + kappa.square() * action.norm_sq()),
                train_proxy=float(train.penalty(action)), audit_proxy=float(audit_value),
                auxiliary_evaluations=self.aux_evaluations)
        return self.pending
