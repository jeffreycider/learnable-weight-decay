"""Opt-in per-checkpoint ascent for a zero relative-ranking-loss constraint.

r_i is the mean nonnegative relative-gap margin loss on checkpoint i's
complete fitting panel. r_i == 0 implies every measured strict top-k relation
has its requested margin. This is stronger than merely retaining the ordering.

The primal ranking term is (base_weight + lambda_i / scale) * r_i.
After its primal update, lambda_i <- lambda_i + dual_lr * r_i / scale,
using the same pre-update measurement. No model replay or scalar search occurs.
Zero tolerance means the multiplier cannot decrease. Finite optimization does
not guarantee feasibility, and fitting-panel feasibility is not generalization.
"""
import copy
import dataclasses
import math

from rewriter.tokenwise_reference import TokenwiseDual


class RankingDual(TokenwiseDual):
    def __init__(self, state_ids, *, config, dual_lr, rank_scale, rank_dual_lr):
        if config.ranking_loss_mode != 'relative_gap' or config.ranking_tie_policy != 'exact_teacher_ties':
            raise ValueError('Ranking ascent requires the relative-gap, exact-teacher-tie loss')
        if config.ranking_weight <= 0:
            raise ValueError('A positive base ranking weight is required')
        if not all(math.isfinite(x) and x > 0 for x in (rank_scale, rank_dual_lr)):
            raise ValueError('Positive finite ranking scale and dual LR required')
        super().__init__(state_ids, config=config, dual_lr=dual_lr)
        self.rank_scale = float(rank_scale)
        self.rank_dual_lr = float(rank_dual_lr)
        self.rank_multipliers = {i: 0. for i in self.multipliers}

    def config_for_state(self, state_id):
        # Unseen IDs never acquire a learned multiplier.
        weight = self.config.ranking_weight + self.rank_multipliers.get(state_id, 0.) / self.rank_scale
        return dataclasses.replace(self.config, ranking_weight=weight)

    def describe_measurement(self, state_id, ranking_penalty):
        weight = self.config_for_state(state_id).ranking_weight
        value = float(ranking_penalty) / weight
        if not math.isfinite(value) or value < 0:
            raise ValueError('Ranking measurement must be finite and nonnegative')
        return dict(ranking_constraint_loss=value, ranking_constraint_tolerance=0.,
                    ranking_effective_weight=weight,
                    ranking_dual_multiplier=self.rank_multipliers.get(state_id, 0.),
                    ranking_dual_linear_term=(weight - self.config.ranking_weight) * value,
                    ranking_base_penalty=self.config.ranking_weight * value)

    def step(self, measurements, *, expected_ids, ranking_measurements):
        expected_ids = list(expected_ids)
        rows = list(ranking_measurements)
        ids = [i for i, _ in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
            raise ValueError('One complete ranking measurement per fitting checkpoint required')
        if not set(ids) <= self.rank_multipliers.keys():
            raise ValueError('Held-out checkpoint cannot update ranking multipliers')
        proposed = dict(self.rank_multipliers)
        for state_id, value in rows:
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError('Ranking measurement must be finite and nonnegative')
            proposed[state_id] += self.rank_dual_lr * value / self.rank_scale
            weight = self.config.ranking_weight + proposed[state_id] / self.rank_scale
            if not math.isfinite(weight):
                raise FloatingPointError('Nonfinite ranking multiplier')
        # Validate ranking values before the parent mutates its KL state/clock.
        super().step(measurements, expected_ids=expected_ids)
        self.rank_multipliers = proposed

    def state_dict(self):
        state = super().state_dict()
        state['ranking_constraint'] = dict(version=1, tolerance=0., scale=self.rank_scale,
                                          dual_lr=self.rank_dual_lr,
                                          multipliers=dict(self.rank_multipliers))
        return state

    def load_state_dict(self, state):
        saved = copy.deepcopy(state)
        rank = saved.pop('ranking_constraint', None)
        expected = self.state_dict()['ranking_constraint']
        if rank is None or any(rank.get(k) != expected[k] for k in ('version', 'tolerance', 'scale', 'dual_lr')):
            raise ValueError('Ranking constraint resume configuration changed')
        values = {int(i): float(v) for i, v in rank['multipliers'].items()}
        if values.keys() != self.rank_multipliers.keys() or not all(math.isfinite(v) and v >= 0 for v in values.values()):
            raise ValueError('Invalid ranking multipliers or checkpoint IDs')
        if not all(math.isfinite(self.config.ranking_weight + v / self.rank_scale) for v in values.values()):
            raise ValueError('Nonfinite effective ranking weight')
        super().load_state_dict(saved)
        self.rank_multipliers = values
