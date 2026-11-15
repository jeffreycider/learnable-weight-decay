"""Per-fitting-checkpoint inequality multipliers; no GPT or autograd state."""
import math


class PrimalDual:
    def __init__(self, state_ids, *, tolerance, rho=1., dual_lr=.01):
        ids=list(state_ids)
        if not ids or len(ids)!=len(set(ids)):
            raise ValueError('unique fitting checkpoint IDs required')
        if not all(math.isfinite(x) and x>0 for x in (tolerance,rho,dual_lr)):
            raise ValueError('positive finite primal-dual scales required')
        self.tolerance=float(tolerance);self.rho=float(rho);self.dual_lr=float(dual_lr)
        self.multipliers={int(i):0. for i in ids};self.updates=0

    def terms(self, kl, state_id=None):
        """Return value and d(value)/d(KL), after the full sequence mean.

        Held-out IDs have no learned multiplier and cannot enter dual ascent.
        Tiny negative floating point KL is clamped to its mathematical bound.
        """
        kl=float(kl)
        if not math.isfinite(kl):raise FloatingPointError('nonfinite KL')
        multiplier=self.multipliers.get(state_id,0.)
        violation=max(kl,0.)/self.tolerance-1.
        positive=max(violation,0.)
        linear=multiplier*violation;quadratic=.5*self.rho*positive**2
        coefficient=(multiplier+self.rho*positive)/self.tolerance if kl>=0 else 0.
        return dict(kl_penalty=linear+quadratic,kl_coefficient=coefficient,
                    constraint_violation=violation,positive_violation=positive,
                    dual_multiplier=multiplier,dual_linear_term=linear,
                    augmented_penalty=quadratic,kl_tolerance=self.tolerance,
                    feasible=violation<=0.)

    def step(self, measurements, *, expected_ids):
        """Call once AFTER Adam, with globally gathered pre-update mean KL.

        Every checkpoint occurs on exactly one rank. Gathering is a union, not
        an additional average or a world-size multiplication.
        """
        rows=list(measurements);ids=[int(i) for i,_ in rows]
        if len(ids)!=len(set(ids)) or set(ids)!=set(expected_ids):
            raise ValueError('dual step requires exactly one global measurement per batch checkpoint')
        if not set(ids)<=self.multipliers.keys():raise ValueError('held-out checkpoint cannot update dual state')
        proposed={i:max(0.,self.multipliers[i]+self.dual_lr*self.terms(k,i)['constraint_violation']) for i,k in rows}
        if not all(math.isfinite(x) for x in proposed.values()):raise FloatingPointError('nonfinite dual update')
        self.multipliers.update(proposed);self.updates+=1

    def state_dict(self):
        return dict(schema=1,tolerance=self.tolerance,rho=self.rho,dual_lr=self.dual_lr,
                    multipliers=dict(self.multipliers),updates=self.updates)

    def load_state_dict(self, state):
        expected=self.state_dict()
        for key in ('schema','tolerance','rho','dual_lr'):
            if state[key]!=expected[key]:raise ValueError('dual resume configuration changed')
        values={int(k):float(v) for k,v in state['multipliers'].items()}
        if values.keys()!=self.multipliers.keys() or not all(math.isfinite(v) and v>=0 for v in values.values()):
            raise ValueError('invalid dual checkpoint IDs or values')
        self.multipliers=values;self.updates=int(state['updates'])


def feasibility_score(rows, tolerance):
    """Stationary stopping score: worst violation first, shrinkage when feasible.

    This never uses learned multipliers. All infeasible scores exceed one;
    feasible scores are nonpositive. It is not the training Lagrangian.
    """
    if not rows:raise ValueError('nonempty evaluation required')
    if not math.isfinite(tolerance) or tolerance<=0:raise ValueError('positive finite tolerance required')
    if any(not math.isfinite(float(r[k])) for r in rows for k in ('mean_kl','mean_group_l2_percentage_reduction')):
        raise FloatingPointError('nonfinite feasibility evaluation')
    worst=max(max(float(r['mean_kl']),0.)/tolerance-1. for r in rows)
    shrinkage=sum(float(r['mean_group_l2_percentage_reduction']) for r in rows)/len(rows)
    return 1.+worst if worst>0 else -max(shrinkage,0.)
