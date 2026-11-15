"""Direct committed-weight cotangent for the pinned PR395 forward envelope."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ForwardEnvelope:
    lookahead_beta:float
    displacement_ema_gamma:float
    anchor_gamma:float
    tail_ema_derivative:float

    @property
    def derivative(self):
        return ((1+self.anchor_gamma)*(1+self.lookahead_beta*(1-self.displacement_ema_gamma))
                -self.anchor_gamma*self.tail_ema_derivative)


def next_forward_envelope(*,action_step,lookahead_stepsize,scheduled_lookahead,lr_ratio,
                          lookahead_ema,prefill_steps,rest_steps,tail_exists,
                          tail_tau,tail_start,tail_end,anchor_strength,anchor_start,anchor_ramp):
    """Evaluate at commit, BEFORE wrapper.it increments and tail EMA is updated.

    The next forward uses the CURRENT optimizer LR: upstream set_hparams for the
    next iteration occurs only after its backward. Tail EMA first initializes
    from the new committed weights (derivative1), then updates with derivative
    1/tau, and finally freezes (derivative0). Evaluation-only blending supplies
    no controller feedback and is not part of this training-forward envelope.
    """
    next_step=action_step+1
    active=next_step+1>prefill_steps and next_step<rest_steps
    beta=lookahead_stepsize*(lr_ratio if scheduled_lookahead else 1.) if active else 0.
    tail_updates=tail_tau>0 and tail_start<=next_step<tail_end
    tail_derivative=(1./tail_tau if tail_exists else 1.) if tail_updates else 0.
    next_tail_exists=tail_exists or tail_updates
    anchor=(anchor_strength*min(1.,max(0.,(next_step-anchor_start)/anchor_ramp))
            if next_tail_exists else 0.)
    return ForwardEnvelope(beta,lookahead_ema,anchor,tail_derivative)


def scope_envelope(scope,step):
    w=scope['optimizers'][0]
    if w.it!=step:raise RuntimeError('wrapper and GPT iteration clocks disagree')
    return next_forward_envelope(action_step=step,lookahead_stepsize=w.lookahead_stepsize,
        scheduled_lookahead=w.use_scheduled_lookahead_stepsize,lr_ratio=w.get_lr_lambda(),
        lookahead_ema=w.lookahead_ema,prefill_steps=w.prefill_steps,rest_steps=w.rest_steps,
        tail_exists=scope['_tailema'] is not None,tail_tau=scope['TAILEMA_TAU'],
        tail_start=scope['TAILEMA_START'],tail_end=scope['TAILEMA_END'],
        anchor_strength=scope['ANCHOR_GAMMA'],anchor_start=scope['ANCHOR_START'],anchor_ramp=scope['ANCHOR_RAMP'])
