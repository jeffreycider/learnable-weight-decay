"""Stop learning residual coordinates after restoring one within-run checkpoint.

This diagnostic freezes H, not Q=s_t H: the detached Shampoo-derived scale still
changes normally. It preserves GPT/base optimizer state, schedules, and moments.
Frozen checkpoints use a distinct schema so the ordinary trainer cannot silently
resume them with learning enabled.
"""
import types
from .lag_one import LagOneHypergradient

FROZEN_SCHEMA=102


def freeze_residual_learning(hyper,completed_steps):
    if not hyper.learned:raise ValueError('freezing requires a learned checkpoint')
    if hasattr(hyper,'frozen_at_step'):raise RuntimeError('residual learning already frozen')
    hyper.frozen_at_step=int(completed_steps)
    hyper.frozen_versions={id(h):h._version for e in hyper.entries.values() for h in (e['hl'],e['hr'])}
    for e in hyper.entries.values():
        e['lag']=LagOneHypergradient()
        e['pending_record']=None;e['pending_coefficient']=None;e['pending_step']=None
    def install_pending(self,e,record,coefficient,step):
        # An unexecuted meta update is not recorded as real gradient feedback.
        e['pending_record']=None;e['pending_coefficient']=None;e['pending_step']=None
    def consume(self,step):
        self.action_difference_sq=None;self.action_cosine_stats=None
        for e in self.entries.values():
            for h in (e['hl'],e['hr']):
                if h._version!=self.frozen_versions[id(h)]:raise RuntimeError('frozen residual coordinates mutated')
        self.metrics={'meta_steps':self.meta_steps,'credit_seconds':0.,
                      'residual_learning_frozen':True,'frozen_at_step':self.frozen_at_step}
    hyper.install_pending=types.MethodType(install_pending,hyper)
    hyper.consume=types.MethodType(consume,hyper)


def install_checkpoint_policy(checkpoint_module,expected_branch_step):
    """Wrap the pinned serializer; ordinary schema2 restore remains strict."""
    original_restore=checkpoint_module.restore;original_payload=checkpoint_module.payload
    def restore(scope,hyper,state):
        if state['schema']==FROZEN_SCHEMA:
            if state.get('frozen_at_step')!=expected_branch_step:raise ValueError('freeze branch mismatch')
            state=dict(state,schema=2)
        elif state['schema']!=2 or state['completed_steps']!=expected_branch_step:
            raise ValueError('fresh freeze branch must start at its declared checkpoint')
        step=original_restore(scope,hyper,state)
        freeze_residual_learning(hyper,expected_branch_step)
        return step
    def payload(scope,hyper,completed_steps):
        state=original_payload(scope,hyper,completed_steps)
        state.update(schema=FROZEN_SCHEMA,frozen_at_step=hyper.frozen_at_step,
                     frozen_quantity='dimensionless H; Q=s_t H remains scale-dependent')
        return state
    checkpoint_module.restore=restore;checkpoint_module.payload=payload
