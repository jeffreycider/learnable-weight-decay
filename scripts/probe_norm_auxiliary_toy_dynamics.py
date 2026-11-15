"""Toy CPU measurement of the opt-in alternating norm auxiliary (root 05:23 request:
"shared Adam moments may increase the auxiliary objective; measure instead of assume").

Uses the test fixtures only (tests/test_online_norm_lifecycle.make, tests/test_transformer_online.task_gradients).
Nothing here is GPT or production evidence; it records, per commit, the continuous
objective J BEFORE the auxiliary Adam tick and the regenerated FINAL J after it, under
(a) the alternating schedule with a task tick in between commits and (b) auxiliary ticks
only (no task credit), so the sign of the final-minus-before difference can be compared
with and without shared task moments.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from tests.test_online_norm_lifecycle import AUX
from tests.test_transformer_action import FactorActor, fixture
from tests.test_transformer_online import task_gradients
from rewriter.transformer_online import TransformerOnlineLifecycle


def make(aux, *, final_action_step):
    # Same construction as tests/test_online_norm_lifecycle.make, with a later terminal step.
    base, snapshot = fixture(torch.float64)
    actor = FactorActor()
    opt = torch.optim.Adam(actor.parameters(), lr=1e-4, betas=(.9, .99))
    life = TransformerOnlineLifecycle(actor, opt, global_token_count=8192, final_action_step=final_action_step,
                                      radius_fraction=.001, gradient_clip=100., norm_auxiliary=aux)
    return life, {n: p.clone().requires_grad_() for n, p in base.items()}, snapshot


def run(aux, *, with_task, steps, first=1000):
    life, live, snapshot = make(aux, final_action_step=first + steps)  # terminal step never reached
    rows = []
    for t in range(first, first + steps):
        if t > first:
            if with_task:
                life.consume(task_gradients(live), gradient_step=t)
            else:
                # Credit the pending action with a zero task gradient: the task tick
                # still happens (Adam step with zero gradient) so clocks stay valid.
                life.consume({n: torch.zeros_like(p) for n, p in live.items()}, gradient_step=t)
        m = life.commit(snapshot, live, action_step=t)
        rows.append(dict(action_step=t,
                         before=m['chisel_norm_auxiliary_fractional_change'],
                         final=m['chisel_norm_auxiliary_final_fractional_change'],
                         unweighted_gradient_norm=m['chisel_norm_auxiliary_unweighted_gradient_norm'],
                         clipped=m['chisel_norm_auxiliary_gradient_clipped'],
                         actor_step_norm=m['chisel_norm_auxiliary_actor_step_norm'],
                         committed_edit_norm=m['committed_edit_norm']))
    diffs = [r['final'] - r['before'] for r in rows]
    return dict(rows=rows, commits=len(rows),
                final_below_before=sum(d < 0 for d in diffs), final_above_before=sum(d > 0 for d in diffs),
                final_equal_before=sum(d == 0 for d in diffs),
                largest_increase=max(diffs), largest_decrease=min(diffs),
                task_updates=life.meta_updates, norm_updates=life.norm_updates,
                adam_step_count=int(life.optimizer.state[life.actor.x]['step']))


def main():
    torch.manual_seed(0)
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('runs/chisel-fable/norm_auxiliary_toy_dynamics.json')
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  scope='toy FactorActor fixture (tests/test_transformer_action.fixture, FP64 base); not GPT, not production settings; AUX from tests/test_online_norm_lifecycle',
                  aux_settings=AUX, first_action_step=1000, commits_per_arm=steps,
                  alternating_with_task=run(AUX, with_task=True, steps=steps),
                  auxiliary_only_zero_task_gradient=run(AUX, with_task=False, steps=steps),
                  reading='final-minus-before < 0 means the regenerated action at the updated actor has a smaller continuous fractional L2 change than the candidate before the auxiliary tick; counts are over commits of this toy only')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    for arm in ('alternating_with_task', 'auxiliary_only_zero_task_gradient'):
        r = result[arm]
        print(arm, 'commits', r['commits'], 'final<before', r['final_below_before'], 'final>before', r['final_above_before'],
              'largest increase %.3e largest decrease %.3e' % (r['largest_increase'], r['largest_decrease']),
              'task/norm/adam steps', r['task_updates'], r['norm_updates'], r['adam_step_count'])
        for row in r['rows']:
            print('  step %d before %.6e final %.6e diff %+.3e grad %.3e clipped %s edit %.4e' % (
                row['action_step'], row['before'], row['final'], row['final'] - row['before'],
                row['unweighted_gradient_norm'], row['clipped'], row['committed_edit_norm']))


if __name__ == '__main__':
    main()
