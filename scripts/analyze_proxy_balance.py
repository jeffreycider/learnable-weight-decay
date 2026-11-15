"""Reconstruct final raw-action proxy terms from recorded scalar diagnostics.

This is detached arithmetic only. It does not evaluate GPT, optimize a candidate,
or use validation. Zero raw action has objective zero by construction.
"""
import argparse
import json
from pathlib import Path
import statistics


def summarize(folder, first=65, last=None):
    folder = Path(folder)
    config = json.loads((folder / 'config.json').read_text())
    c = config.get('rewriter', {})
    rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
    selected = [r for r in rows if r.get('event') == 'train' and r['step'] >= first
                and (last is None or r['step'] <= last) and r.get('kappa', 0) > 0]
    if not selected:
        raise ValueError(f'no executed actions in requested window: {folder}')
    terms = []
    for row in selected:
        k, a2, radius = row['kappa'], row['action_norm'] ** 2, row['radius']
        # committed norm change = 2*k<W,a> + k^2 ||a||^2.
        raw_change = row['norm_change'] / k + (1-k) * a2
        normalization = 2 * row['weight_norm'] * radius + radius**2 + 1e-24
        norm = raw_change / normalization
        preservation = c.get('beta', 1.) * row['train_proxy'] / c.get('loss_budget', .001)**2
        trust = c.get('gamma', .01) * a2 / (radius**2 + 1e-24)
        terms.append(dict(step=row['step'], norm_term=norm, train_term=preservation,
                          trust_term=trust, raw_auxiliary=norm+preservation+trust))
    def mean_key(key):
        values = [r[key] for r in selected if key in r]
        return statistics.mean(values) if values else None
    return dict(run=folder.name, first_step=selected[0]['step'], last_step=selected[-1]['step'],
        executed_actions=len(selected),
        mean_terms={key: statistics.mean(r[key] for r in terms) for key in terms[0] if key != 'step'},
        fraction_raw_objective_above_zero=sum(r['raw_auxiliary'] > 0 for r in terms)/len(terms),
        final_auxiliary_preclip_gradient_mean=mean_key('auxiliary_gradient_norm'),
        task_preclip_gradient_mean=mean_key('task_gradient_norm'),
        final_auxiliary_clip_fraction=sum(r.get('auxiliary_gradient_norm',0) > c.get('gradient_clip_norm',1.)
                                    for r in selected)/len(selected),
        interpretation='Reconstructed final raw objective from FP32 scalar logs; zero action scores zero. '
        'A positive value is worse on this surrogate, not proof of worse future task loss. '
        'Only the last inner auxiliary update per GPT step is logged. '
        'Preclip gradient magnitudes do not determine shared-Adam update magnitudes.',
        per_step=terms)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--first',type=int,default=65)
    parser.add_argument('--last',type=int)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=[summarize(run,args.first,args.last) for run in args.runs]
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    for record in result:
        print(json.dumps({k:v for k,v in record.items() if k!='per_step'}))


if __name__ == '__main__':
    main()
