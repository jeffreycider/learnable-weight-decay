"""Audit recorded zero-residual warmup; no experiment or determinism claim."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root,seed):
    paths=[root/f'soap46-B-uncertainty-{tag}seed{seed}-v1' for tag in ('','frozen-')]
    configs=[json.loads((p/'config.json').read_text()) for p in paths]
    changed={k:[configs[0].get(k),configs[1].get(k)] for k in configs[0].keys()|configs[1].keys()
             if configs[0].get(k)!=configs[1].get(k)}
    if set(changed)-{'warmup_steps','output','executed_sha256'}:
        raise ValueError('Unexpected configuration differences: '+str(set(changed)))
    if changed['warmup_steps']!=[500,2900] or configs[0]['warmup_meta']!='moments' or configs[0]['compile_map']:
        raise ValueError('Unexpected historical warmup protocol')
    logs=[]
    for p in paths:
        rows=[json.loads(line) for line in (p/'metrics.jsonl').read_text().splitlines()]
        logs.append({r['step']:r for r in rows if r.get('event')=='train' and r['step']<=500})
    if any(set(log)!=set(range(1,501)) for log in logs):
        raise ValueError('Need all 500 logged warmup steps in both arms')
    if any(r['basis_residual_norm']!=0 for log in logs for r in log.values()):
        raise ValueError('Warmup residuals were not identically zero')
    differences={t:logs[0][t]['train_loss']-logs[1][t]['train_loss'] for t in range(1,501)}
    first=next((t for t,d in differences.items() if d!=0),None)
    return dict(seed=seed,configuration_differences=changed,
                implementation_sha256=configs[0]['implementation_sha256'],source_sha256=configs[0]['source_sha256'],
                zero_residual_steps_per_arm=500,first_logged_loss_difference_step=first,
                first_loss_difference=differences[first] if first else 0.,
                maximum_absolute_warmup_loss_difference=max(map(abs,differences.values())),
                step500_loss_difference=differences[500],
                input_sha256={str(p/name):digest(p/name) for p in paths for name in ('config.json','metrics.jsonl')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('runs/soap46-v1'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=dict(scope=__doc__,pairs=[audit(args.runs,seed) for seed in (4,5)],
                source_audit='Before step 500 both arms use the same proposal map with zero residuals, zero residual Adam LR, zero applied residual decay, and moment updates enabled. No different warmup forward rule identified.',
                limitations='Logged scalar losses do not prove parameter equality or identify a kernel/nondeterminism source. No subtraction of warmup differences from endpoints; no new benefit claim.',
                next_design='If funded, use the existing exact shared-prefix fork at step500, with a fixed prospective candidate and control. Existing failed confirmations do not select a new tuning cell.',
                source_references=['rewriter/soap46_online.py:effective_lr/consume/propose','rewriter/soap46_prefix.py','tests/test_soap46_prefix.py'],
                gpt_evaluations=0,gpu_leases=0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
