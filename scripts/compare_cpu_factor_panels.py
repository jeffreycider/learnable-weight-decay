"""Compare numerical credit and measured CPU times for matched factor panels."""
import argparse
import json
import math
from pathlib import Path
import statistics
import torch


def difference(a,b):
    if a.keys()!=b.keys():raise ValueError('tensor inventory mismatch')
    diff=ref=0.;largest=0.;none_mismatches=[]
    for name in a:
        if a[name] is None or b[name] is None:
            if (a[name] is None)!=(b[name] is None):none_mismatches.append(name)
            continue
        x=a[name].double();y=b[name].double();delta=x-y
        diff+=float(delta.square().sum());ref+=float(y.square().sum())
        largest=max(largest,float(delta.abs().max()))
    return dict(relative_l2=math.sqrt(diff/ref) if ref else (0. if not diff else None),
                absolute_l2=math.sqrt(diff),max_absolute=largest,none_mismatches=none_mismatches)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dense',type=Path,required=True);p.add_argument('--direct',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    dense=json.loads((a.dense/'complete.json').read_text());direct=json.loads((a.direct/'complete.json').read_text())
    for key in ['checkpoint_sha256','case','sequences','objective','source_sha256','warmups','repeats','torch_version','threads','initial_optimizer','state_weight','teacher_cache_manifest_sha256']:
        if dense['metadata'][key]!=direct['metadata'][key]:raise ValueError('unmatched '+key)
    if dense['metadata'].get('optimizer_hyperparameters')!=direct['metadata'].get('optimizer_hyperparameters'):
        raise ValueError('unmatched optimizer hyperparameters')
    if dense['compute']!=direct['compute']:raise ValueError('compute counts differ')
    d=torch.load(a.dense/'last_update.pt',weights_only=False,map_location='cpu',mmap=True)
    c=torch.load(a.direct/'last_update.pt',weights_only=False,map_location='cpu',mmap=True)
    start=torch.load(dense['metadata']['checkpoint'],weights_only=False,map_location='cpu',mmap=True)
    gradients={role:difference(c['raw_gradients'][role],d['raw_gradients'][role]) for role in ('actor','private')}
    updates={}
    for role in ('actor','private'):
        initial=start['actor'] if role=='actor' else start['private'][dense['metadata']['case']['state_id']]
        names={n for n,v in d[role].items() if torch.is_tensor(v)}
        updates[role]=difference({n:c[role][n].double()-initial[n].double() for n in names},
                                 {n:d[role][n].double()-initial[n].double() for n in names})
    moments={}
    for role in ('shared_optimizer','private_optimizer'):
        if c[role]['param_groups']!=d[role]['param_groups']:raise ValueError('optimizer settings differ')
        if c[role]['state'].keys()!=d[role]['state'].keys():raise ValueError('optimizer state inventory differs')
        states=c[role]['state'].keys()
        if any(not torch.equal(c[role]['state'][i]['step'],d[role]['state'][i]['step']) for i in states):raise ValueError('Adam clocks differ')
        moments[role]={field:difference({i:c[role]['state'][i][field] for i in states},
                                       {i:d[role]['state'][i][field] for i in states}) for field in ('exp_avg','exp_avg_sq')}
    metric_keys=['mean_kl','objective','norm_objective','ranking_penalty','tail_quadratic_penalty',
                 'dual_linear_term','token_kl_quantiles','strict_topk_changes','tie_aware_argmax_changes',
                 'token_count','sequence_count','compute','dual_after']
    metric_mismatches=[dict(repetition=i,field=k) for i,(x,y) in enumerate(zip(dense['rows'],direct['rows'])) for k in metric_keys if x[k]!=y[k]]
    times={name:[r['update_seconds'] for r in result['rows'] if not r['warmup']] for name,result in [('dense',dense),('direct',direct)]}
    medians={name:statistics.median(v) for name,v in times.items()}
    result=dict(scope='Single real GPT checkpoint, two fixed TRAIN sequences, complete weighted-case update; CPU only. Same initial state and fresh Adam restored per repetition. No learning/generalization or GPU result.',
        paths=dict(dense=str(a.dense),direct=str(a.direct)),
        norm_modes={name:result['metadata'].get('norm_mode','dense') for name,result in [('dense',dense),('direct',direct)]},
        timed_seconds=times,median_seconds=medians,
        direct_over_dense_time=medians['direct']/medians['dense'],raw_gradient_difference=gradients,
        optimizer_update_difference=updates,adam_moment_difference=moments,metric_mismatches=metric_mismatches,compute=dense['compute'],
        qualification=dict(raw_gradient_relative_l2_tolerance=1e-4,update_relative_l2_tolerance=1e-3,
            exact_loss_and_counts=not metric_mismatches,
            raw_gradients_within_tolerance=all(not v['none_mismatches'] and v['relative_l2'] is not None and v['relative_l2']<=1e-4 for v in gradients.values()),
            updates_within_tolerance=all(v['relative_l2'] is not None and v['relative_l2']<=1e-3 for v in updates.values())),
        limits=['Sequential CPU measurements; few repetitions; no GPU timing or peak-memory claim.',
                ('Streamed candidate repeats the norm forward scan; its norm VJP discards dense edit graphs.' if direct['metadata'].get('norm_mode')=='streamed' else 'Norm regularizer still has a dense graph.'),
                'Direct factor contractions change finite-precision gradient accumulation.'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
