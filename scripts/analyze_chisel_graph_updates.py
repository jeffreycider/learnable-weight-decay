"""Validate complete full200 evidence and report per-update critical-path timing.

No empirical result is inferred from incomplete trials or local kernel timings.
Periodic production audit/checkpoint costs are excluded from these training rates.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def response_readout(diagnostic):
    """Expose the actual KL-change denominator beside a small-response error.

    A zero reference change makes the relative disagreement undefined, even
    when both updates are identical. No epsilon or 'noise floor' is substituted.
    """
    before, reference, candidate = (diagnostic[k] for k in
        ('kl_before', 'kl_after_reference', 'kl_after_candidate'))
    if not all(math.isfinite(v) for v in (before, reference, candidate)):
        raise ValueError('nonfinite response KL readout')
    reference_change, candidate_change = reference-before, candidate-before
    difference = candidate-reference
    return dict(backend=diagnostic.get('backend','graph'),state_id=diagnostic['state_id'], lr_scale=diagnostic['lr_scale'],
        sequence_ids=diagnostic['sequence_ids'],
        kl_change_reference=reference_change, kl_change_candidate=candidate_change,
        kl_change_difference=difference,
        absolute_kl_change_disagreement=abs(difference),
        relative_kl_change_disagreement=abs(difference)/abs(reference_change)
            if reference_change else None,
        response=diagnostic['response'])


def analyze(directory):
    directory = Path(directory)
    path = directory/'summary.json'
    summary = json.loads(path.read_text())
    if tuple(summary[k] for k in ('world_size','checkpoint_batch','sequences','microbatch')) != (8,200,64,16):
        raise ValueError('full200 x64 micro16 across8ranks required')
    states = summary['states']
    if len(states)!=200 or len(set(states))!=200:
        raise ValueError('200 distinct checkpoint states required')
    trials=summary['trials']
    hybrid=summary.get('graph_comparison',False)
    confirmation=summary.get('benchmark_objective',{}).get('independent_confirmation')
    if confirmation:
        original=confirmation['original_state_ids']
        selected=confirmation['selected_state_ids']
        if (hybrid or confirmation.get('disjoint') is not True or selected!=states
                or len(original)!=200 or len(set(original))!=200 or set(original)&set(states)):
            raise ValueError('independent confirmation requires disjoint original and selected200 states')
        objective=summary['benchmark_objective']
        if (confirmation['kl_abs_limit']!=.1*objective['effective_kl_tolerance']
                or confirmation['objective_abs_floor']!=1e-4 or confirmation['objective_relative_limit']!=1e-3):
            raise ValueError('unknown prospective confirmation gate policy')
    expected=[(0,'reference'),(0,'graph'),(1,'graph'),(1,'reference'),(2,'reference'),(2,'graph')]
    if confirmation:expected=[(0,'reference'),(0,'graph'),(1,'graph'),(1,'reference')]
    if hybrid:expected=[(0,'reference'),(0,'graph'),(0,'hybrid'),(1,'hybrid'),(1,'graph'),(2,'graph'),(2,'hybrid')]
    if [(t['repeat'],t['backend']) for t in trials]!=expected:
        raise ValueError('complete alternating trials required')
    rows=[]
    evidence={}
    for t in trials:
        ranks=t['rank_rows']
        if len(ranks)!=8 or {r['rank'] for r in ranks}!=set(range(8)):
            raise ValueError('exactly eight unique ranks per trial required')
        if t['warmup'] != (t['repeat']==0):raise ValueError('warmup labels differ')
        if not t['all_rank_gates']:raise ValueError('global numerical qualification incomplete')
        for r in ranks:
            if not r['within_gates']:raise ValueError('rank numerical gate failed')
            times=[r[k] for k in ('seconds','accumulation_seconds','setup_seconds')]
            if not all(math.isfinite(v) and v>=0 for v in times) or not r['seconds']>0:
                raise ValueError('invalid rank timing')
            if r['accumulation_seconds']>r['seconds']:raise ValueError('accumulation exceeds update time')
            p=directory/f"qualification_{t['repeat']}_{t['backend']}_rank{r['rank']}.json"
            q=json.loads(p.read_text())
            if not q['within_declared_global_gates']:raise ValueError('rank evidence contradicts summary gate')
            if t['backend']=='reference' and q['reference_repeat_bit_identical'] is not True:
                raise ValueError('baseline repeat is not bit-identical')
            evidence[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        critical=max(ranks,key=lambda r:r['seconds'])
        seconds=critical['seconds']
        if not math.isclose(seconds,t['seconds'],rel_tol=0.,abs_tol=1e-9):
            raise ValueError('summary must use maximum rank wall time')
        rows.append(dict(repeat=t['repeat'],backend=t['backend'],warmup=t['warmup'],seconds=seconds,
            critical_rank=critical['rank'],rank_spread_seconds=seconds-min(r['seconds'] for r in ranks),
            critical_rank_accumulation_seconds=critical['accumulation_seconds'],
            critical_rank_tail_seconds=seconds-critical['accumulation_seconds'],
            initial_setup_max_seconds=max(r['setup_seconds'] for r in ranks),
            timed_recaptures=sum(r['recaptures'] for r in ranks)))
    diagnostics=summary['global_gradient_response_diagnostics']
    expected_pairs={(s,scale) for s in states[:8] for scale in (1.,.1)}
    if hybrid:
        valid_diagnostics=(len(diagnostics)==32 and
            {(d['backend'],d['state_id'],d['lr_scale']) for d in diagnostics}==
            {(b,s,scale) for b in ('graph','hybrid') for s,scale in expected_pairs})
    else:
        valid_diagnostics=(len(diagnostics)==16 and {(d['state_id'],d['lr_scale']) for d in diagnostics}==expected_pairs)
    if not valid_diagnostics:
        raise ValueError('eight fitting checkpoints x two learning rates required')
    for d in diagnostics:
        if len(d['sequence_ids'])!=16 or not d['response']['finite']:
            raise ValueError('response diagnostic workload or finiteness differs')
    medians={b:statistics.median(r['seconds'] for r in rows if r['backend']==b and not r['warmup'])
             for b in (('graph','hybrid') if hybrid else ('reference','graph'))}
    baseline,candidate=(medians['graph'],medians['hybrid']) if hybrid else (medians['reference'],medians['graph'])
    return dict(scope=('four complete200x64micro16 updates across8ranks on disjoint200 fitting states; dense/strict warmups excluded; ONE warm measurement per backend' if confirmation else
                      'seven complete200x64micro16 updates across8ranks; dense oracle and two graph warmups excluded; strict vs approximate hybrid ABBA' if hybrid else
                      'six complete200-checkpoint x64-sequence micro16 updates across8ranks; two warmups excluded'),
        benchmark_objective=summary.get('benchmark_objective'),
        policy_sha256=summary.get('policy_sha256'),
        qualification=('prospective independent gate policy versus dense oracle on all ranks; original failed run remains failed' if confirmation else
                       'declared gates versus dense oracle on all ranks; fresh overridden-objective qualification' if hybrid else
                       'declared gradient/Adam/dual/post-update gates satisfied on all ranks; baseline repeats bit-identical'),
        timing_unit=('one per-trial maximum rank wall time per backend; no repeat variance estimate' if confirmation else
                     'median of per-trial maximum rank wall time; two warm trials per backend'),
        independent_confirmation=confirmation,
        median_seconds=medians,time_reduction_fraction=1-candidate/baseline,throughput_ratio=baseline/candidate,
        training_only_updates_per_hour={b:3600/v for b,v in medians.items()},
        excluded='initial graph capture, numerical readouts, periodic production heldout audits and checkpoint saves',
        trials=rows,response_diagnostics=diagnostics,
        rank_timing_interpretation='accumulation_seconds includes the end-of-accumulation error collective; rank spread after this synchronization cannot identify earlier compute/IO stragglers. Critical-rank tail is the remaining SUM/clip/Adam/dual portion, not all communication time.',
        memory_interpretation='peak_bytes was reset after graph capture; do not infer total device footprint or a memory saving from this counter without separate graph-pool/device-memory measurements.',
        response_readouts=[response_readout(d) for d in diagnostics],
        response_population='eight fitting checkpoints, first16 selected sequences each; actual global200x64 gradient; nominal and0.1LR saved Adam',
        response_interpretation='reported separately; raw-gradient gates do not establish logit-response identity',
        summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rank_evidence_sha256=evidence)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=analyze(args.directory)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('median_seconds','time_reduction_fraction','throughput_ratio','training_only_updates_per_hour')},indent=2))
