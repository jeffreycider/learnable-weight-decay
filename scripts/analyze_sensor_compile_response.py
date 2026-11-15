"""Separate units/denominators in a saved sensor-leaf qualification summary."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rewriter.qualification_metrics import aggregate_recorded_errors


def analyze(summary):
    def groups(records, selectors):
        result = {}
        for name, select in selectors.items():
            chosen = {k:v for k,v in records.items() if select(k)}
            if chosen:
                result[name] = aggregate_recorded_errors(chosen)
        return result

    rows = summary['rows']
    means = {backend:statistics.mean(row['compute_seconds'] for row in rows
             if row['backend'] == backend and row['update'] > 1)
             for backend in ('eager', 'sensor_leaves')}
    result = dict(warm_compute_mean_seconds=means,
        warm_compute_reduction_percent=100*(1-means['sensor_leaves']/means['eager']),
        scope='Updates2/3 of sequential3-eager then3-candidate trajectories; '
              '512seq/524288tokens emulated8ranks on1H100; compute intervals '
              'include observer+optimizer but exclude controller/NCCL/data/audits/CPUcopies.',
        errors={})
    for step, error in summary['errors'].items():
        delta = error['stored_optimizer_delta']['per_parameter']
        state = error['optimizer_state']['per_parameter']
        sensors = error['sensors']['per_parameter']
        inputs = error['next_actor_inputs']['per_parameter']
        matrix = lambda k:k.startswith('blocks.') and k.endswith('.weight')
        result['errors'][step] = dict(
            stored_delta=groups(delta, {
                'all':lambda k:True, 'embedding':lambda k:k=='embed.weight',
                'head':lambda k:k in ('proj.weight','proj.bias'),
                'block_matrices_muon':matrix,
                'other_adam':lambda k:not matrix(k) and k not in ('embed.weight','proj.weight','proj.bias')}),
            moments=groups(state, {
                'adam_exp_avg':lambda k:k.startswith('/0/') and k.endswith('/exp_avg'),
                'adam_exp_avg_sq':lambda k:k.startswith('/0/') and k.endswith('/exp_avg_sq'),
                'adam_step_counters':lambda k:k.startswith('/0/') and k.endswith('/step'),
                'muon_momentum':lambda k:k.startswith('/1/') and k.endswith('/momentum')}),
            sensors=groups(sensors, {
                'all':lambda k:True,
                'positions_only':lambda k:k=='/trajectory_positions',
                'excluding_positions':lambda k:k!='/trajectory_positions',
                'trajectories':lambda k:k=='/trajectories',
                'rms':lambda k:k.endswith(('/input_rms','/output_rms')),
                'ones':lambda k:'/ones' in k,
                'train_audit':lambda k:'/train' in k or '/audit' in k}),
            actor_inputs=groups(inputs, {
                'all':lambda k:True,
                'coordinate_features':lambda k:k.endswith(('/rows','/columns','/coordinates')),
                'trajectories':lambda k:k=='/0/trajectories',
                'weight_norm_radius':lambda k:k in ('/0/weight_norm','/0/radius')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(json.loads(args.summary.read_text())), indent=2)+'\n')
