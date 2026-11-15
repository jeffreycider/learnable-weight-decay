"""Explicit same-prefix FP32 Chisel transition; ordinary resume stays strict.

The CLI verifies the original rank bundle, frozen actor, bank identity and actual
prefix weights, then writes separate rank checkpoints for a shadow or learned
suffix. It never runs GPT, edits model weights, or advances optimizer/data/RNG.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch

from .fp32_transition import promote_bundle
from .checkpoint import save_atomic
from rewriter.gpt_precision import RMS_EPSILON


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):h.update(block)
    return h.hexdigest()


def validate_online_config(config):
    c = config['transformer_chisel']
    if c.get('family') == 'adaptive_grouped_lag_one_v1':
        from .adaptive_transition import validate_adaptive_config
        return validate_adaptive_config(config)
    from rewriter.online_norm import validate_norm_auxiliary
    norm=validate_norm_auxiliary(c.get('norm_auxiliary'))
    from rewriter.online_preservation import validate_replay_maintenance
    replay=validate_replay_maintenance(c.get('replay_maintenance'))
    if replay is not None and norm is not None:raise ValueError('norm auxiliary and replay maintenance are mutually exclusive')
    if config.get('gpt_precision') != 'fp32' or config.get('state_bank'):
        raise ValueError('modern Chisel requires the explicit FP32 suffix, without capture writer')
    if config.get('compile', True):
        raise ValueError('full GPT compilation with online sensors is not yet qualified; set compile:false')
    if c.get('bootstrap') != 'first_ordinary_step_unedited':
        raise ValueError('bootstrap must explicitly leave the first ordinary suffix step unedited')
    if c.get('online_auxiliary') != 'none':
        raise ValueError('this adapter implements only ordinary-next-gradient online credit')
    if type(c.get('prefix_steps')) is not int or not 0 <= c['prefix_steps'] < config['max_steps']:
        raise ValueError('invalid online prefix clock')
    cadence=c.get('replica_check_every',64)
    if type(cadence) is not int or cadence<1:raise ValueError('positive replica check cadence required')
    norm_cadence=c.get('group_norm_metrics_every',64)
    if type(norm_cadence) is not int or norm_cadence<1:raise ValueError('positive group norm metric cadence required')
    for key in ('task_lr', 'representation_lr_multiplier', 'gradient_clip', 'radius_fraction'):
        if not math.isfinite(c[key]) or c[key] <= 0:raise ValueError('positive online optimizer scales required')
    if not config.get('unscored', False):
        if (c['prefix_steps'] != 1000 or config.get('schedule_steps') != 3250 or
                config['batch_tokens'] != 524288 or config.get('world_size') != 8):
            raise ValueError('production online continuation retains prefix1000, schedule3250,8ranks,524288tokens')
    transition = config.get('fp32_prefix_transition', {})
    if transition.get('rms_epsilon') != RMS_EPSILON or transition.get('tf32') is not False:
        raise ValueError('corrected FP32 prefix conversion required')
    if transition.get('completed_steps') != c['prefix_steps']:
        raise ValueError('online and precision transition clocks differ')
    if c['policy_provenance'].get('prefix_bundle_sha256') != transition.get('source_bundle_sha256'):
        raise ValueError('online actor prefix source differs')


def validate_policy(policy, settings):
    m = policy['metadata'];provenance = settings['policy_provenance']
    if m['architecture'] != settings['architecture']:
        raise ValueError('actor architecture differs')
    if m['state_bank_identity'] != provenance['state_bank_identity']:
        raise ValueError('actor belongs to a different state bank or teacher cache')
    options = m.get('options', {})
    if 'eager_chisel' in options:
        expected = ({'backend':'eager'} if options['eager_chisel'] else
                    {'backend':'torch.compile','fullgraph':True,'dynamic':False})
        if settings['actor_execution'] != expected:
            raise ValueError('online actor execution differs from the fitted actor')
    if 'radius_fraction' in options and options['radius_fraction'] != settings['radius_fraction']:
        raise ValueError('initial action radius differs from the qualified policy')
    precision = m['gpt_precision']
    if (precision.get('mode') != 'fp32' or precision.get('rms_epsilon') != RMS_EPSILON or
            precision.get('tf32') is not False or precision.get('attention_backend') != 'math'):
        raise ValueError('actor was not fitted against the corrected FP32 teacher')
    if m.get('diagnostic_prefix_only') or m['state_bank_identity']['selected_states'] != settings['prefix_steps']:
        raise ValueError('actor prefix scope differs from online prefix')


def load_policy_checkpoint(settings, *, device, require_qualification=False):
    if sha256(settings['policy_checkpoint']) != settings['policy_provenance']['policy_checkpoint_sha256']:
        raise ValueError('actor checkpoint SHA256 mismatch')
    policy = torch.load(settings['policy_checkpoint'], map_location=device, weights_only=False)
    validate_policy(policy, settings)
    qualification = settings.get('qualification')
    if require_qualification and not qualification:
        raise ValueError('nonzero live edits require owner-provided frozen-policy qualification evidence')
    if qualification:
        if sha256(qualification['path']) != qualification['sha256']:
            raise ValueError('qualification evidence SHA256 mismatch')
        evidence = json.loads(Path(qualification['path']).read_text())
        if evidence['policy_checkpoint_sha256'] != settings['policy_provenance']['policy_checkpoint_sha256']:
            raise ValueError('qualification evidence refers to another actor')
        if evidence.get('authorized_for_online_smoke') is not True:
            raise ValueError('qualification evidence does not authorize nonzero online smoke')
    return policy


def online_rank_checkpoint(promoted_rank, settings, *, mode):
    """Transition marker; actual controller reconstruction occurs after GPT restore."""
    if mode not in ('shadow', 'learned'):raise ValueError('shadow or learned suffix required')
    if promoted_rank['controller'] is not None or promoted_rank['config'].get('mode') != 'baseline':
        raise ValueError('only a baseline prefix can enter this transition')
    if promoted_rank['step'] != settings['prefix_steps']:
        raise ValueError('prefix clock differs')
    result = dict(promoted_rank)
    config = copy.deepcopy(promoted_rank['config'])
    config.update(mode=mode, transformer_chisel=copy.deepcopy(settings))
    preserve = settings.get('preserve_first_action_checkpoint', False)
    if type(preserve) is not bool:
        raise ValueError('first-action checkpoint retention must be boolean')
    if preserve and mode == 'learned':
        first = settings['prefix_steps'] + 1
        if first >= config['max_steps']:
            raise ValueError('first action needs a later ordinary step for pending credit')
        config['preserve_checkpoint_steps'] = sorted(set(config.get('preserve_checkpoint_steps', [])) | {first})
    validate_online_config(config)
    result['config'] = config
    result['controller'] = dict(schema=1, bootstrap=True, mode=mode, settings=copy.deepcopy(settings),
                                rank=promoted_rank['data_cursor']['rank'], completed_steps=promoted_rank['step'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--policy-sha256', required=True)
    p.add_argument('--state-bank', type=Path, required=True)
    p.add_argument('--settings', type=Path, required=True, help='explicit online learning, execution and qualification settings JSON')
    p.add_argument('--mode', choices=('shadow', 'learned'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--microbatch-sequences', type=int, default=16)
    p.add_argument('--max-steps', type=int, default=3250)
    p.add_argument('--train-shards', type=int, default=18)
    p.add_argument('--compile-loss', action='store_true', help='explicit pure-loss compilation at this new transition only')
    a = p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha256(a.input) != a.expected_sha256 or sha256(a.policy) != a.policy_sha256:
        raise ValueError('input bundle or policy SHA256 mismatch')
    bundle = torch.load(a.input, map_location='cpu', weights_only=False, mmap=True)
    policy = torch.load(a.policy, map_location='cpu', weights_only=False, mmap=True)
    from rewriter.state_bank import StateBankReader
    from scripts.bundle_baseline_checkpoint import extract_rank
    reader = StateBankReader(a.state_bank, mmap_model=True)
    step = bundle['shared']['step'];record = reader.read(step)
    if set(record.model) != set(bundle['shared']['model']):raise ValueError('prefix model inventory differs from bank')
    for name, value in record.model.items():
        other = bundle['shared']['model'][name]
        if value.dtype != other.dtype or not torch.equal(value, other):
            raise ValueError('resumable prefix is not the actor bank endpoint: '+name)
    identity = policy['metadata']['state_bank_identity']
    if identity['manifest_sha256'] != sha256(a.state_bank/'manifest.json'):
        raise ValueError('actor bank manifest differs')
    entries_sha = hashlib.sha256(json.dumps(reader.entries[:step], sort_keys=True).encode()).hexdigest()
    if identity['selected_committed_entries_sha256'] != entries_sha:
        raise ValueError('actor bank committed entries differ')
    settings = json.loads(a.settings.read_text())
    settings.setdefault('replica_check_every',64)
    settings.update(prefix_steps=step, architecture=policy['metadata']['architecture'],
        policy_checkpoint=str(a.policy.resolve()), bootstrap='first_ordinary_step_unedited', online_auxiliary='none',
        policy_provenance=dict(prefix_bundle_sha256=a.expected_sha256, policy_checkpoint_sha256=a.policy_sha256,
            state_bank_identity=identity, source_implementation_sha256=policy['metadata']['implementation_sha256']))
    # Sensor features must use the same prefix construction, including anchors.
    expected_sensors = reader.manifest['provenance']['sensor_config']
    sensor_keys = ('probe_rank','anchor_seed','signs_seed','anchor_refresh_period','trajectory_positions',
                   'train_sequence_probes','audit_sequence_probes')
    defaults = dict(probe_rank=8,anchor_seed=100001,signs_seed=300001,anchor_refresh_period=32,
                    trajectory_positions=64,train_sequence_probes=4,audit_sequence_probes=4)
    settings['sensor_config'] = {k:expected_sensors.get(k,defaults[k]) for k in sensor_keys}
    settings['sensor_config']['radius_fraction'] = settings['radius_fraction']
    load_policy_checkpoint(settings, device='cpu', require_qualification=a.mode=='learned')
    promoted = promote_bundle(bundle, source_sha256=a.expected_sha256,
        microbatch_sequences=a.microbatch_sequences, max_steps=a.max_steps, compile_model=False, training_shards=a.train_shards)
    # This is a new transition, never an ordinary-resume config mutation.
    promoted['shared']['config']['compile_loss'] = a.compile_loss
    records = [online_rank_checkpoint(extract_rank(promoted,r),settings,mode=a.mode)
               for r in range(promoted['shared']['config']['world_size'])]
    a.output.mkdir(parents=True)
    manifest=[]
    for r, state in enumerate(records):
        path=a.output/f'latest_rank{r}.pt';save_atomic(state,path)
        manifest.append(dict(rank=r,file=path.name,sha256=sha256(path)))
    (a.output/'config.json').write_text(json.dumps(records[0]['config'],indent=2)+'\n')
    (a.output/'manifest.json').write_text(json.dumps(dict(prefix_sha256=a.expected_sha256,
        actor_sha256=a.policy_sha256,mode=a.mode,rank_checkpoints=manifest),indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),ranks=len(records),prefix_steps=step,mode=a.mode)))


if __name__ == '__main__':main()
