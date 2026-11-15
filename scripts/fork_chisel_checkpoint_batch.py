"""Explicit scientific batch/LR fork; preserve moments, duals and epoch clocks.

Unlike a performance migration, this intentionally changes the Adam update
frequency. A closed source/finite-update contract is mandatory. It does not stop
the parent run, reset training, or bypass the trainer's strict resume checks.
"""
import argparse
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path

import torch

from scripts.migrate_tokenwise_graph_checkpoint import inventory, sha
from scripts.migrate_offline_systems_checkpoint import numerical_digest


def protected_digest(state):
    """Every nonmetadata field except explicitly permitted group LR scalars."""
    value = copy.deepcopy(state)
    for group in value['optimizer']['param_groups']:
        group['lr'] = '<declared-branch-learning-rate>'
    return numerical_digest(value)


def fork(source, destination, *, contract_path, old_archive, new_archive,
         evidence_root, head_lr):
    source, destination = Path(source), Path(destination)
    contract = json.loads(Path(contract_path).read_text())
    if contract.get('schema') != 'qualified_checkpoint_batch_fork_v1' or contract.get('qualified') is not True:
        raise ValueError('closed batch-fork qualification required')
    if not math.isfinite(head_lr) or head_lr not in contract['head_learning_rates']:
        raise ValueError('unapproved branch learning rate')
    if destination.exists() or source.resolve() == destination.resolve():
        raise ValueError('new destination required')
    if not source.name.startswith('controller_epoch_'):
        raise ValueError('immutable complete-epoch input required')
    old_files = inventory(old_archive, contract['old_archive_sha256'])
    new_files = inventory(new_archive, contract['new_archive_sha256'])
    evidence = contract['qualification_evidence']
    if not evidence or {item['head_lr'] for item in evidence} != set(contract['head_learning_rates']):
        raise ValueError('finite-update evidence required at both branch learning rates')
    for item in evidence:
        path = Path(evidence_root) / item['path']
        if sha(path) != item['sha256']:
            raise ValueError('qualification evidence hash differs')
        data = json.loads(path.read_text())
        for keys, expected in item['required_values']:
            value = data
            for key in keys:
                value = value[key]
            if value != expected:
                raise ValueError('qualification gate did not close')
    raw = source.read_bytes()
    input_sha = hashlib.sha256(raw).hexdigest()
    if input_sha != contract['input_checkpoint_sha256'] or sha(source) != input_sha:
        raise ValueError('input checkpoint hash differs or changed during read')
    state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    if state.get('completed') or state['next_batch'] != 0:
        raise ValueError('unfinished complete-epoch input required')
    if (state['updates'], state['next_epoch']) != (contract['input_updates'], contract['input_next_epoch']):
        raise ValueError('unexpected update or epoch clock')
    metadata = state['metadata']; options = metadata['options']
    if metadata['world_size'] != 8 or options['objective'] != 'tokenwise-v1' or options['offline_backend'] != 'tokenwise_graph':
        raise ValueError('wrong run family')
    if options['checkpoint_batch'] != 200 or contract['target_checkpoint_batch'] != 40:
        raise ValueError('only declared batch200 to40 fork supported')
    if metadata['tokenwise_objective'] != contract['tokenwise_objective'] or metadata['state_bank_identity'] != contract['state_bank_identity']:
        raise ValueError('objective or state/token bank changed')
    sources = metadata['implementation_sha256']
    if any(old_files.get(name) != digest for name, digest in sources.items()):
        raise ValueError('input source identity differs')
    changed = sorted(name for name, digest in sources.items() if new_files.get(name) != digest)
    if changed != ['rewriter/offline_tokenwise_graph.py']:
        raise ValueError('only the reviewed tokenwise batch validator may change')
    # Also reject any unlisted runtime edits in the archive, not just tracked
    # resume dependencies. New qualifier/tests are explicitly enumerated.
    changes = sorted(name for name in old_files.keys() | new_files.keys() if old_files.get(name) != new_files.get(name))
    if changes != sorted(contract['archive_changed_paths']):
        raise ValueError('source archive differs beyond reviewed overlay')
    groups = state['optimizer']['param_groups']
    if [g.get('role') for g in groups] != ['action_heads', 'representation']:
        raise ValueError('unexpected Adam parameter grouping')
    if [g.get('lr_multiplier') for g in groups] != [1., 10.]:
        raise ValueError('representation LR multiplier changed')
    old_lrs = [g['lr'] for g in groups]
    if options['lr'] != contract['input_head_lr'] or any(g['lr'] != options['lr'] * g['lr_multiplier'] for g in groups):
        raise ValueError('saved optimizer LR differs from declared source stage')
    before = protected_digest(state)
    for group in groups:
        group['lr'] = head_lr * group['lr_multiplier']
    updated = copy.deepcopy(metadata)
    updated['options'].update(checkpoint_batch=40, lr=head_lr)
    updated['implementation_sha256'] = {name: new_files[name] for name in sources}
    record = dict(kind=contract['schema'], input_checkpoint=str(source), input_checkpoint_sha256=input_sha,
        old_archive_sha256=contract['old_archive_sha256'], new_archive_sha256=contract['new_archive_sha256'],
        contract_sha256=sha(contract_path), qualification_evidence=evidence,
        changes={'checkpoint_batch': [200, 40], 'head_lr': [options['lr'], head_lr],
                 'optimizer_group_lrs': [old_lrs, [g['lr'] for g in groups]]},
        protected_state_sha256=before, actor_moments_duals_rng_schedule_cursors_preserved=True,
        objective_changed=False, update_frequency_changed=True,
        first_review_completed_epoch=contract['first_review_completed_epoch'],
        first_review_is_stopping_boundary=False,
        prior_source_migration=metadata.get('source_migration'))
    updated['source_migration'] = record; state['metadata'] = updated
    if protected_digest(state) != before:
        raise ValueError('undeclared scientific state change')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with temporary.open('xb') as stream:
        torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
    recovered = torch.load(temporary, map_location='cpu', weights_only=False)
    if protected_digest(recovered) != before or recovered['optimizer']['param_groups'] != groups:
        temporary.unlink(); raise ValueError('state changed during serialization')
    os.link(temporary, destination); temporary.unlink()
    record['output_checkpoint_sha256'] = sha(destination)
    destination.with_suffix(destination.suffix + '.fork.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output', 'contract', 'old-archive', 'new-archive', 'evidence-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--head-lr', type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(fork(args.input, args.output, contract_path=args.contract,
        old_archive=args.old_archive, new_archive=args.new_archive,
        evidence_root=args.evidence_root, head_lr=args.head_lr)))
