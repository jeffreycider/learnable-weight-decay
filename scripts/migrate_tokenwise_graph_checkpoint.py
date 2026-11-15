"""Explicit metadata-only transition to a qualified tokenwise graph backend.

The reviewed contract pins both source archives, the changed dependency list,
and closed GPU evidence. Numerical state is checked after serialization; normal
trainer resume validation remains unchanged. This script never stops training.
"""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

import torch

from scripts.migrate_offline_systems_checkpoint import numerical_digest


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def inventory(path, expected):
    if sha(path) != expected:
        raise ValueError('source archive hash differs')
    result = {}
    with tarfile.open(path) as archive:
        for item in archive:
            if not item.isfile():
                continue
            name = str(Path(item.name))
            if name.startswith('/') or '..' in Path(name).parts or name in result:
                raise ValueError('unsafe or duplicate archive member')
            result[name] = hashlib.sha256(archive.extractfile(item).read()).hexdigest()
    return result


def migrate(source, destination, *, contract_path, old_archive, new_archive, evidence_root):
    source, destination = Path(source), Path(destination)
    contract = json.loads(Path(contract_path).read_text())
    if contract.get('schema') != 'qualified_tokenwise_graph_migration_v1':
        raise ValueError('unknown migration contract')
    if contract.get('qualified') is not True:
        raise ValueError('qualification contract is not closed')
    if destination.exists() or source.resolve() == destination.resolve():
        raise ValueError('new destination required')
    if not source.name.startswith('controller_epoch_'):
        raise ValueError('use an immutable complete-epoch checkpoint')
    old_files = inventory(old_archive, contract['old_archive_sha256'])
    new_files = inventory(new_archive, contract['new_archive_sha256'])
    evidence = []
    if {item['role'] for item in contract['qualification_evidence']} != {'full200_ddp', 'trainer_adapter'}:
        raise ValueError('full200 DDP and actual trainer-adapter evidence required')
    for item in contract['qualification_evidence']:
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
        evidence.append(item)
    if not evidence:
        raise ValueError('closed full-update and adapter evidence required')
    # Read once: digest and deserialization refer to the same immutable bytes.
    raw = source.read_bytes()
    input_sha = hashlib.sha256(raw).hexdigest()
    if sha(source) != input_sha:
        raise ValueError('input changed during read')
    state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    if state.get('completed') or state['next_batch'] != 0:
        raise ValueError('requires an unfinished complete-epoch boundary')
    metadata = state['metadata']
    if metadata['world_size'] != 8 or metadata['options']['objective'] != 'tokenwise-v1':
        raise ValueError('wrong run family')
    if metadata.get('calibration') is not None or metadata['tokenwise_objective'] != contract['tokenwise_objective']:
        raise ValueError('objective or full-bank scope differs')
    if metadata['options']['offline_backend'] != 'reference':
        raise ValueError('predecessor backend differs')
    if metadata['state_bank_identity'] != contract['state_bank_identity']:
        raise ValueError('bank or teacher function differs')
    old_sources = metadata['implementation_sha256']
    for name, digest in old_sources.items():
        if old_files.get(name) != digest:
            raise ValueError('checkpoint source differs: ' + name)
    target_sources = {name: new_files[name] for name in contract['target_implementation_paths']}
    if not set(old_sources) <= set(target_sources):
        raise ValueError('source identity lost a predecessor dependency')
    changed = sorted(name for name, digest in old_sources.items() if target_sources[name] != digest)
    if changed != sorted(contract['changed_existing_dependencies']):
        raise ValueError('unapproved existing dependency change')
    before = numerical_digest(state)
    result = copy.deepcopy(metadata)
    result['options']['offline_backend'] = 'tokenwise_graph'
    result['offline_backend'] = contract['target_backend_metadata']
    result['implementation_sha256'] = target_sources
    record = dict(kind=contract['schema'], input_checkpoint=str(source), input_checkpoint_sha256=input_sha,
                  old_archive_sha256=contract['old_archive_sha256'], new_archive_sha256=contract['new_archive_sha256'],
                  contract_sha256=sha(contract_path), qualification_evidence=evidence,
                  numerical_state_sha256=before, numerical_state_changed=False,
                  objective_or_update_equations_changed=False,
                  prior_source_migration=metadata.get('source_migration'))
    result['source_migration'] = record
    state['metadata'] = result
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with temporary.open('xb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    check = torch.load(temporary, map_location='cpu', weights_only=False)
    if numerical_digest(check) != before:
        temporary.unlink()
        raise ValueError('numerical state changed during serialization')
    os.link(temporary, destination)  # Never replace an existing destination.
    temporary.unlink()
    record['output_checkpoint_sha256'] = sha(destination)
    destination.with_suffix(destination.suffix + '.migration.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('input', 'output', 'contract', 'old-archive', 'new-archive', 'evidence-root'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.input, args.output, contract_path=args.contract,
          old_archive=args.old_archive, new_archive=args.new_archive, evidence_root=args.evidence_root)))
