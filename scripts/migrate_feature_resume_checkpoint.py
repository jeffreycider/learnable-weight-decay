"""One explicit metadata migration for the directional-initialization resume fix.

The two trainer fingerprints are fixed here. Every other recorded source file
must match. This never overwrites the input checkpoint or changes numerical
training state. It is not a general override for source compatibility checks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

SCRIPT = 'scripts/train_multistate_chisel.py'
OLD_SCRIPT = '9cbe43e68566cfc0ee7fab79fa2a90194181766503f7d08566e0d29943810c5a'
NEW_SCRIPT = '620f6ab20d6a3f06d10ef6e5cedd78580be573198ea0ccedb1c9d1ae511bd937'


def migrate(source_checkpoint, destination, new_source):
    source_checkpoint, destination, new_source = map(Path, (source_checkpoint, destination, new_source))
    if destination.exists() or source_checkpoint.resolve() == destination.resolve():
        raise ValueError('migration requires a new output path; original checkpoint stays immutable')
    state = torch.load(source_checkpoint, map_location='cpu', weights_only=False)
    metadata = state['metadata']
    if 'same_run_direction_warmup' not in metadata:
        raise ValueError('this migration is only for a directional-warmup initialized fit')
    sources = dict(metadata['implementation_sha256'])
    if sources.get(SCRIPT) != OLD_SCRIPT:
        raise ValueError('checkpoint does not use the declared predecessor trainer')
    for name, original in sources.items():
        expected = NEW_SCRIPT if name == SCRIPT else original
        actual = hashlib.sha256((new_source / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('unapproved source change: ' + name)
    sources[SCRIPT] = NEW_SCRIPT
    record = dict(kind='direction_warmup_resume_metadata_fix_v1',
                  input_checkpoint_sha256=hashlib.sha256(source_checkpoint.read_bytes()).hexdigest(),
                  old_trainer_sha256=OLD_SCRIPT, new_trainer_sha256=NEW_SCRIPT,
                  reason='Treat directional initialization as initialization-only on resume and retain its provenance',
                  numerical_state_changed=False, objective_or_update_equations_changed=False)
    state['metadata'] = dict(metadata, implementation_sha256=sources, source_migration=record)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    destination.with_suffix(destination.suffix + '.migration.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--new-source', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.input, args.output, args.new_source)))
