"""Explicit hash-pinned migrations for qualified execution-only changes."""
from copy import deepcopy
from pathlib import Path
import yaml
from chisel.storage import sha256

PERFORMANCE_FIELDS = {
    'private_probes.cache_bytes', 'private_probes.buffered_load', 'cache_normalized_sensors',
    'compile_preservation_loss', 'teacher_cache_bytes',
    'checkpoint_io.cache_bytes', 'checkpoint_io.pin_memory',
    'distributed_audit', 'training_audit_inputs', 'private_store',
}
IGNORED = {'resume', 'output_dir', 'stop_step', 'resume_compatibility'}


def validate(saved, config, sources):
    specification = config['resume_compatibility']
    path = Path(specification['path'])
    if sha256(path) != specification['sha256']:
        raise ValueError('resume compatibility manifest SHA256 mismatch')
    manifest = yaml.safe_load(path.read_text())
    fields = set(manifest['configuration_fields'])
    if not fields <= PERFORMANCE_FIELDS:
        raise ValueError('resume migration may only change qualified performance fields')
    if 'private_store' in fields:
        for settings in (saved['config'], config):
            store = settings.get('private_store', {'_target_': 'chisel.offline.ProbeStore'})
            if (store.get('_target_') not in ('chisel.offline.ProbeStore', 'chisel.device_probe_store.DeviceProbeStore')
                    or set(store) - {'_target_', 'cache_bytes'}):
                raise ValueError('unqualified private store migration')
    def comparable(value):
        value = deepcopy(value)
        for key in IGNORED:
            value.pop(key, None)
        for field in fields:
            keys = field.split('.')
            parent = value
            for key in keys[:-1]:
                parent = parent.get(key, {})
            parent.pop(keys[-1], None)
        return value
    if comparable(config) != comparable(saved['config']):
        raise ValueError('resume migration changes training configuration')
    old = saved.get('source_hashes', {})
    changes = manifest['source_changes']
    for name in old.keys() | sources.keys():
        if old.get(name) == sources.get(name):
            continue
        allowed = changes.get(name, {})
        if ('captured' not in allowed or 'current' not in allowed
                or allowed['captured'] != old.get(name)
                or allowed['current'] != sources.get(name)):
            raise ValueError(f'unqualified resume source change: {name}')
