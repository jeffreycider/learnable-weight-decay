"""Explicit seed/local-clock routing over immutable, unchanged state banks.

Global IDs name training examples, never GPT steps. The composite index pins
each original manifest/index and teacher manifest; records and cached features
are not copied, rewritten, requantized, or allowed to cross seed namespaces.
"""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .state_bank import StateBankReader
from .teacher_cache import TeacherCacheReader

SCHEMA = 'chisel_multibank_v1'


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


class MultiStateBankReader:
    def __init__(self, manifest_path, *, verify_once=False, mmap_model=False):
        self.manifest_path = Path(manifest_path)
        self.directory = self.manifest_path.parent
        self.index = json.loads(self.manifest_path.read_text())
        if self.index.get('schema') != SCHEMA:
            raise ValueError('unsupported composite bank schema')
        self.stride = self.index['state_id_stride']
        if type(self.stride) is not int or self.stride < 1:
            raise ValueError('positive state ID stride required')
        banks = self.index['banks']; seeds = [b['seed'] for b in banks]
        if not seeds or seeds != list(range(len(seeds))):
            raise ValueError('ordered contiguous seed namespaces starting at zero required')
        holdout = self.index.get('heldout_online_seeds', [])
        if any(type(s) is not int or s < 0 for s in holdout) or set(holdout) & set(seeds):
            raise ValueError('online holdout seed overlaps pretraining')
        self.readers = {}; self.entries = []; split = {'train': [], 'audit': []}
        common = None
        for item in banks:
            seed = item['seed']; path = Path(item['state_bank'])
            for filename, key in [('manifest.json', 'state_manifest_sha256'), ('index.jsonl', 'state_index_sha256')]:
                if file_sha(path / filename) != item[key]:
                    raise ValueError('composite child state identity differs')
            reader = StateBankReader(path, verify_once=verify_once, mmap_model=mmap_model)
            provenance = reader.manifest['provenance']
            if provenance['seed'] != seed or reader.manifest['expected_steps'] != self.stride:
                raise ValueError('seed or local state count differs')
            signature = dict(model_config=provenance['model_config'],
                schedule_steps=provenance['schedule_steps'], capture_world_size=provenance['capture_world_size'],
                probe_rank=provenance['sensor_config'].get('probe_rank', provenance['sensor_config'].get('rank', 8)),
                training_shape={k:provenance.get('training_config', {}).get(k)
                    for k in ('batch_tokens','microbatch_sequences','seq_len')})
            if common is None: common = signature
            elif signature != common: raise ValueError('incompatible GPT/capture layouts across seeds')
            self.readers[seed] = reader
            for entry in reader.entries:
                local = entry['completed_steps']; uid = seed * self.stride + local
                self.entries.append(dict(entry, completed_steps=uid, local_completed_steps=local, bank_seed=seed))
            for role in split:
                split[role].extend(seed * self.stride + local for local in reader.manifest['split'][role])
        provenance = copy.deepcopy(self.readers[0].manifest['provenance'])
        provenance.update(seed=None, pretraining_seeds=seeds)
        self.manifest = dict(schema=SCHEMA, expected_steps=len(self.entries), provenance=provenance,
            split=split, baseline_identity=True, composite_index=self.index,
            state_clock='training ID=seed*stride+local_step; record/sensor/data clocks remain local')

    def route(self, state_id):
        if type(state_id) is not int or not 1 <= state_id <= len(self.entries):
            raise IndexError('training state outside composite bank')
        return divmod(state_id - 1, self.stride)[0], (state_id - 1) % self.stride + 1

    def read(self, state_id, *, device='cpu', model_device=None):
        seed, local = self.route(state_id)
        record = self.readers[seed].read(local, device=device, model_device=model_device)
        return replace(record, training_state_id=state_id, bank_seed=seed)

    def iter_split(self, split='train', *, device='cpu'):
        if split not in ('train', 'audit'): raise ValueError('unknown state split')
        for uid in self.manifest['split'][split]: yield self.read(uid, device=device)

    def describe_state(self, state_id):
        seed, local = self.route(state_id)
        return dict(state_id=state_id, bank_seed=seed, gpt_completed_steps=local)


class MultiTeacherCacheReader:
    def __init__(self, state_reader, *, verify_once=False, mmap=False):
        if not isinstance(state_reader, MultiStateBankReader):
            raise TypeError('composite state reader required')
        self.states = state_reader; self.readers = {}; self.expected = {}
        execution = None; child_hashes = {}
        for item in state_reader.index['banks']:
            seed = item['seed']; path = Path(item['teacher_cache'])
            if file_sha(path / 'manifest.json') != item['teacher_manifest_sha256']:
                raise ValueError('composite teacher manifest differs')
            reader = TeacherCacheReader(path, verify_once=verify_once, mmap=mmap)
            current = reader.manifest.get('execution', {})
            signature = {k: current.get(k) for k in ('gpt_precision', 'precision', 'microbatch', 'softcap')}
            if execution is None: execution = signature
            elif signature != execution: raise ValueError('teacher arithmetic differs across seeds')
            if reader.manifest['args']['states'] != state_reader.stride:
                raise ValueError('teacher state coverage differs')
            child = state_reader.readers[seed]
            for (local, role), row in reader.expected.items():
                if row['state_sha256'] != child.entries[local - 1]['sha256']:
                    raise ValueError('teacher cache belongs to another state or seed')
                uid = seed * state_reader.stride + local
                self.expected[(uid, role)] = dict(row, training_state_id=uid, bank_seed=seed)
            self.readers[seed] = reader; child_hashes[str(seed)] = item['teacher_manifest_sha256']
        self.manifest = dict(format='composite_exact_head_input_v1', complete=True,
            execution=execution, child_manifest_sha256=child_hashes,
            args={'states': len(state_reader.entries)}, panels=list(self.expected.values()))

    def read(self, state_id, role):
        seed, local = self.states.route(state_id)
        panel = self.readers[seed].read(local, role)
        # Original metadata state_id is the LOCAL GPT clock and remains so.
        return replace(panel, metadata=dict(panel.metadata, training_state_id=state_id, bank_seed=seed))

    def load_engine(self, engine, record):
        seed, local = self.states.route(record.state_id)
        if record.bank_seed != seed or record.completed_steps != local:
            raise ValueError('record namespace or local clock differs')
        return self.readers[seed].load_engine(engine, record)
