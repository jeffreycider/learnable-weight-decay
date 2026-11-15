"""Shared tiny fixtures for Chisel systems tests, importable without any optional
systems module (the owner's 05:27 CUDA component attempt failed because the
fixture lived in tests/test_step_timing.py, whose module-level imports need
rewriter.step_timing that an older checkout lacks). Only production modules here."""
import os
os.environ.setdefault('PROFILE_CHISEL_SKIP_SOURCE_GUARD', '1')  # shared tree is edited concurrently by other agents
import json
import struct
import numpy as np
import torch
from baseline.model import GPT
from rewriter.function_preservation import FunctionalPreservation


def tiny_engine_and_inputs():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = GPT(16, 1, 128).float().requires_grad_(False)
    engine = FunctionalPreservation(model, microbatch_sequences=2, kl_dtype=torch.float64)
    inputs = torch.randint(16, (6, 4))
    return engine, inputs


def tiny_bank_and_cache(tmp_path, gpt_precision='baseline'):
    """Five-state tiny bank plus a complete exact teacher cache (train/audit panels).

    ``gpt_precision='fp32'`` writes the cache with an fp32 engine (production mode), so
    fp32 engines pass ``validate_precision`` when loading it; the default keeps the
    historical baseline-precision fixture every other test was written against."""
    from tests.test_multistate_chisel import _tiny_state_bank
    from scripts.train_multistate_chisel import state_replay, select_sequences, stable_seed
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import write_panel
    bank = tmp_path / 'bank'
    cfg = _tiny_state_bank(bank)
    data = tmp_path / 'fineweb_train_000001.bin'
    tokens = (np.arange(1000) % 16).astype(np.uint16)
    data.write_bytes(struct.pack('<iii', 20240520, 1, len(tokens)) + bytes(1012) + tokens.tobytes())
    reader = StateBankReader(bank)
    cache = tmp_path / 'targets'
    engine = FunctionalPreservation(GPT(**cfg['model']), microbatch_sequences=1, kl_dtype=torch.float64, gpt_precision=gpt_precision)
    names = set(engine.parameters)
    rows = []
    for state_id in range(1, 6):
        record = reader.read(state_id)
        engine.load_teacher_state({n: v for n, v in record.model.items() if n in names},
                                  {n: v for n, v in record.model.items() if n not in names})
        pools = state_replay([str(data)], state_id, cfg, pool_sequences=32, seed=400001, epoch=0, split_seed=500001)
        for role in ('train', 'audit'):
            inputs, ids = select_sequences(pools[role], 2, seed=stable_seed('fixed-panel-test', state_id, role))
            rows.append(write_panel(cache, state_id, role, engine, inputs, ids,
                                    state_sha256=reader.entries[state_id - 1]['sha256'], execution={'test': True}))
    (cache / 'manifest.json').write_text(json.dumps({'complete': True, 'args': {'states': 5}, 'panels': rows}))
    return bank, cache
