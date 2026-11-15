"""Saved real GPT private/Adam snapshot transport and immutable-write gate.

No actor/GPT/CNN forward, backward, optimizer step or policy selection. Every
candidate serializes the same saved private owner and is restored for equality.
"""
from pathlib import Path
import argparse
import json
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.private_probe_slot import PrivateProbeSlot
from rewriter.private_probe_store import PrivateProbeStore, PrivateIdentity
import rewriter.private_probe_store as store_module
import rewriter.owned_cpu_snapshot as snapshot_module
from rewriter.state_bank import StateBankReader
from rewriter.multibank import file_sha
from scripts.continue_coordinate_chisel import stage_origin
from scripts.qualify_coordinate_staging import differences


def benchmark(args):
    if file_sha(args.checkpoint) != args.checkpoint_sha256: raise ValueError('saved fixture SHA changed')
    args.output.mkdir(parents=True, exist_ok=False)
    enable_deterministic_execution();torch.cuda.set_device(0);torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    protocol = stage_origin(saved['metadata'])['protocol']; architecture = dict(protocol['architecture'])
    architecture.pop('name'); architecture.pop('decoder_scale')
    actor = CoordinateChisel(**architecture).cuda();actor.load_state_dict(saved['actor'])
    record = StateBankReader(args.state_bank, verify_once=True, mmap_model=True).read(args.state_id)
    key = ('gpt', args.state_id); original = saved['private'][key]
    private = actor.make_private(record.snapshot, state_key=original['parameters']['_extra_state']['state_key'], seed=1)
    private.load_state_dict(original['parameters']); slot = PrivateProbeSlot(actor, private)
    optimizer = torch.optim.Adam(private.parameters(), lr=1e-4);optimizer.load_state_dict(original['optimizer'])
    identity = PrivateIdentity.from_private(private, state_sha256=record.assert_verified())
    before = snapshot_module.owned_cpu_snapshot(dict(private=private.state_dict(), optimizer=optimizer.state_dict()))
    del actor, record, saved
    phases = {}; original_cpu = store_module._cpu; original_validate = store_module._validate_payload
    original_copy = snapshot_module.owned_cpu_snapshot; original_write = PrivateProbeStore._write
    depth = 0
    def elapsed(name, function, *a, **k):
        start = time.perf_counter()
        try: return function(*a, **k)
        finally: phases[name] = phases.get(name, 0.) + time.perf_counter() - start
    def cpu(value):
        nonlocal depth
        depth += 1
        try: return elapsed('recursive_cpu_copy_outer', original_cpu, value) if depth == 1 else original_cpu(value)
        finally: depth -= 1
    store_module._cpu = cpu
    store_module._validate_payload = lambda *a,**k:elapsed('payload_validation', original_validate, *a, **k)
    snapshot_module.owned_cpu_snapshot = lambda *a,**k:elapsed('pinned_snapshot_copy', original_copy, *a, **k)
    PrivateProbeStore._write = lambda *a,**k:elapsed('immutable_write_including_nested_copy_fsync_sha', original_write, *a, **k)
    export = slot.export_state;slot.export_state = lambda:elapsed('private_export_d2h', export)
    import wandb
    wb = wandb.init(project='chisel', mode='offline', dir=str(args.output), name='private-snapshot-v1', config=vars(args))
    rows = []; backends = ['reference','owned_cpu_v1','pinned_d2h_v1']
    stores = {mode:PrivateProbeStore(args.output/mode, snapshot_backend=mode) for mode in backends}
    try:
        for iteration in range(args.repeats + 1):
            for mode in backends if iteration % 2 == 0 else list(reversed(backends)):
                torch.cuda.synchronize();phases.clear();start=time.perf_counter()
                version=stores[mode].snapshot_slot(identity,slot,optimizer,visits=args.visits)
                seconds=time.perf_counter()-start; measured=dict(phases)
                payload=stores[mode].read_private(identity,version=version)
                diff=differences(before,dict(private=payload['private'],optimizer=payload['optimizer']))
                stride_equal=all(value.stride()==payload['private'][name].stride() for name,value in before['private'].items() if isinstance(value,torch.Tensor))
                stores[mode].stage(identity,slot,optimizer,version=version)
                after=original_copy(dict(private=private.state_dict(),optimizer=optimizer.state_dict()))
                diff += differences(before,after,'restored')
                row=dict(iteration=iteration,warm=iteration>0,backend=mode,seconds=seconds,
                         phases=measured,bytes=version['bytes'],version=version,
                         all_numeric_state_bitexact=not diff,strides_equal=stride_equal,differences=diff)
                rows.append(row); wb.log({'snapshot_seconds':seconds,'backend':mode,'exact':not diff,'warm':iteration>0})
                (args.output/'partial.json').write_text(json.dumps(rows,indent=2))
                if diff or not stride_equal: raise RuntimeError('private snapshot numeric/layout roundtrip failed')
        medians={mode:statistics.median(row['seconds'] for row in rows if row['warm'] and row['backend']==mode) for mode in backends}
        summary=dict(scope='One actual saved GPT private owner plus its Adam; complete D2H/validation/immutable write/fsync/SHA and exact restore check. Timed interval excludes restore/read verification, includes snapshot_slot only. Not complete training throughput.',
                     checkpoint=str(args.checkpoint),checkpoint_sha256=args.checkpoint_sha256,state_id=args.state_id,
                     private_parameters=sum(p.numel() for p in private.parameters()),rows=rows,median_seconds=medians,
                     all_gates=True,model_forwards=0,model_backwards=0,optimizer_steps=0,wandb_directory=str(Path(wb.dir).parent))
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    finally:
        for store in stores.values():store.close()
        store_module._cpu=original_cpu;store_module._validate_payload=original_validate
        snapshot_module.owned_cpu_snapshot=original_copy;PrivateProbeStore._write=original_write
        wb.finish()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--state-bank',type=Path,required=True);p.add_argument('--state-id',type=int,default=6)
    p.add_argument('--visits',type=int,default=256);p.add_argument('--repeats',type=int,default=4)
    p.add_argument('--output',type=Path,required=True);benchmark(p.parse_args())
