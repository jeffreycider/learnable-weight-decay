"""Learn norm-reducing directions from same-run cached probes, without GPT passes.

This preparatory supervised objective does not measure function preservation.
The saved learned network must subsequently be tested/trained on finite KL.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from baseline.execution import enable_deterministic_execution
from rewriter.state_bank import StateBankReader
from rewriter.chisel_factory import add_chisel_arguments,chisel_architecture,build_chisel
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.direction_warmup import direction_targets, direction_loss, direction_metrics, aggregate_direction_metrics
from scripts.train_multistate_chisel import aggregate_controller_gradients, save_atomic, stable_seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_chisel_arguments(p)
    p.add_argument('--state-bank', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prefix-states', type=int, default=0)
    p.add_argument('--updates', type=int, default=1000)
    p.add_argument('--checkpoint-batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--seed', type=int, default=200001)
    p.add_argument('--audit-every', type=int, default=50)
    p.add_argument('--eager', action='store_true')
    p.add_argument('--wandb', action='store_true')
    args = p.parse_args()
    world = int(os.environ.get('WORLD_SIZE', 1)); rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.set_num_threads(4); enable_deterministic_execution()
    torch.cuda.set_device(local); device = torch.device('cuda', local)
    if world > 1: dist.init_process_group('nccl')
    if args.checkpoint_batch % world: p.error('checkpoint batch must be divisible by world size')
    reader = StateBankReader(args.state_bank, require_complete=not args.prefix_states, verify_once=True, mmap_model=True)
    limit = args.prefix_states or reader.manifest['expected_steps']
    probe_rank = reader.manifest['provenance']['sensor_config'].get('probe_rank', 8)
    depth = reader.manifest['provenance']['model_config']['num_layers']
    architecture=chisel_architecture(args,rank=probe_rank,trajectory_dim=(2*depth+1)*2*probe_rank)
    train = [s for s in reader.manifest['split']['train'] if s <= limit]
    audit = [s for s in reader.manifest['split']['audit'] if s <= limit]
    if not train or not audit: p.error('both train and held-out states required')
    if rank == 0: args.output.mkdir(parents=True, exist_ok=False)
    if world > 1: dist.barrier()
    metadata = dict(options={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},architecture=architecture,
        bank_manifest_sha256=hashlib.sha256((args.state_bank/'manifest.json').read_bytes()).hexdigest(),
        bank_provenance=reader.manifest['provenance'], train_states=train, audit_states=audit,
        purpose='same-run learned directional warmup; no GPT evaluation or preservation claim',
        target='canonical projection of negative base weights, equal-role normalized MSE',
        world_size=world, main_model_forwards=0, main_model_backwards=0)
    started = time.perf_counter(); wb = None
    if rank == 0:
        (args.output/'config.json').write_text(json.dumps(metadata, indent=2)+'\n')
        stream = (args.output/'metrics.jsonl').open('w')
        if args.wandb:
            import wandb
            wb = wandb.init(project='chisel', mode='offline', dir=str(args.output), config=metadata)
    def log(event, **values):
        if rank != 0: return
        row = dict(event=event, elapsed_seconds=time.perf_counter()-started, **values)
        print(json.dumps(row), flush=True); stream.write(json.dumps(row)+'\n'); stream.flush()
        if wb: wb.log(row)
    # Ownership is fixed so each snapshot and target resides on only one GPU.
    owned_train = train[rank::world]; owned_audit = audit[rank::world]
    if not owned_train: p.error('too few training states for ranks')
    cache = {}
    for state_id in owned_train + owned_audit:
        record = reader.read(state_id, device=device)
        grouped = GroupedNormObjective(record.model, zero_reference_policy='freeze')
        target = direction_targets(record.snapshot, grouped.groups, grouped.reference_squared,
                                   {n:v.shape for n,v in record.model.items()})
        cache[state_id] = (record.snapshot, target)
        del record, grouped
        if rank == 0: log('loaded', state_id=state_id, resident_states=len(cache))
    torch.cuda.empty_cache()
    with torch.random.fork_rng(devices=[local]):
        torch.manual_seed(args.seed)
        actor = build_chisel(architecture).to(device)
    forward = actor if args.eager else torch.compile(actor, fullgraph=True, dynamic=False)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr, betas=(.9,.99), eps=1e-8)
    local_batch = args.checkpoint_batch // world
    def evaluate(update):
        rows=[]
        # Keep grad-enabled actor arithmetic identical to the training forward.
        for role, ids in [('train', owned_train), ('heldout_state', owned_audit)]:
            for state_id in ids:
                snapshot, target = cache[state_id]
                output = forward(snapshot)
                rows.append(dict(role=role, state_id=state_id, loss=float(direction_loss(output,target).detach()),
                                 groups=direction_metrics(output,target)))
                del output
        if world > 1:
            gathered=[None]*world; dist.all_gather_object(gathered,rows); rows=[r for part in gathered for r in part]
        for row in rows: log('state_audit',update=update,**row)
        for role in ['train','heldout_state']:
            chosen=[r for r in rows if r['role']==role]
            log('audit',update=update,role=role,objective=sum(r['loss'] for r in chosen)/len(chosen),
                groups=aggregate_direction_metrics([r['groups'] for r in chosen]))
    evaluate(0)
    order=[]; cursor=0; epoch=0
    for update in range(1,args.updates+1):
        optimizer.zero_grad(set_to_none=True); total=0.; tick=time.perf_counter()
        for _ in range(local_batch):
            if cursor==len(order):
                generator=torch.Generator().manual_seed(stable_seed(args.seed,rank,epoch))
                order=[owned_train[i] for i in torch.randperm(len(owned_train),generator=generator).tolist()]
                cursor=0; epoch+=1
            state_id=order[cursor];cursor+=1
            snapshot,target=cache[state_id];action=forward(snapshot);loss=direction_loss(action,target)
            (loss/args.checkpoint_batch).backward();total+=float(loss.detach())/args.checkpoint_batch
            del action,loss
        aggregate_controller_gradients(actor,world)
        grad=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True))
        optimizer.step();torch.cuda.synchronize()
        value=torch.tensor(total,device=device)
        if world>1:dist.all_reduce(value)
        log('update',update=update,objective=float(value),gradient_norm=grad,seconds=time.perf_counter()-tick)
        if update%args.audit_every==0 or update==args.updates:
            evaluate(update)
            if rank==0:save_atomic(args.output/'direction_latest.pt',dict(controller=actor.state_dict(),optimizer=optimizer.state_dict(),
                metadata=metadata,updates=update,completed=update==args.updates))
    from torch._dynamo.utils import counters
    log('finished',updates=args.updates,main_model_forwards=0,main_model_backwards=0,
        compilation_counters={key:dict(value) for key,value in counters.items()},
        rank0_peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    if wb:wb.finish()
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
