"""Freeze terminal CNN observations and all 50K FP32 teacher logits. No Chisel updates."""
import argparse,json,os,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.capture_cifar_chisel_bank import load_airbench,SCHEMA as CAPTURE_SCHEMA,PRISTINE_SHA256
from rewriter.cifar_adapter import CifarPreservation,observe_cifar_terminal
from rewriter.cifar_bank import SCHEMA,sha256_file,tensor_tree,normalized_images

def build(args):
    torch.set_num_threads(4);torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    data=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train=torch.load(data/'train.pt',weights_only=False,map_location='cpu')
    split=torch.load(data/'chisel_image_split.pt',weights_only=False,map_location='cpu')
    fitting=split['fitting_indices'];heldout=split['heldout_indices']
    all_indices=torch.cat((fitting,heldout))
    if (len(fitting)!=40000 or len(heldout)!=10000 or all_indices.dtype!=torch.long
        or not torch.equal(all_indices.sort().values,torch.arange(50000))):
        raise ValueError('fixed 40K/10K image split required')
    if train['labels'].dtype!=torch.long or train['labels'].shape!=(50000,) or not ((train['labels']>=0)&(train['labels']<10)).all():
        raise ValueError('50000 integer CIFAR class labels required')
    images=normalized_images(train['images'],args.device);labels=train['labels'].to(args.device)
    module=load_airbench(data,defer_test_readouts=True)
    # Pin terminal teacher inference only; captured base training is unchanged.
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    model=module.CifarNet().to(args.device).float().eval()
    sources=sorted(Path(args.capture).glob('worker*/seed_*.pt'))
    seeds=[int(p.stem.split('_')[1]) for p in sources]
    if sorted(seeds)!=list(range(100)):raise ValueError('requires exactly all 100 independently captured seeds')
    source_hashes={str(p.relative_to(args.capture)):sha256_file(p) for p in sources}
    for p in sources:
        rows=[json.loads(line) for line in (p.parent/'manifest.jsonl').read_text().splitlines() if line]
        matches=[r for r in rows if r['file']==p.name]
        if len(matches)!=1 or matches[0]['sha256']!=source_hashes[str(p.relative_to(args.capture))]:
            raise ValueError('terminal artifact differs from capture manifest')
    config=dict(schema=SCHEMA,source_hashes=source_hashes,train_sha256=sha256_file(data/'train.pt'),
        image_split_sha256=sha256_file(data/'chisel_image_split.pt'),rank=args.rank,anchor_seed_base=args.anchor_seed,
        calibration_images=args.calibration_images,trajectory_images=64,teacher_images=50000,
        teacher_dtype='float32',normalization='uint8/255 -> fixed CIFAR mean/std, all FP32, no augmentation',
        base_training_dtype='upstream mixed precision; terminal observation and teachers strict FP32',
        fitting_images=40000,audit_images=10000,calibration_ids=fitting[:args.calibration_images].tolist(),
        inference=dict(matmul_tf32=False,cudnn_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True,
                       teacher_microbatch_images=args.microbatch_images,input_memory_format='channels_last'),
        script_sha256=sha256_file(__file__),adapter_sha256=sha256_file(Path(__file__).resolve().parents[1]/'rewriter/cifar_adapter.py'))
    header=out/'config.json'
    temporary_header=out/f'config.worker{args.worker}.tmp'
    temporary_header.write_text(json.dumps(config,indent=2))
    try:os.link(temporary_header,header)
    except FileExistsError:pass
    finally:temporary_header.unlink()
    if json.loads(header.read_text())!=config:raise ValueError('resume configuration changed')
    entries=[]
    for p in sources:
        seed=int(p.stem.split('_')[1]);local_id=seed+1;dest=out/f'model_{local_id:04d}.pt';receipt=out/f'model_{local_id:04d}.json'
        if seed%args.workers!=args.worker:continue
        if receipt.exists():
            e=json.loads(receipt.read_text())
            if sha256_file(dest)!=e['sha256']:raise ValueError('corrupt resumed record')
            entries.append(e);continue
        if dest.exists():raise ValueError('unreceipted record; inspect interrupted write before retry')
        start=time.perf_counter();checkpoint=torch.load(p,map_location=args.device,weights_only=False)
        if (checkpoint['seed']!=seed or checkpoint['schema']!=CAPTURE_SCHEMA
            or checkpoint['source']['pristine_sha256']!=PRISTINE_SHA256):
            raise ValueError('terminal seed or pinned source mismatch')
        model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
        engine=CifarPreservation(model,microbatch_images=args.microbatch_images)
        ids=fitting[:args.calibration_images].to(args.device)
        snapshot,metadata=observe_cifar_terminal(engine,images[ids],labels[ids],momentum=checkpoint['momentum'],
            anchor_seed=args.anchor_seed+seed,rank=args.rank,trajectory_images=64)
        teacher=torch.empty(50000,10,dtype=torch.float32,device='cpu')
        with torch.no_grad():
            for lo in range(0,len(images),args.microbatch_images):
                panel=images[lo:lo+args.microbatch_images];valid=len(panel)
                if valid<args.microbatch_images:
                    panel=torch.cat((panel,images[:args.microbatch_images-valid])).contiguous(memory_format=torch.channels_last)
                values=engine._forward(engine.parameters,panel)
                teacher[lo:lo+valid].copy_(values[:valid].cpu())
        if not torch.isfinite(teacher).all():raise RuntimeError('nonfinite teacher logits')
        metadata.update(teacher_forwards=(50000+args.microbatch_images-1)//args.microbatch_images,teacher_images=50000,
            teacher_padding_images=(-50000)%args.microbatch_images,
            capture_source=checkpoint['source'],capture_evaluation_deferred=checkpoint.get('evaluation_deferred',False),
            teacher_role='cache only; later fitting consumes fitting_indices exclusively',source_sha256=source_hashes[str(p.relative_to(args.capture))])
        payload=dict(schema=SCHEMA,seed=seed,local_id=local_id,snapshot=tensor_tree(snapshot),
            state_dict=tensor_tree(engine.model.state_dict()),teacher_logits=teacher,metadata=metadata)
        tmp=dest.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,dest)
        e=dict(local_id=local_id,seed=seed,file=dest.name,sha256=sha256_file(dest),seconds=time.perf_counter()-start)
        temporary_receipt=receipt.with_suffix('.json.tmp');temporary_receipt.write_text(json.dumps(e,indent=2))
        os.replace(temporary_receipt,receipt);entries.append(e)
        print(json.dumps(e),flush=True)
        del engine,snapshot,checkpoint,payload
    # Each worker owns disjoint seeds. Publish only complete all-100 coverage.
    receipts=sorted(out.glob('model_*.json'))
    if len(receipts)!=100:
        os.sync();print(json.dumps(dict(worker=args.worker,worker_complete=True,receipts=len(receipts))),flush=True);return
    entries=[json.loads(p.read_text()) for p in receipts]
    if sorted(e['local_id'] for e in entries)!=list(range(1,101)):raise ValueError('bank identity coverage differs')
    manifest=dict(schema=SCHEMA,config_sha256=sha256_file(header),entries=sorted(entries,key=lambda e:e['local_id']),
        split=dict(train=list(range(1,81)),audit=list(range(81,101))))
    temp=out/f'manifest.worker{args.worker}.tmp';temp.write_text(json.dumps(manifest,indent=2));os.replace(temp,out/'manifest.json')
    os.sync()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--rank',type=int,default=8)
    p.add_argument('--anchor-seed',type=int,default=20261110);p.add_argument('--calibration-images',type=int,default=1024)
    p.add_argument('--microbatch-images',type=int,default=128)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--worker',type=int,default=0)
    args=p.parse_args()
    if not 0<=args.worker<args.workers:raise ValueError('worker must be in range workers')
    build(args)
