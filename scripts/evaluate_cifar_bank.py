"""Reporting-only FP32 terminal accuracy on the official CIFAR-10 test set.

Every terminal model is retained. This readout cannot train or select Chisel or
alter the captured CIFAR optimization trajectories. Single-view, no TTA.
"""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.cifar_bank import CifarBankReader,normalized_images,sha256_file
from rewriter.cifar_adapter import CifarPreservation
from scripts.capture_cifar_chisel_bank import load_airbench

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);module=load_airbench(args.data,defer_test_readouts=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    test_path=Path(args.data)/'test.pt'
    if sha256_file(test_path)!='82af984b820678e6df7fc51cef219e13b85b30668a8be2f8ea9deff61177e464':raise ValueError('official prepared test data hash differs')
    test=torch.load(test_path,weights_only=False);images=normalized_images(test['images'],'cuda');labels=test['labels'].cuda()
    reader=CifarBankReader(args.bank);model=module.CifarNet().cuda().float().eval();rows=[];start=time.perf_counter()
    for entry in reader.entries:
        record=reader.read(entry['local_id']);record.assert_verified();model.load_state_dict(record.model,strict=True)
        engine=CifarPreservation(model);correct=torch.zeros((),device='cuda');loss=torch.zeros((),device='cuda',dtype=torch.float64)
        for lo in range(0,len(images),128):
            panel=images[lo:lo+128];n=len(panel)
            if n<128:panel=torch.cat((panel,images[:128-n])).contiguous(memory_format=torch.channels_last)
            logits=engine._forward(engine.parameters,panel)[:n];target=labels[lo:lo+n]
            correct+=(logits.argmax(-1)==target).sum();loss+=-logits.double().log_softmax(-1).gather(1,target[:,None]).sum()
        row=dict(seed=record.seed,accuracy=float(correct)/len(images),mean_ce=float(loss)/len(images))
        rows.append(row);print(json.dumps(row),flush=True)
    result=dict(scope=__doc__,rows=rows,mean_accuracy=sum(x['accuracy'] for x in rows)/len(rows),
        min_accuracy=min(x['accuracy'] for x in rows),max_accuracy=max(x['accuracy'] for x in rows),elapsed_seconds=time.perf_counter()-start,
        test_images_per_model=len(images),models=len(rows),test_forward_images_per_model=10112,
        chisel_updates=0,model_selection=False,bank_manifest_sha256=sha256_file(Path(args.bank)/'manifest.json'),
        test_sha256=sha256_file(test_path),script_sha256=sha256_file(__file__))
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    import wandb
    run=wandb.init(project='chisel',name='cifar100-terminal-official-test',mode='offline',dir=str(out),config={k:v for k,v in result.items() if k not in ('rows','scope')})
    for row in rows:run.log(row)
    run.finish();os.sync()
if __name__=='__main__':main()
