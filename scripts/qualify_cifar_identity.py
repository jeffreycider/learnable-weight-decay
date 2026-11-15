"""Read-only stored-teacher identity and compiled-forward check on terminal CNNs."""
from pathlib import Path
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.capture_cifar_chisel_bank import load_airbench
from rewriter.cifar_adapter import CifarPreservation
from rewriter.cifar_bank import CifarBankReader,normalized_images,sha256_file
from rewriter.compiled_cifar import CompiledCifarForward
from rewriter.rank_preservation import topk_preservation_flags


@torch.no_grad()
def compare(teacher,student):
    t=teacher.double();s=student.double();tl=t.log_softmax(-1);sl=s.log_softmax(-1)
    forward=(tl.exp()*(tl-sl)).sum(-1);reverse=(sl.exp()*(sl-tl)).sum(-1)
    flags=topk_preservation_flags(t,s,k=t.shape[-1])
    gap=s-t;centered=gap-gap.mean(-1,keepdim=True)
    return dict(bitexact=torch.equal(teacher,student),mean_kl=float(forward.mean()),max_kl=float(forward.max()),
        reverse_mean_kl=float(reverse.mean()),reverse_max_kl=float(reverse.max()),
        centered_logit_rms=float(centered.square().mean().sqrt()),max_logit_error=float(gap.abs().max()),
        **{name:int(value.sum()) for name,value in flags.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--models',type=int,nargs='+',default=[1,21,41,61])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.cuda.set_device(0)
    module=load_airbench(a.data,defer_test_readouts=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    import wandb
    run=wandb.init(project='chisel',mode='offline',dir=str(a.output),config={'scope':'zero-edit CNN identity, no actor/backward/optimizer','models':a.models})
    data=torch.load(a.data/'train.pt',map_location='cpu',weights_only=False)
    split=torch.load(a.data/'chisel_image_split.pt',map_location='cpu',weights_only=False)
    images=normalized_images(data['images'],'cuda:0')
    bank=CifarBankReader(a.bank);cfg=json.loads((a.bank/'config.json').read_text())
    if sha256_file(a.bank/'config.json')!=bank.manifest['config_sha256']:
        raise ValueError('CIFAR cache configuration identity differs')
    if sha256_file(a.data/'train.pt')!=cfg['train_sha256'] or sha256_file(a.data/'chisel_image_split.pt')!=cfg['image_split_sha256']:
        raise ValueError('CIFAR images or fixed split differ from cache')
    batch=int(cfg['inference']['teacher_microbatch_images'])
    if batch!=128:raise ValueError('this identity recipe requires pinned microbatch128')
    ids=split['fitting_indices'][:batch].long()
    # Include the original cache's last80 rows inside a complete128-row shape.
    tail=torch.arange(len(images)-batch,len(images),dtype=torch.long)
    rows=[];compiled=None
    for local_id in a.models:
        record=bank.read(local_id);source_sha=record.assert_verified()
        model=module.CifarNet().to('cuda:0').float().eval();model.load_state_dict(record.model,strict=True)
        engine=CifarPreservation(model,microbatch_images=batch)
        if compiled is None:compiled=CompiledCifarForward(engine)
        for panel_name,index in [('fixed_fitting',ids),('tail_identity_only',tail)]:
            x=images[index.to(images.device)].contiguous(memory_format=torch.channels_last)
            teacher=record.teacher_logits[index].to('cuda:0')
            for backend in ('eager_inference','eager_fitting','compiled_fitting'):
                replacements=engine.parameters if backend=='eager_inference' else {
                    n:v.detach().clone().requires_grad_(True) for n,v in engine.parameters.items()}
                torch.cuda.synchronize();started=time.perf_counter()
                with torch.no_grad() if backend=='eager_inference' else torch.enable_grad():
                    result=(compiled(engine,replacements,x) if backend=='compiled_fitting' else engine._forward(replacements,x))
                torch.cuda.synchronize();elapsed=time.perf_counter()-started
                row=dict(local_id=local_id,seed=record.seed,source_sha256=source_sha,panel=panel_name,
                    backend=backend,seconds=elapsed,images=len(index),image_strides=list(x.stride()),
                    compiler_counters={str(k):dict(v) for k,v in torch._dynamo.utils.counters.items()},
                    **compare(teacher,result))
                rows.append(row)
                with (a.output/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                run.log({'objective':row['mean_kl'],'seconds':elapsed,'strict_changes':row['strict_topk_changed']})
                print(json.dumps(row),flush=True);del result,replacements
        record.assert_verified();del model,engine,record
    eager=[r for r in rows if r['backend'].startswith('eager')]
    fast=[r for r in rows if r['backend']=='compiled_fitting']
    summary=dict(scope='Stored-cache zero-edit baseline, no actor or optimizer; tail panel read-only',
        bank_manifest_sha256=sha256_file(a.bank/'manifest.json'),cache_config=cfg,microbatch_images=batch,
        teacher_forwards=0,student_forwards=len(rows),student_images=sum(r['images'] for r in rows),
        backwards=0,optimizer_updates=0,eager_bitexact=all(r['bitexact'] for r in eager),
        eager_zero_strict_rank_changes=all(r['strict_topk_changed']==0 for r in eager),
        compiled_zero_strict_rank_changes=all(r['strict_topk_changed']==0 for r in fast),
        compiled_max_mean_kl=max(r['mean_kl'] for r in fast),compiled_max_token_kl=max(r['max_kl'] for r in fast),rows=rows)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));run.finish()
    if not summary['eager_bitexact'] or not summary['eager_zero_strict_rank_changes']:
        raise RuntimeError('baseline identity differs; do not attribute it to Chisel edits')


if __name__=='__main__':main()
