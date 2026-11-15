"""Benchmark lossless observation transport only; no GPT or Chisel evaluations."""
import argparse,dataclasses,hashlib,json,statistics,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.prefix_cache import unpack_record


def transfer_packed_observation(record, device):
    """Copy a packed CPU observation in one transfer per dtype.

    The ordinary reader transfers hundreds of small tensors individually. This
    optional path coalesces only their transport; unpack_record still restores
    each original logical shape and stride. No dense GPT weights enter here.
    The returned tensors do not alias the CPU record, even for a CPU target.
    """
    tensors = []
    def collect(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != 'cpu':
                raise ValueError('packed observation transfer requires CPU sources')
            tensors.append(value)
        elif isinstance(value, dict):
            for child in value.values(): collect(child)
        elif isinstance(value, (tuple, list)):
            for child in value: collect(child)
    collect(record)
    by_dtype = {}
    for tensor in tensors:
        by_dtype.setdefault(tensor.dtype, []).append(tensor)
    moved = {}
    for dtype, group in by_dtype.items():
        flat = torch.cat([tensor.detach().reshape(-1) for tensor in group])
        flat = flat.to(device=device, copy=True)
        offset = 0
        for tensor in group:
            moved.setdefault(id(tensor), flat[offset:offset+tensor.numel()].reshape(tensor.shape))
            offset += tensor.numel()
    def replace(value):
        if isinstance(value, torch.Tensor): return moved[id(value)]
        if isinstance(value, dict): return {key: replace(child) for key, child in value.items()}
        if isinstance(value, tuple): return tuple(replace(child) for child in value)
        if isinstance(value, list): return [replace(child) for child in value]
        return value
    return replace(record)



def leaves(value):
    if isinstance(value,torch.Tensor):return [value]
    if dataclasses.is_dataclass(value):return [t for f in dataclasses.fields(value) for t in leaves(getattr(value,f.name))]
    if isinstance(value,dict):return [t for v in value.values() for t in leaves(v)]
    if isinstance(value,(tuple,list)):return [t for v in value for t in leaves(v)]
    return []


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=20);a=p.parse_args()
    if a.repeats<3:raise ValueError('at least three alternating repeats required')
    import wandb
    a.output.parent.mkdir(parents=True,exist_ok=True)
    run=wandb.init(project='learnable-weight-decay',name='observation-transfer-v1',mode='offline',dir=str(a.output.parent),config={'state':str(a.state),'repeats':a.repeats,'gpt_evaluations':0})
    torch.set_num_threads(4)
    record=torch.load(a.state,map_location='cpu',weights_only=False,mmap=True)['observation']
    original=unpack_record(record,device='cuda');coalesced=unpack_record(transfer_packed_observation(record,'cuda'),device='cuda')
    for old,new in zip(leaves(original),leaves(coalesced),strict=True):
        if old.shape!=new.shape or old.stride()!=new.stride() or old.dtype!=new.dtype:raise AssertionError('layout mismatch')
        if not torch.equal(old.contiguous().reshape(-1).view(torch.uint8),new.contiguous().reshape(-1).view(torch.uint8)):raise AssertionError('observation bits changed')
    del original,coalesced
    timings={False:[],True:[]}
    for iteration in range(a.repeats+3):
        for method in ([False,True] if iteration%2==0 else [True,False]):
            torch.cuda.synchronize();start=time.perf_counter()
            result=unpack_record(transfer_packed_observation(record,'cuda') if method else record,device='cuda')
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            if iteration>=3:
                timings[method].append(seconds)
                run.log({'method':'coalesced' if method else 'original','objective':seconds*1000,
                         ('coalesced_ms' if method else 'original_ms'):seconds*1000})
            del result
    source=Path(__file__).resolve()
    summary=dict(scope='packed CPU observations to exact-layout CUDA snapshot; GPT weights, disk reads, and norm references excluded',
        state=str(a.state),observation_step=record['step'],repeats=a.repeats,exact_bits_and_strides=True,
        original_ms=1000*statistics.median(timings[False]),coalesced_ms=1000*statistics.median(timings[True]),
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        main_training_modified=False,raw_seconds={str(k):v for k,v in timings.items()})
    summary['wandb_directory']=run.dir
    a.output.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary));run.finish()
if __name__=='__main__':main()
