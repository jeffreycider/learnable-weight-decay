"""Seed-pure, training-role replay spans with the ordinary shard-tail schedule."""
import hashlib
import json
import re
import struct
from pathlib import Path
import numpy as np
import torch
from scripts.train_multistate_chisel import heldout_sequence, stable_seed


class OnlineReplaySampler:
    version = 'completed_prefix_rejection_v1'

    def __init__(self, files, *, batch_tokens, seq_len, seed, split_seed, manifest_sha256, manifest_path):
        self.files = tuple(Path(p) for p in files)
        if not self.files or list(self.files) != sorted(set(self.files)):
            raise ValueError('replay requires an explicit sorted unique shard list')
        if any(re.search(r'(?:^|[_\-.])(val(?:idation)?|test|eval)(?:[_\-.]|$)', p.name, re.I) for p in self.files):
            raise ValueError('validation/test/evaluation replay is prohibited')
        if batch_tokens < 1 or seq_len < 1 or batch_tokens % seq_len:
            raise ValueError('whole-sequence global batches required')
        self.batch_tokens,self.seq_len,self.seed,self.split_seed = batch_tokens,seq_len,seed,split_seed
        manifest_bytes=Path(manifest_path).read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest()!=manifest_sha256:
            raise ValueError('replay trainer manifest SHA256 mismatch')
        manifest=json.loads(manifest_bytes)
        train=[row for row in manifest if row['kind']=='train']
        if [Path(row['path']).name for row in train]!=[p.name for p in self.files]:
            raise ValueError('replay shard order differs from pinned trainer manifest')
        self.sizes=[]
        self.manifest_sha256=manifest_sha256
        for p,row in zip(self.files,train):
            with p.open('rb') as f: header=f.read(1024)
            if len(header)!=1024:raise ValueError('truncated replay shard header')
            magic,version,tokens=struct.unpack('<iii',header[:12])
            if (magic,version)!=(20240520,1) or tokens<0 or p.stat().st_size!=1024+2*tokens:
                raise ValueError('invalid replay shard header/size')
            if row.get('tokens')!=tokens or row.get('bytes')!=p.stat().st_size or not row.get('sha256'):
                raise ValueError('replay trainer manifest requires full shard content identity')
            digest=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(8<<20),b''):digest.update(chunk)
            if digest.hexdigest()!=row['sha256']:raise ValueError('replay shard content SHA256 mismatch')
            self.sizes.append(tokens)
        self.schedule=[];self._index=0;self._pos=0

    def sample(self, action_step, count):
        """Pool is steps [0, action_step); excludes the just-completed batch."""
        if type(action_step) is not int or action_step<1 or type(count) is not int or count<1:
            raise ValueError('positive replay clock and count required')
        per_step=self.batch_tokens//self.seq_len;total=action_step*per_step
        if count>total:raise ValueError('insufficient completed past sequences')
        while len(self.schedule)<action_step:
            i,pos=self._index,self._pos
            if i>=len(self.files):raise ValueError('replay prefix exceeds training shards')
            if pos+self.batch_tokens+1>=self.sizes[i]:i,pos=i+1,0
            if i>=len(self.files) or pos+self.batch_tokens+1>self.sizes[i]:raise ValueError('replay shard too small')
            self.schedule.append((i,pos));self._index,self._pos=i,pos+self.batch_tokens
        generator=torch.Generator().manual_seed(stable_seed(self.version,self.seed,action_step))
        selected=[];seen=set()
        # Rejection gives uniform eligible IDs without replacement, without a
        # million-entry randperm on every online action. Fail boundedly on tiny pools.
        for _ in range(max(1000,count*100)):
            sid=int(torch.randint(total,(),generator=generator))
            if sid in seen:continue
            seen.add(sid)
            if heldout_sequence(sid,self.split_seed):continue
            selected.append(sid)
            if len(selected)==count:break
        if len(selected)!=count:raise ValueError('insufficient eligible replay IDs in bounded draw')
        rows=[];inputs=[];maps={}
        for sid in selected:
            step,sequence=divmod(sid,per_step);index,pos=self.schedule[step];offset=pos+sequence*self.seq_len
            if index not in maps:maps[index]=np.memmap(self.files[index],mode='r',dtype='<u2',offset=1024,shape=(self.sizes[index],))
            span=np.array(maps[index][offset:offset+self.seq_len],dtype=np.int32,copy=True)
            if len(span)!=self.seq_len:raise ValueError('incomplete replay span')
            inputs.append(torch.from_numpy(span));rows.append(dict(id=sid,step=step,sequence=sequence,shard_index=index,offset=offset))
        value=torch.stack(inputs)
        return dict(inputs=value,ids=selected,provenance=rows,
                    input_sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                    id_sha256=hashlib.sha256(json.dumps(selected).encode()).hexdigest(),
                    action_step=action_step,sampler_version=self.version,manifest_sha256=self.manifest_sha256)
