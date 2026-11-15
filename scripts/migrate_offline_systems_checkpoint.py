"""Explicit, qualified metadata migration to the frozen offline systems source.

The input remains immutable. Actor, Adam, duals, schedule, RNG and data cursors
are byte-checked after serialization. Ordinary trainer resume checks stay strict.
"""
import argparse,hashlib,json,os,statistics,math
from pathlib import Path
import torch

TRAINER='scripts/train_multistate_chisel.py'
OLD_TRAINER='620f6ab20d6a3f06d10ef6e5cedd78580be573198ea0ccedb1c9d1ae511bd937'
NEW_TRAINER='b14e37a7b1c483351a54768f7d6cf6a3b8eee66526132fe0e8489610df3db2dd'
HELPER='rewriter/offline_systems.py'
HELPER_SHA='ec3967220f8667e7007262cb11da935eb6949e354cd66c6a717e9a2f08a6eb8a'
PLOT_HELPER='rewriter/progress_metrics.py'
PLOT_HELPER_SHA='61a2b69394f8587efcba2e406188d3bb332effee994a5a2f22f82b2563b3fd6a'
ARCHIVE_SHA='2e532f29a672fbe7485989d2908be79a4e58d83bfdd25a3c8632b0b2d4cfce50'
CACHE_SHA='89bf42daddbe395b854d6aed99aac32acb58d5169a33a19aef005f274b4a6a4c'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def numerical_digest(state):
    h=hashlib.sha256()
    def walk(x):
        if isinstance(x,torch.Tensor):
            h.update(repr((str(x.dtype),tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=lambda y:(type(y).__name__,repr(y))):walk(k);walk(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode())
            for v in x:walk(v)
        else:h.update(repr((type(x).__name__,x)).encode())
        h.update(b'\0')
    walk({k:v for k,v in state.items() if k!='metadata'})
    return h.hexdigest()

def qualify(directory):
    directory=Path(directory);launch=json.loads((directory/'launch.json').read_text());summary=json.loads((directory/'summary.json').read_text());finished=json.loads((directory/'finished.json').read_text())
    if finished['exit_code']!=0:raise ValueError('GPU qualification did not finish successfully')
    if launch.get('source_archive_sha256')!=ARCHIVE_SHA or launch.get('teacher_cache_manifest_sha256')!=CACHE_SHA or launch.get('rms_epsilon')!=2**-23:raise ValueError('qualification source or corrected teacher function differs')
    if summary.get('world_size')!=8 or summary.get('checkpoint_batch')!=200 or len(summary.get('state_ids',[]))!=200 or len(set(summary.get('state_ids',[])))!=200:raise ValueError('qualification must cover a full eight-rank200-checkpoint update')
    if summary.get('source_policy_sha256')!=launch.get('source_policy_sha256'):raise ValueError('qualification policy hashes differ')
    rows=summary.get('rows',[])
    if len(rows)!=6 or any(not r.get('bit_exact_to_baseline',False) for r in rows):raise ValueError('every warm/measured trial must pass numerical qualification')
    if sorted((r['variant'],r['repeat']) for r in rows)!=sorted((v,i) for v in ['baseline','lazy_untimed_allcache'] for i in range(3)):raise ValueError('qualification trial coverage differs')
    if any(r.get('warmup')!=(r['repeat']==0) or not math.isfinite(r['seconds']) or r['seconds']<=0 for r in rows):raise ValueError('malformed warm/measured trial rows')
    med={v:statistics.median(r['seconds'] for r in rows if r['variant']==v and not r['warmup']) for v in ['baseline','lazy_untimed_allcache']}
    if med!=summary['median_seconds']:raise ValueError('summary medians differ from measured rows')
    if not 0<med['lazy_untimed_allcache']<med['baseline']:raise ValueError('no measured warm full-update benefit')
    return dict(directory=str(directory),launch_sha256=sha(directory/'launch.json'),summary_sha256=sha(directory/'summary.json'),finished_sha256=sha(directory/'finished.json'),source_policy_sha256=summary['source_policy_sha256'],warm_median_seconds=med)

def migrate(source_checkpoint,destination,new_source,qualification):
    source_checkpoint,destination,new_source=map(Path,(source_checkpoint,destination,new_source))
    if destination.exists() or destination.resolve()==source_checkpoint.resolve():raise ValueError('new destination required; never overwrite source checkpoint')
    evidence=qualify(qualification)
    if source_checkpoint.name=='controller_latest.pt':raise ValueError('use an immutable epoch checkpoint or preserved copy')
    with source_checkpoint.open('rb') as stream:
        stamp=os.fstat(stream.fileno());fingerprint=lambda x:(x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
        digest=hashlib.sha256()
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
        stream.seek(0);state=torch.load(stream,map_location='cpu',weights_only=False)
        if fingerprint(stamp)!=fingerprint(os.fstat(stream.fileno())) or fingerprint(stamp)!=fingerprint(source_checkpoint.stat()):raise ValueError('input checkpoint changed during read')
        input_sha=digest.hexdigest()
    metadata=state['metadata'];sources=dict(metadata['implementation_sha256'])
    if state.get('completed',False):raise ValueError('completed trial cannot be resumed')
    if sources.get(TRAINER)!=OLD_TRAINER:raise ValueError('unapproved predecessor trainer')
    if metadata['gpt_precision'].get('rms_epsilon')!=2**-23 or metadata['gpt_precision'].get('tf32') is not False:raise ValueError('checkpoint is not the corrected strict FP32 fit')
    if metadata['state_bank_identity']['teacher_cache_manifest_sha256']!=CACHE_SHA or metadata['world_size']!=8:raise ValueError('checkpoint cache or rank count differs')
    for name,old in sources.items():
        expected=NEW_TRAINER if name==TRAINER else old
        if sha(new_source/name)!=expected:raise ValueError('unapproved source change: '+name)
    if sha(new_source/HELPER)!=HELPER_SHA:raise ValueError('systems helper differs from qualified source')
    if sha(new_source/PLOT_HELPER)!=PLOT_HELPER_SHA:raise ValueError('incremental plotting helper differs from qualified source')
    opts=dict(metadata['options'])
    if opts.get('offline_systems_cache') or opts.get('no_component_timing'):raise ValueError('checkpoint already migrated')
    before=numerical_digest(state);sources[TRAINER]=NEW_TRAINER;sources[HELPER]=HELPER_SHA
    opts.update(offline_systems_cache=True,no_component_timing=True)
    record=dict(kind='qualified_offline_systems_cache_v1',input_checkpoint=str(source_checkpoint),input_checkpoint_sha256=input_sha,old_trainer_sha256=OLD_TRAINER,new_trainer_sha256=NEW_TRAINER,source_archive_sha256=ARCHIVE_SHA,additional_source_sha256={PLOT_HELPER:PLOT_HELPER_SHA},qualification=evidence,numerical_state_sha256=before,numerical_state_changed=False,objective_or_update_equations_changed=False,changes=['verified immutable base/group/radius constants','lazy complete teacher-microbatch selection','disabled component timing','incremental plot parsing'],prior_source_migration=metadata.get('source_migration'))
    state['metadata']=dict(metadata,options=opts,implementation_sha256=sources,source_migration=record)
    temporary=destination.with_suffix(destination.suffix+'.tmp');destination.parent.mkdir(parents=True,exist_ok=True)
    with temporary.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    recovered=torch.load(temporary,map_location='cpu',weights_only=False)
    if numerical_digest(recovered)!=before:temporary.unlink();raise ValueError('serialized numerical state changed')
    temporary.replace(destination);record['output_checkpoint_sha256']=sha(destination)
    destination.with_suffix(destination.suffix+'.migration.json').write_text(json.dumps(record,indent=2)+'\n')
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['input','output','new-source','qualification']:p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args();print(json.dumps(migrate(a.input,a.output,a.new_source,a.qualification)))
