"""Plan/copy byte-identical teacher panels at an owner-controlled boundary.

No trainer mutation, tensor conversion, new cache generation or panel sampling.
The complete original manifest/sidecars remain unchanged. Unselected panels are
absolute symlinks to the original volume; selected panels become regular local
files only after source-stream AND destination SHA verification. Consumers must
wait for stage_complete.json. On tmpfs the owner must check actual cgroup memory
headroom in addition to filesystem capacity before --apply.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def panel_selection(metadata, bank):
    metadata=metadata.get('metadata',metadata)
    limit=metadata['state_bank_identity']['selected_states']
    train=[int(s) for s in bank['split']['train'] if s<=limit]
    heldout=[int(s) for s in metadata['heldout_state_ids']]
    withheld=[int(s) for s in metadata['heldout_sequence_state_ids']]
    if not set(heldout)<=set(bank['split']['audit']) or not set(withheld)<=set(train):
        raise ValueError('saved audit populations do not match the state-bank split')
    return sorted({(s,'train') for s in train}|{(s,'audit') for s in heldout+withheld})


def make_plan(source, selection):
    source=Path(source).resolve()
    manifest=json.loads((source/'manifest.json').read_text())
    if not manifest.get('complete'):raise ValueError('complete original cache required')
    rows={(int(r['state_id']),r['role']):r for r in manifest['panels']}
    if len(rows)!=len(manifest['panels']) or not set(selection)<=rows.keys():
        raise ValueError('duplicate/missing cache panel identity')
    selected=[rows[key] for key in selection]
    for row in rows.values():
        if row['file']!=f"state_{row['state_id']:04d}_{row['role']}.pt":
            raise ValueError('unexpected panel file path')
    return dict(source=str(source),manifest_sha256=digest(source/'manifest.json'),
                selected_panels=[dict(state_id=r['state_id'],role=r['role'],file=r['file'],
                                      bytes=r['bytes'],sha256=r['sha256']) for r in selected],
                selected_bytes=sum(r['bytes'] for r in selected),selected_count=len(selected),
                total_panels=len(rows),unselected_panels=len(rows)-len(selected))


def _atomic_bytes(path, data):
    temporary=path.with_name(path.name+'.stage-partial')
    with temporary.open('wb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def stage_exact(source, destination, plan, *, max_stage_bytes, reserve_bytes=16<<30):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if source==destination or source in destination.parents:
        raise ValueError('destination must be separate from the source cache')
    if plan!=make_plan(source,[(r['state_id'],r['role']) for r in plan['selected_panels']]):
        raise ValueError('source manifest or stage plan changed')
    if plan['selected_bytes']>max_stage_bytes:raise ValueError('declared stage byte limit exceeded')
    destination.mkdir(parents=True,exist_ok=True)
    plan_path=destination/'stage_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('destination has a different stage plan')
    elif any(destination.iterdir()):
        raise FileExistsError('destination must be empty or owned by this exact stage plan')
    else:_atomic_bytes(plan_path,(json.dumps(plan,indent=2)+'\n').encode())
    existing={r['file'] for r in plan['selected_panels'] if (destination/r['file']).is_file()
              and not (destination/r['file']).is_symlink() and digest(destination/r['file'])==r['sha256']}
    remaining=sum(r['bytes'] for r in plan['selected_panels'] if r['file'] not in existing)
    if shutil.disk_usage(destination).free<remaining+reserve_bytes:
        raise OSError('insufficient destination free space including reserve')
    complete=destination/'stage_complete.json'
    complete.unlink(missing_ok=True)
    manifest_bytes=(source/'manifest.json').read_bytes()
    manifest=json.loads(manifest_bytes)
    # Sidecars are validated against the unchanged manifest before publication.
    for row in manifest['panels']:
        path=source/row['file'];sidecar=path.with_suffix('.json')
        data=sidecar.read_bytes();info=json.loads(data)
        for key in ('sha256','state_sha256','input_sha256','feature_sha256','sequence_ids'):
            if info[key]!=row[key]:raise ValueError('sidecar differs from complete manifest')
        _atomic_bytes(destination/sidecar.name,data)
        target=destination/path.name
        if not target.exists() and not target.is_symlink():target.symlink_to(path)
        elif target.is_symlink() and target.resolve()!=path:
            raise ValueError('destination has a conflicting remote panel link')
    _atomic_bytes(destination/'manifest.json',manifest_bytes)
    started=time.perf_counter();copied=0
    with (destination/'stage_progress.jsonl').open('a') as log:
        for row in plan['selected_panels']:
            if row['file'] in existing:continue
            src=source/row['file'];dst=destination/row['file']
            temporary=dst.with_name(dst.name+'.stage-partial')
            h=hashlib.sha256();count=0;stat=src.stat()
            try:
                with src.open('rb') as inp,temporary.open('wb') as out:
                    for block in iter(lambda:inp.read(8<<20),b''):
                        out.write(block);h.update(block);count+=len(block)
                    out.flush();os.fsync(out.fileno())
                after=src.stat()
                if (stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                    raise ValueError('source panel changed during copy')
                if count!=row['bytes'] or h.hexdigest()!=row['sha256'] or digest(temporary)!=row['sha256']:
                    raise ValueError('panel size/SHA verification failed')
                os.replace(temporary,dst)
            finally:
                temporary.unlink(missing_ok=True)
            copied+=count
            log.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),file=row['file'],bytes=count,
                                      copied_bytes=copied,elapsed_seconds=time.perf_counter()-started))+'\n');log.flush()
    if digest(source/'manifest.json')!=plan['manifest_sha256']:
        raise ValueError('source manifest changed during staging')
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=plan['manifest_sha256'],
                selected_count=plan['selected_count'],selected_bytes=plan['selected_bytes'],copied_bytes=copied,
                elapsed_seconds=time.perf_counter()-started,remote_fallback_count=plan['unselected_panels'])
    _atomic_bytes(complete,(json.dumps(result,indent=2)+'\n').encode())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','destination','state-bank-manifest','fit-metadata'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--max-stage-bytes',type=int)
    p.add_argument('--reserve-bytes',type=int,default=16<<30)
    args=p.parse_args()
    metadata=json.loads(args.fit_metadata.read_text());bank=json.loads(args.state_bank_manifest.read_text())
    identity=metadata.get('metadata',metadata)['state_bank_identity']
    if digest(args.state_bank_manifest)!=identity['manifest_sha256']:
        raise ValueError('saved metadata and state bank manifest differ')
    plan=make_plan(args.source,panel_selection(metadata,bank))
    if plan['manifest_sha256']!=identity['teacher_cache_manifest_sha256']:
        raise ValueError('saved metadata and teacher manifest differ')
    print(json.dumps(plan,indent=2),flush=True)
    if args.apply:
        if args.max_stage_bytes is None:p.error('--apply requires explicit --max-stage-bytes')
        print(json.dumps(stage_exact(args.source,args.destination,plan,max_stage_bytes=args.max_stage_bytes,
                                    reserve_bytes=args.reserve_bytes),indent=2),flush=True)


if __name__=='__main__':main()
