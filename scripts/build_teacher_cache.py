"""One-time distributed exact-head-input backfill for a fixed state prefix."""
import argparse,glob,json,os,sys,time,hashlib
from pathlib import Path
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline.model import GPT
from baseline.execution import enable_deterministic_execution,execution_environment
from rewriter.state_bank import StateBankReader
from rewriter.function_preservation import FunctionalPreservation,chunked_teacher_kl
from rewriter.teacher_cache import write_panel,TeacherCacheReader,head_inputs,head_logits,file_hash,bind_engine
from rewriter.gpt_precision import configure_precision,precision_metadata
from scripts.train_multistate_chisel import state_replay,select_sequences,stable_seed


def main():
    p=argparse.ArgumentParser();p.add_argument('--state-bank',required=True);p.add_argument('--train-data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--states',type=int,default=25)
    p.add_argument('--sequences',type=int,default=32);p.add_argument('--microbatch',type=int,default=16)
    p.add_argument('--train-sequences',type=int);p.add_argument('--audit-sequences',type=int)
    p.add_argument('--qualification-states',default='all',help='all or comma-separated state IDs')
    p.add_argument('--wandb-dir');p.add_argument('--resume',action='store_true')
    p.add_argument('--gpt-precision',choices=('baseline','fp32'),default='baseline')
    p.add_argument('--data-seed',type=int,default=400001);p.add_argument('--split-seed',type=int,default=500001)
    args=p.parse_args()
    counts={'train':args.sequences if args.train_sequences is None else args.train_sequences,'audit':args.sequences if args.audit_sequences is None else args.audit_sequences}
    qualify=set(range(1,args.states+1)) if args.qualification_states=='all' else set(map(int,args.qualification_states.split(',')))
    if any(n<1 or n%args.microbatch for n in counts.values()):p.error('positive full microbatch panel counts required')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    enable_deterministic_execution()
    configure_precision(args.gpt_precision)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)));device=torch.device('cuda');torch.set_num_threads(1)
    if world>1:dist.init_process_group('nccl')
    root=Path(args.output)
    if rank==0:root.mkdir(parents=True,exist_ok=args.resume)
    if world>1:dist.barrier()
    reader=StateBankReader(args.state_bank,require_complete=False,verify_once=True,mmap_model=True)
    if len(reader.entries)<args.states:raise ValueError('requested state prefix is not complete')
    provenance=reader.manifest['provenance'];files=sorted(glob.glob(args.train_data))
    model=GPT(**provenance['model_config']).to(device);model.requires_grad_(False)
    engine=FunctionalPreservation(model,kl_dtype=torch.float64,microbatch_sequences=args.microbatch,gpt_precision=args.gpt_precision)
    parameter_names=set(engine.parameters);buffer_names=set(engine.buffers)
    execution=dict(mode='eager_PR323',environment=execution_environment(),softcap='rational15',gpt_precision=args.gpt_precision,precision=precision_metadata(args.gpt_precision),
        feature_dtype='original_head_input_dtype',microbatch=args.microbatch,torch=str(torch.__version__),cuda=torch.version.cuda)
    run=None
    if rank==0 and args.wandb_dir:
        import wandb
        Path(args.wandb_dir).mkdir(parents=True,exist_ok=True)
        run=wandb.init(project='learnable-weight-decay',name='chisel-teacher-cache1000',mode='offline',dir=args.wandb_dir,config=vars(args))
    started=time.time();rows=[]
    cache_reader=TeacherCacheReader(root,require_complete=False,verify_once=True,mmap=True)
    resumed_panels=0;generated_panels=0
    for state_id in range(rank+1,args.states+1,world):
        existing={}
        if args.resume:
            for role in ('train','audit'):
                path=root/f'state_{state_id:04d}_{role}.pt';sidecar=path.with_suffix('.json')
                if path.exists() and sidecar.exists():
                    panel=cache_reader.read(state_id,role);row=json.loads(sidecar.read_text())
                    if (row['state_sha256']!=reader.entries[state_id-1]['sha256']
                        or len(panel.inputs)!=counts[role] or row['microbatch_sequences']!=args.microbatch
                        or row.get('gpt_precision','baseline')!=args.gpt_precision):
                        raise ValueError('existing panel disagrees with resume configuration')
                    policy=row['execution']['environment']
                    if policy['sdpa']!={'flash':True,'cudnn':False,'memory_efficient':False,'math':False} or not policy['deterministic_algorithms']:
                        raise ValueError('existing target execution policy differs')
                    if row['execution'].get('torch')!=str(torch.__version__) or row['execution'].get('cuda')!=torch.version.cuda:
                        raise ValueError('existing target precision runtime differs')
                    existing[role]=row;del panel
                elif path.exists() or sidecar.exists():
                    quarantine=root/'interrupted';quarantine.mkdir(exist_ok=True)
                    for orphan in (path,sidecar):
                        if orphan.exists():os.rename(orphan,quarantine/orphan.name)
        if len(existing)==2 and state_id not in qualify:
            for row in existing.values():row['reused_from_prior_attempt']=True;rows.append(row)
            resumed_panels+=2
            print(json.dumps(dict(event='reused',state=state_id,panels=2,elapsed=time.time()-started)),flush=True)
            continue
        load_started=time.perf_counter()
        record=reader.read(state_id)
        engine.load_teacher_state({n:record.model[n] for n in parameter_names},{n:record.model[n] for n in buffer_names})
        bind_engine(engine,reader.entries[state_id-1]['sha256'])
        torch.cuda.synchronize()
        load_seconds=time.perf_counter()-load_started
        pool=state_replay(files,state_id,provenance['training_config'],pool_sequences=4096,
            seed=args.data_seed,epoch=0,split_seed=args.split_seed)
        for role in ('train','audit'):
            if len(pool[role].inputs)<counts[role]:raise ValueError('panel requires enough unique past sequences')
            inputs,ids=select_sequences(pool[role],counts[role],seed=stable_seed('fixed_teacher_panel',args.data_seed,state_id,role))
            if role in existing:
                row=existing[role]
                if row['sequence_ids']!=list(ids):raise ValueError('resumed sequence sampling changed')
                row['reused_from_prior_attempt']=True;resumed_panels+=1
            else:
                row=write_panel(root,state_id,role,engine,inputs,ids,state_sha256=reader.entries[state_id-1]['sha256'],execution=execution)
                generated_panels+=1
            if state_id in qualify:
                # Qualify exact same production shape and checkpoint on every state/role.
                panel=TeacherCacheReader(root,require_complete=False).read(state_id,role)
                teacher=panel.teacher_batch(engine,0,args.microbatch)
                direct=engine.cache_teacher(inputs[:args.microbatch],data_role='past_training')[0]
                maximum=(teacher.log_probs-direct.log_probs).abs().max().item()
                if maximum!=0:raise AssertionError(f'factorized targets differ: {maximum}')
                # Compare the student-logit cotangent too; no extra GPT backward needed.
                logits=head_logits(engine,panel.features[:args.microbatch])
                left=(logits+.001).requires_grad_();right=left.detach().clone().requires_grad_()
                a=chunked_teacher_kl(left,teacher.log_probs);b=chunked_teacher_kl(right,direct.log_probs)
                ga,=torch.autograd.grad(a,left);gb,=torch.autograd.grad(b,right)
                if not torch.equal(ga,gb):raise AssertionError('target gradient differs')
                row['qualification_max_logprob_error']=maximum
                row['qualification_extra_teacher_forwards']=1
                row['qualification_extra_teacher_tokens']=inputs[:args.microbatch].numel()
                del teacher,direct,left,right,ga,gb,logits
            row['state_load_seconds']=load_seconds
            import resource
            row['worker_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            try:row['node_cgroup_memory_bytes']=int(Path('/sys/fs/cgroup/memory.current').read_text())
            except OSError:row['node_cgroup_memory_bytes']=None
            rows.append(row)
            progress=dict(state=state_id,role=role,bytes=row['bytes'],elapsed=time.time()-started,
                state_load_seconds=load_seconds,teacher_blocks_copy_seconds=row['generation_transformer_and_cpu_copy_seconds'],
                write_hash_seconds=row['write_hash_seconds'],node_cgroup_memory_bytes=row['node_cgroup_memory_bytes'])
            print(json.dumps(progress),flush=True)
            if run:run.log(dict(objective=state_id,**{k:v for k,v in progress.items() if k!='role'}))
        del record
        if (state_id-1)//world%16==15:
            import subprocess
            subprocess.run(['sync','-f',str(root)],check=True)
    gathered=[None]*world
    if world>1:dist.all_gather_object(gathered,rows)
    else:gathered=[rows]
    if rank==0:
        allrows=sorted([r for part in gathered for r in part],key=lambda r:(r['state_id'],r['role']))
        manifest=dict(format='exact_head_input_v1',complete=True,args=vars(args),source_state_manifest=reader.manifest,
            execution=execution,panels=allrows,elapsed_seconds=time.time()-started,
            resumed_from_interrupted_attempt=args.resume,
            prior_attempt_compute_note='Unique panel generation counts below exclude any work lost before persistence; prior attempt also completed6 qualification forwards' if args.resume else None,
            total_bytes=sum(r['bytes'] for r in allrows),source_hashes={name:file_hash(Path(__file__).resolve().parents[1]/name)
            for name in ['rewriter/teacher_cache.py','scripts/build_teacher_cache.py','baseline/model.py','rewriter/function_preservation.py','rewriter/gpt_precision.py']})
        (root/'manifest.json.tmp').write_text(json.dumps(manifest,indent=2)+'\n')
        os.replace(root/'manifest.json.tmp',root/'manifest.json')
        import subprocess
        subprocess.run(['sync','-f',str(root)],check=True)
        if run:
            run.log(dict(objective=args.states,total_bytes=manifest['total_bytes'],elapsed_seconds=manifest['elapsed_seconds']))
            run.finish()
        print(json.dumps(dict(event='complete',total_bytes=manifest['total_bytes'],elapsed=manifest['elapsed_seconds'])),flush=True)
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
