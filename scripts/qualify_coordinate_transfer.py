"""Paired frozen-block transfer readout; reuses the existing heldout routine.

prepare is CPU-only and requires a closed, pinned review checkpoint. run uses
one GPU, optionally one disjoint case shard, and never updates shared parameters.
A new output directory is mandatory; interrupted attempts are not overwritten.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.coordinate_transfer_control import frozen_block_pair
from rewriter.distributed_probe_store import _digest
from rewriter.private_probe_store import PrivateProbeStore, PrivateIdentity
from rewriter.multibank import file_sha, MultiStateBankReader, MultiTeacherCacheReader
from rewriter.mixed_domain import MixedBankCatalog
from rewriter.cifar_bank import CifarBankReader
from rewriter.tokenwise_reference import TokenwiseConfig
from scripts.qualify_coordinate_chisel import FixedContexts
from scripts.train_coordinate_chisel import DOMAINS, describe_states, heldout_readout, immutable_save, rng_state, restore_rng


def sources():
    root=Path(__file__).resolve().parents[1]
    files=[p for folder in ('rewriter','baseline') for p in (root/folder).rglob('*.py')]
    files += [root/'scripts'/name for name in ('qualify_coordinate_transfer.py','train_coordinate_chisel.py','qualify_coordinate_chisel.py','train_multistate_chisel.py','capture_cifar_chisel_bank.py')]
    return {str(p.relative_to(root)):file_sha(p) for p in sorted(files)}


def validate_protocol(plan):
    if plan.get('schema')!='coordinate_frozen_block_transfer_plan_v1':raise ValueError('explicit transfer protocol required')
    if plan['arms']!=['trained_blocks_common_trained_interfaces','initial_blocks_common_trained_interfaces']:
        raise ValueError('only the declared paired block ablation is supported')
    if not plan['shared_frozen'] or plan['shared_optimizer'] is not None:raise ValueError('shared updates forbidden')
    if plan['support']['steps']!=32 or plan['support']['gpt_sequences']!=16 or plan['support']['cifar_images']!=1024:
        raise ValueError('declared support budget changed')
    if plan['query']['readout_steps']!=[0,32] or plan['query']['gpt_sequences']!=64 or plan['query']['cifar_images']!=1024:
        raise ValueError('declared query budget changed')
    if plan['query']['used_for_updates_or_selection'] or not plan['query']['require_disjoint_support_ids']:
        raise ValueError('query feedback forbidden')
    if plan['case_selection']['counts']!={'gpt':4,'cifar10':4}:raise ValueError('declared case counts changed')
    if plan['private_optimizer_initialization']!='fresh_empty' or plan['dual_initialization']!='fresh_zero':
        raise ValueError('fresh private optimizer and dual required')
    if plan['initialization']['architecture'].get('frame_mode')!='tied_orthonormal':raise ValueError('tied private frames required')


def load_pinned(path,sha):
    if file_sha(path)!=sha:raise ValueError('pinned tensor file changed')
    result=torch.load(path,map_location='cpu',weights_only=False)
    if file_sha(path)!=sha:raise ValueError('pinned tensor file changed while loading')
    return result


def catalog_for(base):
    return MixedBankCatalog(base['mixed_index'],dict(gpt=lambda p:MultiStateBankReader(p,verify_once=True,mmap_model=True),cifar10=lambda p:CifarBankReader(p.parent)))


def quartiles(ids):
    ids=sorted(ids)
    if len(ids)<4:raise ValueError('four held-out checkpoints per domain required')
    return [ids[j*(len(ids)-1)//3] for j in range(4)]


def prepare(draft, parent_results, output):
    """Pin inputs before any transfer predictions; no model evaluations."""
    output=Path(output);parent=Path(parent_results);plan=json.loads(Path(draft).read_text())
    validate_protocol(plan)
    if plan['status']!='prepared_not_launched':raise ValueError('explicit draft required')
    metadata=json.loads((parent/'metadata.json').read_text());recipe=metadata['recipe']
    for k in ('private_optimizer','objectives','dual_lr','private_seed'):
        if plan[k]!=recipe[k]:raise ValueError('draft differs from parent '+k)
    if plan['initialization']['architecture']!=recipe['architecture']:raise ValueError('architecture changed')
    with PrivateProbeStore(parent/'joint_store/global',read_only=True) as store:
        _,ref=store._manifest_and_reference();saved=store.restore(manifest_ref=ref)
        update=plan['parent']['review_update']
        if saved['sampler']['next_global_batch']!=update:raise ValueError('parent is not the requested review update')
        if any(r['sampler'].get('stop_reason')!='review_boundary' or r['sampler']['next_global_batch']!=update for r in saved['sampler']['ranks']):
            raise ValueError('parent review checkpoint is not closed')
        expected=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if saved['sampler']['run_identity']!=expected:raise ValueError('parent metadata identity differs')
        base_path=Path(recipe['base_recipe'])
        if file_sha(base_path)!=recipe['base_recipe_sha256']:raise ValueError('base recipe changed')
        base=json.loads(base_path.read_text())
        if file_sha(base['mixed_index'])!=recipe['mixed_index_sha256']:raise ValueError('model/data catalog changed')
        catalog=catalog_for(base);cache=MultiTeacherCacheReader(catalog.readers['gpt'],verify_once=True,mmap=True)
        image_split=torch.load(Path(base['cifar_data'])/'chisel_image_split.pt',map_location='cpu',weights_only=False)
        cases=[]
        for domain in DOMAINS:
            reader=catalog.readers[domain];entries={int(e.get('completed_steps',e.get('local_id'))):e for e in reader.entries}
            for uid in quartiles(reader.manifest['split']['audit']):
                if domain=='gpt':
                    support=cache.expected[(uid,'train')]['sequence_ids'][:plan['support']['gpt_sequences']]
                    query=cache.expected[(uid,'audit')]['sequence_ids'][:plan['query']['gpt_sequences']]
                else:
                    support=image_split['fitting_indices'][:plan['support']['cifar_images']].tolist()
                    query=image_split['heldout_indices'][:plan['query']['cifar_images']].tolist()
                if set(support)&set(query):raise ValueError('support/query overlap')
                cases.append(dict(domain=domain,local_id=uid,state_sha256=entries[uid]['sha256'],support_ids=list(support),query_ids=list(query)))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(plan['initialization']['cpu_seed'])
            initial=CoordinateChisel(**recipe['architecture'])
        trained=copy.deepcopy(initial);trained.load_state_dict(saved['shared']['model'],strict=True)
        _,_,ablation=frozen_block_pair(trained,initial)
        if store._manifest_and_reference()[1]!=ref:raise ValueError('source advanced during preparation')
        output.mkdir(parents=True,exist_ok=False)
        immutable_save(output/'actor.pt',dict(model=trained.state_dict(),parent_manifest_ref=ref,update=update))
        immutable_save(output/'initial_actor.pt',dict(model=initial.state_dict(),architecture=recipe['architecture'],seed=plan['initialization']['cpu_seed']))
    plan.update(status='pinned_ready_not_launched',base_recipe=str(base_path),base_recipe_sha256=file_sha(base_path),mixed_index_sha256=recipe['mixed_index_sha256'],cases=cases,ablation=ablation,source_hashes=sources())
    plan['parent'].update(results=str(parent),metadata_sha256=file_sha(parent/'metadata.json'),global_manifest_ref=ref,actor_file=str(output/'actor.pt'),shared_payload_sha256=file_sha(output/'actor.pt'))
    plan['initialization'].update(path=str(output/'initial_actor.pt'),state_sha256=file_sha(output/'initial_actor.pt'))
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    return plan


class TraceContexts:
    """Trace existing support calls without changing loss, gradients or panels."""
    def __init__(self,inner,cases,write,arm):
        self.inner=inner;self.cases={(c['domain'],c['local_id']):c for c in cases};self.write=write;self.arm=arm;self.ticks={};self.initial={}
    def __getattr__(self,name):return getattr(self.inner,name)
    @property
    def cifar_images(self):return self.inner.cifar_images
    @cifar_images.setter
    def cifar_images(self,value):self.inner.cifar_images=value
    def load(self,domain,uid,**kwargs):
        role=kwargs.get('role','train');purpose=kwargs.get('purpose')
        if purpose!=('private_support' if role=='train' else 'private_query'):raise ValueError('undeclared transfer access')
        result=self.inner.load(domain,uid,**kwargs)
        key='support_ids' if role=='train' else 'query_ids'
        if list(result['ids'])!=self.cases[(domain,uid)][key]:raise ValueError('paired panel identity changed')
        return dict(result,transfer_role=role)
    def measure(self,context,actor,private,config,dual,*,learning,**kwargs):
        key=(context['domain'],context['local_id']);tick=self.ticks.get(key,0)
        if learning!=(context['transfer_role']=='train'):raise ValueError('query cannot supply a gradient')
        if key not in self.initial:self.initial[key]=_digest(private.state_dict())
        before=_digest(private.state_dict()) if not learning else None
        dual_before=_digest(dual.state_dict()) if not learning else None
        started=time.perf_counter();row=self.inner.measure(context,actor,private,config,dual,learning=learning,**kwargs)
        extra={}
        if learning:
            norm=math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in private.parameters() if p.grad is not None))
            if not math.isfinite(norm):raise FloatingPointError('nonfinite private gradient')
            extra['private_preclip_gradient_norm']=norm;self.ticks[key]=tick+1
        elif before!=_digest(private.state_dict()) or dual_before!=_digest(dual.state_dict()):raise RuntimeError('query changed private state/dual')
        self.write(dict(event='support_backward' if learning else 'query',arm=self.arm,private_update=tick,row=row,wall_seconds=time.perf_counter()-started,**extra))
        return row


def execute_pairs(plan,trained,initial,contexts,specs,device,output,*,shard=0,shards=1,write=lambda row:None):
    """Testable wrapper around heldout_readout; only one-rank readout workers."""
    if dist.get_world_size()!=1:raise ValueError('use independent single-rank case shards')
    if not 0<=shard<shards or shards not in (1,2):raise ValueError('only one or two deterministic case shards')
    cases=[c for d in DOMAINS for i,c in enumerate([c for c in plan['cases'] if c['domain']==d]) if i%shards==shard]
    allowed={d:{s.local_id:s for s in specs['audit'][d]} for d in DOMAINS}
    for c in cases:
        if c['local_id'] not in allowed[c['domain']] or allowed[c['domain']][c['local_id']].identity.state_sha256!=c['state_sha256']:raise ValueError('case is not the pinned held-out state')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    learned,control,ablation=frozen_block_pair(trained,initial);actors=(learned,control)
    recipe=dict(private_seed=plan['private_seed'],private_optimizer=plan['private_optimizer'],dual_lr=plan['dual_lr'],readout={'heldout':dict(enabled=True,ids={d:[c['local_id'] for c in cases if c['domain']==d] for d in DOMAINS},support_steps=plan['support']['steps'],query_gpt_sequences=plan['query']['gpt_sequences'])})
    config={d:TokenwiseConfig(**plan['objectives'][d]) for d in DOMAINS};start_rng=rng_state(device);all_rows=[];initial_digests=[]
    before=_digest(trained.state_dict());beg=time.perf_counter()
    for name,actor in zip(plan['arms'],actors):
        directory=output/name;directory.mkdir();restore_rng(start_rng,device)
        trace=TraceContexts(contexts,cases,write,name);captured=[]
        def log(event,**values):
            captured.append(values);write(dict(event=event,arm=name,**values))
        heldout_readout(recipe,trace,actor,specs,config,device,directory,plan['parent']['review_update'],log)
        if any(n!=plan['support']['steps'] for n in trace.ticks.values()) or len(trace.ticks)!=len(cases):raise RuntimeError('support budget mismatch')
        initial_digests.append(trace.initial)
        all_rows.append(dict(arm=name,rows=captured[-1]['rows'],checkpoints={p.name:file_sha(p) for p in directory.glob('*.pt')}))
    if initial_digests[0]!=initial_digests[1]:raise RuntimeError('paired private initialization changed')
    if _digest(trained.state_dict())!=before:raise RuntimeError('source shared actor changed')
    restore_rng(start_rng,device)
    paired=[]
    endpoint={arm['arm']:{(r['domain'],r['local_id']):r for r in arm['rows'] if r['support_steps']==plan['support']['steps']} for arm in all_rows}
    for case in cases:
        key=(case['domain'],case['local_id']);comparison=dict(domain=key[0],local_id=key[1],endpoints={})
        for arm in plan['arms']:
            row=endpoint[arm][key];cfg=plan['objectives'][key[0]]
            complete=all(k in row for k in ('strict_topk_changes','mean_kl','max_token_kl','mean_group_l2_percentage_reduction'))
            passed=(row['strict_topk_changes']==0 and row['mean_kl']<=cfg['mean_tolerance'] and row['max_token_kl']<=cfg['tail_tolerance'] and row['mean_group_l2_percentage_reduction']>0) if complete else None
            comparison['endpoints'][arm]=dict(primary_pass=passed,metrics=row)
        paired.append(comparison)
    summary=dict(paired_endpoints=paired,schema='coordinate_frozen_transfer_results_v1',arms=all_rows,cases=cases,ablation=ablation,private_initial_digests={str(k):v for k,v in initial_digests[0].items()},shared_unchanged=True,compute=dict(contexts.compute),elapsed_seconds=time.perf_counter()-beg,shard=shard,shards=shards,query_controls_updates=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary


def run(plan_path,output,shard,shards):
    started=time.perf_counter();plan=json.loads(Path(plan_path).read_text());validate_protocol(plan)
    if Path(output).exists() or Path(output).with_name(Path(output).name+'-record').exists():raise FileExistsError(output)
    if plan['status']!='pinned_ready_not_launched' or plan['source_hashes']!=sources():raise ValueError('unpinned or changed evaluator source')
    for path,sha in [(plan['parent']['actor_file'],plan['parent']['shared_payload_sha256']),(plan['initialization']['path'],plan['initialization']['state_sha256']),(plan['base_recipe'],plan['base_recipe_sha256'])]:
        if file_sha(path)!=sha:raise ValueError('pinned input changed')
    base=json.loads(Path(plan['base_recipe']).read_text())
    if file_sha(base['mixed_index'])!=plan['mixed_index_sha256']:raise ValueError('catalog changed')
    from baseline.execution import enable_deterministic_execution
    from rewriter.gpt_precision import configure_precision
    enable_deterministic_execution();configure_precision('fp32');torch.cuda.set_device(0);torch.set_num_threads(4);device=torch.device('cuda',0)
    initial=CoordinateChisel(**plan['initialization']['architecture']);initial.load_state_dict(load_pinned(plan['initialization']['path'],plan['initialization']['state_sha256'])['model'])
    trained=copy.deepcopy(initial);trained.load_state_dict(load_pinned(plan['parent']['actor_file'],plan['parent']['shared_payload_sha256'])['model']);trained=trained.to(device);initial=initial.to(device)
    contexts=FixedContexts(base,device,gpt_sequences=plan['support']['gpt_sequences'],cifar_images=plan['support']['cifar_images'])
    trained.requires_grad_(False);trained.zero_grad(set_to_none=True);layouts={}
    for domain in DOMAINS:
        case=next(c for c in plan['cases'] if c['domain']==domain)
        context=contexts.load(domain,case['local_id'],purpose='private_support',shared_actor=trained)
        private=trained.make_private(context['snapshot'],state_key=(domain,'layout'),seed=plan['private_seed'])
        layouts[domain]=PrivateIdentity.from_private(private,state_sha256='0'*64).layout_sha256
    del context,private
    specs=describe_states(contexts.catalog,layouts);out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    rendezvous=out.parent/('.'+out.name+'.gloo')
    dist.init_process_group('gloo',init_method='file://'+str(rendezvous),rank=0,world_size=1)
    import wandb
    # Separate envelope holds metadata before execute_pairs creates its own directory.
    envelope=out.with_name(out.name+'-record');envelope.mkdir(exist_ok=False)
    (envelope/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (envelope/'execution.json').write_text(json.dumps(dict(plan_sha256=file_sha(plan_path),startup_seconds=time.perf_counter()-started,torch_version=torch.__version__,cuda_version=torch.version.cuda,device=torch.cuda.get_device_name(),shard=shard,shards=shards),indent=2))
    root=Path(__file__).resolve().parents[1]
    for name in plan['source_hashes']:
        p=envelope/'source_record'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((root/name).read_bytes())
    wb=wandb.init(project='chisel',mode='offline',dir=str(envelope),name=out.name,config=dict(plan_sha256=file_sha(plan_path),shard=shard,shards=shards,scope=plan['publication_claim']))
    (envelope/'wandb-path.json').write_text(json.dumps(dict(path=str(Path(wb.dir).parent),id=wb.id)))
    with (envelope/'metrics.jsonl').open('x') as log:
        def write(row):
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if row['event']=='query' or row.get('private_update',1)%8==0:
                print(json.dumps({k:v for k,v in row.items() if k!='row' and k!='rows'},allow_nan=False),flush=True)
            wb.log({k:v for k,v in row.get('row',{}).items() if isinstance(v,(int,float))})
        try:
            execute_pairs(plan,trained,initial,contexts,specs,device,out,shard=shard,shards=shards,write=write)
        except BaseException as exc:
            (envelope/'failure.json').write_text(json.dumps(dict(error=repr(exc),wall_seconds=time.perf_counter()-started)))
            raise
        finally:wb.finish();dist.destroy_process_group()
    (envelope/'finished.json').write_text(json.dumps(dict(wall_seconds=time.perf_counter()-started,summary_sha256=file_sha(out/'summary.json'),files={str(p.relative_to(out)):file_sha(p) for p in out.rglob('*') if p.is_file()}),indent=2)+'\n')
    os.sync()


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--draft',required=True);a.add_argument('--parent-results',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('run');a.add_argument('--plan',required=True);a.add_argument('--output',required=True);a.add_argument('--shard',type=int,default=0);a.add_argument('--shards',type=int,default=1)
    args=p.parse_args()
    if args.command=='prepare':prepare(args.draft,args.parent_results,args.output)
    else:run(args.plan,args.output,args.shard,args.shards)

if __name__=='__main__':main()
