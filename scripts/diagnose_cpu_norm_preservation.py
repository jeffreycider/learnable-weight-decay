"""Bounded two-state CPU main-loss diagnostic, not full-bank or generalization.

Uses actual 100*mean group-relative norm change plus tokenwise
primal-dual/tail/ranking loss. Optional ranking ascent constrains zero mean
relative-gap margin loss; the default retains the fixed ranking penalty.
No known-edit target or edit-size penalty enters
this diagnostic. FP32 stored edits use PyTorch's cast derivative, which is not
the derivative of a discrete rounding map. GPT/cotangents are FP32 while KL is
FP64; tiny near-identity KL and relative rank gaps remain numerically sensitive.
Fresh teachers are CPU forwards, never cached GPU features. Supplied seed
initialization and ancestry are recorded separately for each case.
"""
import argparse
from collections import OrderedDict
import copy
import dataclasses
import json
import hashlib
import math
import resource
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.model import GPT
from rewriter.function_preservation import FunctionalPreservation,TeacherBatch
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.prefix_cache import unpack_record
from rewriter.edit_scope import editable_names,editable_weight_norm_sq
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from rewriter.update_accumulation import CompensatedAdam
from rewriter.ranking_dual import RankingDual
from scripts.qualify_coordinate_chisel import FixedContexts,proposal_builder_for_family
from scripts.fit_cpu_edit_teacher import sha,save,exact_state_equal
from scripts.prepare_cpu_teacher_state import validate_train_panel


def tensor_sha(t):
    t=t.detach().cpu().contiguous()
    h=hashlib.sha256(str((tuple(t.shape),str(t.dtype))).encode());h.update(t.numpy().tobytes());return h.hexdigest()


def load_bound_state(engine,state,state_sha):
    """Only this verified-load boundary grants a cache state binding."""
    parameters={n:state[n] for n in engine.parameters};buffers={n:state[n] for n in engine.buffers}
    engine.load_teacher_state(parameters,buffers)
    engine._local_cpu_binding=(state_sha,engine.teacher_id)



# Successful hashes only; metadata strings/integers, never probability tensors.
_TEACHER_CHECKSUM_CACHE=OrderedDict()
_TEACHER_CHECKSUM_CACHE_LIMIT=512


def _teacher_stat(path):
    stat=path.stat()
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)



def _remember_teacher_checksum(key):
    _TEACHER_CHECKSUM_CACHE[key]=None
    _TEACHER_CHECKSUM_CACHE.move_to_end(key)
    while len(_TEACHER_CHECKSUM_CACHE)>_TEACHER_CHECKSUM_CACHE_LIMIT:_TEACHER_CHECKSUM_CACHE.popitem(last=False)


def _hash_created_teacher_file(path):
    """Hash completed producer bytes once; memoize only a stable successful read.

The caller must finish/close the payload writer before calling. No supplied
hash is trusted. Entries are the same bounded metadata-only checks used by readers.
"""
    canonical=Path(path).resolve(strict=True)
    for previous in tuple(_TEACHER_CHECKSUM_CACHE):
        if previous[0]==str(canonical):del _TEACHER_CHECKSUM_CACHE[previous]
    before=_teacher_stat(canonical);actual=sha(canonical)
    if _teacher_stat(canonical)!=before:raise ValueError('teacher probability file changed during checksum')
    _remember_teacher_checksum((str(canonical),actual,before))
    return actual


def _verify_teacher_file(path,expected,counts):
    canonical=Path(path).resolve(strict=True);before=_teacher_stat(canonical)
    key=(str(canonical),expected,before)
    # Changed metadata/digest invalidates old successes for this canonical path.
    for previous in tuple(_TEACHER_CHECKSUM_CACHE):
        if previous[0]==str(canonical) and previous!=key:del _TEACHER_CHECKSUM_CACHE[previous]
    if key in _TEACHER_CHECKSUM_CACHE:
        _TEACHER_CHECKSUM_CACHE.move_to_end(key);counts['hits']+=1
        return canonical,before
    start=time.perf_counter();actual=sha(canonical)
    counts['hash_seconds']+=time.perf_counter()-start
    counts['files_hashed']+=1;counts['bytes_hashed']+=before[2]
    if _teacher_stat(canonical)!=before:raise ValueError('teacher probability file changed during checksum')
    if actual!=expected:raise ValueError('teacher probability file changed')
    _remember_teacher_checksum(key)
    return canonical,before


class DiskTeacherCache:
    """FP64 mmap files with bounded process-local successful checksum memoization.

Trusted local immutable-file contract: canonical path, expected digest and
(dev,inode,size,mtime_ns,ctime_ns) must all match to reuse a hash. Manifest and
state/input/runtime checks always run. This is not protection from adversarial
metadata forgery or writes after an mmap is returned; files must remain immutable.
No probability tensors are retained in the checksum memo.
"""
    @staticmethod
    def create(engine,inputs,ids,state_sha,path):
        if getattr(engine,'_local_cpu_binding',None)!=(state_sha,engine.teacher_id):raise ValueError('CPU teacher must be bound to verified state load')
        if engine.microbatch_sequences!=1 or engine.kl_dtype!=torch.float64 or engine.gpt_precision!='fp32':raise ValueError('explicit CPU FP32/FP64 single-sequence teacher required')
        if engine.device.type!='cpu':raise ValueError('CPU teacher only')
        path=Path(path);path.mkdir(parents=True,exist_ok=False);files=[]
        for i in range(len(inputs)):
            batch=engine.cache_teacher(inputs[i:i+1],data_role='past_training',sequence_ids=ids[i:i+1])[0]
            file=path/f'{i:04d}.pt'
            torch.save(dict(inputs=batch.inputs,log_probs=batch.log_probs,sequence_ids=batch.sequence_ids),file)
            files.append(dict(file=file.name,sha256=sha(file)));del batch
        meta=dict(state_sha256=state_sha,input_sha256=tensor_sha(inputs),sequence_ids=ids,files=files,
            precision='CPU FP32 GPT with FP64 log_softmax',torch_version=torch.__version__,teacher_compute=engine.compute.as_dict())
        (path/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');return meta

    def __init__(self,path,engine,inputs,ids,state_sha):
        self.path=Path(path);self.meta=json.loads((self.path/'manifest.json').read_text());self.engine=engine;self.load_id=engine.teacher_id
        if self.meta.get('representation') is not None:raise ValueError('teacher representation mismatch for fp64_log_probs')
        if getattr(engine,'_local_cpu_binding',None)!=(state_sha,self.load_id):raise ValueError('teacher load binding mismatch')
        if self.meta['state_sha256']!=state_sha or self.meta['input_sha256']!=tensor_sha(inputs) or self.meta['sequence_ids']!=ids or self.meta['torch_version']!=torch.__version__:raise ValueError('cached teacher state/input/runtime identity mismatch')
        if engine.device.type!='cpu' or engine.microbatch_sequences!=1 or engine.kl_dtype!=torch.float64 or engine.gpt_precision!='fp32':raise ValueError('cache runtime requires CPU FP32/FP64 single-sequence engine')
        if self.meta['precision']!='CPU FP32 GPT with FP64 log_softmax' or len(self.meta['files'])!=len(inputs):raise ValueError('cache precision/count mismatch')
        self.inputs=inputs;self.ids=ids
        self.checksum_stats=dict(hits=0,files_hashed=0,bytes_hashed=0,hash_seconds=0.,verification_seconds=0.)
        start=time.perf_counter()
        self.verified_files=[_verify_teacher_file(self.path/row['file'],row['sha256'],self.checksum_stats) for row in self.meta['files']]
        self.checksum_stats['verification_seconds']=time.perf_counter()-start

    def __call__(self,start,stop):
        if self.engine.teacher_id!=self.load_id or getattr(self.engine,'_local_cpu_binding',None)!=(self.meta['state_sha256'],self.load_id):raise ValueError('stale provider after teacher reload')
        if stop!=start+1 or not 0<=start<len(self.inputs):raise ValueError('single-sequence cache access required')
        path,verified=self.verified_files[start]
        if _teacher_stat(path)!=verified:raise ValueError('teacher probability file changed after verification')
        data=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if _teacher_stat(path)!=verified:raise ValueError('teacher probability file changed during mmap load')
        if data['log_probs'].dtype!=torch.float64 or not torch.equal(data['inputs'],self.inputs[start:stop]) or list(data['sequence_ids'])!=self.ids[start:stop]:raise ValueError('cached batch identity mismatch')
        return TeacherBatch(data['inputs'],data['log_probs'],self.load_id,'past_training',tuple(data['sequence_ids']))



def teacher_cache_class(format_name):
    if format_name=='fp64_log_probs':return DiskTeacherCache
    if format_name=='fp32_logits':
        from scripts.experimental_teacher_logits_cache import RawLogitTeacherCache
        return RawLogitTeacherCache
    raise ValueError('unknown teacher cache format')


def read_pool_indices(path,sequences):
    """Explicit ordered TRAIN-panel positions; no inferred evaluation split."""
    if type(sequences) is not int or not 1<=sequences<=16:raise ValueError('bounded positive sequences required')
    if path is None:return None,None
    payload=Path(path).read_bytes();plan=json.loads(payload)
    if not isinstance(plan,dict) or len(plan)!=2:raise ValueError('pool plan requires exactly two state IDs')
    for key,indices in plan.items():
        if not isinstance(key,str) or not key.isdecimal() or str(int(key))!=key:raise ValueError('pool state IDs must be canonical integer strings')
        if not isinstance(indices,list) or not sequences<=len(indices)<=256 or len(indices)%sequences:raise ValueError('pool lengths must be bounded multiples of sequences')
        if any(type(i) is not int or i<0 for i in indices) or len(set(indices))!=len(indices):raise ValueError('pool indices must be unique nonnegative integers')
    if len({len(indices) for indices in plan.values()})!=1:raise ValueError('pool lengths must match across states')
    schedule=dict(mode='cyclic_contiguous_absolute_update_v1',indices_by_state=plan,
        plan_sha256=hashlib.sha256(payload).hexdigest(),sequences_per_update=sequences,
        cycle_updates=len(next(iter(plan.values())))//sequences,final_readout='entire_training_pool')
    return plan,schedule


def panel_positions(pool_length,sequences,completed_updates,*,final=False):
    if any(type(v) is not int for v in (pool_length,sequences,completed_updates)) or completed_updates<0 or sequences<1 or pool_length<sequences or pool_length%sequences:raise ValueError('invalid cyclic panel schedule')
    if final:return tuple(range(pool_length))
    start=(completed_updates*sequences)%pool_length
    return tuple(range(start,start+sequences))


class SubsetTeacherProvider:
    """Map one selected microbatch to the exact full-pool cache entry."""
    def __init__(self,cache,positions):
        self.cache=cache;self.positions=tuple(positions)
        if not self.positions or any(type(i) is not int or not 0<=i<len(cache.inputs) for i in self.positions) or len(set(self.positions))!=len(self.positions):raise ValueError('invalid teacher subset positions')
        self.inputs=cache.inputs[list(self.positions)].clone();self.ids=[cache.ids[i] for i in self.positions]
        self.checksum_stats=cache.checksum_stats
    def __call__(self,start,stop):
        if stop!=start+1 or not 0<=start<len(self.positions):raise ValueError('single-sequence subset access required')
        position=self.positions[start];batch=self.cache(position,position+1)
        if not torch.equal(batch.inputs,self.inputs[start:stop]) or list(batch.sequence_ids)!=self.ids[start:stop]:raise ValueError('teacher subset alignment changed')
        return batch


class LocalCPUContexts(FixedContexts):
    """Reuse measure verbatim; this local fixture has no bank/audit/catalog loader."""
    def __init__(self,panel_backend='reference'):
        if panel_backend not in ('reference','direct_streamed'):raise ValueError('unknown panel backend')
        self.panel_backend=panel_backend
        self.action_family='adaptive_ucv_v1';self.proposal_builder=proposal_builder_for_family(self.action_family)
        self.device=torch.device('cpu');self.compute={}


def objective_config():
    return TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-8,rho=.01,
        ranking_weight=100.,ranking_k=100,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap',loss_backend='reference',dual_update_mode='clipped_relative')


def motion(before,module):
    squared=0.;maxabs=0.
    for n,p in module.named_parameters():
        delta=p.detach().double()-before[n].double();squared+=float(delta.square().sum());maxabs=max(maxabs,float(delta.abs().max()))
    return dict(l2=math.sqrt(squared),max_abs=maxabs)


def snapshot_parameters(module):return {n:p.detach().clone() for n,p in module.named_parameters()}


def pin_sources(output):
    paths=['scripts/experimental_teacher_logits_cache.py','scripts/diagnose_cpu_norm_preservation.py','scripts/qualify_coordinate_chisel.py',
        'rewriter/function_preservation.py','rewriter/tokenwise_reference.py','rewriter/token_preservation.py','rewriter/gradient_components.py',
        'rewriter/ranking_loss.py','rewriter/rank_preservation.py','rewriter/adaptive_preservation.py',
        'rewriter/adaptive_action.py','rewriter/adaptive_coordinate_chisel.py','rewriter/grouped_coordinate_chisel.py',
        'rewriter/coordinate_chisel.py','rewriter/grouped_norm.py','rewriter/gpt_precision.py',
        'rewriter/adaptive_sensing.py','rewriter/edit_scope.py','baseline/model.py',
        'scripts/fit_cpu_edit_teacher.py','scripts/prepare_cpu_teacher_state.py','rewriter/primal_dual.py','rewriter/prefix_cache.py']
    paths=set(paths)
    for module in tuple(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file:
            try:relative=Path(file).resolve().relative_to(Path.cwd())
            except ValueError:continue
            if relative.suffix=='.py' and relative.parts[0] in ('baseline','rewriter','scripts'):paths.add(str(relative))
    digests={}
    for name in sorted(paths):
        p=Path(name);dest=output/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());digests[name]=sha(p)
    return digests


def validate_private_epsilon(value):
    value=float(value)
    if not math.isfinite(value) or value<=0:raise ValueError('private epsilon must be positive and finite')
    return value


def seed_provenance(metadata):
    """Retain supplied seed facts; missing initialization is explicitly unknown."""
    fields=('diagnostic_control','initialization','scope','shared_actor_origin','private_origin',
            'target_provenance','parent_checkpoint','parent_sha256','grouped_fit_sha256',
            'head_gradient_updates','source_sha256','loss','trainable')
    return {key:copy.deepcopy(metadata.get(key)) for key in fields}


def optimizer_configuration(optimizer):
    return [{k:copy.deepcopy(v) for k,v in group.items() if k!='params'} for group in optimizer.param_groups]


def make_diagnostic_adam(parameters, *, update_accumulation='none', **kwargs):
    if update_accumulation not in ('none', 'fp32_remainder'):
        raise ValueError('Unknown update accumulation mode')
    optimizer = torch.optim.Adam if update_accumulation == 'none' else CompensatedAdam
    return optimizer(parameters, **kwargs)


def make_diagnostic_dual(state_ids, config, *, rank_scale=None, rank_dual_lr=None):
    if (rank_scale is None) != (rank_dual_lr is None):
        raise ValueError('Ranking scale and ranking dual LR must be supplied together')
    if rank_scale is None:
        return TokenwiseDual(state_ids, config=config, dual_lr=.01)
    return RankingDual(state_ids, config=config, dual_lr=.01,
                       rank_scale=rank_scale, rank_dual_lr=rank_dual_lr)


def resume_contract(metadata):
    """Identity excludes output location, total horizon and reporting-only options."""
    settings=('sequences','policy','shared_lr','body_lr','private_lr','private_eps','clip','shared_clip_policy','seed','gradient_profile','panel_backend','teacher_cache_format','update_accumulation')
    return dict(settings={k:(validate_private_epsilon(metadata['args'].get(k,1e-12)) if k=='private_eps' else metadata['args'].get(k,'joint') if k=='shared_clip_policy' else metadata['args'].get(k,'reference') if k=='panel_backend' else metadata['args'].get(k,'fp64_log_probs') if k=='teacher_cache_format' else metadata['args'].get(k,'none') if k=='update_accumulation' else metadata['args'].get(k)) for k in settings},
        pool_schedule=metadata.get('pool_schedule'),objective=metadata['objective'],dual_lr=metadata['dual_lr'],
        ranking_constraint=metadata.get('ranking_constraint'),
        cases=[{k:c[k] for k in ('state_id','sha256','sequence_ids','input_sha256','private_identity')} for c in metadata['cases']],
        state_weights=metadata['state_weights'],private_gradient_normalization=metadata['private_gradient_normalization'])


def training_state(actor,cases,optimizer,private_opts,dual,metadata,step):
    return dict(metadata=metadata,completed_updates=step,actor=actor.state_dict(),
        private={c['uid']:c['private'].state_dict() for c in cases},shared_optimizer=optimizer.state_dict(),
        private_optimizers={i:o.state_dict() for i,o in private_opts.items()},dual=dual.state_dict(),rng=torch.get_rng_state())


def restore_training_state(state,actor,cases,optimizer,private_opts,dual,metadata,total_updates):
    if resume_contract(state['metadata'])!=resume_contract(metadata):raise ValueError('resume case/panel/config identity mismatch')
    if isinstance(dual,RankingDual)!=('ranking_constraint' in state['dual']):raise ValueError('resume ranking dual type differs')
    step=state['completed_updates']
    if type(step) is not int or not 0<=step<=total_updates or state['dual']['updates']!=step:raise ValueError('invalid resume update clock')
    ids=set(private_opts)
    if set(state['private'])!=ids or set(state['private_optimizers'])!=ids:raise ValueError('resume private IDs differ')
    # Adam.load_state_dict silently replaces hyperparameters; compare before loading.
    for opt,saved in [(optimizer,state['shared_optimizer'])]+[(private_opts[i],state['private_optimizers'][i]) for i in private_opts]:
        if opt.state_dict()['param_groups']!=saved['param_groups']:raise ValueError('resume optimizer settings differ')
    actor.load_state_dict(state['actor'],strict=True)
    for c in cases:c['private'].load_state_dict(state['private'][c['uid']],strict=True)
    optimizer.load_state_dict(state['shared_optimizer'])
    for i,opt in private_opts.items():opt.load_state_dict(state['private_optimizers'][i])
    dual.load_state_dict(state['dual']);torch.set_rng_state(state['rng'])
    return step


def gradient_norm(parameters):
    return math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in parameters if p.grad is not None))


def clip_shared_gradients(heads,body,max_norm,policy):
    """Clip the two shared roles jointly (legacy) or independently."""
    if policy not in ('joint','by_role'):raise ValueError('unknown shared clipping policy')
    if policy=='joint':
        return float(torch.nn.utils.clip_grad_norm_(heads+body,max_norm,error_if_nonfinite=True))
    before=gradient_norm(heads+body)
    for parameters in (heads,body):
        if parameters:torch.nn.utils.clip_grad_norm_(parameters,max_norm,error_if_nonfinite=True)
    return before


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,required=True,help='JSON list of exactly two state/checkpoint/panel/panel_metadata paths')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--sequences',type=int,default=2)
    parser.add_argument('--updates',type=int,default=2);parser.add_argument('--policy',choices=['full','heads_only'],default='heads_only')
    parser.add_argument('--shared-lr',type=float,required=True,help='Shared output-head Adam LR');parser.add_argument('--body-lr',type=float,help='Required separate body LR for full policy');parser.add_argument('--private-lr',type=float,required=True)
    parser.add_argument('--private-eps',type=validate_private_epsilon,default=1e-12,help='Positive private Adam epsilon; immutable on exact resume')
    parser.add_argument('--update-accumulation',choices=['none','fp32_remainder'],default='none',help='Opt-in FP32 Adam addition remainders; immutable on exact resume')
    parser.add_argument('--ranking-dual-scale',type=float,help='Opt-in dimensionless relative ranking loss normalization; requires ranking-dual-lr')
    parser.add_argument('--ranking-dual-lr',type=float,help='Per-checkpoint ascent LR for the zero relative-ranking-loss constraint')
    parser.add_argument('--panel-backend',choices=['reference','direct_streamed'],default='reference')
    parser.add_argument('--gradient-profile',action='store_true',help='One complete two-case backward with zero optimizer/dual steps; requires updates=0')
    parser.add_argument('--resume',type=Path,help='Exact training checkpoint; updates is total target, output must be a new directory')
    parser.add_argument('--teacher-cache-format',choices=['fp64_log_probs','fp32_logits'],default='fp64_log_probs')
    parser.add_argument('--teacher-cache-root',type=Path,help='Explicit existing root containing teacher_state<ID> caches; never recompute on mismatch')
    parser.add_argument('--pool-indices',type=Path,help='JSON state ID to ordered TRAIN-panel indices; cycles by absolute update, final readout covers full pool')
    parser.add_argument('--gradient-norms',action='store_true',help='Report separate head/body/private norms before clipping')
    parser.add_argument('--gradient-components',action='store_true',help='Extra norm-only VJP to separate norm/preservation gradients; no extra GPT evaluations')
    parser.add_argument('--shared-clip-policy',choices=['joint','by_role'],default='joint',help='Clip shared heads and backbone jointly or separately; fixed across exact resume')
    parser.add_argument('--clip',type=float,default=1.);parser.add_argument('--seed',type=int,default=930067)
    args=parser.parse_args()
    # Validate optional constraint settings before creating output/cache files.
    make_diagnostic_dual([0],objective_config(),rank_scale=args.ranking_dual_scale,rank_dual_lr=args.ranking_dual_lr)
    cache_class=teacher_cache_class(args.teacher_cache_format)
    if not 1<=args.sequences<=16 or not 0<=args.updates<=200 or any(not math.isfinite(v) or v<=0 for v in (args.shared_lr,args.private_lr,args.clip)):raise ValueError('bounded diagnostic settings required')
    if args.resume and args.gradient_profile:raise ValueError('gradient profile cannot resume')
    if args.gradient_profile and args.updates!=0:raise ValueError('gradient profile requires updates=0')
    if args.policy=='full' and (args.body_lr is None or not math.isfinite(args.body_lr) or args.body_lr<=0):raise ValueError('full policy requires positive body LR')
    # Import the optional execution path before pinning all imported sources.
    if args.panel_backend=='direct_streamed':
        from rewriter.experimental_factor_panel import FactorPanelCapture
        from rewriter.experimental_factor_norm import prepare_factor_panel
    args.output.mkdir(parents=True,exist_ok=False);sources=pin_sources(args.output);torch.set_num_threads(2);torch.manual_seed(args.seed);started=time.perf_counter()
    specs=json.loads(args.cases.read_text())
    if len(specs)!=2:raise ValueError('exactly two cases required')
    pool_plan,pool_schedule=read_pool_indices(args.pool_indices,args.sequences)
    if pool_plan is not None:
        pool_state_ids=[]
        # Reject a bad second-case plan before any large teacher cache is made.
        for spec in specs:
            identity=torch.load(spec['state'],map_location='cpu',weights_only=False,mmap=True)['completed_steps']
            available=len(json.loads(Path(spec['panel_metadata']).read_text())['sequence_ids'])
            if str(identity) not in pool_plan or any(i>=available for i in pool_plan[str(identity)]):raise ValueError('pool state/index outside case TRAIN panel')
            pool_state_ids.append(str(identity))
        if set(pool_plan)!=set(pool_state_ids):raise ValueError('pool plan state inventory differs')
    cases=[];actor=None;engine=None;teacher_counts={};cache_provenance=[]
    for spec in specs:
        if set(spec)!=set(('state','checkpoint','panel','panel_metadata')):raise ValueError('case requires exactly state/checkpoint/panel/panel_metadata')
        hashes={k:sha(v) for k,v in spec.items()};state=torch.load(spec['state'],map_location='cpu',weights_only=False,mmap=True)
        ckpt=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False,mmap=True);meta=ckpt['metadata']
        if meta['state_sha256']!=hashes['state']:raise ValueError('actor seed/state mismatch')
        panel=torch.load(spec['panel'],map_location='cpu',weights_only=False,mmap=True);pm=json.loads(Path(spec['panel_metadata']).read_text())
        validate_train_panel(panel,pm,hashes['state'],hashes['panel'])
        if len(panel['inputs'])<args.sequences:raise ValueError('insufficient TRAIN inputs')
        uid=state['completed_steps']
        if pool_plan is not None and str(uid) not in pool_plan:raise ValueError('state absent from pool plan')
        panel_indices=list(range(args.sequences)) if pool_plan is None else pool_plan[str(uid)]
        if any(i>=len(panel['inputs']) or i>=len(pm['sequence_ids']) for i in panel_indices):raise ValueError('pool index outside verified TRAIN panel')
        inputs=panel['inputs'][panel_indices].clone();ids=[pm['sequence_ids'][i] for i in panel_indices]
        snapshot=unpack_record(state['observation'],device='cpu').snapshot
        if actor is None:
            actor=GroupedCoordinateChisel(rank=8,trajectory_dim=400,model_dim=256,num_layers=2,latent_tokens=8,fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
            actor.load_state_dict(ckpt['actor'],strict=True)
        elif not exact_state_equal(actor.state_dict(),ckpt['actor']):raise ValueError('two seeds must use bit-exact shared actor')
        if actor.configuration()!=meta['architecture']:raise ValueError('actor configuration mismatch')
        private=actor.make_private(snapshot,state_key=('cpu-teaching',0,uid,hashes['state']),seed=meta['args']['private_seed']);private.load_state_dict(ckpt['private'],strict=True)
        if engine is None:
            base=state['model'];depth=len({n.split('.')[1] for n in base if n.startswith('blocks.')})
            model=GPT(base['embed.weight'].shape[0],depth,base['embed.weight'].shape[1]).float().eval().requires_grad_(False)
            engine=FunctionalPreservation(model,microbatch_sequences=1,chunk_tokens=128,teacher_cache_device='cpu',kl_dtype=torch.float64,gpt_precision='fp32')
        load_bound_state(engine,state['model'],hashes['state'])
        cache_path=(args.teacher_cache_root or args.output)/f'teacher_state{uid}'
        if args.teacher_cache_root:
            cache_meta=cache_class(cache_path,engine,inputs,ids,hashes['state']).meta
        else:cache_meta=cache_class.create(engine,inputs,ids,hashes['state'],cache_path)
        for k,v in cache_meta['teacher_compute'].items():teacher_counts[k]=teacher_counts.get(k,0)+(0 if args.teacher_cache_root else v)
        cache_provenance.append(dict(state_id=uid,path=str(cache_path.resolve()),manifest_sha256=sha(cache_path/'manifest.json'),format=args.teacher_cache_format,reused=bool(args.teacher_cache_root),original_setup_compute=cache_meta['teacher_compute']))
        cases.append(dict(uid=uid,panel_indices=panel_indices,state=state,snapshot=snapshot,private=private,inputs=inputs,ids=ids,cache=cache_path,hashes=hashes,spec=spec,seed_provenance=seed_provenance(meta)))
        del panel,ckpt
    if len({c['uid'] for c in cases})!=2:raise ValueError('distinct checkpoint identities required')
    if pool_plan is not None and set(pool_plan)!={str(c['uid']) for c in cases}:raise ValueError('pool plan state inventory differs')
    actor.eval()
    for name,p in actor.named_parameters():p.requires_grad_(args.policy=='full' or name.startswith(('matrix_output.','vector_output.')))
    shared=list(p for p in actor.parameters() if p.requires_grad)
    heads=[p for n,p in actor.named_parameters() if p.requires_grad and n.startswith(('matrix_output.','vector_output.'))]
    body=[p for n,p in actor.named_parameters() if p.requires_grad and not n.startswith(('matrix_output.','vector_output.'))]
    groups=[dict(params=heads,lr=args.shared_lr,role='heads')]
    if body:groups.append(dict(params=body,lr=args.body_lr,role='body'))
    optimizer=make_diagnostic_adam(groups,update_accumulation=args.update_accumulation,betas=(.9,.99),eps=1e-8)
    private_opts={c['uid']:make_diagnostic_adam(c['private'].parameters(),update_accumulation=args.update_accumulation,lr=args.private_lr,betas=(.9,.99),eps=args.private_eps) for c in cases}
    config=objective_config();dual=make_diagnostic_dual([c['uid'] for c in cases],config,rank_scale=args.ranking_dual_scale,rank_dual_lr=args.ranking_dual_lr);contexts=LocalCPUContexts(args.panel_backend);training_compute={};readout_compute={}
    metadata=dict(scope=__doc__,case_spec_sha256=sha(args.cases),args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},objective=dataclasses.asdict(config),dual_lr=.01,initial_dual_multiplier=0.,source_sha256=sources,cases=[dict(state_id=c['uid'],paths=c['spec'],sha256=c['hashes'],sequence_ids=c['ids'],input_sha256=tensor_sha(c['inputs']),private_identity=c['private'].get_extra_state(),seed_provenance=c['seed_provenance'],panel_indices=c['panel_indices']) for c in cases],teacher_compute=teacher_counts,teacher_cache_provenance=cache_provenance,teacher_setup_seconds=time.perf_counter()-started,optimizer='fresh Adam shared + distinct private; actual hyperparameters in optimizer_config; no seed moments',state_weights=[.5,.5],private_gradient_normalization='undo state weight before private clipping and step',fixed_train_panels=pool_plan is None,pool_schedule=pool_schedule,full_bank=False,generalization_claim=False)
    metadata['ranking_constraint']=(dict(scale=dual.rank_scale,dual_lr=dual.rank_dual_lr,tolerance=0.,measurement='mean_nonnegative_relative_gap_margin_loss') if isinstance(dual,RankingDual) else None)
    initial_shared=snapshot_parameters(actor);initial_private={c['uid']:snapshot_parameters(c['private']) for c in cases}
    start_step=0
    if args.resume:
        resumed=torch.load(args.resume,map_location='cpu',weights_only=False)
        start_step=restore_training_state(resumed,actor,cases,optimizer,private_opts,dual,metadata,args.updates)
        metadata['resume_lineage']=dict(checkpoint=str(args.resume.resolve()),sha256=sha(args.resume),
            completed_updates=start_step,source_sha256=resumed['metadata']['source_sha256'],
            previous_lineage=resumed['metadata'].get('resume_lineage'))
        metadata['optimizer']='Restored shared/private Adam moments and hyperparameters; duals and CPU RNG restored'
        del resumed
    metadata['optimizer_config']=dict(shared=optimizer_configuration(optimizer),private={uid:optimizer_configuration(opt) for uid,opt in private_opts.items()})
    metadata['segment_start_updates']=start_step
    metadata['teacher_checksum_policy']=dict(scope='process-local successful immutable-file checks',max_entries=_TEACHER_CHECKSUM_CACHE_LIMIT,stat_fields=['dev','inode','size','mtime_ns','ctime_ns'])
    metadata['compute_scope']='current execution segment only; original cache setup counts are provenance, not new compute'
    metadata['segment_initial_dual']=dual.state_dict()
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    def checkpoint(step):
        save(args.output/f'checkpoint_{step:04d}.pt',training_state(actor,cases,optimizer,private_opts,dual,metadata,step))
    checkpoint(start_step)
    with (args.output/'metrics.jsonl').open('w',buffering=1) as log:
        for step in range(start_step,args.updates+1):
            if any(sha(n)!=v for n,v in sources.items()):raise ValueError('pinned execution source changed')
            learning=step<args.updates or args.gradient_profile;advance=learning and not args.gradient_profile;optimizer.zero_grad(set_to_none=True);rows=[];tick=time.perf_counter()
            for c in cases:
                t=time.perf_counter();load_bound_state(engine,c['state']['model'],c['hashes']['state'])
                provider=cache_class(c['cache'],engine,c['inputs'],c['ids'],c['hashes']['state'])
                positions=panel_positions(len(c['inputs']),args.sequences,step,final=not learning) if pool_plan is not None else tuple(range(len(c['inputs'])))
                if pool_plan is not None:provider=SubsetTeacherProvider(provider,positions)
                selected_inputs=provider.inputs;selected_ids=provider.ids
                grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze');editable=editable_names(engine.parameters,'grouped')-grouped.frozen_parameter_names
                context=dict(engine=engine,domain='gpt',local_id=c['uid'],snapshot=c['snapshot'],grouped=grouped,inputs=selected_inputs,ids=selected_ids,teacher=provider,norm_squared=editable_weight_norm_sq(engine.parameters,editable))
                setup=time.perf_counter()-t;private_opts[c['uid']].zero_grad(set_to_none=True);t=time.perf_counter()
                before_compute=dict(contexts.compute)
                case_config=dual.config_for_state(c['uid']) if isinstance(dual,RankingDual) else config
                row=contexts.measure(context,actor,c['private'],case_config,dual,learning=learning,weight=.5,gradient_components=args.gradient_components)
                if isinstance(dual,RankingDual):row.update(dual.describe_measurement(c['uid'],row['ranking_penalty']))
                phase_compute=training_compute if learning else readout_compute
                for name,value in contexts.compute.items():phase_compute[name]=phase_compute.get(name,0)+value-before_compute.get(name,0)
                row.update(context_setup_seconds=setup,measure_seconds=time.perf_counter()-t,teacher_checksum=dict(provider.checksum_stats))
                if pool_plan is not None:row.update(panel_scope='training_subset' if learning else 'full_training_pool',panel_indices=[c['panel_indices'][i] for i in positions],pool_sequences=len(c['inputs']))
                if learning:
                    for p in c['private'].parameters():
                        if p.grad is not None:p.grad.mul_(2.)
                    if args.gradient_norms:row['private_preclip_grad_norm_fp64']=gradient_norm(c['private'].parameters())
                    row['private_preclip_grad_norm']=float(torch.nn.utils.clip_grad_norm_(c['private'].parameters(),float('inf') if args.gradient_profile else args.clip,error_if_nonfinite=True))
                    if advance:private_opts[c['uid']].step()
                rows.append(row)
                del context,provider,grouped
            shared_grad=None;role_gradients=None;role_postclip=None
            if learning:
                if args.gradient_norms:role_gradients=dict(heads=gradient_norm(heads),body=gradient_norm(body))
                shared_grad=clip_shared_gradients(heads,body,float('inf') if args.gradient_profile else args.clip,args.shared_clip_policy)
                if args.gradient_norms:role_postclip=dict(heads=gradient_norm(heads),body=gradient_norm(body))
                if advance:
                    optimizer.step()
                    ranking_kwargs=(dict(ranking_measurements=[(c['uid'],r['ranking_constraint_loss']) for c,r in zip(cases,rows)]) if isinstance(dual,RankingDual) else {})
                    dual.step([(c['uid'],r['mean_kl']) for c,r in zip(cases,rows)],expected_ids=[c['uid'] for c in cases],**ranking_kwargs)
                    checkpoint(step+1)
            report=dict(completed_updates_before_measure=step,updates_after=step+int(advance),learning=learning,gradient_profile=args.gradient_profile,rows=rows,mean_objective=sum(r['objective'] for r in rows)/2,shared_preclip_grad_norm=shared_grad,role_preclip_grad_norms=role_gradients,role_postclip_grad_norms=role_postclip,shared_motion_from_seed=motion(initial_shared,actor),private_motion_from_seed={c['uid']:motion(initial_private[c['uid']],c['private']) for c in cases},dual=dual.state_dict(),compute=dict(contexts.compute),teacher_setup_compute=teacher_counts,gpt_forwards_total=teacher_counts.get('teacher_forwards',0)+contexts.compute.get('student_forwards',0),gpt_backwards_total=contexts.compute.get('student_backwards',0),step_seconds=time.perf_counter()-tick,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            if pool_plan is not None:
                report['phase_compute']=dict(training=dict(training_compute),full_pool_readout=dict(readout_compute))
                report['training_coverage']={c['uid']:dict(completed_updates=step+int(advance),unique_sequences_seen=min(len(c['inputs']),(step+int(advance))*args.sequences),panel_indices_seen=c['panel_indices'][:min(len(c['inputs']),(step+int(advance))*args.sequences)],full_pool_readout=not learning) for c in cases}
            if args.gradient_profile and (report['shared_motion_from_seed']['l2']!=0 or any(v['l2']!=0 for v in report['private_motion_from_seed'].values()) or optimizer.state or any(o.state for o in private_opts.values()) or dual.updates):raise ValueError('gradient profile unexpectedly changed training state')
            log.write(json.dumps(report,allow_nan=False)+'\n');print(json.dumps(report,allow_nan=False),flush=True)
    completion=dict(completed_updates=args.updates,compute=contexts.compute,teacher_setup_compute=teacher_counts,elapsed_seconds=time.perf_counter()-started,final=report)
    if pool_plan is not None:completion['phase_compute']=dict(training=training_compute,full_pool_readout=readout_compute)
    (args.output/'complete.json').write_text(json.dumps(completion,indent=2)+'\n')


if __name__=='__main__':main()
