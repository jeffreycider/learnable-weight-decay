"""Full-bank joint CoordinateChisel training with identity-owned private Adam.

One GPT and one CNN state per rank/step; unequal owner pools are corrected by
0.5 * local_pool_size / domain_size. Private state is staged one layout at a time,
never all checkpoint parameters on GPU. Scientific objectives/stopping/readouts
must be supplied explicitly. This driver never updates a captured base model.
"""
from __future__ import annotations
import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from rewriter.coordinate_chisel import CoordinateChisel, CheckpointProbes
from rewriter.coordinate_optimization import clip_private_gradients, clip_shared_continuation
from rewriter.private_probe_slot import PrivateProbeSlot
from rewriter.private_probe_store import PrivateIdentity
from rewriter.distributed_probe_store import DistributedProbeStore, private_owner, sum_shared_gradients, _digest
from rewriter.tokenwise_reference import TokenwiseConfig, TokenwiseDual
from rewriter.multibank import file_sha
from rewriter.coordinate_verify_ahead import CoordinateVerifyAhead
from scripts.qualify_coordinate_chisel import FixedContexts

DOMAINS = ('gpt', 'cifar10')
SCHEMA = 'coordinate_chisel_fullbank_v1'


def stable_seed(*values):
    return int.from_bytes(hashlib.sha256(json.dumps(values,sort_keys=True).encode()).digest()[:8], 'little') % (2**63-1)


def shared_update_policy(recipe):
    """Resolve explicit ownership without changing legacy recipe semantics."""
    legacy=recipe.get('shared_optimizer_step',True)
    if type(legacy) is not bool:
        raise ValueError('shared_optimizer_step must be an explicit boolean')
    policy=recipe.get('shared_update_policy','full' if legacy else 'frozen')
    if policy not in ('full','frozen','heads_only'):
        raise ValueError('shared_update_policy must be full, frozen, or heads_only')
    if 'shared_update_policy' in recipe and 'shared_optimizer_step' in recipe and legacy != (policy!='frozen'):
        raise ValueError('shared_update_policy conflicts with shared_optimizer_step')
    return policy


def configure_shared_trainability(actor, policy):
    """Keep Adam's complete parameter inventory; frozen owners retain grad=None."""
    if policy not in ('full','frozen','heads_only'):raise ValueError('unknown shared update policy')
    heads={id(p) for head in actor.output_heads for p in head.parameters()}
    if not heads or not heads <= {id(p) for p in actor.parameters()}:
        raise ValueError('action heads must belong to shared actor')
    for p in actor.parameters():
        p.requires_grad_(policy=='full' or (policy=='heads_only' and id(p) in heads))
        if not p.requires_grad:p.grad=None


def validate_recipe(recipe):
    if recipe.get('schema') != SCHEMA: raise ValueError('explicit full-bank coordinate recipe required')
    family=recipe.get('action_family','fixed_anchored_v1')
    if family not in ('fixed_anchored_v1','adaptive_ucv_v1'):raise ValueError('unknown coordinate action family')
    sharing=recipe['architecture'].get('output_head_sharing','shared')
    if sharing not in ('shared','functional_group'):
        raise ValueError('unknown output head sharing')
    if 'output_head_sharing' in recipe['architecture'] and family!='adaptive_ucv_v1':
        raise ValueError('output head sharing requires adaptive actions')
    if sharing=='functional_group' and any(b!='eager' for b in (recipe['actor_backend'],recipe.get('private_actor_backend','eager'))):
        raise ValueError('functional-group output heads require separate graph qualification; use eager')
    if 'residual_precision' in recipe['architecture']:
        residual=recipe['architecture']['residual_precision']
        if family!='adaptive_ucv_v1' or residual not in ('bf16','fp32'):
            raise ValueError('residual_precision requires explicit adaptive bf16 or fp32 mode')
        if residual=='fp32' and any(b!='eager' for b in (recipe['actor_backend'],recipe.get('private_actor_backend','eager'))):
            raise ValueError('FP32 residual actor graphs require separate qualification; use eager')
    if 'factor_precision' in recipe['architecture']:
        if family != 'adaptive_ucv_v1' or recipe['architecture']['factor_precision'] not in ('fp32', 'fp64'):
            raise ValueError('factor_precision requires explicit adaptive fp32 or fp64 mode')
    if 'fresh_weight_core' in recipe['architecture']:
        if family!='adaptive_ucv_v1' or type(recipe['architecture']['fresh_weight_core']) is not bool:
            raise ValueError('fresh weight sensing requires an explicit adaptive boolean flag')
    if recipe.get('student_backend','dense')!='dense':raise ValueError('only dense finite student is qualified here')
    if family=='adaptive_ucv_v1':
        if recipe.get('student_backend')!='dense':raise ValueError('adaptive family requires explicit dense student')
        backends=(recipe['actor_backend'],recipe.get('private_actor_backend','eager'))
        if any(b not in ('eager','adaptive_actor_graph') for b in backends):
            raise ValueError('adaptive family requires explicit adaptive shared/private actors')
        if 'adaptive_actor_graph' in backends and (not recipe['architecture'].get('fresh_weight_core',False) or recipe['architecture'].get('factor_precision','fp32') not in ('fp32','fp64')):
            raise ValueError('adaptive graph requires explicit FP32 or FP64 factors and fresh cores')
        if recipe.get('shared_initialization',{'mode':'random'}).get('mode')!='random':
            raise ValueError('adaptive family requires fresh shared initialization')
        if recipe.get('offline_backend','reference')!='reference':raise ValueError('fixed-anchor offline backends cannot serve adaptive actions')
        if recipe['architecture'].get('frame_mode','tied_orthonormal')!='tied_orthonormal' or recipe['architecture'].get('write_observation_frame','encoder')!='encoder':
            raise ValueError('adaptive family requires its tied read/write coordinate frame')
    if recipe['domain_weights'] != {'gpt':.5,'cifar10':.5}: raise ValueError('equal domain means required')
    if family!='adaptive_ucv_v1' and 'adaptive_actor_graph' in (recipe['actor_backend'],recipe.get('private_actor_backend','eager')):
        raise ValueError('adaptive graph requires adaptive action family')
    if recipe['actor_backend'] not in ('eager','eager_actor_graph','adaptive_actor_graph'): raise ValueError('unqualified actor backend')
    if recipe.get('private_actor_backend','eager') not in ('eager','eager_actor_graph','adaptive_actor_graph'):
        raise ValueError('unqualified private actor backend')
    if recipe.get('private_actor_backend','eager') in ('eager_actor_graph','adaptive_actor_graph') and recipe.get('private_inner_steps',0)<1:
        raise ValueError('private actor capture requires private inner fitting')
    if recipe.get('private_snapshot_backend','reference') not in ('reference','pinned_d2h_v1'):
        raise ValueError('unqualified private snapshot backend')
    if recipe.get('verification_lookahead','off') not in ('off','depth1_v1'):
        raise ValueError('unqualified verification lookahead')
    inner=recipe.get('private_inner_steps',0)
    if type(inner) is not int or inner<0:
        raise ValueError('private_inner_steps must be a nonnegative integer')
    init=recipe.get('shared_initialization',{'mode':'random'})
    if init.get('mode') not in ('random','checkpoint_actor_only'):raise ValueError('explicit shared initialization required')
    if init['mode']=='checkpoint_actor_only':
        if not init.get('path') or len(init.get('sha256',''))!=64 or type(init.get('update')) is not int:
            raise ValueError('shared actor initialization needs path, SHA and update')
    schedule=recipe['private_optimizer'].get('tick_schedule')
    if schedule is not None:
        if set(schedule)!={'cold_steps','warm_lr'} or type(schedule['cold_steps']) is not int or schedule['cold_steps']<1 or not math.isfinite(schedule['warm_lr']) or schedule['warm_lr']<=0:
            raise ValueError('private LR schedule must use positive actual optimizer ticks')
    split=recipe.get('private_inner_panels')
    if split is not None:
        if inner<1 or set(split)!=set(DOMAINS):raise ValueError('explicit inner panel configuration required for both domains')
        for domain,multiple,joint in [('gpt',16,recipe['panels']['gpt_sequences']),('cifar10',128,recipe['panels']['cifar_images'])]:
            s=split[domain]
            if set(s)!={'source_size','inner_size'} or any(type(s[k]) is not int or s[k]<multiple or s[k]%multiple for k in s):raise ValueError('inner/source panel shape differs')
            if s['source_size']<s['inner_size']+joint:raise ValueError('private and joint panels must be disjoint')
    if recipe['panels']['mode'] not in ('fixed','rotating_contiguous'): raise ValueError('explicit panel schedule required')
    for key, multiple in [('gpt_sequences',16),('cifar_images',128)]:
        value=recipe['panels'][key]
        if type(value) is not int or value<multiple or value%multiple: raise ValueError('qualified teacher batch shape required')
    if set(recipe['objectives']) != set(DOMAINS): raise ValueError('both domain objectives required')
    for config in recipe['objectives'].values(): TokenwiseConfig(**config)
    for name in ('shared_optimizer','private_optimizer'):
        config=recipe[name]
        keys=('head_lr','representation_lr','head_eps','representation_eps','clip') if name=='shared_optimizer' else ('lr','eps','clip')
        if any(not math.isfinite(float(config[k])) or config[k]<=0 for k in keys): raise ValueError('positive optimizer settings required')
        if len(config['betas'])!=2 or any(not 0<=b<1 for b in config['betas']): raise ValueError('explicit Adam betas required')
    if recipe['shared_optimizer']['clip_scope'] not in ('joint','separate'): raise ValueError('explicit shared clipping scope required')
    policy=shared_update_policy(recipe)
    experimental_heads=recipe.get('experimental_heads_only_graph',False)
    if type(experimental_heads) is not bool:
        raise ValueError('experimental_heads_only_graph must be an explicit boolean')
    if experimental_heads:
        if (recipe.get('qualification_only') is not True or policy!='heads_only'
                or family!='adaptive_ucv_v1'
                or recipe['architecture'].get('factor_precision','fp32')!='fp32'
                or recipe['actor_backend']!='adaptive_actor_graph'
                or recipe.get('private_actor_backend')!='adaptive_actor_graph'):
            raise ValueError('experimental heads-only graphs require a qualification-only adaptive FP32 joint/private graph recipe')
    if policy=='heads_only' and any(b!='eager' for b in (recipe['actor_backend'],recipe.get('private_actor_backend','eager'))) and not experimental_heads:
        raise ValueError('heads_only requires eager shared and private actors until partial-mask graph qualification')
    if policy=='frozen':
        if recipe['actor_backend'] not in ('eager','adaptive_actor_graph'):
            raise ValueError('frozen shared actor requires eager or explicit adaptive capture')
        if 'adaptive_actor_graph' in (recipe['actor_backend'],recipe.get('private_actor_backend','eager')) and recipe['architecture'].get('factor_precision','fp32')!='fp32':
            raise ValueError('frozen shared adaptive graph experiment supports FP32 factors only')
    stop=recipe['stopping']
    initial_readout=recipe.get('pre_update_fitting_readout_at')
    if initial_readout is not None and (type(initial_readout) is not int or initial_readout<0 or initial_readout>=stop['max_updates']):
        raise ValueError('pre-update fitting readout must pin a batch before the maximum')
    for key in ('max_updates','full_sweep_every','consecutive_full_sweeps'):
        if type(stop[key]) is not int or stop[key]<1: raise ValueError('positive declared stopping/readout budget required')
    if type(stop['stop_when_all_fit']) is not bool: raise ValueError('explicit fitting-only stopping choice required')
    reviews=stop.get('review_after_full_sweeps',[])
    if not isinstance(reviews,list) or len(set(reviews))!=len(reviews) or any(type(n) is not int or n<1 or n*stop['full_sweep_every']>=stop['max_updates'] for n in reviews):
        raise ValueError('review boundaries must name distinct full sweeps before maximum budget')
    if stop['stop_when_all_fit'] and not stop['fitting_criteria']: raise ValueError('fitting criteria cannot be inferred')
    for rule in stop['fitting_criteria']:
        if rule['operator'] not in ('lt','le','eq','ge','gt') or not math.isfinite(rule['value']): raise ValueError('invalid fitting criterion')
        if rule.get('domain') not in (*DOMAINS,None): raise ValueError('unknown criterion domain')
    for key,multiple in [('gpt_sequences',16),('cifar_images',128)]:
        if recipe['readout'][key]<multiple or recipe['readout'][key]%multiple: raise ValueError('readout batch shape mismatch')
    transfer=recipe['readout']['heldout']
    if recipe.get('fitting_subset') is not None and recipe.get('qualification_only') is not True:
        raise ValueError('state subset requires an explicitly labeled qualification-only run')
    if type(transfer['enabled']) is not bool: raise ValueError('explicit held-out protocol required')
    if transfer['enabled']:
        if type(transfer['support_steps']) is not int or transfer['support_steps']<0: raise ValueError('fixed support budget required')
        if set(transfer['ids'])!=set(DOMAINS): raise ValueError('explicit held-out IDs for both domains required')
        if transfer['query_gpt_sequences']<16 or transfer['query_gpt_sequences']%16: raise ValueError('qualified query shape required')


def build_actor(recipe):
    """A recipe family determines both shared and private checkpoint schemas."""
    config=dict(recipe['architecture'])
    if recipe.get('action_family','fixed_anchored_v1')=='adaptive_ucv_v1':
        from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
        sharing=config.pop('output_head_sharing','shared')
        actor_class=AdaptiveCoordinateChisel
        if sharing=='functional_group':
            from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
            actor_class=GroupedCoordinateChisel
        elif sharing!='shared':
            raise ValueError('unknown output head sharing')
        config.pop('frame_mode',None);config.pop('write_observation_frame',None)
        # Device moves preserve the explicit mixed-dtype actor. Never blanket
        # cast it or implicitly promote an old checkpoint at this boundary.
        config.setdefault('factor_precision', 'fp32')
        return actor_class(**config)
    return CoordinateChisel(**config)


def build_shared_adam(actor, config):
    heads=[p for head in actor.output_heads for p in head.parameters()]; ids={id(p) for p in heads}
    body=[p for p in actor.parameters() if id(p) not in ids]
    return torch.optim.Adam([dict(params=heads,lr=config['head_lr'],eps=config['head_eps'],role='action_heads'),
        dict(params=body,lr=config['representation_lr'],eps=config['representation_eps'],role='representation')],
        betas=tuple(config['betas']))


def initialize_shared_actor(actor,recipe):
    """Import shared weights only; this is an explicitly fresh optimizer stage."""
    spec=recipe.get('shared_initialization',{'mode':'random'})
    if recipe.get('action_family','fixed_anchored_v1')=='adaptive_ucv_v1' and spec['mode']!='random':
        raise ValueError('adaptive family requires fresh shared initialization')
    if spec['mode']=='random':return dict(mode='random')
    source=Path(spec['path'])
    if file_sha(source)!=spec['sha256']:raise ValueError('shared initialization SHA differs')
    saved=torch.load(source,map_location='cpu',weights_only=False)
    if saved.get('update')!=spec['update'] or file_sha(source)!=spec['sha256']:raise ValueError('shared initialization clock or bytes changed')
    state=saved['actor'];current=actor.state_dict()
    if state.keys()!=current.keys() or any(not isinstance(v,torch.Tensor) or v.shape!=current[n].shape or not bool(torch.isfinite(v).all()) for n,v in state.items()):
        raise ValueError('shared actor parameter layout or finite values differ')
    actor.load_state_dict(state,strict=True)
    return dict(**spec,shared_optimizer='fresh',private_parameters='fresh per identity',private_optimizer='fresh',duals='fresh_zero')


def set_private_lr(optimizer,config,completed_steps):
    """Absolute LR from this identity's actual private-Adam clock, idempotently."""
    if type(completed_steps) is not int or completed_steps<0:raise ValueError('invalid private optimizer clock')
    schedule=config.get('tick_schedule');lr=config['lr']
    if schedule is not None and completed_steps>=schedule['cold_steps']:lr=schedule['warm_lr']
    for group in optimizer.param_groups:group['lr']=lr
    return lr


def measure_context(contexts,context,*args,**kwargs):
    engine=context.get('engine')
    prior=engine.compute.as_dict() if context.get('domain')=='gpt' and engine is not None else {}
    row=contexts.measure(context,*args,**kwargs)
    for name,value in prior.items():contexts.compute[name]-=value
    return row


def adapt_private(contexts,context,actor,private,optimizer,config,dual,*,steps,clip,completed_steps=0,optimizer_config=None,inner_contexts=None,actor_dispatch=None):
    """First-order private fitting; preserve shared values AND accumulated grads.

    The same fitting panel is reused. These are real finite-model evaluations,
    counted by contexts.measure, not additional independent labels. The following
    joint update treats the adapted private state as a leaf; no inner-loop
    derivative, GPT HVP, or derivative through Adam is computed.
    """
    if steps==0:return []
    flags=[p.requires_grad for p in actor.parameters()]
    rows=[]
    try:
        actor.requires_grad_(False)
        if inner_contexts is not None and len(inner_contexts)!=steps:raise ValueError('private panel count differs')
        for step in range(steps):
            if optimizer_config is not None:set_private_lr(optimizer,optimizer_config,completed_steps+step)
            optimizer.zero_grad(set_to_none=True)
            selected=context if inner_contexts is None else inner_contexts[step]
            active=actor_dispatch.for_visit(selected['domain'],selected,private) if actor_dispatch is not None else actor
            row=measure_context(contexts,selected,active,private,config,dual,learning=True,weight=1.)
            grad=float(clip_private_gradients(private.parameters(),clip))
            if not math.isfinite(grad):raise FloatingPointError('nonfinite private adaptation gradient')
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            dual.step([(row['local_id'],row['mean_kl'])],expected_ids=[row['local_id']])
            rows.append(dict(row,private_preclip_gradient_norm=grad,private_optimizer_step=completed_steps+step+1,private_lr=optimizer.param_groups[0]['lr']))
    finally:
        for parameter,flag in zip(actor.parameters(),flags):parameter.requires_grad_(flag)
    return rows


@dataclasses.dataclass(frozen=True)
class StateSpec:
    domain: str
    local_id: int
    identity: PrivateIdentity


def describe_states(catalog, layouts):
    result={role:{d:[] for d in DOMAINS} for role in ('train','audit')}
    for domain in DOMAINS:
        reader=catalog.readers[domain]
        entries={int(e.get('completed_steps',e.get('local_id'))):e for e in reader.entries}
        for role in result:
            for uid in reader.manifest['split'][role]:
                entry=entries[uid]
                if domain=='gpt':
                    description=reader.describe_state(uid)
                    key=(domain,int(description['bank_seed']),int(description['gpt_completed_steps']))
                else: key=(domain,int(entry['seed']))
                result[role][domain].append(StateSpec(domain,uid,PrivateIdentity(key,entry['sha256'],layouts[domain])))
    return result


class OwnerBalancedSchedule:
    """Uniform domain means despite unequal identity-owner pool sizes.

    Every local cycle is independently shuffled. A sweep length is the largest
    GPT owner pool: it guarantees coverage, but smaller pools repeat. Exact visit
    cycle/position and weights are returned and logged instead of calling this a
    no-repeat global epoch.
    """
    def __init__(self, specs, *, rank, world_size, seed):
        self.rank=rank; self.world=world_size; self.seed=seed; self.specs=specs
        self.pools={d:[s for s in specs[d] if private_owner(s.identity,world_size)==rank] for d in DOMAINS}
        if any(not p for p in self.pools.values()): raise ValueError('each identity owner needs a nonempty pool in each domain')
        self.coverage_batches=max(sum(private_owner(s.identity,world_size)==r for s in specs['gpt']) for r in range(world_size))

    def visit(self, batch, domain):
        pool=self.pools[domain]; cycle,position=divmod(batch,len(pool))
        order=list(pool); random.Random(stable_seed(self.seed,domain,self.rank,cycle)).shuffle(order)
        return order[position], dict(owner_cycle=cycle,owner_position=position,owner_pool_size=len(pool),
            domain_total=len(self.specs[domain]),state_weight=.5*len(pool)/len(self.specs[domain]))


def fitting_pass(rows, specs, rules):
    expected={(s.domain,s.local_id) for d in DOMAINS for s in specs[d]}
    found=[(r['domain'],r['local_id']) for r in rows]
    if len(found)!=len(set(found)) or set(found)!=expected: raise ValueError('full fitting readout coverage differs')
    operators=dict(lt=lambda x,y:x<y,le=lambda x,y:x<=y,eq=lambda x,y:x==y,ge=lambda x,y:x>=y,gt=lambda x,y:x>y)
    for row in rows:
        for rule in rules:
            if rule.get('domain') not in (None,row['domain']): continue
            value=float(row[rule['metric']])
            if not math.isfinite(value): raise FloatingPointError('nonfinite fitting criterion')
            if not operators[rule['operator']](value,rule['value']): return False
    return bool(rules)


def gather_rows(rows):
    gathered=[None]*dist.get_world_size(); dist.all_gather_object(gathered,rows)
    return [r for part in gathered for r in part]


def rng_state(device):
    return dict(torch=torch.get_rng_state(),python=random.getstate(),
                cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)


def restore_rng(state,device):
    torch.set_rng_state(state['torch'].cpu()); random.setstate(state['python'])
    if state['cuda'] is not None: torch.cuda.set_rng_state(state['cuda'].cpu(),device)


def immutable_save(path,value):
    if path.exists(): raise FileExistsError(path)
    temporary=path.with_name('.'+path.name+'.tmp')
    with temporary.open('xb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def load_context(contexts,spec,*,role='train',purpose='fitting',shared_actor=None,gpt_sequences=None,cifar_images=None,block_offset=0):
    previous=contexts.cifar_images
    if cifar_images is not None: contexts.cifar_images=cifar_images
    try:
        return contexts.load(spec.domain,spec.local_id,role=role,purpose=purpose,shared_actor=shared_actor,
                             gpt_sequences=gpt_sequences,block_offset=block_offset)
    finally: contexts.cifar_images=previous


class ActorDispatch:
    def __init__(self,actor,slots,templates,backend):
        from rewriter.coordinate_actor_graph import actor_trainability_signature
        self.shared_trainability=actor_trainability_signature(actor)
        self.actor=actor;self.graphs={};self.graph_replays=0;self.eager_fallbacks=0
        if backend in ('eager_actor_graph','adaptive_actor_graph'):
            if next(actor.parameters()).device.type!='cuda': raise ValueError('actor graph requires qualified CUDA execution')
            from rewriter.coordinate_actor_graph import CoordinateActorGraph
            from rewriter.edit_scope import editable_names
            for domain in DOMAINS:
                actor.zero_grad(set_to_none=True)
                for slot in slots.values(): slot.private.zero_grad(set_to_none=True)
                context=templates[domain]
                editable=editable_names(context['engine'].parameters,'grouped')-context['grouped'].frozen_parameter_names
                if backend=='adaptive_actor_graph':
                    from rewriter.adaptive_actor_graph import capture_adaptive_actor
                    self.graphs[domain]=capture_adaptive_actor(actor,slots[domain],context,editable,warmups=2)
                else:self.graphs[domain]=CoordinateActorGraph(actor,slots[domain],context['snapshot'],editable,warmups=2)
            actor.zero_grad(set_to_none=True)
            for slot in slots.values(): slot.private.zero_grad(set_to_none=True)

    def for_visit(self,domain,context,private):
        graph=self.graphs.get(domain)
        if graph is None:return self.actor
        from rewriter.coordinate_actor_graph import actor_trainability_signature
        if actor_trainability_signature(self.actor)!=self.shared_trainability:
            raise RuntimeError('joint dispatch trainability mask changed; rebuild dispatch at a clean boundary')
        from rewriter.edit_scope import editable_names
        editable=editable_names(context['engine'].parameters,'grouped')-context['grouped'].frozen_parameter_names
        if frozenset(editable)!=graph.editable:
            self.eager_fallbacks+=1;return self.actor
        if hasattr(graph,'bind_context'):graph.bind_context(context)
        graph.bind_scope(private,editable);self.graph_replays+=1
        return graph


def panel_offset(contexts,spec,visits,recipe):
    if recipe['mode']=='fixed':return 0
    if spec.domain=='gpt':
        total=len(contexts.cache.expected[(spec.local_id,'train')]['sequence_ids'])//16
        selected=recipe['gpt_sequences']//16
    else:
        total=len(contexts.split['fitting_indices'])//128;selected=recipe['cifar_images']//128
    if selected>total: raise ValueError('requested panel exceeds cached fitting data')
    return stable_seed(recipe['seed'],spec.identity.token,visits)%(total-selected+1)


def run_training(recipe,output,contexts,device,*,metadata,resume=False,after_commit=None,source_bytes=None):
    """Testable core; contexts supplies the existing verified finite-loss engines."""
    validate_recipe(recipe)
    if getattr(contexts,'action_family','fixed_anchored_v1')!=recipe.get('action_family','fixed_anchored_v1'):
        raise ValueError('finite contexts and recipe action family differ')
    rank,world=dist.get_rank(),dist.get_world_size();output=Path(output)
    if rank==0:
        output.mkdir(parents=True,exist_ok=resume)
        header=output/'metadata.json'
        if resume:
            if json.loads(header.read_text())!=metadata: raise ValueError('resume source/config/data metadata differs')
        else:header.write_text(json.dumps(metadata,indent=2)+'\n')
        if source_bytes is not None:
            for name,value in source_bytes.items():
                relative=Path(name)
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('invalid source record path')
                target=output/'source_record'/relative;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and target.read_bytes()!=value:raise ValueError('source record differs')
                if not target.exists():target.write_bytes(value)
    dist.barrier()
    random.seed(recipe['actor_seed']);torch.manual_seed(recipe['actor_seed']); actor=build_actor(recipe).to(device)
    initialization=initialize_shared_actor(actor,recipe)
    shared_optimizer=build_shared_adam(actor,recipe['shared_optimizer'])
    shared_policy=shared_update_policy(recipe)
    shared_step_enabled=shared_policy!='frozen'
    configure_shared_trainability(actor,shared_policy)
    slots={};optimizers={};fresh_optimizer={};layouts={};templates={}
    cfg=recipe['private_optimizer']
    for domain in DOMAINS:
        uid=max(contexts.catalog.readers[domain].manifest['split']['train'])
        templates[domain]=contexts.load(domain,uid)
        private=actor.make_private(templates[domain]['snapshot'],seed=recipe['private_seed'],state_key=(domain,'template'))
        slots[domain]=PrivateProbeSlot(actor,private)
        layouts[domain]=PrivateIdentity.from_private(private,state_sha256='0'*64).layout_sha256
        optimizers[domain]=torch.optim.Adam(private.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['eps'])
        fresh_optimizer[domain]=copy.deepcopy(optimizers[domain].state_dict())
    specs=describe_states(contexts.catalog,layouts)
    catalog_counts={d:len(specs['train'][d]) for d in DOMAINS}
    if recipe.get('fitting_subset') is not None:
        if set(recipe['fitting_subset'])!=set(DOMAINS):raise ValueError('explicit subset IDs required for both domains')
        for domain in DOMAINS:
            selected=recipe['fitting_subset'][domain];available={s.local_id:s for s in specs['train'][domain]}
            if not selected or len(selected)!=len(set(selected)) or not set(selected)<=available.keys():
                raise ValueError('qualification subset must contain unique fitting-only IDs')
            specs['train'][domain]=[available[i] for i in selected]
    schedule=OwnerBalancedSchedule(specs['train'],rank=rank,world_size=world,seed=recipe['sampler_seed'])
    dispatch=ActorDispatch(actor,slots,templates,recipe['actor_backend'])
    private_dispatch=None
    if recipe.get('private_actor_backend','eager') in ('eager_actor_graph','adaptive_actor_graph'):
        from rewriter.coordinate_actor_graph import PrivateActorDispatch
        private_dispatch=PrivateActorDispatch(actor,slots,templates,backend=recipe['private_actor_backend'])
    templates.clear()
    configs={d:TokenwiseConfig(**recipe['objectives'][d]) for d in DOMAINS}
    duals={d:TokenwiseDual([s.local_id for s in schedule.pools[d]],config=configs[d],dual_lr=recipe['dual_lr']) for d in DOMAINS}
    run_identity=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    log_file=(output/'metrics.jsonl').open('a' if resume else 'w') if rank==0 else None
    wb=None;started=time.perf_counter()
    if rank==0 and recipe['wandb']:
        import wandb
        wb=wandb.init(project='chisel',mode='offline',dir=str(output),name=output.name,
                      config={'recipe':recipe,'run_identity':run_identity,'resume':resume})
        (output/f'wandb-{uuid.uuid4().hex}.json').write_text(json.dumps({'path':str(Path(wb.dir).parent),'id':wb.id}))
    update=0;shared_optimizer_steps=0;streak=0;last_sweep=-1;reason=None;visits={};private_written_bytes=0;heldout_completed=False;last_commit_seconds=0.
    pre_update_readout_completed=None
    def log(event,**values):
        if rank==0:
            row=dict(event=event,update=update,global_batch=update,shared_optimizer_steps=shared_optimizer_steps,shared_update_policy=shared_policy,elapsed_seconds=time.perf_counter()-started,**values)
            text=json.dumps(row,allow_nan=False);print(text,flush=True);log_file.write(text+'\n');log_file.flush()
            if wb:wb.log({k:v for k,v in row.items() if isinstance(v,(int,float))})
    with DistributedProbeStore(output/'joint_store',run_identity=run_identity,
            snapshot_backend=recipe.get('private_snapshot_backend','reference')) as store, CoordinateVerifyAhead(
            recipe.get('verification_lookahead','off'),contexts,schedule,recipe['stopping']['max_updates']) as verification:
        def stage(spec,context):
            domain=spec.domain;slot=slots[domain];optimizer=optimizers[domain]
            if spec.identity.token in store._versions:
                return store.stage(spec.identity,slot,optimizer)
            module=actor.make_private(context['snapshot'],seed=stable_seed(recipe['private_seed'],spec.identity.token),state_key=spec.identity.state_key)
            if PrivateIdentity.from_private(module,state_sha256=spec.identity.state_sha256)!=spec.identity:
                raise ValueError('loaded checkpoint layout differs from pinned private identity')
            slot.stage(module.state_dict(),state_key=spec.identity.state_key)
            optimizer.load_state_dict(copy.deepcopy(fresh_optimizer[domain]));return 0
        def checkpoint(updates):
            nonlocal last_commit_seconds
            begin=time.perf_counter()
            generation=store.commit(shared_state={'model':actor.state_dict(),'optimizer':shared_optimizer.state_dict()},
                rng_state=rng_state(device),sampler_state=dict(next_global_batch=update,shared_optimizer_steps=shared_optimizer_steps,duals={d:v.state_dict() for d,v in duals.items()},
                    compute=dict(contexts.compute),visits=visits,private_written_bytes=private_written_bytes,
                    successful_full_sweeps=streak,last_full_sweep_update=last_sweep,stop_reason=reason,
                    heldout_completed=heldout_completed,
                    **({'pre_update_readout_completed':pre_update_readout_completed} if recipe.get('pre_update_fitting_readout_at') is not None else {})),private_updates=updates)
            last_commit_seconds=time.perf_counter()-begin
            return generation
        if resume:
            saved=store.restore();actor.load_state_dict(saved['shared']['model']);shared_optimizer.load_state_dict(saved['shared']['optimizer'])
            state=saved['sampler'];update=state['next_global_batch']
            # Legacy checkpoints called shared Adam once for every global batch.
            shared_optimizer_steps=state.get('shared_optimizer_steps',update)
            if type(shared_optimizer_steps) is not int or not 0<=shared_optimizer_steps<=update:
                raise ValueError('invalid actual shared optimizer step counter')
            visits=state['visits'];private_written_bytes=state['private_written_bytes']
            contexts.compute=dict(state['compute']);streak=state['successful_full_sweeps'];last_sweep=state['last_full_sweep_update'];reason=state['stop_reason']
            heldout_completed=state['heldout_completed']
            pre_update_readout_completed=state.get('pre_update_readout_completed')
            for domain in DOMAINS:duals[domain].load_state_dict(state['duals'][domain])
            restore_rng(saved['rng'],device)
            if reason=='review_boundary' and update<recipe['stopping']['max_updates']:
                reason=None
        else:checkpoint([])
        def fitting_readout():
            rows=[]
            for domain in DOMAINS:
                for spec in schedule.pools[domain]:
                    context=load_context(contexts,spec,gpt_sequences=recipe['readout']['gpt_sequences'],cifar_images=recipe['readout']['cifar_images'])
                    count=stage(spec,context)
                    row=measure_context(contexts,context,actor,slots[domain].private,configs[domain],duals[domain],learning=False)
                    rows.append(dict(row,private_visits=count,state_sha256=spec.identity.state_sha256))
            rows=gather_rows(rows)
            passed=fitting_pass(rows,specs['train'],recipe['stopping']['fitting_criteria'])
            if rank==0:immutable_save(output/f'full-fitting-{update:06d}-{uuid.uuid4().hex}.pt',dict(rows=rows,uniform_checkpoint_means=True))
            summary={d:dict(states=sum(r['domain']==d for r in rows),
                mean_kl=sum(r['mean_kl'] for r in rows if r['domain']==d)/len(specs['train'][d]),
                mean_group_l2_percentage_reduction=sum(r['mean_group_l2_percentage_reduction'] for r in rows if r['domain']==d)/len(specs['train'][d])) for d in DOMAINS}
            log('full_fitting_sweep',summary=summary,all_fitting_criteria_pass=passed)
            return passed
        def maybe_full_sweep():
            nonlocal streak,last_sweep,reason
            if (update>last_sweep and update>0 and
                    (update%recipe['stopping']['full_sweep_every']==0 or update==recipe['stopping']['max_updates'])):
                passed=fitting_readout();streak=streak+1 if passed else 0;last_sweep=update
                if recipe['stopping']['stop_when_all_fit'] and streak>=recipe['stopping']['consecutive_full_sweeps']:reason='declared_fitting_criteria'
                review_updates={n*recipe['stopping']['full_sweep_every'] for n in recipe['stopping'].get('review_after_full_sweeps',[])}
                if reason is None and update in review_updates:reason='review_boundary'
                checkpoint([])
        log('start',world_size=world,counts={d:len(specs['train'][d]) for d in DOMAINS},owner_pool_sizes=[len(schedule.pools[d]) for d in DOMAINS],
            shared_initialization=initialization,private_lr_tick_schedule=cfg.get('tick_schedule'),
            shared_weighting='0.5*owner_pool_size/domain_total per visit',coverage_batches=schedule.coverage_batches,
            catalog_fitting_counts=catalog_counts,qualification_only=recipe.get('qualification_only',False),
            actor_backend=recipe['actor_backend'],actor_graph_layouts=len(dispatch.graphs),
            actor_capture_trainability=list(dispatch.shared_trainability) if dispatch.graphs else None,
            experimental_heads_only_graph=recipe.get('experimental_heads_only_graph',False),actor_capture_warmups_per_layout=2 if dispatch.graphs else 0,
            private_snapshot_backend=recipe.get('private_snapshot_backend','reference'),
            verification_lookahead=recipe.get('verification_lookahead','off'),
            private_actor_backend=recipe.get('private_actor_backend','eager'),
            private_actor_graph_layouts=len(private_dispatch.graphs) if private_dispatch else 0,
            private_inner_steps=recipe.get('private_inner_steps',0),
            private_adaptation='frozen shared parameters; explicit support/joint TRAIN panels; no differentiation through inner updates',
            shared_optimizer_step_enabled=shared_step_enabled,
            counter_semantics='update/global_batch count data batches; shared_optimizer_steps counts actual Adam.step calls',
            execution=('heads-only shared and private fitting; body/lifts and their Adam states frozen' if shared_policy=='heads_only' else 'joint offline fitting') if shared_step_enabled else 'private-only fitting with frozen shared parameters and shared Adam')
        initial_readout=recipe.get('pre_update_fitting_readout_at')
        if initial_readout is not None and pre_update_readout_completed!=initial_readout:
            if update!=initial_readout:
                raise ValueError('unmeasured precision-stage initial batch differs from its pin')
            # This diagnostic is not a convergence decision. Preserve sampling
            # RNG; count actual evaluation work and persist completion before fit.
            initial_rng=rng_state(device)
            try:fitting_readout()
            finally:restore_rng(initial_rng,device)
            pre_update_readout_completed=initial_readout
            log('pre_update_fitting_readout_completed',pinned_batch=initial_readout,used_for_stopping=False)
            checkpoint([])
        # Recover a crash after the joint commit but before its scheduled readout.
        maybe_full_sweep()
        while update<recipe['stopping']['max_updates'] and reason is None:
            began=time.perf_counter();shared_optimizer.zero_grad(set_to_none=True);local_rows=[];pending=[]
            for domain in DOMAINS:
                spec,sampling=schedule.visit(update,domain)
                verification_wait=verification.wait(update,spec.local_id) if domain=='gpt' else 0.
                count=visits.get(spec.identity.token,0)
                source_panels=dict(recipe['panels']);split=recipe.get('private_inner_panels')
                if split is not None:
                    source_panels['gpt_sequences']=split['gpt']['source_size'];source_panels['cifar_images']=split['cifar10']['source_size']
                offset=panel_offset(contexts,spec,count,source_panels)
                phase=time.perf_counter()
                context=load_context(contexts,spec,block_offset=offset,gpt_sequences=source_panels['gpt_sequences'],cifar_images=source_panels['cifar_images'])
                if device.type=='cuda':torch.cuda.synchronize(device)
                load_seconds=time.perf_counter()-phase;phase=time.perf_counter()
                restored_count=stage(spec,context)
                if device.type=='cuda':torch.cuda.synchronize(device)
                stage_seconds=time.perf_counter()-phase;phase=time.perf_counter()
                if restored_count!=count:raise ValueError('private Adam visit/sampler counter mismatch')
                inner_contexts=None
                if split is not None:
                    from rewriter.coordinate_fitting_panels import split_fitting_panels
                    inner_contexts,context=split_fitting_panels(context,inner_steps=recipe['private_inner_steps'],inner_size=split[domain]['inner_size'],
                        joint_size=recipe['panels']['gpt_sequences' if domain=='gpt' else 'cifar_images'],seed=stable_seed(recipe['panels']['seed'],spec.identity.token,count))
                inner_rows=adapt_private(contexts,context,actor,slots[domain].private,optimizers[domain],
                    configs[domain],duals[domain],steps=recipe.get('private_inner_steps',0),clip=cfg['clip'],
                    completed_steps=count,optimizer_config=cfg,inner_contexts=inner_contexts,actor_dispatch=private_dispatch)
                if device.type=='cuda':torch.cuda.synchronize(device)
                inner_seconds=time.perf_counter()-phase;phase=time.perf_counter()
                active=dispatch.for_visit(domain,context,slots[domain].private)
                private_lr=set_private_lr(optimizers[domain],cfg,count+len(inner_rows))
                row=measure_context(contexts,context,active,slots[domain].private,configs[domain],duals[domain],learning=True,weight=sampling['state_weight'])
                if device.type=='cuda':torch.cuda.synchronize(device)
                measure_seconds=time.perf_counter()-phase;phase=time.perf_counter()
                slots[domain].unweight_gradients(sampling['state_weight'])
                grad=float(clip_private_gradients(slots[domain].private_parameters,cfg['clip']))
                optimizers[domain].step();optimizers[domain].zero_grad(set_to_none=True)
                if device.type=='cuda':torch.cuda.synchronize(device)
                adam_seconds=time.perf_counter()-phase;phase=time.perf_counter()
                private_steps=count+len(inner_rows)+1
                visits[spec.identity.token]=private_steps
                version=store.snapshot_slot(spec.identity,slots[domain],optimizers[domain],visits=private_steps)
                private_written_bytes+=version['bytes'];pending.append(version)
                local_rows.append(dict(row,**sampling,owner_rank=rank,private_visit=private_steps,private_preclip_gradient_norm=grad,
                                       private_lr=private_lr,
                                       private_updates_this_visit=len(inner_rows)+1,private_inner_seconds=inner_seconds,
                                       private_inner_rows=inner_rows,
                                       context_load_seconds=load_seconds,private_stage_seconds=stage_seconds,
                                       verification_wait_seconds=verification_wait,
                                       finite_measure_seconds=measure_seconds,private_adam_seconds=adam_seconds,
                                       private_snapshot_seconds=time.perf_counter()-phase,private_snapshot_bytes=version['bytes'],
                                       panel_block_offset=offset,state_sha256=spec.identity.state_sha256))
            if any(p.grad is not None for p in actor.parameters() if not p.requires_grad):
                raise RuntimeError('frozen shared parameter unexpectedly accumulated gradients')
            if shared_step_enabled:
                sum_shared_gradients(p for p in actor.parameters() if p.requires_grad)
                clip_mode='original_joint' if recipe['shared_optimizer']['clip_scope']=='joint' else 'separate_representation'
                clipping=clip_shared_continuation(shared_optimizer,clip_mode,max_norm=recipe['shared_optimizer']['clip'])
                shared_optimizer.step();shared_optimizer_steps+=1
            else:
                if any(p.grad is not None for p in actor.parameters()):
                    raise RuntimeError('frozen shared actor unexpectedly accumulated gradients')
                clipping={}
            shared_optimizer.zero_grad(set_to_none=True)
            for domain in DOMAINS:
                row=next(r for r in local_rows if r['domain']==domain)
                duals[domain].step([(row['local_id'],row['mean_kl'])],expected_ids=[row['local_id']])
            update+=1;checkpoint(pending)
            rows=gather_rows(local_rows)
            log('joint_update' if shared_step_enabled else 'private_only_batch',shared_optimizer_step_performed=shared_step_enabled,rows=rows,shared_preclip_gradient_norms=clipping,wall_seconds=time.perf_counter()-began,
                commit_seconds_by_rank=gather_rows([dict(rank=rank,seconds=last_commit_seconds)]),
                graph_replays=dispatch.graph_replays,eager_mask_fallbacks=dispatch.eager_fallbacks,
                private_graph_replays=private_dispatch.replays if private_dispatch else 0,
                private_eager_mask_fallbacks=private_dispatch.eager_fallbacks if private_dispatch else 0)
            if after_commit is not None:after_commit(update)
            maybe_full_sweep()
        if reason is None:reason='declared_budget_review_not_convergence'
        # Held-out adaptation/query is deliberately separate and cannot stop or
        # change this fitted shared/private optimizer trajectory.
        if reason!='review_boundary' and recipe['readout']['heldout']['enabled'] and not heldout_completed:
            heldout_readout(recipe,contexts,actor,specs,configs,device,output,update,log)
            heldout_completed=True
        checkpoint([])
        log('finished',reason=reason,full_sweep_update=last_sweep,successful_full_sweeps=streak,
            per_rank=gather_rows([dict(rank=rank,compute=contexts.compute,private_written_bytes=private_written_bytes,
                                      visited_states=len(visits),total_private_visits=sum(visits.values()))]))
    if wb:wb.finish()
    if log_file:log_file.close()
    return dict(update=update,global_batch=update,shared_optimizer_steps=shared_optimizer_steps,reason=reason,actor=actor.state_dict(),visits=visits)


def heldout_readout(recipe,contexts,actor,specs,configs,device,output,update,log):
    protocol=recipe['readout']['heldout'];rank=dist.get_rank();world=dist.get_world_size()
    before=_digest(actor.state_dict());saved_rng=rng_state(device)
    flags=[p.requires_grad for p in actor.parameters()]
    actor.zero_grad(set_to_none=True);actor.requires_grad_(False);rows=[]
    try:
        for domain in DOMAINS:
            allowed={s.local_id:s for s in specs['audit'][domain]}
            if len(set(protocol['ids'][domain]))!=len(protocol['ids'][domain]) or not set(protocol['ids'][domain])<=allowed.keys():
                raise ValueError('held-out protocol IDs are not wholly held-out states')
            for uid in protocol['ids'][domain][rank::world]:
                spec=allowed[uid]
                support=load_context(contexts,spec,purpose='private_support',shared_actor=actor)
                private=actor.make_private(support['snapshot'],seed=stable_seed(recipe['private_seed'],'heldout',spec.identity.token),state_key=(*spec.identity.state_key,'heldout'))
                cfg=recipe['private_optimizer'];optimizer=torch.optim.Adam(private.parameters(),lr=cfg['lr'],eps=cfg['eps'],betas=tuple(cfg['betas']))
                dual=TokenwiseDual([uid],config=configs[domain],dual_lr=recipe['dual_lr']);support_ids=set(support['ids']);queries=[]
                for visit in range(protocol['support_steps']+1):
                    if visit in (0,protocol['support_steps']):
                        query=load_context(contexts,spec,role='audit',purpose='private_query',shared_actor=actor,gpt_sequences=protocol['query_gpt_sequences'])
                        if support_ids & set(query['ids']):raise ValueError('held-out support/query contexts overlap')
                        measured=measure_context(contexts,query,actor,private,configs[domain],dual,learning=False)
                        queries.append(dict(measured,support_steps=visit));rows.append(queries[-1])
                    if visit==protocol['support_steps']:break
                    support=load_context(contexts,spec,purpose='private_support',shared_actor=actor)
                    set_private_lr(optimizer,cfg,visit)
                    optimizer.zero_grad(set_to_none=True)
                    measured=measure_context(contexts,support,actor,private,configs[domain],dual,learning=True,weight=1.)
                    if any(p.grad is not None for p in actor.parameters()):raise RuntimeError('held-out support changed shared gradient')
                    clip_private_gradients(private.parameters(),cfg['clip']);optimizer.step()
                    dual.step([(uid,measured['mean_kl'])],expected_ids=[uid])
                immutable_save(output/f'heldout-{domain}-{uid}-{update}-{uuid.uuid4().hex}.pt',
                    dict(private=private.state_dict(),optimizer=optimizer.state_dict(),dual=dual.state_dict(),query=queries,
                         support_ids=sorted(support_ids),state_sha256=spec.identity.state_sha256))
        if _digest(actor.state_dict())!=before:raise RuntimeError('held-out adaptation changed shared parameters')
        log('heldout_readout',rows=gather_rows(rows),shared_unchanged=True,used_for_stopping=False)
    finally:
        for parameter,flag in zip(actor.parameters(),flags):parameter.requires_grad_(flag)
        actor.zero_grad(set_to_none=True);restore_rng(saved_rng,device)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recipe',required=True);parser.add_argument('--output',required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--qualification-interrupt-after',type=int)
    args=parser.parse_args();recipe=json.loads(Path(args.recipe).read_text());validate_recipe(recipe)
    if args.qualification_interrupt_after is not None and (not recipe.get('qualification_only') or args.resume or not 0<args.qualification_interrupt_after<recipe['stopping']['max_updates']):
        raise ValueError('interruption injection is restricted to an explicit fresh qualification')
    from baseline.execution import enable_deterministic_execution
    from rewriter.gpt_precision import configure_precision
    enable_deterministic_execution();torch.set_num_threads(4)
    local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local);device=torch.device('cuda',local);configure_precision('fp32')
    dist.init_process_group('nccl')
    try:
        base=json.loads(Path(recipe['base_recipe']).read_text())
        if file_sha(recipe['base_recipe'])!=recipe['base_recipe_sha256']:raise ValueError('base data recipe hash differs')
        if file_sha(base['mixed_index'])!=recipe['mixed_index_sha256']:raise ValueError('mixed data index hash differs')
        contexts=FixedContexts(base,device,gpt_sequences=recipe['panels']['gpt_sequences'],cifar_images=recipe['panels']['cifar_images'],
                               action_family=recipe.get('action_family','fixed_anchored_v1'))
        if contexts.catalog.weights!=recipe['domain_weights']:raise ValueError('catalog domain weights differ')
        root=Path(__file__).resolve().parents[1]
        files=[p for folder in ('rewriter','baseline') for p in (root/folder).rglob('*.py')]
        files += [p for p in (root/'baseline/vendor').rglob('*') if p.is_file() and p.suffix!='.py' and '__pycache__' not in p.parts]
        files += [Path(__file__),root/'scripts/qualify_coordinate_chisel.py',root/'scripts/train_multistate_chisel.py',root/'scripts/capture_cifar_chisel_bank.py']
        metadata=dict(recipe=recipe,sources={str(p.relative_to(root)):file_sha(p) for p in sorted(files)},world_size=dist.get_world_size(),
                      mode='offline_joint_shared_private_fit',private_identity='domain,seed,localstep,sourceSHA,layoutSHA',
                      weighting='pervisit 0.5*owner_pool_size/domain_total; fullfittingreadouts uniform percheckpoint')
        source_bytes={str(p.relative_to(root)):p.read_bytes() for p in files} if dist.get_rank()==0 else None
        def interrupt(update):
            if update==args.qualification_interrupt_after:
                marker=Path(args.output)/f'qualification-interrupted-rank{dist.get_rank()}.json'
                marker.write_text(json.dumps(dict(update=update,after_atomic_joint_commit=True))+'\n')
                dist.barrier()
                raise SystemExit(75)
        result=run_training(recipe,args.output,contexts,device,metadata=metadata,resume=args.resume,source_bytes=source_bytes,
                            after_commit=interrupt if args.qualification_interrupt_after is not None else None)
        return result
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
