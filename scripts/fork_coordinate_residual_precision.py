"""Explicit CPU-only residual-arithmetic fork, preserving every numeric owner.

First use the standard same-world immutable copy. Only then publish a child
transaction whose shared `_extra_state` declares the new execution mode. The
parent and private payloads are never edited. No model evaluations or updates.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from rewriter.distributed_probe_fork import fork_distributed_checkpoint
from rewriter.distributed_probe_store import DistributedProbeStore,_digest
from scripts.train_coordinate_chisel import build_actor,validate_recipe,shared_update_policy


def transition_shared(shared,parent_recipe,child_recipe):
    for recipe in (parent_recipe,child_recipe):
        validate_recipe(recipe)
        if recipe.get('action_family')!='adaptive_ucv_v1' or shared_update_policy(recipe)!='frozen':
            raise ValueError('residual fork requires frozen adaptive actors')
        if recipe['actor_backend']!='eager' or recipe.get('private_actor_backend','eager')!='eager':
            raise ValueError('residual fork is eager only')
    old_arch=copy.deepcopy(parent_recipe['architecture']);new_arch=copy.deepcopy(child_recipe['architecture'])
    old_precision=old_arch.pop('residual_precision','bf16');new_precision=new_arch.pop('residual_precision','bf16')
    if old_arch!=new_arch or old_precision!='bf16' or new_precision not in ('bf16','fp32'):
        raise ValueError('only BF16 to BF16/FP32 residual arithmetic is authorized')
    for key in ('objectives','shared_optimizer','private_optimizer','panels','private_inner_panels',
                'private_inner_steps','dual_lr','actor_seed','private_seed','sampler_seed','fitting_subset','readout',
                'base_recipe','base_recipe_sha256','mixed_index_sha256','student_backend','qualification_only',
                'domain_weights','private_snapshot_backend','verification_lookahead'):
        if parent_recipe.get(key)!=child_recipe.get(key):raise ValueError('precision fork altered '+key)
    with torch.random.fork_rng(devices=[]):
        old=build_actor(parent_recipe);new=build_actor(child_recipe)
    if shared['model']['_extra_state']!=old.configuration():raise ValueError('parent shared architecture differs')
    old.load_state_dict(shared['model'],strict=True)
    result=copy.deepcopy(shared);result['model']['_extra_state']=new.configuration()
    new.load_state_dict(result['model'],strict=True)
    stripped=copy.deepcopy(result);stripped['model']['_extra_state']=shared['model']['_extra_state']
    if _digest(stripped)!=_digest(shared):raise AssertionError('numeric shared owner changed')
    return result


def execute(plan,parent_recipe,child_recipe):
    if plan['schema']!='chisel_distributed_checkpoint_fork_plan_v1':raise ValueError('explicit fork plan required')
    if dist.get_world_size()!=plan['world_size']:raise ValueError('world size mismatch')
    args={k:v for k,v in plan.items() if k not in ('schema','world_size')}
    fork=fork_distributed_checkpoint(**args)
    with DistributedProbeStore(plan['destination'],run_identity=plan['destination_run_identity']) as store:
        saved=store.restore();versions=copy.deepcopy(store._versions)
        changed=transition_shared(saved['shared'],parent_recipe,child_recipe)
        store.commit(shared_state=changed,rng_state=saved['rng'],sampler_state=saved['sampler'],private_updates=[])
        final=store.restore()
        for key in ('rng','sampler'):
            if _digest(saved[key])!=_digest(final[key]):raise AssertionError(key+' changed')
        if store._versions!=versions:raise AssertionError('private versions changed')
        if _digest(final['shared'])!=_digest(changed):raise AssertionError('shared commit differs')
        report=dict(schema='coordinate_residual_precision_transition_v1',fork=fork,
                    before_shared_digest=_digest(saved['shared']),after_shared_digest=_digest(changed),
                    before_architecture=saved['shared']['model']['_extra_state'],after_architecture=changed['model']['_extra_state'],
                    all_numeric_owners_unchanged=True,private_payloads_unchanged=True,training_steps=0,
                    global_batch=saved['sampler']['next_global_batch'],shared_optimizer_steps=saved['sampler']['shared_optimizer_steps'],
                    destination_manifest_ref=final['global_manifest_ref'])
    if dist.get_rank()==0:
        path=Path(plan['destination'])/'PRECISION_TRANSITION.json'
        with path.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--parent-recipe',required=True);p.add_argument('--child-recipe',required=True);a=p.parse_args()
    torch.set_num_threads(1);dist.init_process_group('gloo')
    try:
        report=execute(json.loads(Path(a.plan).read_text()),json.loads(Path(a.parent_recipe).read_text()),json.loads(Path(a.child_recipe).read_text()))
        if dist.get_rank()==0:print(json.dumps(report))
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
