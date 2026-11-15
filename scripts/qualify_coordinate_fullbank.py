"""Prepare/compare the bounded two-rank staged-driver qualification.

No lease acquisition. The actual trainer is invoked externally on the declared
two GPUs, with one intentional interruption after its eighth atomic update.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.multibank import MultiStateBankReader,file_sha
from rewriter.mixed_domain import MixedBankCatalog
from rewriter.cifar_bank import CifarBankReader
from rewriter.coordinate_chisel import CheckpointProbes
from rewriter.private_probe_store import PrivateIdentity,PrivateProbeStore
from rewriter.distributed_probe_store import private_owner,_digest
from scripts.train_coordinate_chisel import describe_states,validate_recipe


def prepare(base_recipe,objectives,output):
    base=json.loads(Path(base_recipe).read_text())
    catalog=MixedBankCatalog(base['mixed_index'],dict(gpt=lambda p:MultiStateBankReader(p,verify_once=True,mmap_model=True),cifar10=lambda p:CifarBankReader(p.parent)))
    layouts={}
    for domain,reader in catalog.readers.items():
        record=reader.read(max(reader.manifest['split']['train']))
        private=CheckpointProbes(record.snapshot,model_dim=256,latent_tokens=8,seed=820001,state_key=(domain,'template'))
        layouts[domain]=PrivateIdentity.from_private(private,state_sha256='0'*64).layout_sha256
        del record,private
    specs=describe_states(catalog,layouts)['train'];selected={}
    for domain,values in specs.items():
        pools=[[s.local_id for s in values if private_owner(s.identity,2)==r] for r in range(2)]
        if any(len(p)<8 for p in pools):raise ValueError('need eight distinct fitting checkpoints per owner/domain')
        # Include early states (and their zero-reference masks), without moving
        # a state away from its stable identity owner for load balancing.
        selected[domain]=sorted(pools[0][:8]+pools[1][:8])
    recipe=dict(schema='coordinate_chisel_fullbank_v1',qualification_only=True,fitting_subset=selected,
        base_recipe=str(base_recipe),base_recipe_sha256=file_sha(base_recipe),mixed_index_sha256=file_sha(base['mixed_index']),
        domain_weights={'gpt':.5,'cifar10':.5},actor_backend='eager_actor_graph',
        architecture=dict(rank=8,trajectory_dim=400,model_dim=256,num_layers=2,latent_tokens=8),
        actor_seed=810001,private_seed=820001,sampler_seed=840001,
        shared_optimizer=dict(head_lr=1e-7,representation_lr=1e-4,head_eps=1e-8,representation_eps=1e-12,betas=[.9,.99],clip=1.,clip_scope='separate'),
        private_optimizer=dict(lr=1e-4,eps=1e-12,betas=[.9,.99],clip=1.),
        panels=dict(gpt_sequences=16,cifar_images=1024,mode='fixed',seed=400001),objectives=objectives,dual_lr=.01,wandb=True,
        stopping=dict(max_updates=9,full_sweep_every=9,consecutive_full_sweeps=1,stop_when_all_fit=False,fitting_criteria=[]),
        readout=dict(gpt_sequences=16,cifar_images=1024,heldout=dict(enabled=False)))
    validate_recipe(recipe);Path(output).write_text(json.dumps(recipe,indent=2)+'\n')


def read_joint(directory):
    directory=Path(directory)/'joint_store'
    with PrivateProbeStore(directory/'global',read_only=True) as store:state=store.restore()
    private={};rank_states={}
    for row in state['sampler']['ranks']:
        rank=row['rank'];rank_states[rank]=row['sampler']
        with PrivateProbeStore(directory/'private'/f'rank{rank:04d}',read_only=True) as shard:
            index=torch.load(shard._verify(row['index']),map_location='cpu',weights_only=False)
            for token,version in index['private_versions'].items():
                payload=torch.load(shard._verify(version),map_location='cpu',weights_only=False)
                if token in private:raise ValueError('duplicate private identity across ranks')
                private[token]=payload
    return state['shared'],rank_states,private


def compare(full,resumed,output):
    a=read_joint(full);b=read_joint(resumed)
    shared=_digest(a[0])==_digest(b[0]);private=a[2].keys()==b[2].keys() and all(_digest(a[2][k])==_digest(b[2][k]) for k in a[2])
    rank_states=_digest(a[1])==_digest(b[1])
    rows=[json.loads(line) for line in (Path(full)/'metrics.jsonl').read_text().splitlines()]
    updates=[r for r in rows if r['event']=='joint_update'];visits=[v for r in updates for v in r['rows']]
    distinct={(r['domain'],r['local_id']) for r in visits}
    gates=dict(shared_parameters_Adam_and_rank_RNG_bitexact=shared,private_parameters_Adam_clocks_bitexact=private,
        per_rank_sampler_dual_compute_bitexact=rank_states,updates=len(updates)==9,distinct_state_visits=len(distinct)==32 and len(visits)==36,
        restored_private_Adam_used_in_next_update=sum(v['private_visit']==2 for v in visits)==4,
        domain_weighting=all(abs(sum(v['state_weight'] for v in r['rows'] if v['domain']==d)-.5)<1e-12 for r in updates for d in ('gpt','cifar10')))
    summary=dict(gates=gates,scope='Two-rank actual finite-loss staged-store/graph/resume qualification, not a scientific fit',
        phase_seconds_scope='Summed over all state visits on both ranks; not elapsed wall time',
        graph_counter_scope='Rank zero cumulative counters; other rank is recorded through its state-visit rows',
        full=str(full),resumed=str(resumed),private_written_bytes=sum(v['private_snapshot_bytes'] for v in visits),
        wall_seconds=[r['wall_seconds'] for r in updates],
        phase_seconds={k:sum(v[k] for v in visits) for k in ('context_load_seconds','private_stage_seconds','finite_measure_seconds','private_adam_seconds','private_snapshot_seconds')},
        commit_seconds=[r['commit_seconds_by_rank'] for r in updates],
        eager_mask_fallbacks=updates[-1]['eager_mask_fallbacks'],graph_replays=updates[-1]['graph_replays'])
    Path(output).write_text(json.dumps(summary,indent=2)+'\n')
    if not all(gates.values()):raise RuntimeError('bounded staged-driver qualification failed; inspect summary')


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--base-recipe',required=True);q.add_argument('--objective-recipe',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('compare');q.add_argument('--full',required=True);q.add_argument('--resumed',required=True);q.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.base_recipe,json.loads(Path(a.objective_recipe).read_text())['objectives'],a.output)
    else:compare(a.full,a.resumed,a.output)
