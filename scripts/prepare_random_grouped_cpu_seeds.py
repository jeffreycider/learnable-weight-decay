"""Fresh zero-action grouped Chisel seeds for a local actual-objective diagnostic.

No fitted actor/private values, imitation targets, optimizer moments, teacher
probabilities or GPT execution are used. Existing state/panel files are hash
verified; checkpoint observations provide the declared architecture inputs only.
"""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
from rewriter.prefix_cache import unpack_record
from rewriter.adaptive_sensing import query_weight_cores
from scripts.fit_cpu_edit_teacher import sha,save,exact_state_equal
from scripts.prepare_cpu_teacher_state import validate_train_panel



def residual_output_spec(actor,mode,seed):
    if mode not in ('zero','scaled_random'):raise ValueError('unknown residual output initialization')
    if type(seed) is not int or not 0<=seed<2**63:raise ValueError('residual seed must be a nonnegative signed 64-bit integer')
    layers=len(actor.blocks)
    if layers<1:raise ValueError('residual initialization requires at least one block')
    weights={f'blocks.{i}.{role}.proj.weight':getattr(block,role).proj.weight
             for i,block in enumerate(actor.blocks) for role in ('attn','mlp')}
    return dict(mode=mode,seed=seed,torch_version=torch.__version__,rng='dedicated CPU torch.Generator; never consumes shared/private/caller streams',
        std_formula='sqrt(0.33 / (fan_in * 2 * num_layers))',num_layers=layers,
        std_by_parameter={name:math.sqrt(.33/(w.shape[1]*2*layers)) for name,w in weights.items()},
        changed_parameters=list(weights) if mode=='scaled_random' else [],biases='unchanged zero',final_heads='unchanged exactly zero')


@torch.no_grad()
def initialize_residual_outputs(actor,mode='zero',seed=960067):
    """Prospective CPU initialization ablation; never changes actor architecture."""
    spec=residual_output_spec(actor,mode,seed)
    parameters=dict(actor.named_parameters())
    for name in spec['std_by_parameter']:
        weight=parameters[name];bias=parameters[name[:-6]+'bias']
        if weight.device.type!='cpu':raise ValueError('residual initializer is CPU-only')
        if torch.count_nonzero(weight) or torch.count_nonzero(bias):raise ValueError('residual initialization requires fresh zero projections and biases')
    assert_zero_heads(actor)
    if mode=='scaled_random':
        generator=torch.Generator(device='cpu').manual_seed(seed)
        for name,std in spec['std_by_parameter'].items():parameters[name].normal_(std=std,generator=generator)
    return spec


def fresh_actor(seed,*,residual_output_init='zero',residual_seed=960067):
    """Explicit local draw stream; caller torch RNG is unchanged."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        actor=GroupedCoordinateChisel(rank=8,trajectory_dim=400,model_dim=256,num_layers=2,
            latent_tokens=8,fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
        draw_state=torch.get_rng_state().clone()
    initialize_residual_outputs(actor,residual_output_init,residual_seed)
    return actor,draw_state


def assert_zero_heads(actor):
    for bank in actor.output_heads:
        for head in bank.values():
            if torch.count_nonzero(head.weight) or torch.count_nonzero(head.bias):raise ValueError('fresh output head is not exactly zero')


@torch.no_grad()
def assert_zero_action(actor,snapshot,private,parameters):
    action=actor(snapshot,private,weight_cores=query_weight_cores(parameters,private))
    if any(torch.count_nonzero(f.C) for f in action.matrices.values()) or any(torch.count_nonzero(v) for v in action.vectors.values()):raise ValueError('fresh actor action is not exactly zero')
    return dict(matrix_outputs=len(action.matrices),vector_outputs=len(action.vectors),zero_action_exact=True)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--shared-seed',type=int,default=940067)
    p.add_argument('--residual-output-init',choices=['zero','scaled_random'],default='zero')
    p.add_argument('--residual-output-seed',type=int,default=960067)
    p.add_argument('--private-seeds',type=int,nargs=2,default=[950006,950007])
    p.add_argument('--data',type=Path,default=Path('data/chisel-teaching-diagnostic'));a=p.parse_args()
    if a.private_seeds[0]==a.private_seeds[1]:raise ValueError('distinct private seeds required')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    caller_rng=torch.get_rng_state().clone();actor,draw_state=fresh_actor(a.shared_seed,residual_output_init=a.residual_output_init,residual_seed=a.residual_output_seed);actor.eval();assert_zero_heads(actor)
    shared={n:v.detach().clone() if isinstance(v,torch.Tensor) else v for n,v in actor.state_dict().items()}
    dependencies=[Path(__file__),Path('rewriter/grouped_coordinate_chisel.py'),Path('rewriter/adaptive_coordinate_chisel.py'),Path('rewriter/coordinate_chisel.py'),Path('rewriter/adaptive_sensing.py'),Path('rewriter/prefix_cache.py'),Path('baseline/model.py')]
    sources={str(p):sha(p) for p in dependencies}
    cases=[];rows=[];previous_private=None
    for uid,private_seed in zip((6,7),a.private_seeds):
        state_path=a.data/f'state_{uid:04d}.pt';panel_path=a.data/f'teacher_state{uid}_train.pt';panel_meta_path=panel_path.with_suffix('.json')
        state_sha=sha(state_path);panel_sha=sha(panel_path);panel_meta=json.loads(panel_meta_path.read_text())
        state=torch.load(state_path,map_location='cpu',weights_only=False,mmap=True);panel=torch.load(panel_path,map_location='cpu',weights_only=False,mmap=True)
        validate_train_panel(panel,panel_meta,state_sha,panel_sha)
        if state['completed_steps']!=uid:raise ValueError('state counter mismatch')
        snapshot=unpack_record(state['observation'],device='cpu').snapshot
        private=actor.make_private(snapshot,state_key=('cpu-teaching',0,uid,state_sha),seed=private_seed)
        names=set(snapshot.matrices)|set(snapshot.vectors);params={n:v.detach().float() for n,v in state['model'].items() if n in names}
        checked=assert_zero_action(actor,snapshot,private,params)
        if not exact_state_equal(actor.state_dict(),shared):raise ValueError('actor changed during seed checks')
        if previous_private is not None:
            if exact_state_equal(previous_private,private.state_dict()):raise ValueError('private probes are not independent')
            if not all(not torch.equal(v,private.state_dict()[n]) for n,v in previous_private.items() if n.endswith('.raw')):raise ValueError('a raw private tensor was reused')
        previous_private=private.state_dict()
        dest=a.output/f'state{uid}';dest.mkdir()
        meta=dict(schema='cpu_random_grouped_seed_v1',scope=__doc__,state_sha256=state_sha,
            architecture=actor.configuration(),args=dict(private_seed=private_seed),
            diagnostic_control='random_shared_zero_heads_fresh_private',source_sha256=sources,
            initialization=dict(residual_outputs=residual_output_spec(actor,a.residual_output_init,a.residual_output_seed),shared_seed=a.shared_seed,shared_rng='isolated torch.random.fork_rng CPU stream',private_seed=private_seed,
                private_state_key=['cpu-teaching',0,uid,state_sha],shared_weights='random internals, exactly zero heads; same actor across both cases',
                private_weights='fresh random independently keyed and seeded',imitation=False,fitted_weight_ancestry=False,inherited_optimizer_moments=False),
            private_identity=private.get_extra_state(),panel_sha256=panel_sha,panel_metadata_sha256=sha(panel_meta_path),
            gpt_forwards=0,gpt_backwards=0,actor_forward_checks=1)
        file=dest/'checkpoint_control.pt'
        save(file,dict(metadata=meta,step=0,actor=actor.state_dict(),private=private.state_dict(),diagnostic_control=True,
            optimizer=None,private_optimizer=None,rng=draw_state))
        (dest/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        cases.append(dict(state=str(state_path),checkpoint=str(file),panel=str(panel_path),panel_metadata=str(panel_meta_path)))
        rows.append(dict(state_id=uid,state_sha256=state_sha,checkpoint_sha256=sha(file),panel_sha256=panel_sha,panel_metadata_sha256=sha(panel_meta_path),private_seed=private_seed,first_two_train_ids=panel_meta['sequence_ids'][:2],**checked))
        del state,panel,params
    if not torch.equal(caller_rng,torch.get_rng_state()):raise ValueError('seed preparation changed caller RNG')
    for path,digest in sources.items():
        if sha(path)!=digest:raise ValueError('source changed during seed preparation')
    (a.output/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    result=dict(scope=__doc__,shared_actor_bit_exact_between_cases=True,caller_rng_unchanged=True,source_identity_unchanged=True,source_sha256=sources,shared_seed=a.shared_seed,residual_outputs=residual_output_spec(actor,a.residual_output_init,a.residual_output_seed),cases=rows,gpt_forwards=0,gpt_backwards=0)
    (a.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'source.py').write_text(Path(__file__).read_text())
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
