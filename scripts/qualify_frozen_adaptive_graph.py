"""Bounded scratch-fork lifecycle gate for frozen shared adaptive graphs."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def prepare(parent,output,source):
    from scripts.prepare_coordinate_backend_fork import metadata_from_archive,identity
    from rewriter.private_probe_store import PrivateProbeStore
    parent=Path(parent);output=Path(output);source=Path(source)
    status=json.loads((parent/'status.json').read_text())
    assert status['status']=='completed' and status['exit_code']==0
    old=json.loads((parent/'recipe.json').read_text());meta=json.loads((parent/'results/metadata.json').read_text())
    assert old['shared_optimizer_step'] is False and old['architecture'].get('factor_precision','fp32')=='fp32'
    assert old['action_family']=='adaptive_ucv_v1' and old['architecture']['fresh_weight_core']
    assert old['private_inner_steps']==8
    with PrivateProbeStore(parent/'results/joint_store/global',read_only=True) as store:
        _,ref=store._manifest_and_reference();saved=store.restore(manifest_ref=ref)
    header=saved['sampler'];assert header['run_identity']==identity(meta) and header['world_size']==1
    state=header['ranks'][0]['sampler']
    assert state['next_global_batch']==state['last_full_sweep_update']==128 and state['stop_reason']=='review_boundary'
    assert state['shared_optimizer_steps']==96 and len(state['visits'])==4 and set(state['visits'].values())=={576}
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    for name in ('reference','candidate','resumed'):
        root=output/name;root.mkdir();(root/'results').mkdir()
        r=copy.deepcopy(old);r.update(actor_backend='eager' if name=='reference' else 'adaptive_actor_graph',private_actor_backend='eager' if name=='reference' else 'adaptive_actor_graph',qualification_only=True)
        r['stopping'].update(max_updates=132,full_sweep_every=132,review_after_full_sweeps=[],stop_when_all_fit=False)
        r['readout']['heldout']['enabled']=False
        r['scope']='Scratch frozen-shared backend qualification128to132 only; no science continuation or query.'
        newmeta=metadata_from_archive(source,r,world_size=1,expected_sha=sha)
        (root/'recipe.json').write_text(json.dumps(r,indent=2));(root/'results/metadata.json').write_text(json.dumps(newmeta,indent=2))
        plan=dict(schema='chisel_distributed_checkpoint_fork_plan_v1',world_size=1,source=str(parent/'results/joint_store'),destination=str(root/'results/joint_store'),parent_manifest_ref=ref,source_run_identity=identity(meta),destination_run_identity=identity(newmeta),provenance=dict(source_code_id=status['source_sha256'],destination_code_id=sha,reason='Bounded same-parent frozen-shared adaptive graph qualifier; preserve all numeric owners, no conversion.'))
        (root/'fork_plan.json').write_text(json.dumps(plan,indent=2))
    (output/'parent_binding.json').write_text(json.dumps(dict(parent=str(parent),manifest_ref=ref,shared_optimizer_steps=96,private_ticks=576,source_sha256=sha),indent=2))


def run(recipe,output,interrupt):
    import scripts.train_coordinate_chisel as native
    original=native.run_training
    def controlled(*args,**kwargs):
        if interrupt:
            def after_commit(batch):
                if batch==interrupt:
                    Path(output,'qualification-interrupted-rank0.json').write_text(json.dumps(dict(update=batch,after_atomic_joint_commit=True)))
                    raise RuntimeError('intentional postcommit qualifier interruption')
            kwargs['after_commit']=after_commit
        return original(*args,**kwargs)
    native.run_training=controlled
    sys.argv=['scripts/train_coordinate_chisel.py','--recipe',recipe,'--output',output,'--resume']
    native.main()


def compare(parent,root):
    import torch
    from scripts.qualify_coordinate_fullbank import read_joint
    from scripts.compare_coordinate_inner_program import numeric_visit
    from rewriter.distributed_probe_store import _digest
    root=Path(root);before=read_joint(Path(parent)/'results');states={n:read_joint(root/n/'results') for n in ('reference','candidate','resumed')}
    rows={n:[r for r in map(json.loads,(root/n/'results/metrics.jsonl').read_text().splitlines()) if r['event']=='private_only_batch'] for n in states}
    gates={}
    for a,b in [('reference','candidate'),('candidate','resumed')]:
        gates[a+'_vs_'+b]=dict(shared_Adam_RNG_exact=_digest(states[a][0])==_digest(states[b][0]),rank_duals_sampler_compute_exact=_digest(states[a][1])==_digest(states[b][1]),private_parameters_Adam_ticks_exact=_digest(states[a][2])==_digest(states[b][2]),all_numeric_private_gradient_loss_strict_rows_exact=_digest([[numeric_visit(v) for v in r['rows']] for r in rows[a]])==_digest([[numeric_visit(v) for v in r['rows']] for r in rows[b]]))
    arms={}
    for n,state in states.items():
        checks=dict(shared_parameters_Adam_frozen=_digest({k:state[0][k] for k in ('model','optimizer')})==_digest({k:before[0][k] for k in ('model','optimizer')}),shared_step_counter_frozen=state[1][0]['shared_optimizer_steps']==96,data_clock_advanced=state[1][0]['next_global_batch']==132,private_ticks_advanced=set(state[1][0]['visits'].values())=={594},four_batches=[r['update'] for r in rows[n]]==[129,130,131,132],no_shared_step_or_gradient=all(not r['shared_optimizer_step_performed'] and not r['shared_preclip_gradient_norms'] for r in rows[n]),private_parameters_changed=any(torch.is_tensor(p['private'][k]) and not torch.equal(p['private'][k],before[2][token]['private'][k]) for token,p in state[2].items() for k in p['private']))
        arms[n]=dict(gates=checks,wall_seconds=[r['wall_seconds'] for r in rows[n]],strict_counts=[sum(v['strict_topk_changes'] for v in r['rows']) for r in rows[n]])
    result=dict(scope='OneGPU FP32-factor adaptive frozen shared, actual closed128 owners;8inner+joint; no query/precision conversion.',comparisons=gates,arms=arms,all_gates=all(all(x.values()) for x in gates.values()) and all(all(x['gates'].values()) for x in arms.values()))
    (root/'summary.json').write_text(json.dumps(result,indent=2));assert result['all_gates'],'preserve failed frozen-shared gate'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','compare']);p.add_argument('--parent');p.add_argument('--output',required=True);p.add_argument('--source');p.add_argument('--recipe');p.add_argument('--interrupt',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.parent,Path(a.output),a.source)
    elif a.mode=='run':run(a.recipe,a.output,a.interrupt)
    else:compare(a.parent,a.output)
