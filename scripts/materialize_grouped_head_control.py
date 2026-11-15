"""Explicit diagnostic conversion: unchanged global body, fitted typed heads.

This is a one-checkpoint supervised linear solve, not norm+preservation training
or a reusable pretrained model. Ordinary Adam resume must reject this artifact.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
from rewriter.adaptive_edit_target import edit_target_terms
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.edit_scope import editable_names,editable_weight_norm_sq,scoped_snapshot
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.prefix_cache import unpack_record
from scripts.fit_cpu_edit_teacher import make_target,physical_action,relative_target_error,error_metrics,stored_readout,sha,save,exact_state_equal


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','fit','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gradient-fit',action='store_true',help='Import iterative head-fit endpoint, without requiring exact target reconstruction.')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    parent=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    meta=parent['metadata'];source=meta['args']
    coefficient_path=args.fit/('checkpoint.pt' if args.gradient_fit else 'coefficients.pt')
    fitted=torch.load(coefficient_path,map_location='cpu',weights_only=False)
    report=json.loads((args.fit/'complete.json').read_text())
    if sha(args.checkpoint)!=fitted['source_sha256'][str(args.checkpoint)]:
        raise ValueError('group fit uses a different parent')
    if sha(source['state'])!=meta['state_sha256'] or sha(source['selection'])!=meta['selection_sha256']:
        raise ValueError('target/state identity changed')
    state=torch.load(source['state'],map_location='cpu',weights_only=False,mmap=True)
    snapshot=unpack_record(state['observation'],device='cpu').snapshot
    names=set(snapshot.matrices)|set(snapshot.vectors)
    parameters={n:v.detach().float() for n,v in state['model'].items() if n in names}
    grouped=GroupedNormObjective(parameters,zero_reference_policy='freeze')
    editable=editable_names(parameters,'grouped')-grouped.frozen_parameter_names
    snapshot=scoped_snapshot(snapshot,editable_weight_norm_sq(parameters,editable),meta['radius_fraction'],scope='grouped')
    with torch.random.fork_rng(devices=[]):
        actor=GroupedCoordinateChisel(rank=8,trajectory_dim=400,model_dim=256,num_layers=2,
            latent_tokens=8,fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
        private=actor.make_private(snapshot,state_key=('cpu-teaching',0,state['completed_steps'],meta['state_sha256']),seed=source['private_seed'])
    destination=actor.state_dict()
    def body_key(name):return name!='_extra_state' and not name.startswith(('matrix_output.','vector_output.'))
    oldkeys={n for n in parent['actor'] if body_key(n)}
    if oldkeys!={n for n in destination if body_key(n)}:raise ValueError('shared body inventory changed')
    for n in oldkeys:
        if destination[n].shape!=parent['actor'][n].shape or destination[n].dtype!=parent['actor'][n].dtype:
            raise ValueError('shared body shape/precision mismatch')
        destination[n]=parent['actor'][n]
    actor.load_state_dict(destination,strict=True)
    private.load_state_dict(parent['private'],strict=True)
    if set(fitted['grouped_coefficients'])!=set(grouped.groups):raise ValueError('incomplete group coefficient inventory')
    for name,coeff in fitted['grouped_coefficients'].items():
        head=actor.output_head(name)
        joined=coeff.reshape(head.out_features,head.in_features+1)
        head.weight.copy_(joined[:,:-1]);head.bias.copy_(joined[:,-1])
    actor.eval();actor.requires_grad_(False);private.requires_grad_(False)
    raw=actor(snapshot,private,weight_cores=query_weight_cores(parameters,private))
    action,norm,guard=physical_action(raw,grouped,snapshot)
    if float(guard)!=1.:raise ValueError('unexpected active guard')
    target,_=make_target(parameters,snapshot,json.loads(Path(source['selection']).read_text())['rows'],editable)
    target_squared=target.norm_sq()
    relative_squared=relative_target_error(edit_target_terms(target.scaled(0.),target),grouped)
    metrics=error_metrics(edit_target_terms(action,target),target_squared,relative_squared,grouped)
    # The old design used a norm subtraction for the span floor, so near-zero
    # cancellation is not a valid equality oracle. Check actual core outputs.
    projected_error=0.
    for name,f in action.matrices.items():
        t=target.matrices[name]
        wanted=(f.U.T@t.U)@t.C@(t.V.T@f.V)
        projected_error+=float((f.C-wanted).square().sum())
    if not args.gradient_fit and projected_error>1e-18:raise ValueError('grouped actor failed fitted matrix-core reconstruction')
    if not args.gradient_fit and max(metrics['normalized_edit_error'],metrics['relative_normalized_edit_error'])>1e-18:
        raise ValueError('actual grouped action does not reconstruct the tight target')
    for name in oldkeys:
        if not torch.equal(actor.state_dict()[name],parent['actor'][name]):raise ValueError('body changed during conversion')
    if not exact_state_equal(private.state_dict(),parent['private']):raise ValueError('private state changed during conversion')
    converted=copy.deepcopy(meta)
    converted.update(architecture=actor.configuration(),diagnostic_control=('functional_group_head_gradient' if args.gradient_fit else 'functional_group_head_lstsq'),
        parent_checkpoint=str(args.checkpoint),parent_sha256=sha(args.checkpoint),
        grouped_fit_sha256=sha(coefficient_path),control_source_sha256=sha(__file__),
        head_gradient_updates=report['steps'] if args.gradient_fit else 0,
        shared_head_training_cases=report.get('shared_training_cases') if args.gradient_fit else None,
        trainable='conditional group-head fit; original body/private unchanged; converted artifact has no optimizer state',
        loss='known physical edit target, not learned norm/preservation objective')
    converted['args']=dict(source,output=str(args.output),objective='balanced')
    save(args.output/'checkpoint_control.pt',dict(metadata=converted,step=parent['step'],
        control_updates=parent.get('control_updates'),actor=actor.state_dict(),private=private.state_dict(),
        diagnostic_control=True,optimizer=None,private_optimizer=None,rng=parent['rng']))
    result=dict(scope=__doc__,actual_actor_metrics=metrics,projected_matrix_error_squared=projected_error,
        fitted_design_metrics=report['final'] if args.gradient_fit else report['grouped'],body_private_exactly_unchanged=True,guard=float(guard),
        raw_action_norm=float(norm),gpt_forwards=0,gpt_backwards=0,
        source_sha256={str(f):sha(f) for f in (args.checkpoint,coefficient_path,Path(__file__),
            Path('rewriter/grouped_coordinate_chisel.py'),Path('rewriter/adaptive_coordinate_chisel.py'))})
    (args.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'stored_readout.json').write_text(json.dumps(stored_readout(action,target,parameters,grouped),indent=2)+'\n')
    print(json.dumps(result['actual_actor_metrics']))


if __name__=='__main__':main()
