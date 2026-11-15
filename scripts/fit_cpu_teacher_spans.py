"""Private-frame projection control; analytic best cores, not the learned actor."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from rewriter.prefix_cache import unpack_record
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.edit_scope import editable_names
from scripts.fit_cpu_edit_teacher import make_target,sha,save


def projection_error(u,v,left,right):
    """Stable ||T-UU'TVV'||² for T=left@right.T; thin contractions only."""
    projected=u.T@left
    row_residual=left-u@projected
    column_residual=right-v@(v.T@right)
    return ((row_residual.T@row_residual)*(right.T@right)).sum()+((projected.T@projected)*(column_residual.T@column_residual)).sum()


def vector_projection_error(q,target):
    return (target-q@(q.T@target)).square().sum()


@torch.no_grad()
def frame_diagnostics(frames):
    rows={}
    for name,frame in frames.items():
        raw=frame.raw.detach();singular=torch.linalg.svdvals(raw)
        rows[name]=dict(raw_norm=float(raw.norm()),raw_column_norm_min=float(raw.norm(dim=0).min()),raw_column_norm_max=float(raw.norm(dim=0).max()),
            raw_sigma_min=float(singular[-1]),raw_sigma_max=float(singular[0]),raw_condition=float(singular[0]/singular[-1]))
    return rows


def restore_projection_adam(optimizer,parent,frame_names):
    if not parent.get('diagnostic_control') or parent['metadata'].get('diagnostic_control')!='private_span_projection_only':
        raise ValueError('requires a private-span projection control')
    if 'projection_optimizer' not in parent:raise ValueError('missing retained projection Adam state')
    if 'projection_parameter_names' in parent and parent['projection_parameter_names']!=list(frame_names):raise ValueError('frame optimizer order mismatch')
    optimizer.load_state_dict(parent['projection_optimizer'])
    steps=sorted({int(v['step']) for v in optimizer.state.values()})
    if steps!=[parent['control_updates']]:raise ValueError('projection optimizer counters mismatch')
    return steps


def continuation_schedule(spec,steps):
    schedule=[]
    for item in spec:
        count,lr=item.split(':');count=int(count);lr=float(lr)
        if count<=0 or not math.isfinite(lr) or lr<=0:raise ValueError('invalid continuation schedule')
        schedule.append(dict(updates=count,lr=lr))
    if sum(r['updates'] for r in schedule)!=steps:raise ValueError('schedule must cover declared steps')
    return schedule


def scheduled_rate(schedule,step):
    boundary=0
    for row in schedule:
        boundary+=row['updates']
        if step<boundary:return row['lr']
    return schedule[-1]['lr']


def reached_training_target(global_floor,relative_floor,threshold):
    return threshold is not None and global_floor<=threshold and relative_floor<=threshold


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--polish',action='store_true');p.add_argument('--steps',type=int,default=1000);p.add_argument('--seconds',type=float,default=300.);p.add_argument('--continuation-schedule',nargs='+',help='Retain projection Adam moments; explicit updates:lr segments');p.add_argument('--target-stop',type=float,help='Stop and save current state when both training projection errors meet this threshold');a=p.parse_args()
    if not 1<=a.steps<=1000 or not 0<a.seconds<=300:raise ValueError('bounded control required')
    if a.polish and (a.steps!=1000 or a.seconds>180):raise ValueError('polish requires500+500 steps and at most180seconds')
    if a.continuation_schedule and (a.polish or a.seconds>180):raise ValueError('explicit continuation excludes polish flag and is capped at180seconds')
    if a.target_stop is not None and (not a.continuation_schedule or not math.isfinite(a.target_stop) or a.target_stop<=0):raise ValueError('positive target stop requires explicit continuation')
    continuing=a.polish or bool(a.continuation_schedule)
    schedule=continuation_schedule(a.continuation_schedule,a.steps) if a.continuation_schedule else ([dict(updates=500,lr=1e-4),dict(updates=500,lr=1e-5)] if a.polish else [dict(updates=a.steps,lr=1e-3)])
    threshold=a.target_stop if a.continuation_schedule else (None if a.polish else 1e-10)
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');started=time.perf_counter()
    parent=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True);meta=parent['metadata'];args=meta['args']
    if sha(args['state'])!=meta['state_sha256'] or sha(args['selection'])!=meta['selection_sha256']:raise ValueError('parent identity mismatch')
    state=torch.load(args['state'],map_location='cpu',weights_only=False,mmap=True);snapshot=unpack_record(state['observation'],device='cpu').snapshot
    names=set(snapshot.matrices)|set(snapshot.vectors);params={n:v.detach().float() for n,v in state['model'].items() if n in names}
    grouped=GroupedNormObjective(params,zero_reference_policy='freeze');editable=editable_names(params,'grouped')-grouped.frozen_parameter_names
    target,_=make_target(params,snapshot,json.loads(Path(args['selection']).read_text())['rows'],editable)
    actor=AdaptiveCoordinateChisel(rank=8,trajectory_dim=400,model_dim=256,num_layers=2,latent_tokens=8,fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
    if actor.configuration()!=meta['architecture']:raise ValueError('architecture mismatch')
    private=actor.make_private(snapshot,state_key=('cpu-teaching',0,state['completed_steps'],meta['state_sha256']),seed=args['private_seed']);private.load_state_dict(parent['private']);private.requires_grad_(False);del actor
    groups={n:g for g,ns in grouped.groups.items() for n in ns};references={n:grouped.reference_squared[g].detach() for n,g in groups.items()};matrix=[];vectors=[];frames={};energy={}
    for i,name in enumerate(private.matrix_names):
        if name not in target.matrices:continue
        f=target.matrices[name];e=f.norm_sq().detach()
        if float(e)==0:continue
        u=private.matrix_rows[i];v=private.matrix_columns[i]
        if min(u.raw.shape[0],v.raw.shape[0])<8:raise ValueError('column-orthonormal frames required')
        frames[name+'/rows']=u;frames[name+'/columns']=v;matrix.append((name,u,v,(f.U@f.C).detach(),f.V.detach()));energy[name]=e
    for i,name in enumerate(private.vector_names):
        if name not in target.vectors:continue
        t=target.vectors[name].detach();e=t.square().sum()
        if float(e)==0:continue
        q=private.vector_frames[i];frames[name+'/vector']=q;vectors.append((name,q,t));energy[name]=e
    for frame in frames.values():frame.raw.requires_grad_(True)
    denominator=sum(energy.values());relative_denominator=sum(v/references[n] for n,v in energy.items())
    optimizer=torch.optim.Adam([f.raw for f in frames.values()],lr=1e-3,betas=(.9,.99),eps=1e-12)
    inherited_steps=restore_projection_adam(optimizer,parent,frames) if continuing else []
    start_control_step=parent['control_updates'] if continuing else 0
    initial_frame_stats=frame_diagnostics(frames)
    record=dict(scope=__doc__,parent_checkpoint=str(a.checkpoint),parent_sha256=sha(a.checkpoint),source_sha256=sha(__file__),
        objective='global normalized best-projected-core physical error',adam=dict(lr=schedule[0]['lr'],betas=[.9,.99],eps=1e-12,fresh_moments=not continuing,clipping=False),
        selected_matrices=len(matrix),selected_vectors=len(vectors),trainable_frames=len(frames),threads=2,
        zero_target_outputs='best core/vector exactly zero; their frames fixed, zero contribution',transformer_forwards=0,gpt_forwards=0,gpt_backwards=0,
        max_updates=a.steps,max_training_seconds=a.seconds,parent_actor_unchanged=True,actual_actor_output_fitted=False,
        inherited_optimizer_steps=inherited_steps,start_control_step=start_control_step,
        schedule=schedule,target_stop=threshold,target_stop_scope='both training-target projection errors; inclusive threshold, save current state before any further update',
        projection_parameter_names=list(frames),optimizer_order_contract='matrix inventory rows then columns, followed by vector inventory; original checkpoint lacks explicit names but uses same fixed construction')
    (a.output/'source.py').write_text(Path(__file__).read_text())
    (a.output/'metadata.json').write_text(json.dumps(record,indent=2)+'\n')
    training_started=time.perf_counter();reason='update_cap';initial=None
    with (a.output/'metrics.jsonl').open('w',buffering=1) as stream:
        for step in range(a.steps+1):
            tic=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=scheduled_rate(schedule,step)
            stop=(step==a.steps or time.perf_counter()-training_started>=a.seconds)
            if stop and step<a.steps:reason='time_cap'
            with torch.set_grad_enabled(not stop):
                terms={n:projection_error(u.orthonormal_matrix(),v.orthonormal_matrix(),left,right) for n,u,v,left,right in matrix}
                terms.update({n:vector_projection_error(q.orthonormal_matrix(),t) for n,q,t in vectors})
                loss=sum(terms.values())/denominator;relative=sum(v/references[n] for n,v in terms.items())/relative_denominator
            if not torch.isfinite(loss) or not torch.isfinite(relative):raise FloatingPointError('nonfinite span loss')
            row=dict(step=start_control_step+step,segment_step=step,learning_rate=optimizer.param_groups[0]['lr'],global_floor=float(loss.detach()),relative_floor=float(relative.detach()),by_parameter_error_squared={n:float(v.detach()) for n,v in terms.items()})
            if initial is None:initial=dict(row)
            if reached_training_target(row['global_floor'],row['relative_floor'],threshold):stop=True;reason=('both_training_floors_at_or_below_target' if a.continuation_schedule else 'both_floors_below_1e-10')
            if not stop:
                loss.backward()
                if any(not torch.isfinite(f.raw.grad).all() for f in frames.values()):raise FloatingPointError('nonfinite frame gradient')
                optimizer.step()
            row['compute_seconds']=time.perf_counter()-tic;row['elapsed_training_seconds']=time.perf_counter()-training_started
            stream.write(json.dumps(row)+'\n')
            if step%25==0 or stop:print(json.dumps({k:v for k,v in row.items() if k!='by_parameter_error_squared'}),flush=True)
            if stop:break
    final_frames=frame_diagnostics(frames)
    result=dict(**record,initial=initial,final=row,completed_updates=step,total_control_updates=start_control_step+step,stop_reason=reason,initial_frame_diagnostics=initial_frame_stats,final_frame_diagnostics=final_frames,elapsed_seconds=time.perf_counter()-started,learned_preservation_claim=False)
    outmeta=copy.deepcopy(meta);outmeta.update(diagnostic_control='private_span_projection_only',parent_checkpoint=str(a.checkpoint),parent_sha256=sha(a.checkpoint),control_source_sha256=sha(__file__))
    outmeta['projection_continuation']=record
    outmeta['trainable']='projection-control frames only; actor unchanged; analytic optimal cores not actor output'
    save(a.output/'checkpoint_control.pt',dict(metadata=outmeta,step=parent['step'],control_updates=start_control_step+step,projection_parameter_names=list(frames),actor=parent['actor'],private=private.state_dict(),diagnostic_control=True,optimizer=None,private_optimizer=None,projection_optimizer=optimizer.state_dict(),rng=parent['rng'],projection_stop_row=row,projection_stop_reason=reason))
    (a.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='complete',steps=step,reason=reason,global_floor=row['global_floor'],relative_floor=row['relative_floor'])),flush=True)


if __name__=='__main__':main()
