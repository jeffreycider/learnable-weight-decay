"""Fitting-only norm/preservation parameter gradient balance, no optimizer steps.

One full model backward per state plus one cached-algebra norm-only actor VJP.
Preservation is derived as total minus norm; no second GPT/CNN backward.
"""
from pathlib import Path
import argparse,copy,json,sys,time
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.gpt_precision import configure_precision
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.coordinate_optimization import configure_shared_continuation,clip_shared_continuation,clip_private_gradients
from rewriter.meta_optimizer import build_meta_adam
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from rewriter.multibank import file_sha
from scripts.qualify_coordinate_chisel import FixedContexts
from scripts.train_multistate_chisel import build_proposal


def gradients(module,scale=1.):
    return {n:(torch.zeros_like(p) if p.grad is None else p.grad.detach()/scale).cpu().clone() for n,p in module.named_parameters()}


def component_stats(norm,total,names,clip_factor=1.):
    names=list(names)
    if not names:return None
    a=torch.cat([norm[n].double().flatten() for n in names]);t=torch.cat([total[n].double().flatten() for n in names]);b=t-a
    an=float(a.norm());bn=float(b.norm());tn=float(t.norm());dot=float(a@b)
    return dict(norm_reward_gradient_norm=an,preservation_gradient_norm=bn,total_gradient_norm=tn,preservation_over_norm=bn/an if an else None,cosine=dot/(an*bn) if an*bn else None,dot=dot,parameter_elements=a.numel(),combined_clip_factor=clip_factor,norm_reward_after_combined_clip=an*clip_factor,preservation_after_combined_clip=bn*clip_factor)


def families(norm,total,owner,clip_factors=None):
    clip_factors=clip_factors or {};names=list(total)
    if owner=='shared':
        groups={'heads':[n for n in names if n.startswith(('matrix_output.','vector_output.'))], 'body':[n for n in names if n.startswith('blocks.')], 'other_representation':[n for n in names if not n.startswith(('matrix_output.','vector_output.','blocks.'))], 'all':names}
        factors={n:clip_factors.get('action_heads' if n=='heads' else 'representation',1.) for n in groups};factors['all']=1.
    else:
        groups={'encoder':[n for n in names if n.endswith('.encoder')], 'decoder':[n for n in names if n.endswith('.decoder')], 'trajectory':[n for n in names if n.startswith('trajectory_input.')], 'all':names};factors={n:clip_factors.get('private',1.) for n in groups}
    return {name:component_stats(norm,total,selected,factors[name]) for name,selected in groups.items()}


def original_metadata(metadata):
    while 'protocol' not in metadata:
        if 'parent_metadata' not in metadata:raise ValueError('missing original coordinate protocol')
        metadata=metadata['parent_metadata']
    return metadata


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','base-recipe','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--ranking-weight',type=float,default=.1);a=p.parse_args()
    if file_sha(a.checkpoint)!=a.checkpoint_sha256:raise ValueError('fixture SHA differs')
    a.output.mkdir(parents=True,exist_ok=False);enable_deterministic_execution();torch.cuda.set_device(0);torch.set_num_threads(1);configure_precision('fp32');device=torch.device('cuda:0')
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);metadata=original_metadata(saved['metadata']);protocol=metadata['protocol'];stage=protocol['joint_pilot'];architecture=dict(protocol['architecture']);architecture.pop('name')
    if architecture.pop('decoder_scale')!='sqrt_coordinates_over_latent_tokens':raise ValueError('decoder scale differs')
    if file_sha(a.base_recipe)!=metadata['base_recipe_sha256']:raise ValueError('base recipe differs')
    actor=CoordinateChisel(**architecture).to(device);actor.load_state_dict(saved['actor']);weight=stage['state_weight']
    contexts=FixedContexts(json.loads(a.base_recipe.read_text()),device,gpt_sequences=16,cifar_images=1024)
    optimizer=build_meta_adam(actor,lr=stage['shared_head_lr'],representation_lr_multiplier=stage['shared_representation_multiplier']);optimizer.load_state_dict(copy.deepcopy(saved['shared_optimizer']));configure_shared_continuation(optimizer,'separate_representation')
    import wandb
    wb=wandb.init(project='chisel',mode='offline',dir=str(a.output),config={'checkpoint_sha256':a.checkpoint_sha256,'ranking_weight':a.ranking_weight,'scope':'Parameter gradient diagnostics only; no updates/query'})
    rows=[];evidence={};shared_norm={};shared_total={};beg=time.perf_counter()
    for key in [('gpt',6),('gpt',499),('cifar10',1),('cifar10',41)]:
        domain,uid=key;c=contexts.load(domain,uid);state=saved['private'][key]['parameters'];private=actor.make_private(c['snapshot'],state_key=state['_extra_state']['state_key'],seed=1);private.load_state_dict(state)
        cfg=replace(TokenwiseConfig(**saved['duals'][domain]['config']),ranking_loss_mode='relative_gap',loss_backend='tiled_relative_fp64_v1',ranking_weight=a.ranking_weight)
        old=saved['duals'][domain];dual=TokenwiseDual(old['multipliers'],config=cfg,dual_lr=old['dual_lr']);dual.multipliers=dict(old['multipliers']);dual.updates=old['updates']
        actor.zero_grad(set_to_none=True);private.zero_grad(set_to_none=True)
        pr=build_proposal(lambda snap:actor(snap,private),c['snapshot'],c['engine'],c['grouped'],radius_fraction=.01,norm_squared=c['norm_squared'],shrinkage_coefficient=1.)
        (pr[1]*weight).backward();original,leaves=pr[4];pairs=[(o,l.grad) for o,l in zip(original,leaves) if o.requires_grad and l.grad is not None]
        if pairs:torch.autograd.backward([o for o,g in pairs],[g for o,g in pairs])
        ns=gradients(actor);np=gradients(private,weight);del pr,pairs,original,leaves
        actor.zero_grad(set_to_none=True);private.zero_grad(set_to_none=True)
        measured=contexts.measure(c,actor,private,cfg,dual,learning=True,weight=weight)
        ts=gradients(actor);tp=gradients(private,weight)
        for parameter in private.parameters():
            if parameter.grad is not None:parameter.grad.div_(weight)
        private_preclip=float(clip_private_gradients(private.parameters(),stage['private_clip']));private_factor=min(1.,stage['private_clip']/(private_preclip+1e-6))
        for n in ts:
            if n not in shared_total:shared_total[n]=ts[n].clone();shared_norm[n]=ns[n].clone()
            else:shared_total[n].add_(ts[n]);shared_norm[n].add_(ns[n])
        row=dict(domain=domain,state=uid,context_ids=c['ids'],mean_kl=measured['mean_kl'],norm_objective=measured['norm_objective'],linear_kl=measured['dual_linear_term'],tail=measured['tail_quadratic_penalty'],ranking=measured['ranking_penalty'],strict_topk_changes=measured['strict_topk_changes'],state_weight=weight,shared_weighted=families(ns,ts,'shared'),private_unweighted=families(np,tp,'private',{'private':private_factor}),private_preclip_norm=private_preclip)
        rows.append(row);evidence[key]=dict(shared_norm=ns,shared_total=ts,private_norm=np,private_total=tp);print(json.dumps({k:v for k,v in row.items() if k!='context_ids'}),flush=True);del c,private
    for n,parameter in actor.named_parameters():parameter.grad=shared_total[n].to(device)
    clipping=clip_shared_continuation(optimizer,'separate_representation',stage['shared_clip']);factors={name:min(1.,stage['shared_clip']/(value+1e-6)) for name,value in clipping.items()}
    shared=families(shared_norm,shared_total,'shared',factors)
    unchanged=all(torch.equal(v.detach().cpu(),saved['actor'][n]) for n,v in actor.state_dict().items())
    if not unchanged:raise RuntimeError('diagnostic mutated shared actor')
    summary=dict(checkpoint_sha256=a.checkpoint_sha256,checkpoint_update=saved['update'],ranking_weight=a.ranking_weight,scope='Immutable saved policy, same fitting GPT6/499 first16, CNN0/40 first1024; one full model backward plus cached-algebra norm actor VJP per state; preservation is total-minus-norm in FP64; no optimizer/query',rows=rows,shared_global_weighted=shared,shared_preclip_norms=clipping,shared_clip_factors=factors,counts=contexts.compute,additional_norm_only_actor_forwards=4,additional_norm_only_actor_backwards=4,optimizer_steps=0,shared_unchanged=unchanged,seconds=time.perf_counter()-beg,peak_bytes=torch.cuda.max_memory_allocated())
    torch.save(evidence,a.output/'gradient_components.pt');(a.output/'summary.json').write_text(json.dumps(summary,indent=2));wb.log({'seconds':summary['seconds'],'body_preservation_over_norm':shared['body']['preservation_over_norm']});wb.finish()
if __name__=='__main__':main()
