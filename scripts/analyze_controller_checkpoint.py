"""Read-only controller analysis on cached observations; never evaluates GPT.

--extract writes a small standalone actor/snapshot artifact from a same-run
checkpoint. It is for analysis only, never an initialization for another trial.
Perturbations below test learned global dependence, not counterfactual GPT loss.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rewriter.controller import GlobalRewriter


def extract(path):
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if 'actor_state' in state:
        return state
    ctrl = state['controller']
    if ctrl is None or ctrl['pending'] is None:
        raise ValueError('need a nonterminal learned checkpoint with pending action-time inputs')
    return dict(actor_state=ctrl['controller'], snapshot=ctrl['pending']['snapshot'],
                config=state['config'], action_step=state['step']-1,
                meta_updates=ctrl['updates'], origin_checkpoint=str(path))


def build(config):
    c=config['rewriter']; rank=c.get('probe_rank',8)
    return GlobalRewriter(rank=rank,trajectory_dim=(2*config['model']['num_layers']+1)*2*rank,
        coordinate_width=c.get('coordinate_width',64),global_width=c.get('global_width',128),
        pool_queries=c.get('pool_queries',4),latents=c.get('global_latents',64),
        blocks=c.get('global_blocks',3),heads=c.get('attention_heads',4),ff_width=c.get('ff_width',512))


@torch.no_grad()
def analyze(state):
    torch.set_num_threads(2)
    model=build(state['config']).train()  # dropout is identically zero; retain training operator path
    model.load_state_dict(state['actor_state'])
    snapshot=state['snapshot']; original=model(snapshot)
    rank=model.rank
    sources=[n for n in ['embed.weight','blocks.0.attn.q.weight','blocks.6.mlp.fc.weight','proj.weight']
             if n in snapshot.matrices]
    experiments=[]
    for source in sources:
        obs=snapshot.matrices[source]
        # An explicit intervention on normalized weight-probe inputs only.
        # It is not asserted to describe a physically consistent alternative GPT.
        rows=obs.rows.clone(); columns=obs.columns.clone()
        rows[:,:rank] *= 1.1; columns[:,:rank] *= 1.1
        matrices=dict(snapshot.matrices)
        matrices[source]=replace(obs,rows=rows,columns=columns)
        modified=model(replace(snapshot,matrices=matrices))
        changes={}
        for name, before in original.matrices.items():
            if name==source: continue
            after=modified.matrices[name]
            delta=((after.P-before.P).square().sum()+(after.Q-before.Q).square().sum()).sqrt()
            changes[name]=float(delta/before.norm_sq().sqrt().clamp_min(1e-12))
        values=torch.tensor(list(changes.values()))
        experiments.append(dict(source=source,intervention='multiply normalized W probes by1.1; all other inputs fixed',
            other_matrix_relative_change_median=float(values.median()),
            other_matrix_relative_change_max=float(values.max()),
            nonzero_other_matrices=int((values>0).sum()),per_matrix=changes))
    broad=[]
    for target in sources:
        # Keep the target's local observation identical, changing only inputs
        # elsewhere. This distinguishes global influence from a local decoder.
        matrices={}
        for name, obs in snapshot.matrices.items():
            if name == target:
                matrices[name]=obs
            else:
                rows=obs.rows.clone(); columns=obs.columns.clone()
                rows[:,:rank]=0; columns[:,:rank]=0
                matrices[name]=replace(obs,rows=rows,columns=columns)
        action=model(replace(snapshot,matrices=matrices))
        before=original.matrices[target]; after=action.matrices[target]
        delta=((after.P-before.P).square().sum()+(after.Q-before.Q).square().sum()).sqrt()
        broad.append(dict(target=target,
            intervention='zero normalized W probes in all other matrices; target local observation unchanged',
            relative_change=float(delta/before.norm_sq().sqrt().clamp_min(1e-12))))
    return dict(action_step=state['action_step'],meta_updates=state['meta_updates'],
        controller_parameters=sum(p.numel() for p in model.parameters()),
        raw_action_norm=float(original.norm_sq().sqrt()),
        raw_predicted_squared_norm_change=float(original.norm_change(snapshot.weights,snapshot.vector_weights)),
        perturbations=experiments, broad_context_interventions=broad,
        interpretation='Raw actor dependence only; no guard contribution and no additional GPT evaluation or function-preservation label.')


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--extract');p.add_argument('--output')
    args=p.parse_args();state=extract(args.checkpoint)
    if args.extract:
        torch.save(state,args.extract)
        print(json.dumps(dict(extracted=args.extract,action_step=state['action_step'])))
    if args.output:
        result=analyze(state);Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='perturbations'}))
    if not args.extract and not args.output: p.error('specify --extract or --output')

if __name__=='__main__': main()
