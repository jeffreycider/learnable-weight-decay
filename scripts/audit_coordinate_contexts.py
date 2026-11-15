"""Frozen first-zero actor/private readout: verify seen panels, then query once."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.gpt_precision import configure_precision
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.distributed_probe_store import _digest
from rewriter.multibank import file_sha
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from scripts.continue_coordinate_chisel import stage_origin,resolve_actor_architecture,read_parent_readout,parent_readout_differences
from scripts.qualify_coordinate_chisel import FixedContexts


def run(recipe_path,output):
    r=json.loads(Path(recipe_path).read_text());source=Path(r['checkpoint'])
    if file_sha(source)!=r['checkpoint_sha256']:raise ValueError('checkpoint SHA differs')
    saved=torch.load(source,map_location='cpu',weights_only=False)
    if saved['update']!=1280 or file_sha(source)!=r['checkpoint_sha256']:raise ValueError('first-zero checkpoint differs')
    expected=read_parent_readout(r['parent_readout'],1280)
    origin=stage_origin(saved['metadata']);base=json.loads(Path(r['base_recipe']).read_text())
    if file_sha(r['base_recipe'])!=origin['base_recipe_sha256'] or file_sha(base['mixed_index'])!=origin['mixed_index_sha256']:raise ValueError('data changed')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    enable_deterministic_execution();torch.cuda.set_device(0);torch.set_num_threads(4);configure_precision('fp32');device=torch.device('cuda',0)
    _,architecture=resolve_actor_architecture({},saved['metadata']);architecture.pop('name');architecture.pop('decoder_scale')
    actor=CoordinateChisel(**architecture).to(device);actor.load_state_dict(saved['actor']);actor.requires_grad_(False)
    contexts=FixedContexts(base,device,gpt_sequences=16,cifar_images=1024)
    configs={d:TokenwiseConfig(**v['config']) for d,v in saved['duals'].items()}
    private={};duals={}
    for d in configs:
        duals[d]=TokenwiseDual([k[1] for k in saved['private'] if k[0]==d],config=configs[d],dual_lr=saved['duals'][d]['dual_lr'])
        duals[d].load_state_dict(saved['duals'][d])
    for key,payload in saved['private'].items():
        ctx=contexts.load(*key);p=actor.make_private(ctx['snapshot'],state_key=payload['parameters']['_extra_state']['state_key'],seed=0)
        p.load_state_dict(payload['parameters']);p.requires_grad_(False);private[key]=p
    before=_digest({'actor':actor.state_dict(),'private':{k:p.state_dict() for k,p in private.items()},'duals':{d:v.state_dict() for d,v in duals.items()}})
    torch.set_rng_state(saved['rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    root=Path(__file__).resolve().parents[1]
    sources={str(p.relative_to(root)):file_sha(p) for folder in ('rewriter','baseline') for p in (root/folder).rglob('*.py')}
    sources['scripts/audit_coordinate_contexts.py']=file_sha(__file__)
    metadata=dict(recipe=r,sources=sources,scope='Frozen trained models and private probes; fitting verification then previously unused contexts. No optimizer or dual updates; no selection.')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    import wandb
    wb=wandb.init(project='chisel',mode='offline',dir=str(output),name=r['name'],config=metadata)
    (output/'wandb-path.json').write_text(json.dumps(dict(path=str(Path(wb.dir).parent),id=wb.id)))
    rows={};started=time.perf_counter()
    for role in ('train','audit'):
        rr=[]
        for (domain,uid),p in private.items():
            ctx=contexts.load(domain,uid,role=role,gpt_sequences=16 if role=='train' else 64)
            if role=='audit':
                previous=next(x['context_ids'] for x in rows['train'] if x['domain']==domain and x['local_id']==uid)
                if set(previous)&set(ctx['ids']):raise ValueError('seen and query panels overlap')
            row=contexts.measure(ctx,actor,p,configs[domain],duals[domain],learning=False);rr.append(row)
            wb.log({role+'/'+domain+'/'+str(uid)+'/'+k:v for k,v in row.items() if isinstance(v,(int,float))})
        rows[role]=rr
        if role=='train':
            differences=parent_readout_differences(rr,expected)
            if differences:
                (output/'seen_identity_failure.json').write_text(json.dumps(differences,indent=2));raise RuntimeError('seen readout does not exactly reproduce parent')
        (output/(role+'.json')).write_text(json.dumps(rr,indent=2))
    after=_digest({'actor':actor.state_dict(),'private':{k:p.state_dict() for k,p in private.items()},'duals':{d:v.state_dict() for d,v in duals.items()}})
    if before!=after or any(p.grad is not None for p in actor.parameters()) or any(p.grad is not None for m in private.values() for p in m.parameters()):raise RuntimeError('frozen audit changed state or generated parameter gradients')
    summary=dict(rows=rows,seen_exact=True,all_state_unchanged=True,optimizer_updates=0,dual_updates=0,compute=contexts.compute,elapsed_seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));wb.finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.recipe,a.output)
