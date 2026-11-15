"""Read-only same-logit component cotangents on a frozen actor and cached panel."""
from pathlib import Path
import argparse,hashlib,json,math,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from baseline.model import GPT
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader,head_logits
from rewriter.offline_systems import CachedBaseNormPreservation,VerifiedStateConstants,select_panel_blocks
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.chisel_factory import build_chisel
from rewriter.edit_scope import scoped_snapshot
from rewriter.tiled_teacher_kl import tiled_teacher_kl
from rewriter.tiled_ranking_loss import tiled_topk_margin_loss
from rewriter.token_preservation import token_preservation_terms
from scripts.train_multistate_chisel import build_proposal
from scripts.qualify_graph_offline import record_qualification_source

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('state-bank','teacher-cache','policy','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--rank-arithmetic',choices=('fp32','fp64'),default='fp32');p.add_argument('--policy-sha256',required=True);p.add_argument('--state-id',type=int,default=999);args=p.parse_args()
    assert hashlib.sha256(args.policy.read_bytes()).hexdigest()==args.policy_sha256
    args.output.mkdir(parents=True,exist_ok=False);record_qualification_source(args.output,vars(args));enable_deterministic_execution();torch.set_num_threads(1);device=torch.device('cuda:0')
    import wandb
    run=wandb.init(project='chisel',mode='offline',dir=str(args.output),config={k:str(v) for k,v in vars(args).items()})
    reader=StateBankReader(args.state_bank,verify_once=True,mmap_model=True);cache=TeacherCacheReader(args.teacher_cache,verify_once=True,mmap=True)
    if args.state_id not in reader.manifest['split']['train']:raise ValueError('fitting state required')
    record=reader.read(args.state_id,model_device='cpu',device=device)
    engine=CachedBaseNormPreservation(GPT(**reader.manifest['provenance']['model_config']).to(device).requires_grad_(False),microbatch_sequences=16,teacher_cache_device=device,kl_dtype=torch.float64,gpt_precision='fp32')
    constants=VerifiedStateConstants(engine,AcceleratedGroupedNormObjective);constants.load(cache,record);grouped=constants.grouped();norm=constants.radius_squared(grouped)
    snapshot=scoped_snapshot(record.snapshot,norm,.01,scope='grouped');saved=torch.load(args.policy,map_location='cpu',weights_only=False)
    actor=build_chisel(saved['metadata']['architecture']).to(device);actor.load_state_dict(saved['controller']);forward=torch.compile(actor,fullgraph=True,dynamic=False)
    with torch.enable_grad():proposal=build_proposal(forward,snapshot,engine,grouped,radius_fraction=.01,norm_squared=norm,shrinkage_coefficient=1.)
    edited={n:v.detach() for n,v in proposal[0].items()};stats=proposal[2];del proposal
    panel=select_panel_blocks(cache.read(args.state_id,'train'),range(4));total=panel.inputs.numel();names=('unit_multiplier_linear','tail_rho_0.01','ranking_weight_1')
    sums={n:0. for n in names};dots={f'{a}:{b}':0. for a in names for b in names};kl_rows=[];rank_rows=[];started=time.perf_counter()
    for start in range(0,len(panel.inputs),16):
        stop=min(start+16,len(panel.inputs))
        with torch.no_grad():
            teacher=head_logits(engine,panel.features[start:stop].to(device))
            logits=engine._forward(edited,panel.inputs[start:stop].to(device))
        logits=logits.detach().requires_grad_();kl=tiled_teacher_kl(logits,teacher,reduction='none');rank=tiled_topk_margin_loss(teacher,logits,k=10,margin_fraction=.5,arithmetic=args.rank_arithmetic)
        terms=token_preservation_terms(kl,multiplier=1.,mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-8,rho=.01,ranking_loss=rank,ranking_weight=1.)
        values=dict(zip(names,(terms['dual_linear'],terms['tail_quadratic'],terms['ranking'])))
        gradients={}
        for i,(name,value) in enumerate(values.items()):
            sums[name]+=float(value.detach().double().sum())/total
            gradients[name],=torch.autograd.grad(value.sum()/total,logits,retain_graph=i<len(values)-1)
        for i,a in enumerate(names):
            for b in names[i:]:
                d=float((gradients[a].double()*gradients[b].double()).sum());dots[f'{a}:{b}']+=d
                if a!=b:dots[f'{b}:{a}']+=d
        kl_rows.append(kl.detach().flatten().cpu());rank_rows.append(rank.detach().flatten().cpu())
        del gradients,values,terms,kl,rank,logits,teacher
    norms={n:math.sqrt(max(0,dots[f'{n}:{n}'])) for n in names};cos={k:v/max(1e-300,norms[k.split(':')[0]]*norms[k.split(':')[1]]) for k,v in dots.items()};allkl=torch.cat(kl_rows);allrank=torch.cat(rank_rows)
    result=dict(ranking_arithmetic=args.rank_arithmetic,scope=f'Frozen scheduled policy{args.policy.name}, fitting state{args.state_id} train64;4strictFP32student+4cachedteacherhead forwards; three logit-only cotangents reuse each produced logits;0GPT/actor backwards and0updates',policy_path=str(args.policy),policy_sha256=args.policy_sha256,state_id=args.state_id,state_sha256=record.assert_verified(),sequence_ids=panel.sequence_ids,token_count=total,coefficients=dict(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-8,rho=.01,fresh_stage_multiplier=0.,reported_linear_basis_multiplier=1.,ranking_weight=1.,ranking_margin_fraction=.5),component_mean_losses=sums,logit_cotangent_l2_norms=norms,logit_cotangent_dot_products=dots,logit_cotangent_cosines=cos,mean_kl=float(allkl.mean()),max_token_kl=float(allkl.max()),p99_token_kl=float(torch.quantile(allkl,.99)),ranking_mean=float(allrank.double().mean()),ranking_nonzero_tokens=int((allrank>0).sum()),ranking_max=float(allrank.max()),actor_norm_stats=stats,elapsed_seconds=time.perf_counter()-started,peak_bytes=torch.cuda.max_memory_allocated(),no_weight_choice=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);run.log({**sums,**{'norm_'+k:v for k,v in norms.items()}});run.finish()
if __name__=='__main__':main()
