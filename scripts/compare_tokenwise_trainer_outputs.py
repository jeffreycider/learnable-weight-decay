"""Read-only direct output comparison of two saved tokenwise trainer policies."""
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from baseline.model import GPT
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from rewriter.offline_systems import CachedBaseNormPreservation,VerifiedStateConstants,select_panel_blocks
from rewriter.accelerated_grouped_norm import AcceleratedGroupedNormObjective
from rewriter.chisel_factory import build_chisel
from scripts.train_multistate_chisel import build_proposal

def compare(a,b):
    a=a.flatten(0,-2);b=b.flatten(0,-2);n=len(a);v=a.shape[-1]
    witnesses=[]
    sums=torch.zeros(7,device=a.device,dtype=torch.float64);maxima=torch.zeros(3,device=a.device,dtype=torch.float64)
    for start in range(0,n,128):
        x=a[start:start+128].double();y=b[start:start+128].double();lp=x.log_softmax(-1);lq=y.log_softmax(-1)
        p=(lp.exp()*(lp-lq)).sum(-1);q=(lq.exp()*(lq-lp)).sum(-1);d=x-y;center=d-d.mean(-1,keepdim=True)
        ti=x.topk(10,dim=-1).indices;si=y.topk(10,dim=-1).indices
        changed=(ti!=si).any(-1)|(x.argmax(-1)!=y.argmax(-1))
        for row in changed.nonzero().flatten().tolist():
            top=x[row].topk(11).values
            witnesses.append(dict(token=start+row,reference_top1_gap=float(top[0]-top[1]),reference_min_top10_adjacent_gap=float((top[:9]-top[1:10]).min()),reference_top10_membership_gap=float(top[9]-top[10]),reference_top10=ti[row].tolist(),candidate_top10=si[row].tolist()))
        sums+=torch.stack((p.sum(),q.sum(),center.square().sum(),(x.argmax(-1)!=y.argmax(-1)).sum(),(ti!=si).any(-1).sum(),(ti.sort(-1).values!=si.sort(-1).values).any(-1).sum(),d.square().sum()))
        maxima=torch.maximum(maxima,torch.stack((p.max(),q.max(),d.abs().max())))
    z=sums.cpu().tolist();m=maxima.cpu().tolist()
    return dict(rank_change_witnesses=witnesses,tokens=n,reference_to_tiled_kl=z[0]/n,tiled_to_reference_kl=z[1]/n,max_reference_to_tiled_token_kl=m[0],max_tiled_to_reference_token_kl=m[1],centered_logit_rms=(z[2]/(n*v))**.5,argmax_changes=int(z[3]),top10_order_changes=int(z[4]),top10_membership_changes=int(z[5]),logit_rms=(z[6]/(n*v))**.5,max_abs_logit_difference=m[2])

def main():
    p=argparse.ArgumentParser()
    for k in ('reference','candidate','state-bank','teacher-cache','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--sequences',type=int,default=16);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    enable_deterministic_execution();torch.set_num_threads(1);device=torch.device('cuda:0')
    import wandb
    run=wandb.init(project='chisel',mode='offline',dir=str(a.output),config={k:str(v) for k,v in vars(a).items()})
    saved=[torch.load(path,map_location='cpu',weights_only=False) for path in (a.reference,a.candidate)]
    reader=StateBankReader(a.state_bank,verify_once=True,mmap_model=True);cache=TeacherCacheReader(a.teacher_cache,verify_once=True,mmap=True)
    engine=CachedBaseNormPreservation(GPT(**reader.manifest['provenance']['model_config']).to(device).requires_grad_(False),microbatch_sequences=16,teacher_cache_device=device,kl_dtype=torch.float64,gpt_precision='fp32')
    constants=VerifiedStateConstants(engine,AcceleratedGroupedNormObjective)
    actor=build_chisel(saved[0]['metadata']['architecture']).to(device);forward=torch.compile(actor,fullgraph=True,dynamic=False)
    rows=[];start=time.perf_counter()
    for state in (6,251,499,999):
        record=reader.read(state,model_device='cpu',device=device);constants.load(cache,record);grouped=constants.grouped();norm=constants.radius_squared(grouped)
        panel=select_panel_blocks(cache.read(state,'train'),range(a.sequences//16));edited=[]
        for policy in saved:
            actor.load_state_dict(policy['controller'])
            with torch.enable_grad():proposal=build_proposal(forward,record.snapshot,engine,grouped,radius_fraction=.01,norm_squared=norm,shrinkage_coefficient=1.)
            edited.append({n:v.detach() for n,v in proposal[0].items()});del proposal
        for offset in range(0,a.sequences,16):
            inputs=panel.inputs[offset:offset+16].to(device)
            with torch.no_grad():x=engine._forward(edited[0],inputs);y=engine._forward(edited[1],inputs);metrics=compare(x,y)
            rows.append(dict(state_id=state,sequence_ids=panel.sequence_ids[offset:offset+16],**metrics));del x,y
        del edited
        print(json.dumps(rows[-1]),flush=True)
    result=dict(scope='Direct final-policy output comparison after two actual tokenwise trainer updates; fitting panels; diagnostic, not retrospective gate promotion',reference=str(a.reference),candidate=str(a.candidate),reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),student_forwards=2*len(rows),teacher_forwards=0,backwards=0,updates=0,rows=rows,elapsed_seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated())
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));run.log({'objective':max(r['reference_to_tiled_kl'] for r in rows)});run.finish();print(json.dumps(result),flush=True)
if __name__=='__main__':main()
