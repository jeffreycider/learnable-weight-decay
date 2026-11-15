"""Two real checkpoint contributions with reused literal workspace, one Adam.

This qualifies a local two-checkpoint accumulation boundary, not the scientific
200-checkpoint/eight-GPU update. Reference evaluations precede plan allocation
to avoid overlapping backend activation pools.
"""
import argparse,copy,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline.model import GPT
from baseline.execution import enable_deterministic_execution
from rewriter.transformer_chisel import TransformerChisel
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from rewriter.function_preservation import FunctionalPreservation
from rewriter.persistent_offline import PersistentOfflineTrainer,PersistentActorOptimizer
from scripts.qualify_persistent_offline_chisel import reference_step
from scripts.benchmark_persistent_mlp import discrepancy

class NoOptimizer:
    def step(self):pass
    def zero_grad(self,**kwargs):pass

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--teacher-cache',type=Path,required=True);p.add_argument('--states',default='100,101')
    p.add_argument('--batch',type=int,default=16);p.add_argument('--head-std',type=float,default=.0002)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import wandb
    wb=wandb.init(project='chisel-persistent-kernel',mode='offline',dir=str(args.output))
    enable_deterministic_execution();torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=False
    model=GPT(50304,12,768).cuda().requires_grad_(False)
    reference=TransformerChisel(model_dim=512,patch_size=64).cuda()
    with torch.no_grad():
        for head in reference.output_heads:head.weight.normal_(std=args.head_std)
    actual=copy.deepcopy(reference);reader=StateBankReader(args.bank);cache=TeacherCacheReader(args.teacher_cache)
    engine=FunctionalPreservation(model,microbatch_sequences=16,teacher_cache_device='cuda',kl_dtype=torch.float64)
    states=list(map(int,args.states.split(',')));expected=[]
    def load(state):
        record=reader.read(state,device='cuda',model_device='cpu');record.load_model(model);cache.load_engine(engine,record)
        panel=cache.read(state,'train');teacher=panel.teacher_batch(engine,0,16)
        return record,teacher.inputs[:args.batch].cuda(),teacher.log_probs[:args.batch]
    for state in states:
        record,inputs,teacher=load(state);reference.zero_grad(set_to_none=True)
        torch.cuda.synchronize();started=time.perf_counter()
        values,grads,logits,parameters=reference_step(reference,record.snapshot,engine,teacher,inputs,NoOptimizer(),
            radius_fraction=.01,budget=1e-4,clip=float('inf'),microbatch_sequences=16,state_weight=1/200)
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        expected.append(dict(state=state,values=values,gradient=torch.cat([v.flatten() for v in grads.values()]).cpu(),seconds=seconds))
        del record,inputs,teacher,grads,logits,parameters
    summed=sum(x['gradient'] for x in expected);offset=0
    for parameter in reference.parameters():
        parameter.grad=summed[offset:offset+parameter.numel()].reshape_as(parameter).cuda();offset+=parameter.numel()
    optimizer=torch.optim.Adam(reference.parameters(),lr=1e-5,betas=(.9,.99),eps=1e-8)
    torch.nn.utils.clip_grad_norm_(reference.parameters(),1.);optimizer.step();optimizer.zero_grad(set_to_none=True)
    expected_parameters=torch.cat([v.detach().flatten() for v in reference.parameters()]).cpu()
    del summed;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    accumulated={n:torch.zeros_like(v) for n,v in actual.named_parameters()};rows=[];program=None
    initial_actor=torch.cat([v.detach().flatten() for v in actual.parameters()]).cpu()
    for index,state in enumerate(states):
        record,inputs,teacher=load(state)
        torch.cuda.synchronize();started=time.perf_counter()
        if program is None:
            program=PersistentOfflineTrainer(actual,model,record.snapshot,inputs,teacher,microbatch_sequences=16,
                state_weight=1/200,gradient_accumulator=accumulated,apply_optimizer=False)
            pointers=[v.data_ptr() for v in program.plan.storage];stage_pointer=program.plan.stages.data_ptr();kernel=program.plan.kernel
        else:
            program.load_checkpoint(model,record.snapshot);program.load_panel(inputs,teacher)
            assert pointers==[v.data_ptr() for v in program.plan.storage]
            assert stage_pointer==program.plan.stages.data_ptr() and kernel is program.plan.kernel
        torch.cuda.synchronize();load_seconds=time.perf_counter()-started
        started=time.perf_counter();program.step();torch.cuda.synchronize();seconds=time.perf_counter()-started
        gradient=torch.cat([v.flatten() for v in program.gradients.values()]).cpu()
        row=dict(state=state,initial_build=index==0,checkpoint_and_panel_refresh_seconds=load_seconds,
            kernel_seconds=seconds,reference=expected[index]['values'],literal_kl=float(program.kl),
            gradient_error=discrepancy(expected[index]['gradient'],gradient),workspace_pointers_unchanged=True,
            source_actor_unchanged=torch.equal(torch.cat([v.detach().flatten() for v in actual.parameters()]).cpu(),initial_actor))
        assert row['source_actor_unchanged']
        rows.append(row);print(json.dumps(row),flush=True);wb.log(dict(state=state,kernel_seconds=seconds,gradient_error=row['gradient_error']['relative_l2']))
        del record,inputs,teacher,gradient
    update=PersistentActorOptimizer(actual,accumulated,lr=1e-5,kernel=program.plan.kernel)
    update.step();torch.cuda.synchronize()
    updated=torch.cat([v.detach().flatten() for v in actual.parameters()]).cpu()
    result=dict(scope=__doc__,states=states,batch=args.batch,state_weight=1/200,rows=rows,
        optimizer_updates=int(update.optimizer_state()['step']),updated_actor=discrepancy(expected_parameters,updated),
        peak_bytes=torch.cuda.max_memory_allocated(),limitations='Only two local checkpoint contributions; no distributed reduction. Panel target reconstruction and source staging are excluded from refresh timing.')
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);wb.finish()
    assert result['optimizer_updates']==1
    assert max(x['gradient_error']['relative_l2'] for x in rows)<.05
    assert result['updated_actor']['relative_l2']<.001

if __name__=='__main__':main()
