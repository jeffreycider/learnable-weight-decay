"""Eight-GPU readiness check, not a Chisel training experiment.

Exercises NCCL, CUDA graphs, torch.compile, and the actual grouped Chisel ->
edited tiny GPT -> preservation loss -> shared/private Adam path on synthetic
inputs. Saves disposable smoke checkpoints; no research checkpoint is loaded.
"""
import datetime
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from baseline.model import GPT
from rewriter.adaptive_preservation import build_adaptive_proposal
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.cifar_bank import tensor_tree
from rewriter.function_preservation import FunctionalPreservation
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual,tokenwise_panel
from tests.test_adaptive_preservation_actor import actual_actor_case


def main():
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_device(rank)
    device=torch.device('cuda',rank)
    started=time.perf_counter()
    dist.init_process_group('nccl',device_id=device,timeout=datetime.timedelta(seconds=180))
    try:
        value=torch.tensor(float(rank+1),device=device,dtype=torch.float64)
        dist.all_reduce(value)
        assert value.item()==world*(world+1)/2
        # A tiny compiled kernel checks this runtime's Triton/A100 combination.
        from torch._dynamo.utils import counters
        def op(x,y):return (x@y).tanh()
        x=torch.randn(64,64,device=device);y=torch.randn_like(x)
        compiled=torch.compile(op,fullgraph=True)
        reference=op(x,y);observed=compiled(x,y)
        torch.testing.assert_close(observed,reference,rtol=2e-5,atol=2e-5)
        graph_count=counters['stats']['unique_graphs']
        for _ in range(3):compiled(x,y)
        assert counters['stats']['unique_graphs']==graph_count
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):op(x,y)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=op(x,y)
        graph.replay();torch.testing.assert_close(captured,reference,rtol=0,atol=0)

        cpu,_,_,snapshot,_,inputs=actual_actor_case()
        model=GPT(32,1,128).float().to(device).eval().requires_grad_(False)
        engine=FunctionalPreservation(model,parameter_state=tensor_tree(cpu.parameters,device),
            buffer_state=tensor_tree(cpu.buffers,device),microbatch_sequences=1,
            kl_dtype=torch.float64,gpt_precision='fp32',teacher_cache_device=device)
        del cpu
        snapshot=tensor_tree(snapshot,device);inputs=inputs.to(device)
        actor=GroupedCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2,
            fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32').to(device)
        private=actor.make_private(snapshot,state_key=('lambda-smoke',rank),seed=77).to(device)
        # Nonzero heads exercise private-probe gradients on the first step.
        with torch.no_grad():
            for bank in actor.output_heads:
                for head in bank.values():head.weight.normal_(std=1e-6)
        for p in actor.parameters():dist.broadcast(p.data,src=0)
        initial=actor.output_head('head.weight').weight.detach().clone()
        shared=torch.optim.Adam(actor.parameters(),lr=1e-7,betas=(.9,.99))
        private_opt=torch.optim.Adam(private.parameters(),lr=1e-7,betas=(.9,.99))
        grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
        config=TokenwiseConfig(mean_tolerance=1e-10,tail_tolerance=1e-10,scale=1e-5,rho=.01,
            ranking_weight=.01,ranking_k=10,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap')
        dual=TokenwiseDual([rank],config=config,dual_lr=.01)
        teachers=engine.cache_teacher(inputs,data_role='synthetic')
        rows=[]
        for step in range(2):
            shared.zero_grad(set_to_none=True);private_opt.zero_grad(set_to_none=True)
            def propose(s):return actor(s,private,weight_cores=query_weight_cores(engine.parameters,private))
            parameters,regularizer,stats,_,credit=build_adaptive_proposal(propose,snapshot,engine,grouped,radius_fraction=.01)
            row=tokenwise_panel(engine,parameters,regularizer,inputs,config=config,dual=dual,state_id=rank,
                actor_credit=credit,teacher_provider=lambda a,b:teachers[a],backward=True)
            shared_grads=[p.grad for p in actor.parameters() if p.grad is not None]
            private_grads=[p.grad for p in private.parameters() if p.grad is not None]
            assert shared_grads and private_grads
            assert all(torch.isfinite(g).all() for g in shared_grads+private_grads)
            assert sum(float(g.double().square().sum()) for g in private_grads)>0
            for g in shared_grads:dist.all_reduce(g);g.div_(world)
            torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
            torch.nn.utils.clip_grad_norm_(private.parameters(),1.)
            shared.step();private_opt.step()
            rows.append(dict(step=step,objective=row['objective'],mean_kl=row['mean_kl']))
        assert shared.state and private_opt.state
        assert not torch.equal(initial,actor.output_head('head.weight').weight)
        checkpoint=dict(actor=actor.state_dict(),private=private.state_dict(),shared=shared.state_dict(),private_optimizer=private_opt.state_dict())
        torch.save(checkpoint,output/f'rank{rank}_smoke.pt')
        restored=torch.load(output/f'rank{rank}_smoke.pt',map_location=device,weights_only=False)
        actor.load_state_dict(restored['actor']);private.load_state_dict(restored['private'])
        shared.load_state_dict(restored['shared']);private_opt.load_state_dict(restored['private_optimizer'])
        assert all(p.grad is None for p in engine.parameters.values())
        torch.cuda.synchronize()
        record=dict(rank=rank,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
            nccl_allreduce=True,compile_graph_count=graph_count,compile_repeated_without_new_graph=True,
            cuda_graph_replay=True,chisel_gpt_updates=2,checkpoint_reload=True,rows=rows,
            seconds=time.perf_counter()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(),
            scope=__doc__)
        (output/f'rank{rank}.json').write_text(json.dumps(record,indent=2)+'\n')
        dist.barrier()
        if rank==0:
            records=[json.loads((output/f'rank{i}.json').read_text()) for i in range(world)]
            (output/'complete.json').write_text(json.dumps(dict(world_size=world,records=records),indent=2)+'\n')
            print(json.dumps(dict(status='completed',world_size=world,max_seconds=max(r['seconds'] for r in records))),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
