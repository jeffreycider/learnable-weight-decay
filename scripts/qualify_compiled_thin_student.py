"""Isolated replay K1 oracle plus warm local AOT student and allocator probe.

Uses the existing retained-state qualification unchanged, replacing only its
local raw KL/cotangent boundary in this process. No production backend changes.
"""
from pathlib import Path
import argparse,gc,hashlib,importlib.util,json,statistics,sys,time,weakref
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rewriter.compiled_thin_student import CompiledThinStudent
from rewriter.online_preservation import ReplayMaintenance
from scripts.qualify_training_gemm import comparison
from scripts.qualify_graph_offline import record_qualification_source
from scripts.qualify_compiled_training_response import cpu_copy,state_tensors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--expected-sha256',required=True)
    p.add_argument('--settings',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--compile-boundary',choices=('student_loss','student'),default='student_loss')
    p.add_argument('--backend',choices=('eager','compiled'),default='compiled')
    p.add_argument('--oracle-only',action='store_true')
    p.add_argument('--comparison',type=Path)
    p.add_argument('--rebind-executed-base',action='store_true')
    p.add_argument('--repaired-actor',type=Path)
    p.add_argument('--expected-repaired-sha256')
    p.add_argument('--raw-actor-oracle',action='store_true')
    p.add_argument('--restore-repaired-dual',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    record_qualification_source(a.output,vars(a))
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('retained_replay_oracle',root/'runs/chisel-astra/qualify_replay_maintenance.py')
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    override=None
    if a.restore_repaired_dual and not a.repaired_actor:raise ValueError('restored diagnostic dual requires exact repaired checkpoint')
    if a.repaired_actor:
        if hashlib.sha256(a.repaired_actor.read_bytes()).hexdigest()!=a.expected_repaired_sha256:raise ValueError('repaired actor SHA differs')
        override=torch.load(a.repaired_actor,map_location='cpu',weights_only=False,mmap=True)
        if override['original_checkpoint_sha256']!=a.expected_sha256:raise ValueError('repair provenance differs')
        original_torch=oracle.torch
        class DiagnosticTorchProxy:
            def __getattr__(self,name):return getattr(original_torch,name)
            def load(self,path,*args,**kwargs):
                value=original_torch.load(path,*args,**kwargs)
                if Path(path).resolve()!=Path(a.checkpoint).resolve():return value
                value=dict(value);value['controller']=dict(value['controller'])
                learner=dict(value['controller']['learner']);value['controller']['learner']=learner
                learner['actor']=override['actor'];learner['optimizer']=override['optimizer']
                return value
        oracle.torch=DiagnosticTorchProxy()
    original=ReplayMaintenance._student;original_run=ReplayMaintenance.run
    retained={};helpers={}
    def student(self,wire,teachers,editable,*,backward):
        if a.backend=='compiled' and id(self) not in helpers:
            helpers[id(self)]=CompiledThinStudent(self.engine,editable,
                global_tokens=self.settings['sequences']*self.seq_len,embedding_names=self.embedding_names,compile_kl=a.compile_boundary=='student_loss')
        retained.update(replay=self,wire=wire,teachers=teachers,editable=editable)
        if a.backend=='eager':return original(self,wire,teachers,editable,backward=backward)
        return helpers[id(self)].step(wire,teachers,editable,backward=backward)
    def run(self,lifecycle,*args,**kwargs):
        retained['lifecycle']=lifecycle
        if a.raw_actor_oracle:retained['snapshot']=args[0]
        if a.restore_repaired_dual:self.dual.load_state_dict(override['dual'])
        return original_run(self,lifecycle,*args,**kwargs)
    ReplayMaintenance._student=student;ReplayMaintenance.run=run
    inventory=oracle.source_inventory
    def sources():
        result=inventory()
        for name in ('rewriter/compiled_thin_student.py','scripts/qualify_compiled_thin_student.py'):
            result[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
        return result
    oracle.source_inventory=sources
    import wandb
    telemetry=wandb.init(project='chisel',mode='offline',dir=str(a.output),config=dict(kind='compiled_local_replay_no_cudagraph',production_changed=False))
    sys.argv=['qualify_replay_maintenance.py','--checkpoint',a.checkpoint,'--expected-sha256',a.expected_sha256,
              '--settings',a.settings,'--output',str(a.output/'correctness.json'),'--device','cuda']
    try:oracle.main()
    finally:ReplayMaintenance._student=original;ReplayMaintenance.run=original_run
    if override is not None:
        for path in (a.output/'correctness.json',a.output/'correctness.maintenance.json'):
            result=json.loads(path.read_text())
            result['diagnostic_actor_override']=dict(sha256=a.expected_repaired_sha256,path=str(a.repaired_actor),repair_updates=override['update'],
                restored_diagnostic_dual=a.restore_repaired_dual,
                scope='Fixed repaired actor+savedAdam; original1500 base/snapshot/panel retained. Saved diagnostic dual restored ONLY for this disposed continuation.' if a.restore_repaired_dual else 'Actor and savedAdam only; diagnostic windup dual NOT transferred')
            if 'actor_meta_updates_at_start' in result:
                result['original_checkpoint_actor_meta_updates']=result.pop('actor_meta_updates_at_start')
            path.write_text(json.dumps(result,indent=2)+'\n')
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    actual=cpu_copy(retained['lifecycle'].actor.state_dict());initial=override['actor'] if override is not None else saved['controller']['learner']['actor']
    response=dict(checkpoint_sha256=a.expected_sha256,settings_sha256=hashlib.sha256(Path(a.settings).read_bytes()).hexdigest(),
        actor=actual,stored_actor_delta={n:v-initial[n] for n,v in actual.items()},optimizer=cpu_copy(retained['lifecycle'].optimizer.state_dict()),
        correctness=json.loads((a.output/'correctness.json').read_text()),repaired_actor_sha256=a.expected_repaired_sha256,restored_diagnostic_dual=a.restore_repaired_dual,
        repaired_actor_scope='Fixed actor+savedAdam override; original1500 post-base/snapshot/panel. See restored_diagnostic_dual; no state is transferred to online training.' if override is not None else None)
    torch.save(response,a.output/'actor_response.pt')
    response_error=None
    if a.comparison:
        ref=torch.load(a.comparison,map_location='cpu',weights_only=False)
        for key in ('checkpoint_sha256','settings_sha256','repaired_actor_sha256'):
            if response[key]!=ref[key]:raise ValueError('response reference binding differs: '+key)
        if response['restored_diagnostic_dual']!=ref.get('restored_diagnostic_dual',False):raise ValueError('diagnostic dual restoration differs')
        for key in ('chisel_replay_input_sha256','chisel_replay_panel_id_sha256'):
            if response['correctness']['metrics'][key]!=ref['correctness']['metrics'][key]:raise ValueError('response panel differs')
        response_error=dict(stored_actor_delta=comparison(response['stored_actor_delta'],ref['stored_actor_delta']),
            final_kl_abs_difference=abs(response['correctness']['metrics']['chisel_replay_final_kl']-ref['correctness']['metrics']['chisel_replay_final_kl']))
        got_state=state_tensors(response['optimizer']);ref_state=state_tensors(ref['optimizer'])
        response_error['adam']={kind:comparison({n:v for n,v in got_state.items() if n.endswith('/'+kind)},
            {n:v for n,v in ref_state.items() if n.endswith('/'+kind)}) for kind in ('exp_avg','exp_avg_sq','step')}
        (a.output/'response_comparison.json').write_text(json.dumps(response_error,indent=2)+'\n')
    del response,actual,initial
    if a.oracle_only:
        telemetry.finish();return
    replay=retained['replay'];helper=helpers[id(replay)]
    wire=retained['wire'];teachers=retained['teachers'];editable=retained['editable']
    def invoke(backend):
        for leaf in wire[2]:leaf.grad=None
        if backend=='eager':return original(replay,wire,teachers,editable,backward=True)
        return helper.step(wire,teachers,editable,backward=True)
    def counters():return {k:dict(v) for k,v in torch._dynamo.utils.counters.items()}
    # Oracle already warmed the candidate. Warm both again outside measurements.
    for backend in ('eager','compiled'):invoke(backend)
    before=counters();rows=[];references={};errors=[]
    for repeat in range(a.repeats):
        for backend in (('eager','compiled') if repeat%2==0 else ('compiled','eager')):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            value,gradients=invoke(backend);torch.cuda.synchronize();seconds=time.perf_counter()-started
            rows.append(dict(repeat=repeat,backend=backend,seconds=seconds,kl=float(value),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            references[backend]=(float(value),{str(key):g.detach().cpu().clone() for key,g in zip(wire[3],gradients)})
            telemetry.log(dict(objective=seconds,**rows[-1]));print(json.dumps(rows[-1]),flush=True)
        error=comparison(references['compiled'][1],references['eager'][1])
        errors.append(dict(repeat=repeat,raw_cotangent=error,kl_abs_difference=abs(references['compiled'][0]-references['eager'][0])))
    after=counters()
    summary=dict(scope=__doc__,rows=rows,errors=errors,compilation_counters_before=before,compilation_counters_after=after,
        compile_counters_stable=before==after,median_seconds={b:statistics.median(r['seconds'] for r in rows if r['backend']==b) for b in ('eager','compiled')},
        teacher_bytes=sum(t.log_probs.numel()*t.log_probs.element_size() for t in teachers),cudagraphs=False,
        numerical_oracle=json.loads((a.output/'correctness.json').read_text())['qualification'],response_comparison=response_error)
    if a.raw_actor_oracle:
        from rewriter.transformer_action import build_action_reference_cache,prepare_transformer_action
        from rewriter.online_preservation import _broadcast_action
        lifecycle=retained['lifecycle'];cache=build_action_reference_cache(replay.engine.parameters);raw={}
        actor_parameters=dict(lifecycle.actor.named_parameters())
        for backend in ('dense','compiled'):
            prepared=prepare_transformer_action(lifecycle.actor,retained['snapshot'],replay.engine.parameters,
                radius_fraction=lifecycle.config['radius_fraction'],embedding_names=replay.embedding_names,
                actor_forward=lifecycle.actor_forward,reference_cache=cache)
            if backend=='dense':
                proposal=prepared.materialize();leaves={n:v.detach().requires_grad_(n in editable) for n,v in proposal.items()}
                kl=torch.zeros((),dtype=torch.float64,device=replay.device)
                for teacher in teachers:
                    loss=replay.engine.student_parameter_loss(leaves,teacher)*(teacher.tokens/(replay.settings['sequences']*replay.seq_len))
                    kl+=loss.detach();loss.backward()
                outputs=[proposal[n] for n in sorted(editable)]
                cotangents=[torch.zeros_like(leaves[n]) if leaves[n].grad is None else leaves[n].grad for n in sorted(editable)]
            else:
                raw_wire=_broadcast_action(prepared,replay.device)
                kl,raw_gradients=helper.step(raw_wire,teachers,editable,backward=True)
                outputs=[];cotangents=[]
                for (name,role),gradient in zip(raw_wire[3],raw_gradients):
                    value=prepared.scale if role=='scale' else prepared.action.vectors[name] if role=='v' else getattr(prepared.action.matrices[name],role)
                    if value.requires_grad:outputs.append(value);cotangents.append(gradient)
            actor_gradients=torch.autograd.grad(outputs,tuple(actor_parameters.values()),grad_outputs=cotangents,allow_unused=True)
            raw[backend]=dict(kl=float(kl),gradients={n:torch.zeros_like(p,device='cpu') if g is None else g.detach().cpu().clone()
                for (n,p),g in zip(actor_parameters.items(),actor_gradients)})
            del prepared,outputs,cotangents,actor_gradients
            if backend=='dense':del proposal,leaves,loss
            else:del raw_wire,raw_gradients
        summary['raw_actor_kl']=dict(comparison=comparison(raw['compiled']['gradients'],raw['dense']['gradients']),
            dense_kl=raw['dense']['kl'],compiled_kl=raw['compiled']['kl'],kl_abs_difference=abs(raw['dense']['kl']-raw['compiled']['kl']),
            scope='Unweighted raw KL actor VJP; no deadband coefficient or norm reward; fresh actor graph per branch')
        del cache,raw,kl,actor_parameters
        retained.pop('snapshot')
    if a.rebind_executed_base:
        before_rebind=counters()
        replay.engine.load_teacher_state({n:saved['model'][n] for n in replay.engine.parameters},
                                        {n:saved['model'][n] for n in replay.engine.buffers})
        teachers=replay.engine.cache_teacher(teachers[0].inputs,data_role='past_training')
        retained['teachers']=teachers
        rebound={}
        for backend in ('eager','compiled'):
            val,grads=invoke(backend)
            rebound[backend]=(float(val),{str(key):g.detach().cpu().clone() for key,g in zip(wire[3],grads)})
        summary['base_reuse']=dict(scope='Actual retained1500 post-base values replaced in place by retained1500 executed values; same fixed thin action/panel, no actor update',
            kl_abs_difference=abs(rebound['compiled'][0]-rebound['eager'][0]),raw_cotangent=comparison(rebound['compiled'][1],rebound['eager'][1]),
            compile_counters_stable=before_rebind==counters(),base_addresses_unchanged=helper.base_layout=={n:(v.data_ptr(),v.shape,v.stride(),v.dtype,v.device) for n,v in replay.engine.parameters.items()})
        del val,grads
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Release teacher/factor arguments while retaining compiled helper, actor and
    # shared Adam. No pending actor autograd graph or ordinary sensors in scope.
    inputs=teachers[0].inputs.detach().to(replay.device,copy=True);targets=inputs.long().roll(-1,1)
    teacher_refs=[weakref.ref(t.log_probs) for t in teachers]
    held_bytes=torch.cuda.memory_allocated();del teachers,wire,gradients,value
    retained.pop('teachers');retained.pop('wire');gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    released_bytes=torch.cuda.memory_allocated()
    from baseline.fp32_execution import enable_fp32_execution
    model=replay.model;model.requires_grad_(True);enable_fp32_execution(model,compile_loss=False)
    ordinary=[]
    for _ in range(2):
        model.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        loss=model(inputs,targets);loss.backward();torch.cuda.synchronize()
        ordinary.append(dict(seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),loss=float(loss.detach())))
    summary['allocator_probe']=dict(teacher_objects_released=all(ref() is None for ref in teacher_refs),allocated_before_release=held_bytes,
        allocated_after_release=released_bytes,ordinary_micro16=ordinary,
        retained='compiled helper/engine, actor and shared Adam; no private CUDA graph pool',
        excluded='pending actor graph and sensors; synthetic rolled targets; no ordinary optimizer update')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['allocator_probe']),flush=True);telemetry.finish()


if __name__=='__main__':main()
