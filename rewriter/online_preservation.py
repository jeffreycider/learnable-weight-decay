"""Opt-in all-rank replay maintenance; one rank-zero actor and persistent dual.

No GPT/base optimizer updates. Candidate forwards use stored FP32 weights with
existing thin anchored backward; final student evaluation does not consume the
connected actor graph retained for the next ordinary task gradient.
"""
import copy
import math
import time
import torch
from .algebra import Anchors, Factors, RawAction
from .function_preservation import FunctionalPreservation, chunked_teacher_kl
from .factorized_fp32 import factorized_logits
from .primal_dual import PrimalDual
from .transformer_action import prepare_transformer_action, build_action_reference_cache
from .online_replay import OnlineReplaySampler


def validate_replay_maintenance(settings):
    if settings is None or settings=={'mode':'off'}:return None
    keys={'mode','steps','sequences','microbatch_sequences','seed','split_seed','train_files',
          'manifest_sha256','manifest_path','teacher_cache_device','learning_rate','representation_lr_multiplier',
          'gradient_clip','shrinkage_coefficient','tolerance','rho','dual_lr'}
    if (not isinstance(settings,dict) or not keys <= set(settings) <= keys|{'student_backend','tokenwise_objective'}
            or settings['mode']!='replay_primal_dual'):
        raise ValueError('explicit replay_primal_dual settings required')
    if settings.get('student_backend','eager') not in ('eager','compiled_student_v1_experimental'):
        raise ValueError('unknown replay student backend')
    for name in ('steps','sequences','microbatch_sequences'):
        if type(settings[name]) is not int or settings[name]<1:raise ValueError('positive replay '+name+' required')
    for name in ('seed','split_seed'):
        if type(settings[name]) is not int or settings[name]<0:raise ValueError('nonnegative replay seed required')
    for name in ('learning_rate','representation_lr_multiplier','gradient_clip','tolerance','rho','dual_lr'):
        x=settings[name]
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0:raise ValueError('positive finite replay '+name+' required')
    x=settings['shrinkage_coefficient']
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0:raise ValueError('nonnegative finite shrinkage coefficient required')
    if settings['teacher_cache_device'] not in ('cpu','device'):raise ValueError('explicit cpu/device teacher cache location required')
    if settings.get('student_backend')=='compiled_student_v1_experimental' and settings['teacher_cache_device']!='device':
        raise ValueError('compiled replay requires resident device teacher cache')
    if 'tokenwise_objective' in settings:
        from .tokenwise_reference import TokenwiseConfig
        try:
            config=TokenwiseConfig(**settings['tokenwise_objective'])
        except (TypeError,ValueError) as error:
            raise ValueError('invalid replay tokenwise objective') from error
        if config.mean_tolerance!=settings['tolerance'] or config.rho!=settings['rho']:
            raise ValueError('tokenwise and replay tolerance/rho must agree')
        if getattr(config,'loss_backend','reference')!='reference':
            raise ValueError('online tokenwise loss requires separately qualified reference backend')
        if settings.get('student_backend','eager')!='eager':
            raise ValueError('tokenwise online replay currently requires eager backend')
    if not isinstance(settings['train_files'],list) or not settings['train_files'] or any(not isinstance(x,str) for x in settings['train_files']):raise ValueError('explicit training shard list required')
    if not isinstance(settings['manifest_path'],str) or not settings['manifest_path']:raise ValueError('pinned trainer manifest path required')
    sha=settings['manifest_sha256']
    if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):raise ValueError('replay manifest SHA256 required')
    return copy.deepcopy(settings)


def _distributed():return torch.distributed.is_initialized()
def _rank():return torch.distributed.get_rank() if _distributed() else 0
def _world():return torch.distributed.get_world_size() if _distributed() else 1


def _root(fn,device):
    from .transformer_online import authoritative_call
    return authoritative_call(fn,device=device)


def _local(fn,device):
    from .transformer_online import collective_local_call
    return collective_local_call(fn,device=device)


def _sum(value):
    if _distributed():torch.distributed.all_reduce(value,op=torch.distributed.ReduceOp.SUM)
    return value


def _allocate_wire_payload(description,device):
    return [torch.empty(shape,dtype=dtype,device=device) for _,_,shape,dtype,_ in description]


def _broadcast_action(prepared,device):
    """All ranks receive independent leaves; no actor replica or cap detachment."""
    def pack():
        payload=[];description=[]
        for name,obs in sorted(prepared.snapshot.matrices.items()):
            factor=prepared.action.matrices.get(name)
            p=factor.P if factor is not None else torch.zeros_like(obs.anchors.A)
            q=factor.Q if factor is not None else torch.zeros_like(obs.anchors.B)
            for role,value in [('P',p),('Q',q),('A',obs.anchors.A),('B',obs.anchors.B)]:
                description.append((name,role,tuple(value.shape),value.dtype,name in prepared.editable and role in ('P','Q')))
                payload.append(value.detach().contiguous().clone())
        for name,obs in sorted(prepared.snapshot.vectors.items()):
            value=prepared.action.vectors.get(name,torch.zeros_like(obs.weight))
            description.append((name,'v',tuple(value.shape),value.dtype,name in prepared.editable));payload.append(value.detach().contiguous().clone())
        description.append(('', 'scale', (),prepared.scale.dtype,True));payload.append(prepared.scale.detach().clone())
        return description,payload
    packed=_root(pack,device)
    objects=[packed[0] if _rank()==0 else None]
    if _distributed():torch.distributed.broadcast_object_list(objects,src=0,device=device)
    description=objects[0]
    payload=_local(lambda:packed[1] if _rank()==0 else
        _allocate_wire_payload(description,device),device)
    # All allocation/packing succeeded on EVERY rank before tensor collectives.
    for value in payload:
        if _distributed():torch.distributed.broadcast(value,src=0)
    def unpack():
        matrices={};vectors={};slots={}
        for desc,value in zip(description,payload):
            name,role,_,_,grad=desc;value.requires_grad_(grad);slots[(name,role)]=value
            if role=='v':vectors[name]=value
        for name,role,*_ in description:
            if role=='P':matrices[name]=Factors(slots[name,'P'],slots[name,'Q'],Anchors(slots[name,'A'],slots[name,'B']))
        leaves=[v for v in payload if v.requires_grad]
        keys=[(n,r) for n,r,_,_,grad in description if grad]
        return RawAction(matrices,vectors),slots['','scale'],leaves,keys
    return _local(unpack,device)


def _stored_parameters(base,action,scale,editable,embedding_names):
    result={}
    for name,value in base.items():
        if name not in editable:result[name]=value;continue
        if name in action.matrices:
            delta=action.matrices[name].dense()
            if name in embedding_names:delta=delta.T
        else:delta=action.vectors[name]
        result[name]=(value.float()+scale*delta).to(value.dtype)
    return result


class ReplayMaintenance:
    def __init__(self,model,settings,*,batch_tokens,seq_len,embedding_names):
        self.settings=validate_replay_maintenance(settings)
        if self.settings is None:raise ValueError('enabled replay configuration required')
        self.device=next(model.parameters()).device;self.model=model
        self.world,self.rank=_world(),_rank();self.embedding_names=frozenset(embedding_names)
        if self.settings['sequences']%self.world:raise ValueError('replay sequences must divide all ranks')
        self.sampler=OnlineReplaySampler(self.settings['train_files'],batch_tokens=batch_tokens,seq_len=seq_len,
            seed=self.settings['seed'],split_seed=self.settings['split_seed'],manifest_sha256=self.settings['manifest_sha256'],manifest_path=self.settings['manifest_path'])
        self.engine=FunctionalPreservation(model,microbatch_sequences=self.settings['microbatch_sequences'],
            teacher_cache_device=self.device if self.settings['teacher_cache_device']=='device' else 'cpu',
            kl_dtype=torch.float64,gpt_precision='fp32')
        self.tokenwise_config=None
        if 'tokenwise_objective' in self.settings:
            from .tokenwise_reference import TokenwiseConfig,TokenwiseDual
            self.tokenwise_config=TokenwiseConfig(**self.settings['tokenwise_objective'])
            self.dual=TokenwiseDual([0],config=self.tokenwise_config,dual_lr=self.settings['dual_lr']) if self.rank==0 else None
        else:
            self.dual=PrimalDual([0],tolerance=self.settings['tolerance'],rho=self.settings['rho'],dual_lr=self.settings['dual_lr']) if self.rank==0 else None
        self._active_multiplier=0.
        self._last_token_metrics=None
        self.updates=0;self.blocks=0;self.last_panel=None
        self.compute=dict(teacher_forwards=0,teacher_tokens=0,student_forwards=0,student_tokens=0,student_backwards=0,student_backward_tokens=0)
        self.seq_len=seq_len

    def state_dict(self):
        return dict(schema=1,settings=copy.deepcopy(self.settings),updates=self.updates,blocks=self.blocks,
                    dual=None if self.dual is None else self.dual.state_dict(),compute=dict(self.compute),last_panel=copy.deepcopy(self.last_panel))

    def load_state_dict(self,state,lifecycle):
        if state['schema']!=1 or state['settings']!=self.settings:raise ValueError('replay resume configuration changed')
        updates=state['updates'];blocks=state['blocks']
        if type(updates) is not int or type(blocks) is not int or min(updates,blocks)<0 or updates!=blocks*self.settings['steps']:raise ValueError('replay checkpoint clock mismatch')
        if self.rank==0:
            if blocks!=lifecycle.meta_updates+int(lifecycle.action is not None):raise ValueError('replay/task checkpoint clocks differ')
            self.dual.load_state_dict(state['dual'])
            if self.dual.updates!=updates:raise ValueError('replay dual clock differs')
        elif state['dual'] is not None:raise ValueError('nonroot may not own dual')
        if set(state['compute'])!=set(self.compute) or any(type(v) is not int or v<0 for v in state['compute'].values()):raise ValueError('invalid replay compute counters')
        self.updates,self.blocks=updates,blocks;self.compute=dict(state['compute']);self.last_panel=copy.deepcopy(state['last_panel'])

    def _student(self,wire,teachers,editable,*,backward):
        if self.tokenwise_config is not None:
            return self._student_tokenwise(wire,teachers,editable,backward=backward)
        if self.settings.get('student_backend','eager')=='compiled_student_v1_experimental':
            from .compiled_replay import compiled_replay_student
            return compiled_replay_student(self,wire,teachers,editable,backward=backward)
        action,scale,leaves,keys=wire
        with torch.no_grad():stored=_stored_parameters(self.engine.parameters,action,scale,editable,self.embedding_names)
        total=torch.zeros((),dtype=torch.float64,device=self.device)
        for teacher in teachers:
            with torch.set_grad_enabled(backward):
                logits=factorized_logits(self.model,self.engine.parameters,self.engine.buffers,teacher.inputs.to(self.device),
                    action,scale,realized_parameters=stored,linear_backend='ieee')
                loss=chunked_teacher_kl(logits,teacher.log_probs,self.engine.chunk_tokens)
                weighted=loss*(teacher.tokens/(self.settings['sequences']*self.seq_len))
                total+=weighted.detach()
                self.engine.compute.student_forwards+=1;self.engine.compute.student_tokens+=teacher.tokens
                if backward:
                    weighted.backward();self.engine.compute.student_backwards+=1;self.engine.compute.student_backward_tokens+=teacher.tokens
            del logits,loss,weighted
        if not bool(torch.isfinite(total)):raise FloatingPointError('nonfinite replay KL')
        gradients=[]
        if backward:
            for leaf in leaves:
                gradient=torch.zeros_like(leaf) if leaf.grad is None else leaf.grad.detach()
                if not bool(torch.isfinite(gradient).all()):raise FloatingPointError('nonfinite replay thin cotangent')
                gradients.append(gradient)
        return total,gradients

    def _sync_tokenwise_multiplier(self):
        if self.tokenwise_config is None:return
        # The complete objective coefficient is fixed before any rank starts
        # its microbatches. Only rank zero owns/updates the dual state.
        value=_root(lambda:float(self.dual.multipliers[0]),self.device)
        wire=torch.tensor(value if self.rank==0 else 0.,device=self.device,dtype=torch.float64)
        if _distributed():torch.distributed.broadcast(wire,src=0)
        self._active_multiplier=float(wire)

    def _student_tokenwise(self,wire,teachers,editable,*,backward):
        from .tokenwise_reference import logit_terms_and_cotangent,TOKEN_SUM_FIELDS
        action,scale,leaves,keys=wire
        with torch.no_grad():stored=_stored_parameters(self.engine.parameters,action,scale,editable,self.embedding_names)
        total=torch.zeros((),device=self.device,dtype=torch.float64)
        sums=torch.zeros(len(TOKEN_SUM_FIELDS),device=self.device,dtype=torch.float64)
        maximum=torch.full((),-torch.inf,device=self.device,dtype=torch.float64)
        exceeded=torch.zeros((),device=self.device,dtype=torch.int64)
        global_tokens=self.settings['sequences']*self.seq_len
        for teacher in teachers:
            if teacher.data_role!='past_training':raise ValueError('online replay may learn only from past training data')
            if teacher.teacher_id!=self.engine.teacher_id:raise ValueError('online replay teacher identity mismatch')
            with torch.set_grad_enabled(backward):
                logits=factorized_logits(self.model,self.engine.parameters,self.engine.buffers,teacher.inputs.to(self.device),
                    action,scale,realized_parameters=stored,linear_backend='ieee')
            cotangent,parts,raw=logit_terms_and_cotangent(logits,teacher.log_probs,
                config=self.tokenwise_config,multiplier=self._active_multiplier,
                panel_tokens=global_tokens,chunk_tokens=self.engine.chunk_tokens,backward=backward)
            total+=raw.sum().to(self.device)/global_tokens
            sums+=parts
            maximum=torch.maximum(maximum,raw.max().to(self.device))
            exceeded+=(raw>self.tokenwise_config.tail_tolerance).sum().to(self.device)
            self.engine.compute.student_forwards+=1;self.engine.compute.student_tokens+=teacher.tokens
            if backward:
                logits.backward(cotangent)
                self.engine.compute.student_backwards+=1;self.engine.compute.student_backward_tokens+=teacher.tokens
            del logits,cotangent,parts,raw
        self._last_token_sums=sums.detach()
        self._last_token_max=maximum.detach()
        self._last_token_exceeded=exceeded.detach()
        gradients=[]
        if backward:
            for leaf in leaves:
                gradient=torch.zeros_like(leaf) if leaf.grad is None else leaf.grad.detach()
                if not bool(torch.isfinite(gradient).all()):raise FloatingPointError('nonfinite tokenwise replay cotangent')
                gradients.append(gradient)
        if not bool(torch.isfinite(total)):raise FloatingPointError('nonfinite tokenwise replay KL')
        return total,gradients

    def _reduce_tokenwise_metrics(self):
        if self.tokenwise_config is None:return
        from .tokenwise_reference import TOKEN_SUM_FIELDS
        _sum(self._last_token_sums);_sum(self._last_token_exceeded)
        if _distributed():torch.distributed.all_reduce(self._last_token_max,op=torch.distributed.ReduceOp.MAX)
        count=self.settings['sequences']*self.seq_len
        values=dict(zip(TOKEN_SUM_FIELDS,self._last_token_sums.cpu().tolist()))
        self._last_token_metrics=dict(
            dual_linear_term=values['dual_linear']/count,tail_quadratic_penalty=values['tail_quadratic']/count,
            ranking_penalty=values['ranking']/count,
            kl_penalty=sum(values[k] for k in ('dual_linear','tail_quadratic','ranking'))/count,
            argmax_changes=int(values['argmax_changes']),top10_order_changes=int(values['top10_order_changes']),
            top10_membership_changes=int(values['top10_membership_changes']),maximum_token_kl=float(self._last_token_max),
            token_tail_exceedances=int(self._last_token_exceeded),token_count=count,
            objective_version=self.tokenwise_config.version,
            measurement_scope='same past replay contexts; final measurements are in-sample')

    def _actor_step(self,lifecycle,prepared,keys,gradients,kl,cache):
        if lifecycle.action is not None or lifecycle.lag.pending:raise RuntimeError('replay update with pending task graph')
        opt=lifecycle.optimizer
        if not isinstance(opt,torch.optim.Adam):raise ValueError('replay requires shared Adam')
        if any(g.get('role') not in (None,'action_heads','representation') for g in opt.param_groups):raise ValueError('unsupported replay optimizer group role')
        terms=self.dual.terms(float(kl),0)
        if self.tokenwise_config is not None:
            terms.update(self._last_token_metrics)
            coefficient=1.  # Already applied per token before the single GPT backward.
        else:coefficient=terms['kl_coefficient']
        proposal=prepared.materialize();norm_result=cache.grouped.evaluate(proposal)
        regularizer=self.settings['shrinkage_coefficient']*100*norm_result.mean_relative_change
        outputs=[regularizer];cotangents=[torch.ones_like(regularizer)]
        for (name,role),gradient in zip(keys,gradients):
            value=(prepared.scale if role=='scale' else prepared.action.vectors[name] if role=='v' else getattr(prepared.action.matrices[name],role))
            if value.requires_grad:outputs.append(value);cotangents.append(gradient*coefficient)
        opt.zero_grad(set_to_none=True)
        torch.autograd.backward(outputs,cotangents)
        magnitude=torch.nn.utils.clip_grad_norm_(lifecycle.actor.parameters(),self.settings['gradient_clip'],error_if_nonfinite=True)
        metric=dict(candidate_replay_kl=float(kl),norm_percentage_change=float(100*norm_result.mean_relative_change.detach()),
                    gradient_norm=float(magnitude),gradient_clipped=bool(magnitude>self.settings['gradient_clip']),
                    gradient_clip_threshold=self.settings['gradient_clip'],**terms)
        old=[g['lr'] for g in opt.param_groups]
        try:
            for g in opt.param_groups:g['lr']=self.settings['learning_rate']*(self.settings['representation_lr_multiplier'] if g.get('role')=='representation' else 1.)
            opt.step()
        finally:
            for g,lr in zip(opt.param_groups,old):g['lr']=lr
        self.dual.step([(0,float(kl))],expected_ids=[0])
        return metric

    def run(self,lifecycle,snapshot,parameters,collector,*,action_step):
        """All ranks call, after base step, before any live edit. Returns root action."""
        started=time.perf_counter()
        def setup():
            if collector.active or collector.ctx is not None:raise RuntimeError('ordinary sensors must be dormant during replay')
            if any(p.dtype!=torch.float32 for p in parameters.values()):raise ValueError('strict FP32 replay requires FP32 GPT parameters')
            self.engine.load_teacher_state(parameters,dict(self.model.named_buffers()))
            panel=self.sampler.sample(action_step,self.settings['sequences'])
            local=self.settings['sequences']//self.world;start=self.rank*local
            with torch.no_grad():teachers=self.engine.cache_teacher(panel['inputs'][start:start+local],data_role='past_training',sequence_ids=panel['ids'][start:start+local])
            return panel,teachers
        panel,teachers=_local(setup,self.device)
        # Every rank must agree on sampling/provenance BEFORE a shared actor update.
        if _distributed():
            signatures=[None]*self.world;torch.distributed.all_gather_object(signatures,(panel['id_sha256'],panel['input_sha256']))
            if len(set(signatures))!=1:raise RuntimeError('replay panels differ across ranks')
        def references():
            if lifecycle.action is not None or lifecycle.lag.pending:raise RuntimeError('consume task credit before replay')
            if lifecycle.last_consumed_step is not None and lifecycle.last_consumed_step!=action_step:raise ValueError('replay action clock differs')
            lifecycle._check_action_budget(action_step)
            return build_action_reference_cache(self.engine.parameters)
        cache=_root(references,self.device)
        def prepare():return prepare_transformer_action(lifecycle.actor,snapshot,self.engine.parameters,
            radius_fraction=lifecycle.config['radius_fraction'],embedding_names=self.embedding_names,
            actor_forward=lifecycle.actor_forward,reference_cache=cache,
            action_budget=lifecycle.current_action_budget)
        candidates=[]
        for _ in range(self.settings['steps']):
            prepared=_root(prepare,self.device)
            wire=_broadcast_action(prepared,self.device)
            editable=set(n for n,r in wire[3] if r!='scale')
            self._sync_tokenwise_multiplier()
            kl,gradients=_local(lambda:self._student(wire,teachers,editable,backward=True),self.device)
            _sum(kl)
            self._reduce_tokenwise_metrics()
            for gradient in gradients:_sum(gradient)
            metric=_root(lambda:self._actor_step(lifecycle,prepared,wire[3],gradients,kl,cache),self.device)
            if self.rank==0:candidates.append(metric)
            self.updates+=1
            del prepared,wire,gradients
        final=_root(prepare,self.device)  # Connected actor graph survives final GPT evaluation.
        wire=_broadcast_action(final,self.device);editable=set(n for n,r in wire[3] if r!='scale')
        self._sync_tokenwise_multiplier()
        final_kl,_=_local(lambda:self._student(wire,teachers,editable,backward=False),self.device);_sum(final_kl)
        self._reduce_tokenwise_metrics()
        counts=self.engine.compute.as_dict()
        for key,value in counts.items():self.compute[key]+=value
        self.blocks+=1
        self.last_panel={k:copy.deepcopy(v) for k,v in panel.items() if k!='inputs'}
        self.last_panel['final_replay_kl']=float(final_kl)
        if collector.active or collector.ctx is not None:raise RuntimeError('replay contaminated sensor state')
        cache_bytes=sum(t.storage_bytes for t in teachers)
        memory={}
        if self.device.type=='cuda':
            memory=dict(chisel_replay_cuda_allocated_bytes=torch.cuda.memory_allocated(self.device),
                chisel_replay_cuda_reserved_bytes=torch.cuda.memory_reserved(self.device),
                chisel_replay_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                chisel_replay_cuda_peak_scope='process peak since last external reset; not reset by maintenance')
        metrics=dict(**memory,chisel_replay_candidate_metrics=candidates,chisel_replay_final_kl=float(final_kl),
            chisel_replay_panel_id_sha256=panel['id_sha256'],chisel_replay_input_sha256=panel['input_sha256'],
            chisel_replay_sequence_ids=panel['ids'],chisel_replay_global_tokens=self.settings['sequences']*self.seq_len,
            chisel_replay_updates=self.updates,chisel_replay_blocks=self.blocks,
            chisel_replay_compute_scope='per-rank GPT calls/tokens; actor calls only on rank zero',chisel_replay_local_compute=counts,chisel_replay_local_cumulative_compute=dict(self.compute),
            chisel_replay_teacher_cache_bytes=cache_bytes,chisel_replay_teacher_cache_device=self.settings['teacher_cache_device'],
            chisel_replay_actor_forwards=self.settings['steps']+1,chisel_replay_actor_backwards=self.settings['steps'],
            chisel_extra_gpt_forwards=counts['teacher_forwards']+counts['student_forwards'],chisel_extra_gpt_backwards=counts['student_backwards'],
            chisel_online_auxiliary='past-replay KL primal-dual + stored 15-group percentage norm; final KL is in-sample',
            chisel_replay_wall_seconds=time.perf_counter()-started)
        if self.rank==0:
            metrics['chisel_replay_dual_multiplier']=self.dual.multipliers[0]
            metrics['chisel_actor_optimizer_step_calls']=lifecycle.meta_updates+self.updates
        if self.tokenwise_config is not None:
            metrics['chisel_replay_final_token_metrics']=dict(self._last_token_metrics)
            metrics['chisel_online_auxiliary']='tokenwise-v1 past-replay preservation + stored 15-group percentage norm; final measurements are in-sample'
        return final,metrics
