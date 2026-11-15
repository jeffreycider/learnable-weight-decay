"""One-rank and two-rank CPU/gloo staging for adaptive ordinary-pass lag-one credit.

This is not eight-rank qualification and adds no norm/preservation objective.
An exact activation transition commits the teaching-snapshot edit before the
first suffix forward. Both online optimizers start fresh; exact resume restores
both, including the retained pending action graph.
"""
import copy
import torch
from baseline.adaptive_transition import FAMILY,validate_adaptive_config,load_sources,assert_activation,sha256
from .adaptive_online import AdaptiveOnlineBinding,AdaptiveOnlineLifecycle
from .grouped_coordinate_chisel import GroupedCoordinateChisel
from .prefix_cache import unpack_record
from .state_capture import DistributedStateObserver
from .transformer_online import authoritative_call,collective_local_call,assert_replicated_before_edit


class AdaptiveChiselAdapter:
    def __init__(self,model,optimizers,config,*,mode):
        validate_adaptive_config(config)
        self.distributed=torch.distributed.is_initialized()
        self.rank=torch.distributed.get_rank() if self.distributed else 0
        world=torch.distributed.get_world_size() if self.distributed else 1
        if world!=config['world_size']:raise ValueError('configured and actual world sizes differ')
        if world==2 and (torch.distributed.get_backend()!='gloo' or next(model.parameters()).device.type!='cpu'):
            raise ValueError('two-rank adaptive staging is CPU/gloo only, not NCCL qualification')
        if mode not in ('shadow','learned') or mode!=config.get('mode'):raise ValueError('adaptive adapter mode mismatch')
        self.config=config;self.c=copy.deepcopy(config['transformer_chisel']);self.mode=mode
        self.observer=DistributedStateObserver(model,optimizers,config,sensor_config=self.c['sensor_config'],start_step=self.c['prefix_steps'])
        self.collector=self.observer.collector;self.collector.active=False
        self.params=self.observer.params;self.device=self.observer.device
        self.lifecycle=None;self.metrics={};self.initialized=False

    def _initialize(self):
        policy,state,case=load_sources(self.c,device=self.device)
        snapshot=unpack_record(state['observation'],device=self.device).snapshot
        architecture=self.c['architecture']
        devices=[self.device.index or 0] if self.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            actor=GroupedCoordinateChisel(**{k:architecture[k] for k in ('rank','trajectory_dim','model_dim','num_layers','latent_tokens','fresh_weight_core','factor_precision','residual_precision')}).to(self.device)
            if actor.configuration()!=architecture:raise ValueError('unsupported grouped architecture')
            actor.load_state_dict(policy['actor'],strict=True)
            private=actor.make_private(snapshot,state_key=tuple(case['private_identity']['state_key']),seed=0)
            private.load_state_dict(policy['private'][self.c['case_id']],strict=True)
            binding=AdaptiveOnlineBinding.from_fitted(actor,private,snapshot,run_id=self.c['run_id'])
        head=[];body=[]
        for n,p in binding.shared.named_parameters():
            (head if n.startswith(('matrix_output.','vector_output.')) else body).append(p)
        shared=torch.optim.Adam([dict(params=head,lr=self.c['head_lr'],role='heads'),dict(params=body,lr=self.c['body_lr'],role='body')],betas=tuple(self.c['betas']),eps=self.c['shared_eps'])
        private_opt=torch.optim.Adam(binding.private.parameters(),lr=self.c['private_lr'],betas=tuple(self.c['betas']),eps=self.c['private_eps'])
        self.lifecycle=AdaptiveOnlineLifecycle(binding,shared,private_opt,global_token_count=self.config['batch_tokens'],
            final_action_step=self.config['max_steps']-1,radius_fraction=self.c['radius_fraction'],shared_clip=self.c['shared_clip'],
            private_clip=self.c['private_clip'],parameter_view=self.c['parameter_view'],embedding_names=sorted(self.observer.embedding_names))
        return snapshot,state

    def before_step(self,t,model,optimizers):
        if not self.initialized:raise RuntimeError('restore activation or online checkpoint before ordinary pass')
        self.metrics={}
        collective_local_call(lambda:self.observer.before_step(t,model,optimizers),device=self.device)

    def before_microbatch(self,t,sequence_ids,inputs):self.observer.before_microbatch(t,sequence_ids,inputs)

    def after_backward(self,t,model):
        def check_gradients():
            if any(p.grad is None for p in self.params.values()):
                raise ValueError('ordinary observer requires full gradients, as baseline/train.py does')
        collective_local_call(check_gradients,device=self.device)
        self.observer.after_backward(t,model)
        def consume():
            if self.mode=='learned' and self.lifecycle.pending is not None:
                gradients={n:p.grad if p.grad is not None else torch.zeros_like(p) for n,p in self.params.items()}
                self.metrics.update(self.lifecycle.consume(gradients,gradient_step=t))
        authoritative_call(consume,device=self.device)

    def after_step(self,t,model,optimizers):
        if self.distributed:self.metrics.update(assert_replicated_before_edit(model,device=self.device))
        momentum=self.observer.momentum_projections()
        def act():
            snapshot,_,error=self.observer.snapshot_from_momentum(t,momentum)
            self.metrics['chisel_allones_relative_error']=error
            if self.mode=='learned':self.metrics.update(self.lifecycle.commit(snapshot,self.params,action_step=t,base_step_completed=t))
            else:self.metrics.update(shadow_identity=True,meta_updates=0)
            self.metrics.update(chisel_extra_gpt_forwards=0,chisel_extra_gpt_backwards=0,
                chisel_online_auxiliary='none; ordinary-next-task credit only',chisel_family=FAMILY)
        authoritative_call(act,device=self.device)
        self._broadcast_executed()
        self.observer.release_step(t)

    def _broadcast_executed(self):
        if self.distributed and self.mode=='learned':
            for p in self.params.values():torch.distributed.broadcast(p.detach(),src=0)
            authoritative_call(lambda:self.lifecycle.rebind_after_broadcast(self.params),device=self.device)

    def state_dict(self):
        if not self.initialized:raise RuntimeError('uninitialized adaptive adapter')
        observation=collective_local_call(self.observer.state_dict,device=self.device)
        return dict(schema=FAMILY,bootstrap=False,rank=self.rank,mode=self.mode,settings=copy.deepcopy(self.c),
            observer=observation,learner=authoritative_call(lambda:self.lifecycle.state_dict(),device=self.device))

    def load_state_dict(self,saved):
        def check_local():
            if (saved.get('schema')!=FAMILY or saved.get('mode')!=self.mode or saved.get('settings')!=self.c or
                    saved.get('rank',0)!=self.rank):
                raise ValueError('adaptive adapter checkpoint identity/config/rank mismatch')
            if self.initialized:raise RuntimeError('adapter already initialized')
            if saved.get('bootstrap'):
                if saved.get('completed_steps')!=self.c['prefix_steps']:raise ValueError('activation clock mismatch')
                if sha256(self.c['teaching_state_path'])!=self.c['teaching_state_sha256']:raise ValueError('teaching state SHA256 mismatch')
                state=torch.load(self.c['teaching_state_path'],map_location=self.device,weights_only=False,mmap=True)
                assert_activation(self.observer.model.state_dict(),state['model'])
            else:
                next_step=saved['observer']['next_step']
                if type(next_step) is not int or not self.c['prefix_steps']<=next_step<=self.config['max_steps']:
                    raise ValueError('invalid adaptive observer checkpoint clock')
                if self.rank==0:
                    learner=saved['learner']
                    if self.mode=='learned' and (learner['last_committed_step']!=next_step-1 or
                            learner['meta_updates']!=next_step-self.c['prefix_steps']):
                        raise ValueError('adaptive observer and learner checkpoint clocks differ')
                elif saved['learner'] is not None:raise ValueError('non-root checkpoint must not own learner')
        collective_local_call(check_local,device=self.device)
        if self.distributed:assert_replicated_before_edit(self.observer.model,device=self.device)
        def initialize():
            snapshot,state=self._initialize()
            if saved.get('bootstrap'):
                if saved.get('completed_steps')!=self.c['prefix_steps']:raise ValueError('activation clock mismatch')
                # Includes buffers as well as parameters; restore already loaded GPT.
                assert_activation(self.observer.model.state_dict(),state['model'])
                if self.mode=='learned':
                    step=self.c['prefix_steps']-1
                    self.metrics.update(self.lifecycle.commit(snapshot,self.params,action_step=step,base_step_completed=step))
            else:self.lifecycle.load_state_dict(saved['learner'],self.params)
        authoritative_call(initialize,device=self.device)
        if not saved.get('bootstrap'):
            collective_local_call(lambda:self.observer.load_state_dict(saved['observer']),device=self.device)
        self._broadcast_executed()
        self.initialized=True

    def close(self):self.observer.close()
