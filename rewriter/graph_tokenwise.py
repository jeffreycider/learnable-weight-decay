"""Experimental strict tokenwise graph, isolated from the deployed trainer.

Reuses the verified checkpoint/address/dual lifecycle of NativeThinGradientGraph.
One rowwise aggregate traverses each student microbatch once. This object never
clips gradients, updates Adam/duals or selects states. No production registration.
"""
import torch
from .graph_offline import NativeThinGradientGraph
from .algebra import Factors,RawAction
from .edit_scope import scoped_action,keep_readonly_parameters
from .factorized_fp32 import factorized_logits
from .tiled_teacher_kl import tiled_teacher_kl
from .tiled_ranking_loss import tiled_topk_margin_loss


def device_tokenwise_terms(kl,ranking,config,multiplier):
    """Captured algebra; immutable config and multiplier validated before replay."""
    if multiplier.dtype!=torch.float64 or multiplier.ndim or multiplier.device!=kl.device:
        raise ValueError('resident FP64 multiplier scalar required')
    measured=kl.clamp_min(0.)
    linear=multiplier*((measured-config.mean_tolerance)/config.scale)
    tail=.5*config.rho*((measured-config.tail_tolerance)/config.scale).relu().square()
    rank=config.ranking_weight*ranking
    return linear,tail,rank


class TokenwiseThinGradientGraph(NativeThinGradientGraph):
    def _validate_ranking_scope(self, config):
        if getattr(config,'ranking_k',10)!=10 or getattr(config,'ranking_tie_policy','legacy')!='legacy':
            raise ValueError('tokenwise graph is qualified only for legacy top10 ranking')

    def __init__(self,*args,token_config,teacher_representation='log_probs',**kwargs):
        self._validate_ranking_scope(token_config)
        self.token_config=token_config
        if teacher_representation not in ('log_probs','logits'):raise ValueError('unknown teacher representation')
        self.teacher_representation=teacher_representation
        if kwargs.get('raw_kl_only',False):raise ValueError('tokenwise graph has one aggregate objective')
        if any(kwargs.get(k,default)!=default for k,default in
               [('dx_backend','ieee'),('linear_backend','ieee'),('activation_backend','native'),('fused_softcap',False)]):
            raise ValueError('tokenwise graph qualification is strict arithmetic only')
        if kwargs.get('kl_backend','chunked')!='chunked':raise ValueError('exact FP64 teacher logprobs required')
        super().__init__(*args,**kwargs)
        if teacher_representation=='logits' and self.teacher_features is None:raise ValueError('raw teacher requires verified head features')
        if self.engine.gpt_precision!='fp32' or getattr(self._dual,'config',None)!=token_config:
            raise ValueError('matching tokenwise configuration and FP32 GPT required')

    @torch.no_grad()
    def _teacher_slice(self,start,stop):
        if self.teacher_representation=='logits':
            from .teacher_cache import head_logits
            return head_logits(self.engine,self.teacher_features[start:stop])
        return super()._teacher_slice(start,stop)

    def _check_token_config(self,dual):
        if getattr(dual,'config',None)!=self.token_config:
            self.binding_valid=False
            raise ValueError('captured tokenwise objective changed')

    def bind_dual(self,dual,state_id):
        self._check_token_config(dual);return super().bind_dual(dual,state_id)

    def load_checkpoint(self,cache_reader,record,panel,dual,**kwargs):
        self._check_token_config(dual)
        return super().load_checkpoint(cache_reader,record,panel,dual,**kwargs)

    def _student_options(self, realized):
        """Strict defaults; isolated research subclasses may override explicitly."""
        return {}

    def _ranking_loss(self, teacher, logits):
        return tiled_topk_margin_loss(teacher,logits,k=10,margin_fraction=self.token_config.margin_fraction,
                                     arithmetic='fp64',return_teacher_topk=True)

    def _ranking_counts(self, teacher, logits, teacher_topk):
        t=teacher.flatten(0,1);s=logits.flatten(0,1);ti=teacher_topk[1];si=s.topk(10,dim=-1).indices
        return torch.stack(((t.argmax(-1)!=s.argmax(-1)).sum(),(ti!=si).any(-1).sum(),
            (ti.sort(-1).values!=si.sort(-1).values).any(-1).sum()))

    def _body(self):
        action=scoped_action(self.forward(self.snapshot),self.structural_editable if self.dynamic_zero_groups else self.editable)
        if self.dynamic_zero_groups:
            def masked(name,value):return torch.where(self.edit_masks[name],value,torch.zeros_like(value)) if name in self.edit_masks else value
            action=RawAction({n:Factors(masked(n,f.P),masked(n,f.Q),f.anchors) for n,f in action.matrices.items()},
                             {n:masked(n,v) for n,v in action.vectors.items()})
        action=RawAction({n:Factors(f.P*self.scales[n],f.Q*self.scales[n],f.anchors) for n,f in action.matrices.items()},
                         {n:v*self.scales[n] for n,v in action.vectors.items()})
        raw_norm=action.norm_sq().clamp_min(1e-24).sqrt()
        scale=torch.minimum(raw_norm.new_ones(()),self.snapshot.radius/(raw_norm+1e-12))
        realized=keep_readonly_parameters(self.engine.edited_parameters(action,scale),self.engine.parameters,
            self.structural_editable if self.dynamic_zero_groups else self.editable)
        if self.dynamic_zero_groups:
            realized={n:torch.where(self.edit_masks[n],v,self.engine.parameters[n]) if n in self.edit_masks else v for n,v in realized.items()}
        norm_result=self.grouped.evaluate(realized,edit_masks=self.edit_masks if self.dynamic_zero_groups else None)
        unweighted=100*norm_result.mean_relative_change;regularizer=self.shrinkage_coefficient*unweighted
        originals=[];leaves=[]
        def leaf(value):
            detached=value.detach().requires_grad_(value.requires_grad);originals.append(value);leaves.append(detached);return detached
        measured=RawAction({n:Factors(leaf(f.P),leaf(f.Q),f.anchors) for n,f in action.matrices.items()},
                           {n:leaf(v) for n,v in action.vectors.items()})
        scale_leaf=leaf(scale);rows=[];components=[];counts=[];cfg=self.token_config
        student_options=self._student_options(realized)
        for start in range(0,len(self.inputs),self.engine.microbatch_sequences):
            stop=min(start+self.engine.microbatch_sequences,len(self.inputs));teacher=self._teacher_slice(start,stop)
            logits=factorized_logits(self.engine.view.gpt,self.engine.parameters,self.engine.buffers,
                self.inputs[start:stop],measured,scale_leaf,realized_parameters=realized,**student_options)
            kl=tiled_teacher_kl(logits,teacher,reduction='none',teacher_is_log_probs=self.teacher_representation=='log_probs')
            ranking,teacher_topk=self._ranking_loss(teacher,logits)
            linear,tail,rank=device_tokenwise_terms(kl,ranking,cfg,self.dual_inputs.multiplier)
            total=(linear+tail+rank).sum()/self.inputs.numel()
            with torch.no_grad():
                counts.append(self._ranking_counts(teacher,logits,teacher_topk))
                rows.append(kl.detach().flatten());components.append(torch.stack((linear.sum(),tail.sum(),rank.sum())).detach())
            total.backward()
            del logits,teacher,total,kl,ranking,linear,tail,rank
        raw=torch.cat(rows);parts=torch.stack(components).sum(0)/self.inputs.numel();penalty=parts.sum();mean=raw.mean()
        outputs=[regularizer];cotangents=[self.norm_weight]
        for original,detached in zip(originals,leaves):
            if original.requires_grad and detached.grad is not None:
                outputs.append(original);cotangents.append(detached.grad*self.state_weight)
        torch.autograd.backward(outputs,cotangents)
        if self.retain_metrics:
            self.metric_tensors=dict(new_norms={n:v.detach() for n,v in norm_result.new_norms.items()},edit_norm_sq=norm_result.edit_norm_sq.detach())
            self.metric_action=RawAction({n:Factors(f.P.detach(),f.Q.detach(),f.anchors) for n,f in action.matrices.items()},
                                        {n:v.detach() for n,v in action.vectors.items()})
        return dict(mean_kl=mean,regularizer=regularizer.detach(),unweighted_norm_term=unweighted.detach(),
            objective=regularizer.detach()+penalty,finite=torch.isfinite(raw).all()&torch.isfinite(penalty)&torch.isfinite(regularizer.detach()),
            token_kl=raw,token_loss_components=parts,rank_change_counts=torch.stack(counts).sum(0),
            group_relative_change=torch.stack(tuple(norm_result.relative_changes.values())).detach(),raw_action_norm=raw_norm.detach(),global_scale=scale.detach())
