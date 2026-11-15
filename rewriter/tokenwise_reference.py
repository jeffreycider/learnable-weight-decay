"""Experimental eager tokenwise-v1 objective, independent of the legacy path.

Loss-row autograd produces a logit cotangent in bounded chunks. Exactly one
backward traverses the existing student GPT per microbatch. No extra GPT pass
is used for tail, ranking, or CE diagnostics. Teacher log probabilities can be
used as ranking targets because per-row additive offsets cancel in margins.
"""
from dataclasses import asdict, dataclass
import math

import torch

from rewriter.primal_dual import PrimalDual
from rewriter.ranking_loss import topk_margin_loss, tie_aware_topk_margin_loss
from rewriter.rank_preservation import topk_preservation_flags
from rewriter.token_preservation import token_preservation_terms

TOKEN_SUM_FIELDS=('dual_linear','tail_quadratic','ranking','argmax_changes',
    'top10_order_changes','top10_membership_changes','student_nll_sum',
    'teacher_nll_sum','logprob_shift_squared_sum','centered_logit_shift_squared_sum','ce_tokens',
    'tail_logit_cotangent_squared_sum','unit_rank_logit_cotangent_squared_sum',
    'tie_aware_argmax_changes','strict_order_changes','strict_membership_changes','strict_topk_changes')


@dataclass(frozen=True)
class TokenwiseConfig:
    mean_tolerance: float
    tail_tolerance: float
    scale: float
    rho: float
    ranking_weight: float = 0.
    margin_fraction: float = .5
    version: str = 'tokenwise-v1'
    dual_update_mode: str = 'unclipped_scaled'
    loss_backend: str = 'reference'
    ranking_k: int = 10
    ranking_tie_policy: str = 'legacy'
    ranking_loss_mode: str = 'absolute'

    def __post_init__(self):
        if type(self.ranking_k) is not int or self.ranking_k<1:raise ValueError('positive integer ranking_k required')
        if self.ranking_tie_policy not in ('legacy','exact_teacher_ties'):raise ValueError('invalid ranking tie policy')
        if self.loss_backend not in ('reference','tiled_fp64_v1','tiled_tie_fp64_v1','tiled_relative_fp64_v1','reference_micro_max_v1'):raise ValueError('invalid tokenwise loss backend')
        if self.ranking_loss_mode not in ('absolute','relative_gap','relative_gap_micro_max'):raise ValueError('invalid ranking loss mode')
        if self.ranking_loss_mode=='relative_gap' and (self.ranking_tie_policy!='exact_teacher_ties' or self.loss_backend not in ('reference','tiled_relative_fp64_v1')):
            raise ValueError('relative_gap requires exact teacher ties and its explicit backend')
        if self.loss_backend=='tiled_relative_fp64_v1' and self.ranking_loss_mode!='relative_gap':raise ValueError('relative backend requires relative_gap objective')
        if (self.ranking_loss_mode=='relative_gap_micro_max') != (self.loss_backend=='reference_micro_max_v1'):
            raise ValueError('micro-max requires its explicit objective and backend')
        if self.ranking_loss_mode=='relative_gap_micro_max' and self.ranking_tie_policy!='exact_teacher_ties':
            raise ValueError('micro-max requires exact teacher ties')
        # Normalize the two names requested by the concurrent online API review.
        # Serialized config and trainer CLI use the owner's canonical spellings.
        aliases={'scale-unclipped':'unclipped_scaled','tolerance-clipped':'clipped_relative'}
        object.__setattr__(self,'dual_update_mode',aliases.get(self.dual_update_mode,self.dual_update_mode))
        values=(self.mean_tolerance,self.tail_tolerance,self.scale,self.rho,self.ranking_weight,self.margin_fraction)
        if not all(math.isfinite(x) for x in values):raise ValueError('finite tokenwise coefficients required')
        if min(self.mean_tolerance,self.tail_tolerance,self.ranking_weight)<0 or self.scale<=0 or self.rho<=0:
            raise ValueError('invalid tokenwise tolerances, scale, rho or ranking weight')
        if not 0<self.margin_fraction<=1 or self.version!='tokenwise-v1':raise ValueError('invalid tokenwise version or margin')
        if self.dual_update_mode not in ('unclipped_scaled','clipped_relative'):raise ValueError('invalid dual update mode')
        if self.dual_update_mode=='clipped_relative' and self.mean_tolerance<=0:raise ValueError('clipped dual needs positive mean tolerance')


class TokenwiseDual(PrimalDual):
    """Same per-fitting-state lifecycle; fixed scale rather than tolerance units."""
    def __init__(self,state_ids,*,config,dual_lr):
        # Parent validates the state IDs and learning rate. Its legacy terms are
        # never used. Zero tolerances are legal for this experimental version.
        super().__init__(state_ids,tolerance=max(config.mean_tolerance,1e-300),rho=config.rho,dual_lr=dual_lr)
        self.config=config
        self.tolerance=config.mean_tolerance

    def terms(self,kl,state_id=None):
        kl=float(kl)
        if not math.isfinite(kl):raise FloatingPointError('nonfinite KL')
        violation=(max(kl,0.)-self.config.mean_tolerance)/self.config.scale
        multiplier=self.multipliers.get(state_id,0.)
        direction=violation if self.config.dual_update_mode=='unclipped_scaled' else max(-1.,min(1.,(kl-self.config.mean_tolerance)/self.config.mean_tolerance))
        return dict(constraint_violation=violation,positive_violation=max(violation,0.),
            dual_multiplier=multiplier,kl_tolerance=self.config.mean_tolerance,
            normalization_scale=self.config.scale,feasible=violation<=0.,dual_ascent_direction=direction)

    def step(self,measurements,*,expected_ids):
        rows=list(measurements);ids=[int(i) for i,_ in rows]
        if len(ids)!=len(set(ids)) or set(ids)!=set(expected_ids):raise ValueError('dual step needs one measurement per fitting state')
        if not set(ids)<=self.multipliers.keys():raise ValueError('held-out checkpoint cannot update dual state')
        proposed={i:max(0.,self.multipliers[i]+self.dual_lr*self.terms(k,i)['dual_ascent_direction']) for i,k in rows}
        if not all(math.isfinite(v) for v in proposed.values()):raise FloatingPointError('nonfinite dual update')
        self.multipliers.update(proposed);self.updates+=1

    def state_dict(self):
        return dict(schema='tokenwise-v1',config=asdict(self.config),dual_lr=self.dual_lr,
            multipliers=dict(self.multipliers),updates=self.updates)

    def load_state_dict(self,state):
        if 'ranking_constraint' in state:
            raise ValueError('Ranking constraint state requires RankingDual')
        state=dict(state)
        state['config']=_canonical_backend_config(state.get('config'))
        if any(state.get(k)!=self.state_dict()[k] for k in ('schema','config','dual_lr')):
            raise ValueError('tokenwise dual resume configuration changed')
        values={int(k):float(v) for k,v in state['multipliers'].items()}
        if values.keys()!=self.multipliers.keys() or not all(math.isfinite(v) and v>=0 for v in values.values()):
            raise ValueError('invalid tokenwise dual IDs or values')
        if type(state['updates']) is not int or state['updates']<0:raise ValueError('invalid dual clock')
        self.multipliers=values;self.updates=state['updates']


def tokenwise_config_from_options(args):
    """Fail closed on unsupported backends before touching a checkpoint/GPU."""
    backend=getattr(args,'token_loss_backend',None)
    extras=(getattr(args,'ranking_k',None),getattr(args,'ranking_tie_policy',None),backend,args.token_tail_tolerance,args.token_normalization_scale,args.ranking_weight,args.ranking_margin_fraction,getattr(args,'token_dual_update',None))
    if args.objective!='tokenwise-v1':
        if any(x is not None for x in extras):raise ValueError('tokenwise flags require --objective tokenwise-v1')
        return None
    if args.offline_backend not in ('reference','tokenwise_graph'):raise ValueError('tokenwise-v1 requires reference backend or explicit tokenwise_graph')
    if args.gpt_precision!='fp32':raise ValueError('tokenwise-v1 requires strict FP32 GPT')
    if args.gradient_diagnostics:raise ValueError('separate component VJPs are not implemented for tokenwise-v1')
    if args.token_tail_tolerance is None or args.token_normalization_scale is None:
        raise ValueError('tokenwise-v1 requires explicit tail tolerance and normalization scale')
    return TokenwiseConfig(args.kl_tolerance,args.token_tail_tolerance,args.token_normalization_scale,
        args.augmented_rho,0. if args.ranking_weight is None else args.ranking_weight,
        .5 if args.ranking_margin_fraction is None else args.ranking_margin_fraction,
        dual_update_mode=getattr(args,'token_dual_update',None) or 'unclipped_scaled',loss_backend=backend or 'reference',
        ranking_k=getattr(args,'ranking_k',None) or 10,ranking_tie_policy=getattr(args,'ranking_tie_policy',None) or 'legacy')


def _canonical_backend_config(config):
    # Old checkpoints without this opt-in used the unchanged reference algebra.
    if config is None:return None
    value=dict(config);value.setdefault('loss_backend','reference');value.setdefault('ranking_k',10);value.setdefault('ranking_tie_policy','legacy');value.setdefault('ranking_loss_mode','absolute');return value


def validate_objective_resume(metadata,config):
    expected=None if config is None else asdict(config)
    if _canonical_backend_config(metadata.get('tokenwise_objective'))!=expected:
        raise ValueError('objective version/config changed; use explicit actor-only --primal-init')


def tokenwise_feasibility_score(rows,mean_tolerance):
    """Stationary score: mean feasibility first, then fixed tail/rank/norm loss.

    Learned dual multipliers never enter model/LR selection. The mean-infeasible
    branch is unchanged from the existing trainer, in tolerance-relative units.
    """
    if not rows or mean_tolerance<=0:raise ValueError('nonempty readout and positive mean tolerance required')
    worst=max(max(r['mean_kl'],0.)/mean_tolerance-1 for r in rows)
    stationary=sum(r['norm_objective']+r['tail_quadratic_penalty']+r['ranking_penalty'] for r in rows)/len(rows)
    # Strictly monotone within the feasible class and below every infeasible
    # score, even when a feasible-mean panel has a very large tail penalty.
    return 1+worst if worst>0 else (2/math.pi)*math.atan(stationary)


def logit_terms_and_cotangent(student,teacher_log_probs,*,config,multiplier,panel_tokens,
                              chunk_tokens=128,backward=True,inputs=None,balance=False,diagnostics=None):
    """Reference API for systems qualification. Cotangent includes 1/panel_tokens.

    No checkpoint-batch weight here: apply it once when propagating the edited
    parameter bank to the actor. Metrics are additive sums except the raw KL
    vector, which permits exact panel quantiles. The diagnostics vector is
    ordered by TOKEN_SUM_FIELDS. Inputs optionally enable CE.
    """
    if config.loss_backend=='reference_micro_max_v1':
        from .tokenwise_micro_max import logit_terms_and_cotangent as micro_max
        return micro_max(student,teacher_log_probs,config=config,multiplier=multiplier,panel_tokens=panel_tokens,
            chunk_tokens=chunk_tokens,backward=backward,inputs=inputs,balance=balance,diagnostics=diagnostics)
    if config.loss_backend in ('tiled_fp64_v1','tiled_tie_fp64_v1','tiled_relative_fp64_v1'):
        if config.loss_backend=='tiled_relative_fp64_v1':
            from .tokenwise_tiled_relative import logit_terms_and_cotangent as tiled
        elif config.loss_backend=='tiled_tie_fp64_v1':
            from .tokenwise_tiled_tie import logit_terms_and_cotangent as tiled
        else:
            from .tokenwise_tiled import logit_terms_and_cotangent as tiled
        return tiled(student,teacher_log_probs,config=config,multiplier=multiplier,panel_tokens=panel_tokens,
                     chunk_tokens=chunk_tokens,backward=backward,inputs=inputs,balance=balance)
    if student.shape!=teacher_log_probs.shape or student.dtype!=torch.float32 or teacher_log_probs.dtype!=torch.float64:
        raise ValueError('matching FP32 student/FP64 teacher arrays required')
    flat=student.detach().reshape(-1,student.shape[-1]);teacher=teacher_log_probs.detach().reshape_as(flat)
    k=min(config.ranking_k,student.shape[-1])
    if panel_tokens<len(flat) or chunk_tokens<1 or (config.ranking_tie_policy=='legacy' and k>=student.shape[-1]):raise ValueError('invalid token normalization or legacy ranking vocabulary')
    rank_loss=tie_aware_topk_margin_loss if config.ranking_tie_policy=='exact_teacher_ties' else topk_margin_loss
    if config.ranking_loss_mode=='relative_gap':
        from .relative_ranking_loss import relative_tie_aware_topk_margin_loss
        rank_loss=relative_tie_aware_topk_margin_loss
    gradient=torch.empty_like(flat) if backward else None
    sums=torch.zeros(len(TOKEN_SUM_FIELDS),device=student.device,dtype=torch.float64);raw=[]
    labels=inputs.roll(-1,1).flatten() if inputs is not None else None
    length=inputs.shape[1] if inputs is not None else None
    for start in range(0,len(flat),chunk_tokens):
        stop=min(start+chunk_tokens,len(flat))
        # Root helpers retain only this row chunk's graph, never a full-vocab
        # FP64 activation bank across the entire GPT microbatch.
        with torch.enable_grad() if backward or balance else torch.no_grad():
            s=flat[start:stop].requires_grad_(backward or balance)
            t=teacher[start:stop].to(student.device)
            lp=s.double().log_softmax(-1)
            kl=(t.exp()*(t-lp)).sum(-1)
            teacher_topk=None
            if (config.ranking_weight or balance) and config.ranking_loss_mode=='relative_gap':
                ranking,teacher_topk=rank_loss(t,s.double(),k=k,margin_fraction=config.margin_fraction,return_teacher_topk=True)
            else:
                ranking=rank_loss(t,s.double(),k=k,margin_fraction=config.margin_fraction) if config.ranking_weight or balance else None
            terms=token_preservation_terms(kl,multiplier=multiplier,mean_tolerance=config.mean_tolerance,
                tail_tolerance=config.tail_tolerance,scale=config.scale,rho=config.rho,
                ranking_loss=ranking,ranking_weight=config.ranking_weight)
            if balance:
                tail_g=torch.autograd.grad(terms['tail_quadratic'].sum()/panel_tokens,s,retain_graph=True)[0]
                rank_g=torch.autograd.grad(ranking.sum()/panel_tokens,s,retain_graph=backward)[0]
                sums[11]+=tail_g.double().square().sum();sums[12]+=rank_g.double().square().sum()
            if backward:gradient[start:stop]=torch.autograd.grad(terms['total'].sum()/panel_tokens,s)[0]
        with torch.no_grad():
            if teacher_topk is None:teacher_topk=t.topk(k,dim=-1)
            student_topk=s.topk(k,dim=-1)
            ti=teacher_topk[1];si=student_topk.indices
            if config.ranking_tie_policy=='exact_teacher_ties':
                flags=topk_preservation_flags(t,s,k=k,teacher_topk=teacher_topk,student_topk=student_topk)
                sums[13:17]+=torch.stack([flags[name].sum() for name in ('tie_aware_argmax_changed','strict_order_changed','strict_membership_changed','strict_topk_changed')]).double()
            counts=torch.stack(((t.argmax(-1)!=s.argmax(-1)).sum(),(ti!=si).any(-1).sum(),
                (ti.sort(-1).values!=si.sort(-1).values).any(-1).sum())).double()
            sums[:3]+=torch.stack([terms[k].sum() for k in ('dual_linear','tail_quadratic','ranking')])
            sums[3:6]+=counts
            if inputs is not None:
                valid=torch.arange(start,stop,device=student.device)%length<length-1
                target=labels[start:stop,None].long()
                sums[6]+=(-lp.gather(1,target).squeeze(1))[valid].sum()
                sums[7]+=(-t.gather(1,target).squeeze(1))[valid].sum()
                gap=lp-t;sums[8]+=gap.square().sum();sums[9]+=(gap-gap.mean(-1,keepdim=True)).square().sum();sums[10]+=valid.sum()
            raw.append(kl.detach())
    return gradient.reshape_as(student) if backward else None,sums.detach(),torch.cat(raw).cpu()


def tokenwise_panel(engine,parameters,regularizer,inputs,*,config,dual,state_id,
                    state_weight=1.,actor_credit=None,teacher_provider=None,backward=True,return_ce=False,balance=False):
    if inputs.ndim!=2 or inputs.numel()==0 or state_weight<=0:raise ValueError('nonempty panel and positive state weight required')
    # Immutable coefficients are read once for the complete checkpoint panel.
    multiplier=float(dual.multipliers.get(state_id,0.))
    leaves={n:v.detach().requires_grad_(v.requires_grad and backward) for n,v in parameters.items()}
    sums=torch.zeros(len(TOKEN_SUM_FIELDS),device=engine.device,dtype=torch.float64);raw=[];rank_diagnostics=[]
    for start in range(0,len(inputs),engine.microbatch_sequences):
        selected=inputs[start:start+engine.microbatch_sequences]
        teacher=teacher_provider(start,start+len(selected)) if teacher_provider else engine.cache_teacher(selected,data_role='past_training')[0]
        if teacher.teacher_id!=engine.teacher_id or teacher.log_probs.dtype!=torch.float64:raise ValueError('teacher identity/arithmetic mismatch')
        if teacher.data_role not in ('past_training','synthetic','fresh_next_training_audit') or (backward and teacher.data_role=='fresh_next_training_audit'):
            raise ValueError('audit-only or invalid teacher role used for learning')
        if tuple(teacher.inputs.shape)!=tuple(selected.shape) or not torch.equal(teacher.inputs.to(selected.device),selected):
            raise ValueError('teacher inputs differ from requested panel')
        engine._validate_replacements(leaves)
        selected=selected.to(engine.device)
        with torch.enable_grad() if backward else torch.no_grad():logits=engine._forward(leaves,selected)
        vocabulary=logits.shape[-1]
        engine.compute.student_forwards+=1;engine.compute.student_tokens+=selected.numel()
        diagnostic={}
        cotangent,parts,kl=logit_terms_and_cotangent(logits,teacher.log_probs,config=config,multiplier=multiplier,
            panel_tokens=inputs.numel(),chunk_tokens=engine.chunk_tokens,backward=backward,inputs=selected if return_ce else None,balance=balance,diagnostics=diagnostic)
        if diagnostic:rank_diagnostics.append(diagnostic)
        if backward:
            logits.backward(cotangent)
            engine.compute.student_backwards+=1;engine.compute.student_backward_tokens+=selected.numel()
        sums+=parts;raw.append(kl)
        del teacher,logits,cotangent,parts,kl
    values=torch.cat(raw);parts=sums.cpu().tolist()
    if not torch.isfinite(values).all() or not all(math.isfinite(x) for x in parts):raise FloatingPointError('nonfinite tokenwise measurements')
    if backward:
        outputs=[regularizer];cotangents=[regularizer.new_tensor(state_weight)]
        for name,value in parameters.items():
            if value.requires_grad:
                g=leaves[name].grad
                outputs.append(value);cotangents.append((torch.zeros_like(value) if g is None else g)*state_weight)
        torch.autograd.backward(outputs,cotangents)
        if actor_credit is not None:
            original,factor_leaves=actor_credit
            pairs=[(o,l.grad) for o,l in zip(original,factor_leaves) if o.requires_grad and l.grad is not None]
            if pairs:torch.autograd.backward([o for o,g in pairs],[g for o,g in pairs])
    count=inputs.numel();linear,tail,ranking=(x/count for x in parts[:3]);penalty=linear+tail+ranking
    mean=float(values.mean());constraint=dual.terms(mean,state_id)
    result=dict(mean_kl=mean,kl_penalty=penalty,objective=float(regularizer.detach())+penalty,
        dual_linear_term=linear,augmented_penalty=tail,tail_quadratic_penalty=tail,ranking_penalty=ranking,
        norm_objective=float(regularizer.detach()),token_count=count,sequence_count=len(inputs),
        token_kl_quantiles=dict(zip(('min','median','p90','p99','p999','max'),torch.quantile(values,torch.tensor([0.,.5,.9,.99,.999,1.],dtype=torch.float64)).tolist())),
        token_tail_exceedances=int((values>config.tail_tolerance).sum()),negative_token_kl_count=int((values<0).sum()),
        mean_nonnegative_token_kl=float(values.clamp_min(0).mean()),
        argmax_changes=int(parts[3]),ranking_k=min(config.ranking_k,vocabulary),ranking_tie_policy=config.ranking_tie_policy,
        topk_order_changes=int(parts[4]),topk_membership_changes=int(parts[5]),
        **constraint)
    if rank_diagnostics:
        result.update(ranking_reduction='mean_micro_max_relation_context',unweighted_max_relative_rank=max(r['unweighted_max_relative_rank'] for r in rank_diagnostics),ranking_micro_diagnostics=rank_diagnostics)
    if result['ranking_k']==10:
        result.update(top10_order_changes=int(parts[4]),top10_membership_changes=int(parts[5]))
    if config.ranking_tie_policy=='exact_teacher_ties':
        result.update({name:int(parts[i]) for i,name in enumerate(TOKEN_SUM_FIELDS[13:],13)})
    if balance:
        if parts[12]<=0:raise ValueError('ranking calibration needs nonzero unit-rank logit cotangent')
        result.update(tail_logit_cotangent_norm=parts[11]**.5,unit_rank_logit_cotangent_norm=parts[12]**.5,
            gamma_balance_ratio=(parts[11]/parts[12])**.5)
    if return_ce:
        if parts[10]==0:raise ValueError('CE needs sequence length at least two')
        result.update(student_ce=parts[6]/parts[10],teacher_ce=parts[7]/parts[10],ce_change=(parts[6]-parts[7])/parts[10],
            ce_tokens=int(parts[10]),logprob_rms_shift=(parts[8]/(count*vocabulary))**.5,
            centered_logit_rms_shift=(parts[9]/(count*vocabulary))**.5,
            ce_scope='known next tokens within inputs; final token excluded')
    return result
