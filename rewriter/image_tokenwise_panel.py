"""Image-domain panel objective using the same per-output preservation algebra.

Each image is one context, regardless of its pixel count. The supplied teacher
provider reads immutable cached FP32 logits and converts to FP64 log probabilities;
this function never evaluates a teacher model or uses test images for fitting.
"""
import math
import torch
from .tokenwise_reference import logit_terms_and_cotangent,TOKEN_SUM_FIELDS


def image_tokenwise_panel(engine,parameters,regularizer,images,labels,*,config,dual,state_id,
                         teacher_provider,state_weight=1.,actor_credit=None,backward=True):
    if images.ndim!=4 or len(images)<1 or labels.shape!=(len(images),) or labels.dtype!=torch.long:
        raise ValueError('nonempty NCHW images and matching integer class labels required')
    if not math.isfinite(state_weight) or state_weight<=0:raise ValueError('positive finite state weight required')
    multiplier=float(dual.multipliers.get(state_id,0.));count=len(images)
    leaves={n:v.detach().requires_grad_(v.requires_grad and backward) for n,v in parameters.items()}
    sums=torch.zeros(len(TOKEN_SUM_FIELDS),device=engine.device,dtype=torch.float64);raw=[]
    student_ce=teacher_ce=0.;calls=0;rank_diagnostics=[]
    for start in range(0,count,engine.microbatch_sequences):
        stop=min(count,start+engine.microbatch_sequences);selected=images[start:stop].to(engine.device)
        target=labels[start:stop].to(engine.device);teacher=teacher_provider(start,stop)
        if teacher.teacher_id!=engine.teacher_id or teacher.log_probs.dtype!=torch.float64:
            raise ValueError('teacher identity/arithmetic differs')
        if teacher.data_role not in ('past_training','synthetic','withheld_training','fresh_audit') or (backward and teacher.data_role not in ('past_training','synthetic')):
            raise ValueError('audit-only or invalid image panel used for learning')
        if teacher.inputs.shape!=selected.shape or not torch.equal(teacher.inputs.to(engine.device),selected):
            raise ValueError('teacher images differ')
        if config.ranking_tie_policy=='legacy' and config.ranking_k>=teacher.log_probs.shape[-1]:
            raise ValueError('image ranking_tie_policy must be exact_teacher_ties when ranking_k covers all classes')
        engine._validate_replacements(leaves)
        with torch.enable_grad() if backward else torch.no_grad():logits=engine._forward(leaves,selected)
        if logits.ndim!=2 or len(logits)!=len(selected):raise ValueError('image model must return N by class logits')
        diagnostic={}
        cotangent,parts,kl=logit_terms_and_cotangent(logits,teacher.log_probs,config=config,multiplier=multiplier,
            panel_tokens=count,chunk_tokens=getattr(engine,'chunk_tokens',128),backward=backward,diagnostics=diagnostic)
        if diagnostic:rank_diagnostics.append(diagnostic)
        if backward:logits.backward(cotangent)
        with torch.no_grad():
            student_ce+=float(-logits.double().log_softmax(-1).gather(1,target[:,None]).sum())
            teacher_ce+=float(-teacher.log_probs.to(engine.device).gather(1,target[:,None]).sum())
        sums+=parts;raw.append(kl);calls+=1
    if backward:
        outputs=[regularizer];cotangents=[regularizer.new_tensor(state_weight)]
        for name,value in parameters.items():
            if value.requires_grad:
                outputs.append(value);g=leaves[name].grad
                cotangents.append((torch.zeros_like(value) if g is None else g)*state_weight)
        torch.autograd.backward(outputs,cotangents)
        if actor_credit is not None:
            original,factor_leaves=actor_credit
            pairs=[(o,l.grad) for o,l in zip(original,factor_leaves) if o.requires_grad and l.grad is not None]
            if pairs:torch.autograd.backward([o for o,g in pairs],[g for o,g in pairs])
    values=torch.cat(raw);parts=sums.cpu().tolist()
    if not torch.isfinite(values).all() or not all(math.isfinite(x) for x in parts):raise FloatingPointError('nonfinite image objective')
    linear,tail,ranking=(x/count for x in parts[:3]);penalty=linear+tail+ranking;mean=float(values.mean())
    result=dict(domain='cifar10',context_unit='image',context_count=count,image_count=count,
        # Compatibility aliases describe output contexts, never image pixels.
        token_count=count,sequence_count=count,mean_kl=mean,kl_penalty=penalty,
        objective=float(regularizer.detach())+penalty,norm_objective=float(regularizer.detach()),
        dual_linear_term=linear,augmented_penalty=tail,tail_quadratic_penalty=tail,ranking_penalty=ranking,
        student_ce=student_ce/count,teacher_ce=teacher_ce/count,ce_change=(student_ce-teacher_ce)/count,
        token_kl_quantiles=dict(zip(('min','median','p90','p99','p999','max'),torch.quantile(values,torch.tensor([0.,.5,.9,.99,.999,1.],dtype=torch.float64)).tolist())),
        token_tail_exceedances=int((values>config.tail_tolerance).sum()),negative_token_kl_count=int((values<0).sum()),
        argmax_changes=int(parts[3]),topk_order_changes=int(parts[4]),topk_membership_changes=int(parts[5]),
        ranking_k=min(config.ranking_k,logits.shape[-1]),ranking_tie_policy=config.ranking_tie_policy,
        compute=dict(teacher_forwards=0,teacher_images=0,cached_teacher_logit_images=count,
            student_forwards=calls,student_images=count,student_backwards=calls if backward else 0,
            student_backward_images=count if backward else 0),**dual.terms(mean,state_id))
    if rank_diagnostics:
        result.update(ranking_reduction='mean_micro_max_relation_context',unweighted_max_relative_rank=max(r['unweighted_max_relative_rank'] for r in rank_diagnostics),ranking_micro_diagnostics=rank_diagnostics)
    if result['ranking_k']==10:result.update(top10_order_changes=int(parts[4]),top10_membership_changes=int(parts[5]))
    if config.ranking_tie_policy=='exact_teacher_ties':result.update({name:int(parts[i]) for i,name in enumerate(TOKEN_SUM_FIELDS[13:],13)})
    return result
