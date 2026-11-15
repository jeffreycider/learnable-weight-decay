"""Literal single-launch offline Chisel update with frozen teacher targets.

All actor, edit, student GPT, loss, reverse-mode and Adam stages run inside one
cooperative CUDA launch. Construction stages immutable inputs and allocates the
plan. Each update supports one checkpoint and a fixed batch of sequences.
"""
from __future__ import annotations
import math
import os
import torch
from .algebra import Factors,RawAction
from .edit_scope import editable_names,editable_weight_norm_sq,scoped_snapshot
from .grouped_norm import GroupedNormObjective
from .persistent_controller import append_global_rewriter,_staged_snapshot
from .persistent_chisel import append_transformer_chisel
from .persistent_gpt import append_gpt,append_gpt_body,unary,rounded
from .persistent_plan import PersistentTrainingPlan,float_bits


def typed_empty(plan,shape,dtype):
    value=torch.empty(shape,device=plan.device,dtype=dtype);plan.storage.append(value);return value


def constants(plan,values,dtype=torch.int64):
    value=torch.tensor(values,device=plan.device,dtype=dtype);plan.storage.append(value);return value


def pointer_list(plan,values):return constants(plan,[v.data_ptr() for v in values])


def double_reduce(plan,value,rows,cols):
    out=typed_empty(plan,(cols,),torch.float64)
    plan.append([24,value.data_ptr(),out.data_ptr(),0,0,0,rows,cols]);return out


def pair_stats(plan,a,b=None,mode=0):
    chunk=4096;parts=(a.numel()+chunk-1)//chunk
    partial=typed_empty(plan,(parts,2),torch.float64)
    plan.append([25,a.data_ptr(),0 if b is None else b.data_ptr(),partial.data_ptr(),0,0,a.numel(),chunk,mode])
    return double_reduce(plan,partial,parts,2)


def append_kl(plan,logits,teacher_log_probs,*,softcap_gradient_scale=None,softcap_fp32=False):
    rows,cols=logits.shape
    if teacher_log_probs.shape!=logits.shape or teacher_log_probs.dtype!=torch.float64:
        raise ValueError('cached teacher targets must be full FP64 log probabilities with matching shape')
    derivative=plan.empty(logits.shape);partial=typed_empty(plan,(rows,),torch.float64)
    plan.append([23,logits.data_ptr(),teacher_log_probs.data_ptr(),derivative.data_ptr(),partial.data_ptr(),0,rows,cols,
                 int(softcap_gradient_scale is not None),float_bits(softcap_gradient_scale or 0.),int(softcap_fp32)])
    return double_reduce(plan,partial,rows,1),derivative


def append_student_kl_chunks(plan,parameters,buffers,inputs,teacher,*,num_layers,chunk_tokens,fraction,gpt_precision='baseline',compensated_backward=False,thin_factors=None):
    """Bound vocabulary storage while retaining one complete GPT microbatch.

    Teacher targets stay untouched. Head weight/bias contributions accumulate
    in FP32 and round once, matching the original BF16 parameter-cast VJP.
    """
    fp32=gpt_precision=='fp32'
    if fp32:
        from .persistent_fp32 import append_body
        x,body_back=append_body(plan,parameters,buffers,inputs,num_layers=num_layers,compensated_backward=compensated_backward,thin_factors=thin_factors)
    else:x,body_back=append_gpt_body(plan,parameters,buffers,inputs,num_layers=num_layers)
    rows,width=x.shape;weight=parameters['proj.weight'];bias=parameters['proj.bias'];vocab=weight.shape[0]
    w=weight if fp32 else rounded(plan,weight,out=plan.scratch(weight.shape,'chunk-head-weight'))
    b=bias if fp32 else rounded(plan,bias,out=plan.scratch(bias.shape,'chunk-head-bias'))
    factor=None if thin_factors is None else thin_factors.get('proj.weight')
    if factor is None:dw_total=plan.scratch(weight.shape,'chunk-head-dw-total')
    else:dw_total=Factors(plan.scratch(factor.P.shape,'chunk-head-dp-total'),
                         plan.scratch(factor.Q.shape,'chunk-head-dq-total'),factor.anchors)
    db_total=plan.scratch(bias.shape,'chunk-head-db-total')
    dx=plan.empty(x.shape);losses=[];fractions=[]
    for start in range(0,rows,chunk_tokens):
        stop=min(rows,start+chunk_tokens);count=stop-start
        with plan.temporary_arena('vocab-head-'+str(count)):
            fused=os.environ.get('CHISEL_FUSED_HEAD_KL')=='1'
            if fused:
                raw=plan.gemm(x[start:stop],w,tb=True,epilogue=1,bias=b,bf16=not fp32,ieee=fp32,round_output=not fp32,immutable_b=True)
                kl,draw=append_kl(plan,raw,teacher[start:stop],softcap_gradient_scale=fraction*count/rows,softcap_fp32=fp32)
                dlogits=draw
                if stop==rows:logits=unary(plan,raw,3,out=plan.scratch(raw.shape,'debug-head-logits'))
            else:
                raw=plan.gemm(x[start:stop],w,tb=True,epilogue=1,bias=b,bf16=not fp32,ieee=fp32,immutable_b=True)
                if not fp32:raw=rounded(plan,raw)
                logits=unary(plan,raw,3)
                kl,dlogits=append_kl(plan,logits,teacher[start:stop])
                dlogits=unary(plan,dlogits,5,scale=fraction*count/rows)
                draw=unary(plan,raw,8 if fp32 else 4,dlogits)
            if factor is not None:
                from .persistent_fp32 import linear_vjp
                dx_raw,dw,db=linear_vjp(plan,x[start:stop],w,draw,compensated_backward=compensated_backward,factors=factor,split_dx=True)
            else:
                dx_raw=plan.gemm(draw,w,bf16=not fp32,split=True,ieee=fp32 and not compensated_backward)
                dw=plan.gemm(draw,x[start:stop],ta=True,bf16=not fp32,ieee=fp32)
                db=plan.bias_gradient(draw)
            if fp32:plan.copy_into(dx_raw,dx[start:stop])
            else:rounded(plan,dx_raw,out=dx[start:stop])
            pairs=[(dw, dw_total)] if factor is None else [(dw.P,dw_total.P),(dw.Q,dw_total.Q)]
            pairs.append((db,db_total))
            if start==0:
                for value,total in pairs:plan.copy_into(value,total)
            else:
                for value,total in pairs:
                    plan.append([8,total.data_ptr(),value.data_ptr(),total.data_ptr(),0,0,total.numel(),float_bits(1.),float_bits(1.)])
        losses.append(kl);fractions.append(count/rows)
    loss=typed_empty(plan,(1,),torch.float64);pointers=pointer_list(plan,losses);weights=constants(plan,fractions,torch.float64)
    plan.append([40,pointers.data_ptr(),loss.data_ptr(),0,0,0,len(losses),float_bits(1.),weights.data_ptr()])
    dw=dw_total if fp32 else rounded(plan,dw_total);db=db_total if fp32 else rounded(plan,db_total)
    gradients=body_back(dx);gradients.update({'proj.weight':dw,'proj.bias':db})
    return loss,gradients,logits,dlogits


def append_edits(plan,action,base,grouped,editable,radius,*,fp32=False,retain_dense_deltas=True):
    scales={};scaled_matrices={};scaled_vectors={};norm_parts=[]
    plan.checkpoint_scale_slots=[];plan.checkpoint_radius_slots=[]
    rank=next(iter(action.matrices.values())).P.shape[1]
    for group,names in grouped.groups.items():
        count=sum(base[n].numel() for n in names)
        rms=float(grouped.reference_norms[group])/math.sqrt(count)
        for name in names:
            value=base[name];dimension=rank*(sum(value.shape)-rank) if value.ndim==2 else value.numel()
            scales[name]=rms*math.sqrt(value.numel()/dimension)
    for name,factor in action.matrices.items():
        if name not in editable:continue
        plan.checkpoint_scale_slots.append((len(plan.instructions),8,name));p=unary(plan,factor.P,5,scale=scales[name])
        plan.checkpoint_scale_slots.append((len(plan.instructions),8,name));q=unary(plan,factor.Q,5,scale=scales[name])
        scaled_matrices[name]=Factors(p,q,factor.anchors);norm_parts.extend((pair_stats(plan,p),pair_stats(plan,q)))
    for name,value in action.vectors.items():
        if name not in editable:continue
        plan.checkpoint_scale_slots.append((len(plan.instructions),8,name))
        v=unary(plan,value,5,scale=scales[name]);scaled_vectors[name]=v;norm_parts.append(pair_stats(plan,v))
    norm_pointers=pointer_list(plan,norm_parts);radius_info=plan.empty((2,))
    plan.checkpoint_radius_slots.append((len(plan.instructions),7))
    plan.append([26,norm_pointers.data_ptr(),0,radius_info.data_ptr(),0,0,len(norm_parts),float_bits(radius)])
    replacements=dict(base);deltas={}
    plan.scaled_edit_action=RawAction(scaled_matrices,scaled_vectors)
    for name in base:
        if name not in editable:continue
        if name in scaled_matrices:
            factor=scaled_matrices[name];a,b=factor.anchors.A,factor.anchors.B
            first=None if retain_dense_deltas else plan.scratch(base[name].shape,'edit-delta-first')
            second=None if retain_dense_deltas else plan.scratch(base[name].shape,'edit-delta-second')
            if name=='embed.weight':
                first=plan.gemm(b,factor.P,tb=True,ieee=fp32,out=first)
                second=plan.gemm(factor.Q,a,tb=True,ieee=fp32,out=second)
            else:
                first=plan.gemm(factor.P,b,tb=True,ieee=fp32,out=first)
                second=plan.gemm(a,factor.Q,tb=True,ieee=fp32,out=second)
            if retain_dense_deltas:delta=plan.add(first,second)
            else:
                delta=first
                plan.append([8,first.data_ptr(),second.data_ptr(),delta.data_ptr(),0,0,delta.numel(),float_bits(1.),float_bits(1.)])
        else:delta=scaled_vectors[name]
        value=plan.empty(base[name].shape)
        plan.append([27,base[name].data_ptr(),delta.data_ptr(),radius_info.data_ptr(),value.data_ptr(),0,
                     value.numel(),int(name=='embed.weight' and not fp32)])
        replacements[name]=value;deltas[name]=delta
    def backward(parameter_cotangents=None,*,thin_cotangents=None):
        if thin_cotangents is None:
            if not retain_dense_deltas:raise ValueError('reused edit scratch requires thin-coordinate cotangents')
            contractions=[pair_stats(plan,parameter_cotangents[n],deltas[n],mode=1) for n in deltas]
        else:
            contractions=[]
            for name,factor in scaled_matrices.items():
                gradient=thin_cotangents[name]
                contractions.extend((pair_stats(plan,gradient.P,factor.P,mode=1),pair_stats(plan,gradient.Q,factor.Q,mode=1)))
            contractions.extend(pair_stats(plan,thin_cotangents[n],v,mode=1) for n,v in scaled_vectors.items())
        contraction_pointers=pointer_list(plan,contractions);radial=plan.empty((1,))
        plan.checkpoint_radius_slots.append((len(plan.instructions),7))
        plan.append([32,contraction_pointers.data_ptr(),radius_info.data_ptr(),radial.data_ptr(),0,0,
                     len(contractions),float_bits(radius)])
        def finish(value,gradient,name):
            result=plan.empty(value.shape)
            plan.checkpoint_scale_slots.append((len(plan.instructions),7,name))
            plan.append([33,value.data_ptr(),gradient.data_ptr(),radius_info.data_ptr(),radial.data_ptr(),result.data_ptr(),
                         value.numel(),float_bits(scales[name])])
            return result
        matrices={};vectors={}
        for name,factor in action.matrices.items():
            if name not in editable:
                p=torch.zeros_like(factor.P);q=torch.zeros_like(factor.Q);plan.storage.extend((p,q))
            else:
                scaled=scaled_matrices[name];a,b=factor.anchors.A,factor.anchors.B
                if thin_cotangents is not None:dp=thin_cotangents[name].P;dq=thin_cotangents[name].Q
                else:
                    gradient=parameter_cotangents[name]
                    if name=='embed.weight':dp=plan.gemm(gradient,b,ta=True,ieee=fp32);dq=plan.gemm(gradient,a,ieee=fp32)
                    else:dp=plan.gemm(gradient,b,ieee=fp32);dq=plan.gemm(gradient,a,ta=True,ieee=fp32)
                p=finish(scaled.P,dp,name);q=finish(scaled.Q,dq,name)
            matrices[name]=Factors(p,q,factor.anchors)
        for name,value in action.vectors.items():
            if name not in editable:vectors[name]=torch.zeros_like(value);plan.storage.append(vectors[name])
            else:vectors[name]=finish(scaled_vectors[name],(thin_cotangents if thin_cotangents is not None else parameter_cotangents)[name],name)
        return RawAction(matrices,vectors)
    return replacements,radius_info,backward


def append_group_objective(plan,parameters,base,grouped):
    groups=list(grouped.groups);indices=[];stats=[]
    for index,group in enumerate(groups):
        for name in grouped.groups[group]:
            indices.append(index);stats.append(pair_stats(plan,parameters[name],base[name],mode=2))
    pointers=pointer_list(plan,stats);group_indices=constants(plan,indices)
    refs=constants(plan,[float(grouped.reference_squared[g]) for g in groups],torch.float64)
    plan.checkpoint_group_references=refs
    values=typed_empty(plan,(len(groups),3),torch.float64)
    plan.append([28,pointers.data_ptr(),group_indices.data_ptr(),refs.data_ptr(),values.data_ptr(),0,len(stats),len(groups)])
    return values,{name:index for index,group in enumerate(groups) for name in grouped.groups[group]}


class PersistentOfflineTrainer:
    """Complete single-checkpoint update; kernel is experimental until qualification passes."""
    def __init__(self,actor,model,snapshot,inputs,teacher_log_probs,*,radius_fraction=.01,
                 kl_budget=1e-4,gradient_clip=1.,lr=2e-4,betas=(.9,.99),eps=1e-8,optimizer_state=None,kernel=None,
                 microbatch_sequences=None,state_weight=1.,gradient_accumulator=None,apply_optimizer=True,head_chunk_tokens=None,
                 gpt_precision='baseline',compensated_backward=False,raw_kl_only=False,kl_multiplier=None,augmented_rho=.01,trust_weight=.01,thin_backward=False,parameter_lrs=None):
        if gpt_precision not in ('baseline','fp32'):raise ValueError('unknown persistent GPT precision')
        if compensated_backward and gpt_precision!='fp32':raise ValueError('compensated backward requires FP32 GPT')
        if thin_backward and (gpt_precision!='fp32' or trust_weight!=0):raise ValueError('thin backward requires FP32 and the current zero-trust objective')
        if trust_weight<0 or augmented_rho<=0 or (kl_multiplier is not None and kl_multiplier<0):raise ValueError('invalid objective controls')
        self.gpt_precision=gpt_precision;fp32=gpt_precision=='fp32'
        if inputs.ndim!=2:raise ValueError('expected (sequence_batch,tokens) inputs')
        if radius_fraction<=0 or kl_budget<=0 or gradient_clip<=0 or state_weight<=0:raise ValueError('positive objective scales required')
        microbatch_sequences=microbatch_sequences or inputs.shape[0]
        if inputs.shape[0]%microbatch_sequences:raise ValueError('sequence panel must contain complete fixed microbatches')
        if gradient_accumulator is not None and apply_optimizer:raise ValueError('accumulation must defer clipping and Adam until global reduction')
        self.applies_optimizer=apply_optimizer;self.microbatch_sequences=microbatch_sequences
        head_chunk_tokens=head_chunk_tokens or int(os.environ.get('CHISEL_HEAD_CHUNK_TOKENS','0'))
        self.actor=actor;self.original={n:p.detach() for n,p in model.named_parameters()}
        self.grouped=GroupedNormObjective(self.original,zero_reference_policy='freeze')
        self.editable=editable_names(self.original,'grouped')-self.grouped.frozen_parameter_names
        norm_sq=editable_weight_norm_sq(self.original,self.editable)
        self.radius_fraction=radius_fraction
        self.snapshot=_staged_snapshot(scoped_snapshot(snapshot,norm_sq,radius_fraction,scope='grouped'),copy=True)
        self.radius=float(self.snapshot.radius)
        self.base={n:p.to(dtype=torch.float32,copy=True).contiguous() for n,p in self.original.items()}
        self.buffers={n:b.detach().to(dtype=torch.float32,copy=True).contiguous() for n,b in model.named_buffers()}
        self.inputs=inputs.detach().long().contiguous().clone()
        self.teacher=teacher_log_probs.detach().double().reshape(inputs.numel(),-1).contiguous()
        plan=self.plan=PersistentTrainingPlan(inputs.device,kernel=kernel)
        append_actor=append_transformer_chisel if getattr(actor,'architecture','')=='nanogpt_blocks_patch_io_v1' else append_global_rewriter
        self.action,actor_back=append_actor(plan,actor,self.snapshot)
        self.parameters,self.radius_info,edits_back=append_edits(plan,self.action,self.base,self.grouped,self.editable,self.radius,fp32=fp32,retain_dense_deltas=not thin_backward)
        thin_factors=plan.scaled_edit_action.matrices if thin_backward else None
        parts=inputs.shape[0]//microbatch_sequences
        self.gpt_gradients={}
        for name,value in self.parameters.items():
            if name not in self.editable:continue
            if thin_factors is not None and name in thin_factors:
                f=thin_factors[name];self.gpt_gradients[name]=Factors(plan.empty(f.P.shape),plan.empty(f.Q.shape),f.anchors)
            else:self.gpt_gradients[name]=plan.empty(value.shape)
        kl_parts=[]
        for micro in range(parts):
            start=micro*microbatch_sequences;stop=start+microbatch_sequences
            # Every temporary is dead after the VJP is copied into the gradient
            # bank. Reusing this arena bounds memory by the GPT microbatch.
            with plan.temporary_arena('student_gpt'):
                selected_teacher=self.teacher[start*inputs.shape[1]:stop*inputs.shape[1]]
                if head_chunk_tokens:
                    kl,micro_gradients,self.logits,self.logit_gradient=append_student_kl_chunks(plan,self.parameters,self.buffers,
                        self.inputs[start:stop],selected_teacher,num_layers=len(model.blocks),chunk_tokens=head_chunk_tokens,fraction=1/parts,
                        gpt_precision=gpt_precision,compensated_backward=compensated_backward,thin_factors=thin_factors)
                else:
                    if fp32:
                        from .persistent_fp32 import append_gpt as append_fp32_gpt
                        self.logits,gpt_back=append_fp32_gpt(plan,self.parameters,self.buffers,self.inputs[start:stop],num_layers=len(model.blocks),compensated_backward=compensated_backward,thin_factors=thin_factors)
                    else:self.logits,gpt_back=append_gpt(plan,self.parameters,self.buffers,self.inputs[start:stop],num_layers=len(model.blocks))
                    kl,dlogits=append_kl(plan,self.logits,selected_teacher)
                    if parts>1:dlogits=unary(plan,dlogits,5,scale=1/parts)
                    self.logit_gradient=dlogits
                    micro_gradients=gpt_back(dlogits)
            kl_parts.append(kl)
            for name,total in self.gpt_gradients.items():
                gradient=micro_gradients[name]
                pairs=[(gradient.P,total.P),(gradient.Q,total.Q)] if isinstance(total,Factors) else [(gradient,total)]
                for contribution,target in pairs:
                    if micro==0:plan.copy_into(contribution,target)
                    else:
                        plan.append([8,target.data_ptr(),contribution.data_ptr(),target.data_ptr(),0,0,target.numel(),float_bits(1.),float_bits(1.)])
                        # The baseline embedding leaf accumulates in BF16.
                        if name=='embed.weight' and not fp32:plan.append([16,target.data_ptr(),0,target.data_ptr(),0,0,target.numel(),0,float_bits(1.)])
        self.kl=typed_empty(plan,(1,),torch.float64);kl_pointers=pointer_list(plan,kl_parts)
        plan.append([40,kl_pointers.data_ptr(),self.kl.data_ptr(),0,0,0,parts,float_bits(1/parts)])
        group_values,group_ids=append_group_objective(plan,self.parameters,self.base,self.grouped)
        self.objective=typed_empty(plan,(5,),torch.float64);coefficients=typed_empty(plan,(2,),torch.float64)
        budget=constants(plan,kl_budget,torch.float64);radius=constants(plan,self.radius,torch.float64)
        self.radius_buffer=radius
        self.kl_budget_buffer=budget
        self.objective_mode=2 if raw_kl_only else 1 if kl_multiplier is not None else 0
        self.primal_dual_config=dict(kl_budget=kl_budget,kl_multiplier=kl_multiplier,augmented_rho=augmented_rho)
        self.objective_controls=constants(plan,[trust_weight,kl_multiplier or 0.,augmented_rho,
                                               2 if raw_kl_only else 1 if kl_multiplier is not None else 0],torch.float64)
        plan.append([29,self.kl.data_ptr(),group_values.data_ptr(),self.objective.data_ptr(),coefficients.data_ptr(),
                     budget.data_ptr(),len(self.grouped.groups),0,radius.data_ptr(),self.objective_controls.data_ptr()])
        parameter_cotangents={}
        if thin_backward:
            def combine(klgrad,normgrad,name):
                result=plan.empty(klgrad.shape)
                plan.append([47,klgrad.data_ptr(),normgrad.data_ptr(),result.data_ptr(),group_values.data_ptr(),coefficients.data_ptr(),
                             result.numel(),group_ids[name],int(raw_kl_only),float_bits(state_weight)])
                return result
            for name in self.editable:
                gradient=self.gpt_gradients[name];value=self.parameters[name]
                if isinstance(gradient,Factors):
                    a,b=gradient.anchors.A,gradient.anchors.B
                    # Norm derivative is c*W, so project W into the two thin
                    # coordinate spaces directly instead of materializing dW.
                    dp=plan.gemm(value,b,ta=name=='embed.weight',ieee=True)
                    dq=plan.gemm(value,a,ta=name!='embed.weight',ieee=True)
                    parameter_cotangents[name]=Factors(combine(gradient.P,dp,name),combine(gradient.Q,dq,name),gradient.anchors)
                else:parameter_cotangents[name]=combine(gradient,value,name)
            self.action_cotangents=edits_back(thin_cotangents=parameter_cotangents)
        else:
            for name in self.editable:
                gradient=plan.empty(self.base[name].shape);parameter_cotangents[name]=gradient
                plan.append([30,self.base[name].data_ptr(),self.parameters[name].data_ptr(),self.gpt_gradients[name].data_ptr(),
                             group_values.data_ptr(),coefficients.data_ptr(),gradient.numel(),group_ids[name],int(name=='embed.weight' and not fp32),gradient.data_ptr(),float_bits(state_weight),int(raw_kl_only)])
            self.action_cotangents=edits_back(parameter_cotangents)
        self.parameter_cotangents=parameter_cotangents
        self.gradients=actor_back(self.action_cotangents)
        if gradient_accumulator is not None:
            if gradient_accumulator.keys()!=self.gradients.keys():raise ValueError('actor accumulator parameter names differ')
            for name,gradient in self.gradients.items():
                target=gradient_accumulator[name]
                if target.shape!=gradient.shape or target.dtype!=torch.float32 or target.device!=gradient.device or not target.is_contiguous():
                    raise ValueError('actor accumulator shape/dtype/device/layout differs')
                plan.append([8,target.data_ptr(),gradient.data_ptr(),target.data_ptr(),0,0,target.numel(),float_bits(1.),float_bits(1.)])
        if apply_optimizer:
            append_actor_optimizer(plan,actor,self.gradients,gradient_clip=gradient_clip,lr=lr,betas=betas,eps=eps,optimizer_state=optimizer_state,parameter_lrs=parameter_lrs)
        plan.finalize()

    def step(self):self.plan.step()

    @torch.no_grad()
    def load_primal_dual(self,*,kl_multiplier,kl_budget=None,augmented_rho=None):
        """Refresh checkpoint-specific dual controls in existing device storage.

        Call between completed launches; this never changes the actor, gradient
        accumulator or Adam history. Raw-KL and legacy objective modes are fixed
        at construction because their norm-gradient stages differ.
        """
        import math
        if self.objective_mode!=1:raise ValueError('plan was not built for a primal-dual objective')
        config=dict(self.primal_dual_config,kl_multiplier=kl_multiplier)
        if kl_budget is not None:config['kl_budget']=kl_budget
        if augmented_rho is not None:config['augmented_rho']=augmented_rho
        if any(not math.isfinite(float(v)) for v in config.values()) or config['kl_multiplier']<0 or config['kl_budget']<=0 or config['augmented_rho']<=0:
            raise ValueError('invalid primal-dual controls')
        self.kl_budget_buffer.fill_(config['kl_budget'])
        self.objective_controls[1].fill_(config['kl_multiplier'])
        self.objective_controls[2].fill_(config['augmented_rho'])
        self.primal_dual_config=config

    @torch.no_grad()
    def load_panel(self,inputs,teacher_log_probs):
        """Copy a same-shaped panel into the existing resident buffers.

        The constructor adopts the provided teacher-probability storage; this
        method overwrites that storage. It allocates no GPT activation buffers.
        Teacher reconstruction/source staging is the caller's separate job.
        """
        if inputs.shape!=self.inputs.shape or teacher_log_probs.numel()!=self.teacher.numel():
            raise ValueError('panel shape changed; construct a new plan')
        if inputs.dtype not in (torch.int32,torch.int64) or teacher_log_probs.dtype!=torch.float64:
            raise ValueError('panel inputs must be integer tokens and exact targets float64')
        self.inputs.copy_(inputs);self.teacher.copy_(teacher_log_probs.reshape_as(self.teacher))

    @torch.no_grad()
    def load_checkpoint(self,model,snapshot,*,grouped=None,kl_multiplier=None):
        """Refresh fixed checkpoint buffers and constants without recompilation.

        Requires identical parameter/observation shapes and editable-group
        structure. A changed zero-reference group is rejected before mutation.
        Call between completed launches, on the launch stream. This refresh
        does not clear accumulated actor gradients or change Adam history.
        Primal-dual plans require the new checkpoint's explicit multiplier;
        retaining the previous checkpoint's dual state is never implicit.
        """
        if self.objective_mode==1:
            if kl_multiplier is None or not math.isfinite(float(kl_multiplier)) or kl_multiplier<0:
                raise ValueError('checkpoint refresh requires its nonnegative finite KL multiplier')
        elif kl_multiplier is not None:raise ValueError('KL multiplier requires a primal-dual plan')
        if not hasattr(self.plan,'chisel_observation_inputs'):
            raise ValueError('checkpoint reuse supports current TransformerChisel only')
        original={n:p.detach() for n,p in model.named_parameters()}
        if original.keys()!=self.original.keys() or any(v.shape!=self.base[n].shape for n,v in original.items()):
            raise ValueError('checkpoint parameter structure changed')
        grouped=grouped or GroupedNormObjective(original,zero_reference_policy='freeze')
        editable=editable_names(original,'grouped')-grouped.frozen_parameter_names
        if grouped.groups!=self.grouped.groups or editable!=self.editable:
            raise ValueError('checkpoint editable/zero-reference group structure changed')
        old=self.snapshot
        if tuple(old.matrices)!=tuple(snapshot.matrices) or tuple(old.vectors)!=tuple(snapshot.vectors):
            raise ValueError('checkpoint observation names changed')
        pairs=[]
        for name,obs in old.matrices.items():
            new=snapshot.matrices[name]
            pairs.extend((getattr(obs,key),getattr(new,key)) for key in ('rows','columns'))
            pairs.extend(((obs.anchors.A,new.anchors.A),(obs.anchors.B,new.anchors.B)))
        pairs.extend((obs.coordinates,snapshot.vectors[name].coordinates) for name,obs in old.vectors.items())
        pairs.append((old.trajectories,snapshot.trajectories.reshape_as(old.trajectories)))
        buffers=dict(model.named_buffers())
        if buffers.keys()!=self.buffers.keys():raise ValueError('checkpoint model buffers changed')
        pairs.extend((self.buffers[n],value) for n,value in buffers.items())
        if any(a.shape!=b.shape or a.device!=b.device for a,b in pairs):
            raise ValueError('checkpoint observation shape/device changed')
        scoped=scoped_snapshot(snapshot,editable_weight_norm_sq(original,editable),self.radius_fraction,scope='grouped')
        radius=float(scoped.radius);rank=self.actor.rank;scales={}
        for group,names in grouped.groups.items():
            rms=float(grouped.reference_norms[group])/math.sqrt(sum(original[n].numel() for n in names))
            for name in names:
                value=original[name];dimension=rank*(sum(value.shape)-rank) if value.ndim==2 else value.numel()
                scales[name]=rms*math.sqrt(value.numel()/dimension)
        for name,value in original.items():self.base[name].copy_(value)
        for target,value in pairs:target.copy_(value)
        matrix=self.plan.chisel_observation_inputs['matrix'];offset=0
        for obs in old.matrices.values():
            for value in (obs.rows,obs.columns):
                patch=self.actor._patches(value);matrix[offset:offset+len(patch)].copy_(patch);offset+=len(patch)
        if old.vectors:
            vector=self.plan.chisel_observation_inputs['vector'];offset=0
            for obs in old.vectors.values():
                patch=self.actor._patches(obs.coordinates);vector[offset:offset+len(patch)].copy_(patch);offset+=len(patch)
        self.radius=radius;self.radius_buffer.fill_(radius)
        old.radius.copy_(scoped.radius);old.weight_norm.copy_(scoped.weight_norm)
        self.plan.checkpoint_group_references.copy_(torch.tensor([float(grouped.reference_squared[g]) for g in grouped.groups],dtype=torch.float64,device=self.inputs.device))
        for index,slot,name in self.plan.checkpoint_scale_slots:self.plan.instructions[index][slot]=float_bits(scales[name])
        for index,slot in self.plan.checkpoint_radius_slots:self.plan.instructions[index][slot]=float_bits(radius)
        self.plan.stages.copy_(torch.tensor(self.plan.instructions,dtype=torch.int64,device='cpu'))
        self.grouped=grouped
        if self.objective_mode==1:self.load_primal_dual(kl_multiplier=kl_multiplier)

    def optimizer_state(self):
        """Retain these buffers when constructing a plan for another checkpoint.

        The actor and Adam history are shared across checkpoint states/epochs;
        only immutable observations, frozen GPT weights and cached labels change.
        A plan rebuild currently incurs allocation/setup overhead outside step().
        """
        if not self.applies_optimizer:raise ValueError('accumulation-only plan has no optimizer state')
        return self.plan.optimizer_state()


def append_actor_optimizer(plan,actor,gradients,*,gradient_clip=1.,lr=1e-5,betas=(.9,.99),eps=1e-8,optimizer_state=None,parameter_lrs=None):
    """Append global clipping/Adam after the caller's cross-rank SUM."""
    norms=[pair_stats(plan,g) for g in gradients.values()];pointers=pointer_list(plan,norms)
    for gradient in gradients.values():
        plan.append([34,pointers.data_ptr(),gradient.data_ptr(),0,0,0,gradient.numel(),len(norms),float_bits(gradient_clip)])
    if parameter_lrs is not None:
        if parameter_lrs.keys()!=dict(actor.named_parameters()).keys():raise ValueError('Adam learning-rate names differ')
        lr=tuple(parameter_lrs[n] for n,_ in actor.named_parameters())
    plan.adam(tuple(actor.parameters()),tuple(gradients.values()),lr=lr,betas=betas,eps=eps,state=optimizer_state)


def export_torch_adam(actor,optimizer):
    """Copy a saved Adam state in actor order, preserving per-group learning rates.

    The returned buffers are independent of the source optimizer. This handles
    representation/action-head groups whose optimizer order differs from module
    order; unsupported optimizer semantics fail explicitly.
    """
    if not isinstance(optimizer,torch.optim.Adam):raise ValueError('ordinary Adam required')
    groups={};common=None
    for group in optimizer.param_groups:
        if group.get('amsgrad') or group.get('maximize') or group.get('weight_decay',0):
            raise ValueError('persistent Adam does not implement these optimizer options')
        config=(tuple(group['betas']),float(group['eps']))
        if common is not None and config!=common:raise ValueError('Adam groups must share betas and epsilon')
        common=config
        for p in group['params']:
            if id(p) in groups:raise ValueError('duplicate Adam parameter')
            groups[id(p)]=group
    named=dict(actor.named_parameters())
    if {id(p) for p in named.values()}!=groups.keys():raise ValueError('Adam and actor parameter sets differ')
    moments=[];variances=[];steps=[];rates={}
    for name,p in named.items():
        state=optimizer.state.get(p,{})
        moments.append(state['exp_avg'].detach().clone() if state else torch.zeros_like(p))
        variances.append(state['exp_avg_sq'].detach().clone() if state else torch.zeros_like(p))
        steps.append(int(state['step']) if state else 0)
        rates[name]=float(groups[id(p)]['lr'])
    if len(set(steps))!=1:raise ValueError('Adam per-parameter step numbers differ')
    step=torch.tensor(steps[0],device=next(actor.parameters()).device,dtype=torch.int64)
    return dict(step=step,exp_avg=tuple(moments),exp_avg_sq=tuple(variances)),rates,common[0],common[1]


class PersistentActorOptimizer:
    """One optimizer launch after globally summed checkpoint gradients.

    Per-checkpoint plans use apply_optimizer=False and state_weight=1/200 for
    the scientific200-state batch. The external distributed SUM happens before
    this launch; neither clipping nor Adam is applied in the checkpoint plans.
    """
    def __init__(self,actor,gradients,*,gradient_clip=1.,lr=1e-5,betas=(.9,.99),eps=1e-8,optimizer_state=None,kernel=None,parameter_lrs=None):
        expected=dict(actor.named_parameters())
        if gradients.keys()!=expected.keys():raise ValueError('optimizer gradient names differ')
        self.plan=PersistentTrainingPlan(next(actor.parameters()).device,kernel=kernel)
        ordered={name:gradients[name] for name in expected}
        append_actor_optimizer(self.plan,actor,ordered,gradient_clip=gradient_clip,lr=lr,betas=betas,eps=eps,optimizer_state=optimizer_state,parameter_lrs=parameter_lrs)
        self.plan.finalize()
    def step(self):self.plan.step()
    def optimizer_state(self):return self.plan.optimizer_state()
