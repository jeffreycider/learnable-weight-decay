"""Literal IEEE FP32 GPT forward/VJP using the cooperative stage interpreter.

The default uses IEEE FP32 GPT forward operands. It retains the original RMS
epsilon and materializes math attention. Optional compensated backward GEMMs
and explicitly opted-in forward emulation must be qualified separately. The BF16 actor path is not
changed. This module is an implementation under qualification, not a fit backend.
"""
import math
import torch
from .gpt_precision import RMS_EPSILON
from .algebra import Factors
from .persistent_gpt import unary
from .persistent_plan import PersistentTrainingPlan, float_bits


def linear_vjp(plan,x,weight,dy,*,compensated_backward=False,factors=None,split_dx=False):
    import os
    dx=plan.gemm(dy,weight,ieee=not compensated_backward,split=split_dx,
                 split_chunk=int(os.environ.get('CHISEL_HEAD_DX_SPLIT_K','512')) if split_dx else 512)
    if factors is None:dw=plan.gemm(dy,x,ta=True,split=True,ieee=True)
    else:
        xb=plan.gemm(x,factors.anchors.B,ieee=True)
        ga=plan.gemm(dy,factors.anchors.A,ieee=True)
        dp=plan.gemm(dy,xb,ta=True,split=True,ieee=True)
        dq=plan.gemm(x,ga,ta=True,split=True,ieee=True)
        dw=Factors(dp,dq,factors.anchors)
    return dx,dw,plan.bias_gradient(dy)


def append_linear(plan,x,weight,bias,*,compensated_backward=False,factors=None):
    out=plan.gemm(x,weight,tb=True,bias=bias,epilogue=1,ieee=True,immutable_b=True)
    def backward(dy):
        return linear_vjp(plan,x,weight,dy,compensated_backward=compensated_backward,factors=factors)
    return out,backward


def append_norm(plan,x,gamma=None):
    rows,cols=x.shape;out=plan.empty(x.shape);stats=plan.empty((rows,))
    ptr=0 if gamma is None else gamma.data_ptr()
    plan.append([17,x.data_ptr(),ptr,0,out.data_ptr(),stats.data_ptr(),rows,cols,float_bits(RMS_EPSILON),0,1])
    def backward(dy):
        dx=plan.empty(x.shape)
        plan.append([18,x.data_ptr(),ptr,dy.data_ptr(),dx.data_ptr(),stats.data_ptr(),rows,cols,0,0,1])
        return dx
    return out,backward


def rotary(plan,x,frequency,heads,width,tokens,*,backward=False):
    out=plan.empty(x.shape)
    plan.append([19,x.data_ptr(),frequency.data_ptr(),out.data_ptr(),0,0,
                 x.shape[0],heads,width,int(backward),tokens,1])
    return out


def append_attention(plan,q,k,v,*,scale=.12):
    heads,tokens,width=q.shape
    # PyTorch's math SDPA scales both operands before its score GEMM.
    root=math.sqrt(scale)
    qs=unary(plan,q,5,scale=root);ks=unary(plan,k,5,scale=root)
    # Score and score-cotangent buffers die inside each attention invocation.
    # Reuse them across layers/microbatches; retain only probabilities for VJP.
    score_buffer=plan.scratch((heads,tokens,tokens),'fp32-attention-scores')
    scores=plan.gemm(qs,ks,tb=True,ieee=True,causal_mode=1,out=score_buffer)
    probability=plan.empty(scores.shape)
    plan.append([4,scores.data_ptr(),0,probability.data_ptr(),0,0,heads*tokens,tokens,1,tokens])
    output=plan.gemm(probability,v,ieee=True,causal_mode=2)
    def backward(dy):
        dv=plan.gemm(probability,dy,ta=True,ieee=True,causal_mode=3)
        dp=plan.gemm(dy,v,tb=True,ieee=True,causal_mode=1,out=score_buffer)
        ds=plan.scratch(dp.shape,'fp32-attention-score-cotangent')
        plan.append([5,probability.data_ptr(),dp.data_ptr(),ds.data_ptr(),0,0,heads*tokens,tokens,1,tokens])
        dq=plan.gemm(ds,ks,ieee=True,factor=root,causal_mode=2)
        dk=plan.gemm(ds,qs,ta=True,ieee=True,factor=root,causal_mode=3)
        return dq,dk,dv
    return output,backward


def append_block(plan,parameters,buffers,index,x,*,head_dim=128,batch=1,compensated_backward=False,thin_factors=None):
    prefix=f'blocks.{index}.';rows,width=x.shape;heads=width//head_dim;tokens=rows//batch
    def layout(value,mode):return plan.head_layout(value,heads,tokens,head_dim,mode,batch=batch)
    def linear(name,value):return append_linear(plan,value,parameters[prefix+name+'.weight'],parameters[prefix+name+'.bias'],
        compensated_backward=compensated_backward,factors=None if thin_factors is None else thin_factors.get(prefix+name+'.weight'))
    norm,norm_back=append_norm(plan,x,parameters[prefix+'norm1.gains'])
    q,q_back=linear('attn.q',norm);k,k_back=linear('attn.k',norm);v,v_back=linear('attn.v',norm)
    qn,qn_back=append_norm(plan,q.view(rows*heads,head_dim))
    kn,kn_back=append_norm(plan,k.view(rows*heads,head_dim))
    frequency=buffers[prefix+'attn.rotary.angular_freq']
    qr=rotary(plan,qn.view(rows,width),frequency,heads,head_dim,tokens)
    kr=rotary(plan,kn.view(rows,width),frequency,heads,head_dim,tokens)
    attended,attention_back=append_attention(plan,layout(qr,3),layout(kr,3),layout(v,3))
    projected,project_back=linear('attn.proj',layout(attended,2))
    residual=plan.add(x,projected)
    norm2,norm2_back=append_norm(plan,residual,parameters[prefix+'norm2.gains'])
    hidden,fc_back=linear('mlp.fc',norm2)
    activated=unary(plan,hidden,6)
    mlp,mlp_back=linear('mlp.proj',activated)
    out=plan.add(residual,mlp)
    def backward(dy):
        gradients={}
        def vjp(name,fn,grad):
            dx,dw,db=fn(grad)
            gradients[prefix+name+'.weight']=dw;gradients[prefix+name+'.bias']=db
            return dx
        da=vjp('mlp.proj',mlp_back,dy)
        dn=vjp('mlp.fc',fc_back,unary(plan,hidden,7,da))
        dr=plan.add(dy,norm2_back(dn))
        dm=vjp('attn.proj',project_back,dr)
        dq,dk,dv=attention_back(layout(dm,3))
        dq=rotary(plan,layout(dq,2),frequency,heads,head_dim,tokens,backward=True)
        dk=rotary(plan,layout(dk,2),frequency,heads,head_dim,tokens,backward=True)
        dq=qn_back(dq.view(rows*heads,head_dim)).view(rows,width)
        dk=kn_back(dk.view(rows*heads,head_dim)).view(rows,width)
        dxq=vjp('attn.q',q_back,dq);dxk=vjp('attn.k',k_back,dk);dxv=vjp('attn.v',v_back,layout(dv,2))
        dx=plan.add(dr,norm_back(plan.add(plan.add(dxv,dxk),dxq)))
        return dx,gradients
    return out,backward


def append_body(plan,parameters,buffers,inputs,*,num_layers,head_dim=128,compensated_backward=False,thin_factors=None):
    if inputs.ndim!=2:raise ValueError('GPT inputs must be sequence matrices')
    batch=inputs.shape[0];rows=inputs.numel();embedding=parameters['embed.weight'];vocab,width=embedding.shape
    x=plan.empty((rows,width))
    plan.append([20,embedding.data_ptr(),inputs.data_ptr(),x.data_ptr(),0,0,rows,vocab,width])
    x,initial_back=append_norm(plan,x,parameters['norm1.gains']);backs=[]
    for index in range(num_layers):
        x,back=append_block(plan,parameters,buffers,index,x,head_dim=head_dim,batch=batch,compensated_backward=compensated_backward,thin_factors=thin_factors)
        backs.append(back)
    x,final_back=append_norm(plan,x,parameters['norm2.gains'])
    def backward(dx):
        dx=final_back(dx);gradients={}
        for back in reversed(backs):
            dx,block_grad=back(dx);gradients.update(block_grad)
        dx=initial_back(dx)
        factor=None if thin_factors is None else thin_factors.get('embed.weight')
        if factor is None:
            de=plan.empty(embedding.shape)
            plan.append([41,de.data_ptr(),0,0,0,0,de.numel()])
            plan.append([42,dx.data_ptr(),inputs.data_ptr(),de.data_ptr(),0,0,rows,vocab,width])
        else:
            rank=factor.P.shape[1];selected_b=plan.empty((rows,rank))
            plan.append([20,factor.anchors.B.data_ptr(),inputs.data_ptr(),selected_b.data_ptr(),0,0,
                         rows,vocab,rank,*factor.anchors.B.stride()])
            dp=plan.gemm(dx,selected_b,ta=True,split=True,ieee=True)
            projected=plan.gemm(dx,factor.anchors.A,ieee=True);dq=plan.empty(factor.Q.shape)
            plan.append([41,dq.data_ptr(),0,0,0,0,dq.numel()])
            plan.append([42,projected.data_ptr(),inputs.data_ptr(),dq.data_ptr(),0,0,rows,vocab,rank])
            de=Factors(dp,dq,factor.anchors)
        gradients['embed.weight']=de
        return gradients
    return x,backward


def append_gpt(plan,parameters,buffers,inputs,*,num_layers,head_dim=128,compensated_backward=False,thin_factors=None):
    x,body_back=append_body(plan,parameters,buffers,inputs,num_layers=num_layers,head_dim=head_dim,compensated_backward=compensated_backward,thin_factors=thin_factors)
    raw,head_back=append_linear(plan,x,parameters['proj.weight'],parameters['proj.bias'],compensated_backward=compensated_backward,
        factors=None if thin_factors is None else thin_factors.get('proj.weight'))
    logits=unary(plan,raw,3)
    def backward(dlogits):
        dx,dw,db=head_back(unary(plan,raw,8,dlogits))
        gradients=body_back(dx)
        gradients.update({'proj.weight':dw,'proj.bias':db})
        return gradients
    return logits,backward


class PersistentFP32GPTVJP:
    def __init__(self,model,inputs,cotangent,*,kernel=None,compensated_backward=False):
        self.parameters={n:p.detach().float().clone().contiguous() for n,p in model.named_parameters()}
        self.buffers={n:b.detach().float().clone().contiguous() for n,b in model.named_buffers()}
        self.inputs=inputs.detach().long().contiguous();self.cotangent=cotangent.detach().float().contiguous()
        self.plan=PersistentTrainingPlan(inputs.device,kernel=kernel)
        self.logits,backward=append_gpt(self.plan,self.parameters,self.buffers,self.inputs,
            num_layers=len(model.blocks),compensated_backward=compensated_backward)
        self.gradients=backward(self.cotangent);self.plan.finalize()
    def step(self):self.plan.step()
