"""Student GPT stages for the literal cooperative training kernel.

BF16 values are represented in FP32 buffers, with explicit round-to-nearest-even
at the baseline's BF16 operation boundaries. No teacher transformer is evaluated.
The materialized attention implementation is qualified against baseline Flash
attention separately; it must not redefine cached teacher targets.
"""
from __future__ import annotations
import math
import os
import torch
from .persistent_plan import PersistentTrainingPlan,float_bits


def unary(plan,x,mode,dy=None,scale=1.,out=None):
    out=plan.empty(x.shape) if out is None else out
    plan.append([16,x.data_ptr(),0 if dy is None else dy.data_ptr(),out.data_ptr(),0,0,
                 x.numel(),mode,float_bits(scale)])
    return out


def rounded(plan,x,*,out=None):return unary(plan,x,0,out=out)


def append_bf16_linear(plan,x,weight,bias):
    # Weights are unchanged until the final optimizer stage, so their BF16
    # casts can be recomputed during reverse instead of retained per layer.
    w=rounded(plan,weight,out=plan.scratch(weight.shape,'linear-weight'))
    b=rounded(plan,bias,out=plan.scratch(bias.shape,'linear-bias'))
    raw=plan.scratch((x.shape[0],weight.shape[0]),'linear-forward')
    out=rounded(plan,plan.gemm(x,w,tb=True,epilogue=1,bias=b,bf16=True,out=raw))
    def backward(dy):
        w=rounded(plan,weight,out=plan.scratch(weight.shape,'linear-weight'))
        dx=plan.gemm(dy,w,bf16=True,out=plan.scratch(x.shape,'linear-dx'))
        dx=rounded(plan,dx)
        dw=plan.gemm(dy,x,ta=True,split=True,bf16=True,out=plan.scratch(weight.shape,'linear-dw'))
        dw=rounded(plan,dw);db=rounded(plan,plan.bias_gradient(dy))
        return dx,dw,db
    return out,backward


def append_rms_norm(plan,x,gamma=None,eps=1.1920928955078125e-7,*,train_gain=False):
    rows,cols=x.shape;out=plan.empty(x.shape);stats=plan.empty((rows,))
    pointer=0 if gamma is None else gamma.data_ptr()
    plan.append([17,x.data_ptr(),pointer,0,out.data_ptr(),stats.data_ptr(),rows,cols,float_bits(eps)])
    def backward(dy):
        dx=plan.empty(x.shape)
        contributions=plan.empty(x.shape) if train_gain else None
        plan.append([18,x.data_ptr(),pointer,dy.data_ptr(),dx.data_ptr(),stats.data_ptr(),rows,cols,0,
                     contributions.data_ptr() if contributions is not None else 0])
        if train_gain:return dx,rounded(plan,plan.bias_gradient(contributions))
        return dx
    return out,backward


def rotary(plan,x,frequencies,heads,width,backward=False,*,sequence_tokens=None):
    out=plan.empty(x.shape)
    plan.append([19,x.data_ptr(),frequencies.data_ptr(),out.data_ptr(),0,0,
                 x.shape[0],heads,width,int(backward),sequence_tokens or x.shape[0]])
    return out


def append_bf16_attention(plan,q,k,v,*,scale=.12):
    heads,tokens,width=q.shape
    # Match the pinned FlashAttention split-KV heuristic for the initial B=1
    # target. Each split rounds its own unnormalized probabilities, so using a
    # single global row maximum changes the actual finite-precision function.
    sm_count=torch.cuda.get_device_properties(plan.device).multi_processor_count*2
    query_blocks=heads*math.ceil(tokens/64);key_blocks=math.ceil(tokens/128)
    splits=1
    if query_blocks<.8*sm_count:
        choices=[]
        for split in range(1,min(128,sm_count,key_blocks)+1):
            if split>1 and math.ceil(key_blocks/split)==math.ceil(key_blocks/(split-1)):continue
            waves=query_blocks*split/sm_count;choices.append((split,waves/math.ceil(waves)))
        maximum=max(eff for _,eff in choices)
        splits=next(split for split,eff in choices if eff>=.85*maximum)
    chunk=128 if splits>1 else 64
    blocks_per_split=math.ceil(key_blocks/splits) if splits>1 else math.ceil(tokens/chunk)
    scratch_shape=(heads,tokens,tokens)
    stream_mode=os.environ.get('CHISEL_STREAM_ATTENTION','0')
    scratch0=plan.scratch(scratch_shape,'attention0') if stream_mode not in ('2','3') else None
    scratch1=plan.scratch(scratch_shape,'attention1') if stream_mode not in ('2','3') else None
    probability=plan.empty((heads,tokens) if stream_mode=='3' else scratch_shape)
    packed_attention=os.environ.get('CHISEL_PACKED_ATTENTION')=='1'
    def attention_inputs(*values):
        if not packed_attention:return values
        outputs=[]
        for index,value in enumerate(values):
            packed=plan.scratch(((value.numel()+1)//2,),'attention-packed-'+str(index))
            plan.append([43,value.data_ptr(),packed.data_ptr(),0,0,0,value.numel()//width,width,width,0])
            outputs.append(packed)
        return outputs
    if stream_mode in ('1','2','3'):
        if width!=128:raise ValueError('streaming attention requires head width128')
        output=plan.empty(q.shape)
        pq,pk,pv=attention_inputs(q,k,v)
        plan.append([36,pq.data_ptr(),pk.data_ptr(),pv.data_ptr(),output.data_ptr(),probability.data_ptr(),tokens,heads,float_bits(scale),int(stream_mode=='3'),0,0,0,0,int(packed_attention)])
        # Normalize the saved raw scores in place for the unchanged reverse.
        if stream_mode!='3':plan.append([4,probability.data_ptr(),0,probability.data_ptr(),0,0,heads*tokens,tokens,1,tokens])
    else:
        scores=plan.gemm(q,k,tb=True,bf16=True,out=scratch0,causal_mode=1)
        effective=scratch1
        plan.append([35,scores.data_ptr(),effective.data_ptr(),probability.data_ptr(),0,0,
                     heads*tokens,tokens,chunk,float_bits(scale),blocks_per_split])
        output=rounded(plan,plan.gemm(effective,v,causal_mode=2))
    def backward(dy):
        if stream_mode in ('2','3'):
            delta=plan.empty((heads,tokens));dq=plan.empty(q.shape);dk=plan.empty(k.shape);dv=plan.empty(v.shape)
            plan.append([39,output.data_ptr(),dy.data_ptr(),delta.data_ptr(),0,0,heads*tokens])
            pq,pk,pv,pdy=attention_inputs(q,k,v,dy)
            plan.append([37,pdy.data_ptr(),pv.data_ptr(),pk.data_ptr(),probability.data_ptr(),delta.data_ptr(),tokens,heads,float_bits(scale),dq.data_ptr(),pq.data_ptr(),int(stream_mode=='3'),0,0,int(packed_attention)])
            plan.append([38,pdy.data_ptr(),pv.data_ptr(),pq.data_ptr(),probability.data_ptr(),delta.data_ptr(),tokens,heads,float_bits(scale),dk.data_ptr(),dv.data_ptr(),pk.data_ptr(),int(stream_mode=='3'),0,int(packed_attention)])
            return dq,dk,dv
        plan.append([16,probability.data_ptr(),0,scratch0.data_ptr(),0,0,probability.numel(),0,float_bits(1.)])
        dv=rounded(plan,plan.gemm(scratch0,dy,ta=True,bf16=True,causal_mode=3))
        dp=plan.gemm(dy,v,tb=True,bf16=True,out=scratch0,causal_mode=1);ds=scratch1
        # Flash-style derivative uses the saved BF16 output for row delta and
        # rounds the scaled score cotangent before the Q/K tensor-core products.
        plan.append([22,output.data_ptr(),dy.data_ptr(),probability.data_ptr(),dp.data_ptr(),ds.data_ptr(),
                     heads*tokens,tokens,width,float_bits(scale)])
        dq=rounded(plan,plan.gemm(ds,k,bf16=True,causal_mode=2));dk=rounded(plan,plan.gemm(ds,q,ta=True,bf16=True,causal_mode=3))
        return dq,dk,dv
    return output,backward


def append_gpt_block(plan,parameters,buffers,index,inputs,*,head_dim=128,train_norms=False,batch=1):
    prefix=f'blocks.{index}.';width=inputs.shape[-1];tokens=inputs.shape[0];heads=width//head_dim
    sequence_tokens=tokens//batch
    if tokens%batch:raise ValueError('GPT sequence batch does not divide token rows')
    def layout(value,mode):return plan.head_layout(value,heads,sequence_tokens,head_dim,mode,batch=batch)
    def linear(name,x):return append_bf16_linear(plan,x,parameters[prefix+name+'.weight'],parameters[prefix+name+'.bias'])
    normalized,norm1_back=append_rms_norm(plan,inputs,parameters[prefix+'norm1.gains'],train_gain=train_norms)
    q,q_back=linear('attn.q',normalized);k,k_back=linear('attn.k',normalized);v,v_back=linear('attn.v',normalized)
    qnorm,qnorm_back=append_rms_norm(plan,q.view(tokens*heads,head_dim))
    knorm,knorm_back=append_rms_norm(plan,k.view(tokens*heads,head_dim))
    frequency=buffers[prefix+'attn.rotary.angular_freq']
    qrot=rotary(plan,qnorm.view(tokens,width),frequency,heads,head_dim,sequence_tokens=sequence_tokens)
    krot=rotary(plan,knorm.view(tokens,width),frequency,heads,head_dim,sequence_tokens=sequence_tokens)
    qh=layout(qrot,3)
    kh=layout(krot,3)
    vh=layout(v,3)
    attended,attention_back=append_bf16_attention(plan,qh,kh,vh)
    merged=layout(attended,2)
    projected,projection_back=linear('attn.proj',merged)
    residual=rounded(plan,plan.add(inputs,projected))
    normalized2,norm2_back=append_rms_norm(plan,residual,parameters[prefix+'norm2.gains'],train_gain=train_norms)
    hidden,fc_back=linear('mlp.fc',normalized2)
    activation=unary(plan,hidden,1)
    mlp_out,mlp_back=linear('mlp.proj',activation)
    output=rounded(plan,plan.add(residual,mlp_out))
    def backward(dy):
        gradients={}
        def vjp(name,fn,d):
            dx,dw,db=fn(d);gradients[prefix+name+'.weight']=dw;gradients[prefix+name+'.bias']=db
            return dx
        dactivation=vjp('mlp.proj',mlp_back,dy)
        dhidden=unary(plan,hidden,2,dactivation)
        dnorm2=vjp('mlp.fc',fc_back,dhidden)
        dnorm2_input=norm2_back(dnorm2)
        if train_norms:dnorm2_input,gradients[prefix+'norm2.gains']=dnorm2_input
        dresidual=rounded(plan,plan.add(dy,dnorm2_input))
        dmerged=vjp('attn.proj',projection_back,dresidual)
        dattended=layout(dmerged,3)
        dqh,dkh,dvh=attention_back(dattended)
        dqrot=layout(dqh,2)
        dkrot=layout(dkh,2)
        dv=layout(dvh,2)
        dqnorm=rotary(plan,dqrot,frequency,heads,head_dim,True,sequence_tokens=sequence_tokens)
        dknorm=rotary(plan,dkrot,frequency,heads,head_dim,True,sequence_tokens=sequence_tokens)
        dq=qnorm_back(dqnorm.view(tokens*heads,head_dim)).view(tokens,width)
        dk=knorm_back(dknorm.view(tokens*heads,head_dim)).view(tokens,width)
        # PyTorch's shared input accumulates BF16 branch cotangents. This fixed
        # order is deterministic; compare sensitivity to its execution order.
        dxq=vjp('attn.q',q_back,dq);dxk=vjp('attn.k',k_back,dk);dxv=vjp('attn.v',v_back,dv)
        dnormalized=rounded(plan,plan.add(rounded(plan,plan.add(dxv,dxk)),dxq))
        dnorm1_input=norm1_back(dnormalized)
        if train_norms:dnorm1_input,gradients[prefix+'norm1.gains']=dnorm1_input
        dx=rounded(plan,plan.add(dresidual,dnorm1_input))
        return dx,gradients
    return output,backward


def append_gpt_body(plan,parameters,buffers,inputs,*,num_layers,head_dim=128):
    """Frozen GPT through final normalized head input, then its complete VJP."""
    if inputs.ndim!=2:raise ValueError('literal GPT kernel requires (batch,tokens) inputs')
    if any(p.dtype!=torch.float32 for p in parameters.values()):raise ValueError('GPT plan parameters use FP32 storage with explicit BF16 casts')
    batch=inputs.shape[0];tokens=inputs.numel();embedding=parameters['embed.weight'];vocab,width=embedding.shape
    x=plan.empty((tokens,width))
    plan.append([20,embedding.data_ptr(),inputs.data_ptr(),x.data_ptr(),0,0,tokens,vocab,width])
    x,initial_back=append_rms_norm(plan,x,parameters['norm1.gains'])
    block_backs=[]
    for index in range(num_layers):
        x,backward=append_gpt_block(plan,parameters,buffers,index,x,head_dim=head_dim,batch=batch);block_backs.append(backward)
    x,final_back=append_rms_norm(plan,x,parameters['norm2.gains'])
    def backward(dx):
        gradients={};dx=final_back(dx)
        for back in reversed(block_backs):
            dx,block_grad=back(dx);gradients.update(block_grad)
        dx=initial_back(dx);dembedding=plan.empty(embedding.shape)
        plan.append([41,dembedding.data_ptr(),0,0,0,0,dembedding.numel()])
        plan.append([42,dx.data_ptr(),inputs.data_ptr(),dembedding.data_ptr(),0,0,tokens,vocab,width])
        rounded(plan,dembedding,out=dembedding)
        gradients['embed.weight']=dembedding
        return gradients
    return x,backward


def append_gpt(plan,parameters,buffers,inputs,*,num_layers,head_dim=128):
    x,body_back=append_gpt_body(plan,parameters,buffers,inputs,num_layers=num_layers,head_dim=head_dim)
    logits,head_back=append_bf16_linear(plan,x,parameters['proj.weight'],parameters['proj.bias'])
    softened=unary(plan,logits,3)
    def backward(dlogits):
        draw=unary(plan,logits,4,dlogits)
        dx,dw,db=head_back(draw)
        gradients=body_back(dx);gradients.update({'proj.weight':dw,'proj.bias':db})
        return gradients
    return softened,backward


class PersistentGPTVJP:
    """Qualification harness for complete student forward and supplied-logit VJP."""
    def __init__(self,model,inputs,cotangent):
        self.parameters={n:p.detach().float().clone().contiguous() for n,p in model.named_parameters()}
        self.buffers={n:b.detach().float().clone().contiguous() for n,b in model.named_buffers()}
        self.inputs=inputs.detach().long().contiguous();self.cotangent=cotangent.detach().float().contiguous()
        self.plan=PersistentTrainingPlan(inputs.device)
        self.logits,backward=append_gpt(self.plan,self.parameters,self.buffers,self.inputs,num_layers=len(model.blocks))
        self.gradients=backward(self.cotangent);self.plan.finalize()
    def step(self):self.plan.step()
