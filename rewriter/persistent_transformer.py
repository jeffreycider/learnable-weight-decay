"""Single-launch complete Chisel TransformerEncoderLayer training stage.

Executes both prenorms, QKV projection, multihead self-attention, residuals,
GELU feed-forward, all first-order gradients, and Adam. An external output
cotangent is still required; this is not yet the entire offline Chisel update.
"""
import torch
from .persistent_plan import PersistentTrainingPlan


class PersistentTransformerTrainer:
    def __init__(self, layer, inputs, cotangent, *, lr=2e-4, betas=(.9,.99), eps=1e-8, kernel=None):
        if (not layer.norm_first or not layer.self_attn.batch_first or layer.self_attn.dropout
            or layer.dropout.p or layer.dropout1.p or layer.dropout2.p
            or layer.activation_relu_or_gelu!=2):
            raise ValueError('persistent block requires prenorm, batch-first, exact GELU, zero dropout')
        if inputs.ndim!=2 or inputs.shape!=cotangent.shape:
            raise ValueError('persistent block expects (tokens,width) inputs and cotangent')
        parameters=dict(layer.named_parameters())
        if any(t.device!=inputs.device or t.dtype!=torch.float32 or not t.is_contiguous()
               for t in (*parameters.values(),inputs,cotangent)) or inputs.device.type!='cuda':
            raise ValueError('persistent block requires contiguous CUDA FP32 tensors')
        self.inputs=inputs;self.cotangent=cotangent;self.parameters=parameters
        tokens,width=inputs.shape;heads=layer.self_attn.num_heads;head_width=width//heads
        plan=self.plan=PersistentTrainingPlan(inputs.device,kernel=kernel)
        n1,n1stats=plan.layer_norm(inputs,layer.norm1.weight,layer.norm1.bias,layer.norm1.eps)
        qkv,_=plan.linear(n1,layer.self_attn.in_proj_weight,layer.self_attn.in_proj_bias)
        qkv_heads=plan.head_layout(qkv,heads,tokens,head_width,0)
        q,k,v=qkv_heads.unbind(0)
        attention,probabilities=plan.attention(q,k,v)
        merged=plan.head_layout(attention,heads,tokens,head_width,2)
        projected,_=plan.linear(merged,layer.self_attn.out_proj.weight,layer.self_attn.out_proj.bias)
        residual=plan.add(inputs,projected)
        n2,n2stats=plan.layer_norm(residual,layer.norm2.weight,layer.norm2.bias,layer.norm2.eps)
        hidden,pre=plan.linear(n2,layer.linear1.weight,layer.linear1.bias,gelu=True)
        feedforward,_=plan.linear(hidden,layer.linear2.weight,layer.linear2.bias)
        self.output=plan.add(residual,feedforward)
        gradients={}
        dh,gradients['linear2.weight'],gradients['linear2.bias']=plan.linear_vjp(
            hidden,layer.linear2.weight,cotangent,input_gelu_preactivation=pre)
        dn2,gradients['linear1.weight'],gradients['linear1.bias']=plan.linear_vjp(n2,layer.linear1.weight,dh)
        dr,gradients['norm2.weight'],gradients['norm2.bias']=plan.layer_norm_vjp(residual,dn2,layer.norm2.weight,n2stats)
        dy=plan.add(cotangent,dr)
        dm,gradients['self_attn.out_proj.weight'],gradients['self_attn.out_proj.bias']=plan.linear_vjp(
            merged,layer.self_attn.out_proj.weight,dy)
        da=plan.head_layout(dm,heads,tokens,head_width,3)
        dq,dk,dv=plan.attention_vjp(q,k,v,probabilities,da)
        dqkv_heads=plan.concatenate_head_gradients((dq,dk,dv))
        dqkv=plan.head_layout(dqkv_heads,heads,tokens,head_width,1)
        dn1,gradients['self_attn.in_proj_weight'],gradients['self_attn.in_proj_bias']=plan.linear_vjp(
            n1,layer.self_attn.in_proj_weight,dqkv)
        dx,gradients['norm1.weight'],gradients['norm1.bias']=plan.layer_norm_vjp(inputs,dn1,layer.norm1.weight,n1stats)
        self.input_gradient=plan.add(dy,dx)
        self.gradients={name:gradients[name] for name in parameters}
        plan.adam(tuple(parameters.values()),tuple(self.gradients.values()),lr=lr,betas=betas,eps=eps)
        plan.finalize()

    def step(self):self.plan.step()


def append_transformer(plan, layer, inputs):
    """Append forward now and return a closure that appends reverse later."""
    tokens,width=inputs.shape;heads=layer.self_attn.num_heads;head_width=width//heads
    n1,n1stats=plan.layer_norm(inputs,layer.norm1.weight,layer.norm1.bias,layer.norm1.eps)
    qkv,_=plan.linear(n1,layer.self_attn.in_proj_weight,layer.self_attn.in_proj_bias)
    qkv_heads=plan.head_layout(qkv,heads,tokens,head_width,0)
    q,k,v=qkv_heads.unbind(0)
    attention,probabilities=plan.attention(q,k,v)
    merged=plan.head_layout(attention,heads,tokens,head_width,2)
    projected,_=plan.linear(merged,layer.self_attn.out_proj.weight,layer.self_attn.out_proj.bias)
    residual=plan.add(inputs,projected)
    n2,n2stats=plan.layer_norm(residual,layer.norm2.weight,layer.norm2.bias,layer.norm2.eps)
    hidden,pre=plan.linear(n2,layer.linear1.weight,layer.linear1.bias,gelu=True)
    feedforward,_=plan.linear(hidden,layer.linear2.weight,layer.linear2.bias)
    output=plan.add(residual,feedforward)
    def vjp(cotangent):
        gradients={}
        dh,gradients['linear2.weight'],gradients['linear2.bias']=plan.linear_vjp(
            hidden,layer.linear2.weight,cotangent,input_gelu_preactivation=pre)
        dn2,gradients['linear1.weight'],gradients['linear1.bias']=plan.linear_vjp(n2,layer.linear1.weight,dh)
        dr,gradients['norm2.weight'],gradients['norm2.bias']=plan.layer_norm_vjp(residual,dn2,layer.norm2.weight,n2stats)
        dy=plan.add(cotangent,dr)
        dm,gradients['self_attn.out_proj.weight'],gradients['self_attn.out_proj.bias']=plan.linear_vjp(
            merged,layer.self_attn.out_proj.weight,dy)
        da=plan.head_layout(dm,heads,tokens,head_width,3)
        dq,dk,dv=plan.attention_vjp(q,k,v,probabilities,da)
        dqkv_heads=plan.concatenate_head_gradients((dq,dk,dv))
        dqkv=plan.head_layout(dqkv_heads,heads,tokens,head_width,1)
        dn1,gradients['self_attn.in_proj_weight'],gradients['self_attn.in_proj_bias']=plan.linear_vjp(
            n1,layer.self_attn.in_proj_weight,dqkv)
        dx,gradients['norm1.weight'],gradients['norm1.bias']=plan.layer_norm_vjp(inputs,dn1,layer.norm1.weight,n1stats)
        return plan.add(dy,dx),gradients
    return output,vjp


class PersistentTransformerStackTrainer:
    def __init__(self, layers, inputs, cotangent, *, lr=2e-4, betas=(.9,.99), eps=1e-8, kernel=None):
        layers=tuple(layers)
        if not layers:raise ValueError('nonempty transformer stack required')
        if inputs.ndim!=2 or inputs.shape!=cotangent.shape:
            raise ValueError('persistent stack expects (tokens,width) inputs and cotangent')
        for layer in layers:
            if (not layer.norm_first or not layer.self_attn.batch_first or layer.self_attn.dropout
                or layer.dropout.p or layer.dropout1.p or layer.dropout2.p or layer.activation_relu_or_gelu!=2):
                raise ValueError('persistent stack requires prenorm, batch-first, GELU, zero dropout')
        self.inputs=inputs;self.cotangent=cotangent
        self.parameters={f'{index}.{name}':p for index,layer in enumerate(layers) for name,p in layer.named_parameters()}
        if any(t.device!=inputs.device or t.dtype!=torch.float32 or not t.is_contiguous()
               for t in (*self.parameters.values(),inputs,cotangent)) or inputs.device.type!='cuda':
            raise ValueError('persistent stack requires contiguous CUDA FP32 tensors')
        plan=self.plan=PersistentTrainingPlan(inputs.device,kernel=kernel)
        output=inputs;backwards=[]
        for layer in layers:
            output,vjp=append_transformer(plan,layer,output);backwards.append(vjp)
        self.output=output
        gradient=cotangent;gradients={}
        for index in reversed(range(len(layers))):
            gradient,layer_gradients=backwards[index](gradient)
            gradients.update({f'{index}.{name}':value for name,value in layer_gradients.items()})
        self.input_gradient=gradient
        self.gradients={name:gradients[name] for name in self.parameters}
        plan.adam(tuple(self.parameters.values()),tuple(self.gradients.values()),lr=lr,betas=betas,eps=eps)
        plan.finalize()

    def step(self):self.plan.step()
