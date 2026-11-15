"""Pure compiled PR323 kernels separated by ordinary-pass observation hooks.

No sensor runs inside Dynamo, and no transformer computation is replayed. The
operation order, dtypes, attention implementation and summed loss are identical
to baseline.model; compilation boundaries differ from the monolithic baseline.
"""
from types import MethodType
import torch
import torch.nn.functional as F
from .model import Linear,RMSNorm,CausalSelfAttention,MLP
from .execution import _forward_with_compiled_loss,softcap_cross_entropy


def linear_kernel(x,weight,bias):
    return F.linear(x,weight.type_as(x),bias.type_as(x))


def norm_kernel(x,gains):
    return F.rms_norm(x,(x.size(-1),),weight=gains.type_as(x))


def relu_squared_kernel(x):return x.relu().square()


def attention_kernel(q,k,v,angular_freq,num_heads,head_dim):
    B,T=q.shape[:2]
    q=q.view(B,T,num_heads,head_dim);k=k.view(B,T,num_heads,head_dim);v=v.view(B,T,num_heads,head_dim)
    q,k=F.rms_norm(q,(q.size(-1),)),F.rms_norm(k,(k.size(-1),))
    def rotary(x):
        pos=torch.arange(x.size(1),dtype=torch.float32,device=x.device)
        theta=torch.outer(pos,angular_freq)[None,:,None,:]
        cos,sin=theta.cos(),theta.sin()
        x1,x2=x.to(dtype=torch.float32).chunk(2,dim=-1)
        y1=x1*cos+x2*sin;y2=x1*(-sin)+x2*cos
        return torch.cat((y1,y2),3).type_as(x)
    q,k=rotary(q),rotary(k)
    y=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
        scale=.12,is_causal=True).transpose(1,2)
    return y.contiguous().view(B,T,num_heads*head_dim)


def enable_capture_execution(model,*,backend=None):
    if hasattr(model,'_compiled_loss'):raise ValueError('capture execution already installed or incompatible loss wrapper')
    kwargs=dict(fullgraph=True,dynamic=False)
    if backend is not None:kwargs['backend']=backend
    linear=torch.compile(linear_kernel,**kwargs);norm=torch.compile(norm_kernel,**kwargs)
    attention=torch.compile(attention_kernel,**kwargs);relu_squared=torch.compile(relu_squared_kernel,**kwargs)
    def linear_forward(self,x):return linear(x,self.weight,self.bias)
    def norm_forward(self,x):return norm(x,self.gains)
    def attention_forward(self,x):
        return self.proj(attention(self.q(x),self.k(x),self.v(x),self.rotary.angular_freq,self.num_heads,self.head_dim))
    def mlp_forward(self,x):return self.proj(relu_squared(self.fc(x)))
    for module in model.modules():
        if isinstance(module,Linear):module.forward=MethodType(linear_forward,module)
        elif isinstance(module,RMSNorm):module.forward=MethodType(norm_forward,module)
        elif isinstance(module,CausalSelfAttention):module.forward=MethodType(attention_forward,module)
        elif isinstance(module,MLP):module.forward=MethodType(mlp_forward,module)
    model._compiled_loss=torch.compile(softcap_cross_entropy,**kwargs)
    model.forward=MethodType(_forward_with_compiled_loss,model)
    return model
