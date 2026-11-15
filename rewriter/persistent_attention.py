"""Actual one-launch attention forward/VJP using the cooperative interpreter.

This is the FP32 Chisel attention primitive. It materializes score/probability
matrices and is not a FlashAttention replacement for long-sequence GPT yet.
"""
import struct
import torch
from .persistent_mlp import CooperativeTrainingKernel


class PersistentAttentionVJP:
    def __init__(self, query, key, value, cotangent, *, scale=None, causal=False, kernel=None):
        if any(t.device!=query.device or t.device.type!='cuda' or t.dtype!=torch.float32
               or not t.is_contiguous() for t in (query,key,value,cotangent)):
            raise ValueError('persistent attention requires contiguous CUDA FP32 tensors')
        heads,queries,width=query.shape;keys=key.shape[1]
        if key.shape!=(heads,keys,width) or value.shape!=key.shape or cotangent.shape!=query.shape:
            raise ValueError('attention head/query/key/value shapes differ')
        self.kernel=kernel or CooperativeTrainingKernel()
        self.query=query;self.key=key;self.value=value;self.cotangent=cotangent
        self.output=torch.empty_like(query)
        self.query_gradient=torch.empty_like(query);self.key_gradient=torch.empty_like(key)
        self.value_gradient=torch.empty_like(value)
        self.scores=torch.empty((heads,queries,keys),device=query.device)
        self.probabilities=torch.empty_like(self.scores)
        self.probability_gradient=torch.empty_like(self.scores);self.score_gradient=torch.empty_like(self.scores)
        scale=width**-.5 if scale is None else scale
        stages=[]
        def gemm(a,b,c,M,N,K,*,ta=False,tb=False,factor=1.):
            stages.append([0,a.data_ptr(),b.data_ptr(),c.data_ptr(),0,0,M,N,K,a.shape[-1],b.shape[-1],
                int(ta)+2*int(tb),1,((K+7)//8)*8,heads,a[0].numel(),b[0].numel(),c[0].numel(),0,
                struct.unpack('<I',struct.pack('<f',factor))[0],0,0,0,0])
        gemm(query,key,self.scores,queries,keys,width,tb=True,factor=scale)
        stages.append([4,self.scores.data_ptr(),0,self.probabilities.data_ptr(),0,0,heads*queries,keys,int(causal),queries]+[0]*14)
        gemm(self.probabilities,value,self.output,queries,width,keys)
        gemm(self.probabilities,cotangent,self.value_gradient,keys,width,queries,ta=True)
        gemm(cotangent,value,self.probability_gradient,queries,keys,width,tb=True)
        stages.append([5,self.probabilities.data_ptr(),self.probability_gradient.data_ptr(),self.score_gradient.data_ptr(),
                       0,0,heads*queries,keys,int(causal),queries]+[0]*14)
        gemm(self.score_gradient,key,self.query_gradient,queries,width,keys,factor=scale)
        gemm(self.score_gradient,query,self.key_gradient,keys,width,queries,ta=True,factor=scale)
        self.stages=torch.tensor(stages,dtype=torch.int64,device=query.device)
        self.cursor=torch.zeros((),dtype=torch.int32,device=query.device)
        self.step_number=torch.zeros((),dtype=torch.int64,device=query.device)
        self.clocks=torch.empty(len(stages)+1,dtype=torch.int64,device=query.device)

    def step(self):
        self.kernel.launch(self.stages,self.cursor,self.step_number,self.clocks)
