"""Host construction of a device-resident training stage plan.

Plan construction allocates buffers but performs no model evaluation. run()
submits exactly one cooperative CUDA kernel for every stage in the plan.
"""
import struct
import os
from contextlib import contextmanager
import torch
from .persistent_mlp import CooperativeTrainingKernel


def float_bits(value):
    return struct.unpack('<I',struct.pack('<f',value))[0]


class PersistentTrainingPlan:
    def __init__(self, device, *, kernel=None):
        self.device=device;self.kernel=kernel
        self.instructions=[];self.storage=[];self.scratch_storage={};self.temporary_storage={};self._arena=None;self._emulation_b_cache={}

    def empty(self, shape):
        if self._arena is not None:
            name,index=self._arena;self._arena=(name,index+1);key=(name,index)
            if key in self.temporary_storage:
                value=self.temporary_storage[key]
                if tuple(value.shape)!=tuple(shape):raise ValueError('temporary arena shape changed across microbatches')
                return value
        value=torch.empty(shape,device=self.device,dtype=torch.float32)
        self.storage.append(value)
        if self._arena is not None:self.temporary_storage[key]=value
        return value

    @contextmanager
    def temporary_arena(self,name):
        """Reuse same-shaped temporaries only after their last device consumer.

        Caller must finish the entire forward/reverse and copy retained results
        before beginning the next use of this named arena.
        """
        previous=self._arena
        self._arena=(name,0)
        try:yield
        finally:self._arena=previous

    def scratch(self, shape, slot):
        key=(tuple(shape),slot)
        if key not in self.scratch_storage:
            arena=self._arena;self._arena=None
            try:self.scratch_storage[key]=self.empty(shape)
            finally:self._arena=arena
        return self.scratch_storage[key]

    def append(self, values):
        if len(values)>24:raise ValueError('persistent instruction exceeds descriptor width')
        self.instructions.append(values+[0]*(24-len(values)))

    def reduce(self, value, rows, cols, *, factor=1.):
        output=self.empty((cols,))
        self.append([1,value.data_ptr(),output.data_ptr(),0,0,0,rows,cols,float_bits(factor)])
        return output

    def bias_gradient(self, value):
        rows,cols=value.shape;chunk=256;parts=(rows+chunk-1)//chunk
        partial=self.scratch((parts,cols),'bias-partial')
        self.append([2,value.data_ptr(),partial.data_ptr(),0,0,0,rows,cols,chunk])
        return self.reduce(partial,parts,cols)

    def gemm(self, a, b, *, ta=False, tb=False, factor=1., epilogue=0, bias=None, aux=None, split=False, bf16=False, out=None, causal_mode=0, round_output=False, ieee=False,split_chunk=512,immutable_b=False):
        if ieee and (bf16 or round_output):raise ValueError('IEEE GEMM cannot quantize inputs or output')
        M=a.shape[-1] if ta else a.shape[-2]
        K=a.shape[-2] if ta else a.shape[-1]
        bK=b.shape[-1] if tb else b.shape[-2]
        N=b.shape[-2] if tb else b.shape[-1]
        if K!=bK or a.ndim!=b.ndim:raise ValueError('persistent GEMM dimensions differ')
        batch=a.shape[0] if a.ndim==3 else 1
        if a.ndim==3 and b.shape[0]!=batch:raise ValueError('persistent GEMM batch dimensions differ')
        if ieee and split and epilogue:raise ValueError('IEEE split-K epilogue must follow reduction')
        split=split or (batch==1 and K>512 and M*N<16384 and not (ieee and epilogue))
        if split and batch!=1:raise ValueError('split K currently requires a single matrix pair')
        if split_chunk<=0 or split_chunk%16:raise ValueError('split K chunk must be a positive multiple of16')
        parts=(K+split_chunk-1)//split_chunk if split else 1
        chunk=split_chunk if split else ((K+15)//16)*16 if bf16 else ((K+7)//8)*8
        shape=(batch,M,N) if a.ndim==3 else (M,N)
        output=self.empty(shape) if out is None else out
        if output.shape!=shape:raise ValueError('GEMM output shape differs')
        target=self.scratch((parts,M,N),'gemm-split') if parts>1 else output
        pointer=lambda v:0 if v is None else v.data_ptr()
        def layout(value,transposed):
            if value.stride(-1)==1:return value.stride(-2),transposed
            if value.stride(-2)==1:return value.stride(-1),not transposed
            raise ValueError('persistent GEMM requires row-major or column-major matrix storage')
        lda,physical_ta=layout(a,ta);ldb,physical_tb=layout(b,tb)
        emulate_tf32=os.environ.get('CHISEL_FP32_EMULATION')=='tf32x3' and ieee and batch==1 and not causal_mode and M>=128 and N>=64 and K>=64 and M%4==N%4==K%4==lda%4==ldb%4==0 and epilogue<=1
        use_cutlass=bool(os.environ.get('CHISEL_CUTLASS_INCLUDE')) and bf16 and batch==1 and not causal_mode and M>=32 and N>=32 and K%8==0 and epilogue<=1
        use_cutlass_ieee=bool(os.environ.get('CHISEL_CUTLASS_IEEE')) and ieee and M%128==0 and N%64==0 and K%8==0 and epilogue<=1
        use_cutlass_tf32=bool(os.environ.get('CHISEL_CUTLASS_TF32X3')) and (not ieee or emulate_tf32) and not bf16 and not causal_mode and M>=32 and N>=32 and M%4==N%4==K%4==lda%4==ldb%4==0 and epilogue<=1
        emulate=os.environ.get('CHISEL_FP32_EMULATION')=='bf16x9' and ieee and batch==1 and not causal_mode and M>=128 and N>=64 and K>=64 and K%8==0 and epilogue<=1
        ap,bp=a.data_ptr(),b.data_ptr()
        if use_cutlass or emulate:
            # Packed BF16 storage is reused between GEMMs; all conversion and
            # asynchronous tensor-core tiles remain inside the same launch.
            components=3 if emulate else 1
            pa=self.scratch((components*((M*K+1)//2),),'cutlass-a')
            cache_b=emulate and immutable_b and os.environ.get('CHISEL_EMULATION_CACHE_WEIGHTS')=='1'
            key=(bp,N,K,ldb,physical_tb)
            cached=cache_b and key in self._emulation_b_cache
            pb=self._emulation_b_cache[key] if cached else self.scratch((components*((K*N+1)//2),),('emulation-b',key) if cache_b else 'cutlass-b')
            self.append([49 if emulate else 43,ap,pa.data_ptr(),0,0,0,M,K,lda,int(physical_ta)])
            if not cached:
                # One pack stage per immutable operand, replayed on EVERY launch.
                # No data-dependent work is moved out of the measured update.
                self.append([49 if emulate else 43,bp,pb.data_ptr(),0,0,0,N,K,ldb,int(not physical_tb)])
                if cache_b:self._emulation_b_cache[key]=pb
            ap,bp=pa.data_ptr(),pb.data_ptr()
        self.append([50 if emulate else 48 if emulate_tf32 else 46 if use_cutlass_ieee else 45 if ieee else 44 if use_cutlass else 48 if use_cutlass_tf32 else 0,ap,bp,target.data_ptr(),pointer(bias),pointer(aux),
            M,N,K,lda,ldb,int(physical_ta)+2*int(physical_tb)+4*epilogue,parts,chunk,
            batch,a.stride(0) if a.ndim==3 else a.numel(),b.stride(0) if b.ndim==3 else b.numel(),parts*M*N,0,float_bits(factor),int(bf16),int(causal_mode),int(round_output and parts==1)])
        if parts>1:
            self.append([1,target.data_ptr(),output.data_ptr(),0,0,0,parts,M*N,float_bits(1.)])
        if round_output and (not use_cutlass or parts>1):
            self.append([16,output.data_ptr(),0,output.data_ptr(),0,0,output.numel(),0,float_bits(1.)])
        return output

    def linear(self, x, weight, bias, *, gelu=False):
        pre=self.empty((x.shape[0],weight.shape[0])) if gelu else None
        output=self.gemm(x,weight,tb=True,epilogue=2 if gelu else 1,bias=bias,aux=pre)
        return output,pre

    def linear_vjp(self, x, weight, dy, *, input_gelu_preactivation=None):
        dw=self.gemm(dy,x,ta=True,split=True)
        db=self.bias_gradient(dy)
        dx=self.gemm(dy,weight,epilogue=3 if input_gelu_preactivation is not None else 0,
                     aux=input_gelu_preactivation)
        return dx,dw,db

    def add(self, a, b, *, alpha=1., beta=1.):
        if a.shape!=b.shape:raise ValueError('persistent residual shapes differ')
        output=self.empty(a.shape)
        self.append([8,a.data_ptr(),b.data_ptr(),output.data_ptr(),0,0,a.numel(),float_bits(alpha),float_bits(beta)])
        return output

    def copy_into(self, source, target):
        if source.numel()!=target.numel():raise ValueError('copy element counts differ')
        self.append([12,source.data_ptr(),target.data_ptr(),0,0,0,source.numel()])

    def concatenate(self, values):
        values=tuple(values)
        if not values:raise ValueError('nonempty concatenation required')
        if any(v.shape[1:]!=values[0].shape[1:] for v in values):raise ValueError('concatenation trailing dimensions differ')
        result=self.empty((sum(v.shape[0] for v in values),*values[0].shape[1:]))
        offset=0
        for value in values:
            self.copy_into(value,result[offset:offset+value.shape[0]]);offset+=value.shape[0]
        return result

    def broadcast(self, vector, rows, *, factor=1.):
        output=self.empty((rows,vector.numel()))
        self.append([13,vector.data_ptr(),output.data_ptr(),0,0,0,rows,vector.numel(),float_bits(factor)])
        return output

    def context_concat(self, local, context):
        output=self.empty((local.shape[0],local.shape[1]+context.numel()))
        self.append([10,local.data_ptr(),context.data_ptr(),output.data_ptr(),0,0,
                     local.shape[0],local.shape[1],context.numel()])
        return output

    def context_vjp(self, gradient, local_width):
        rows,total=gradient.shape;context_width=total-local_width
        local=self.empty((rows,local_width));contexts=self.empty((rows,context_width))
        self.append([11,gradient.data_ptr(),local.data_ptr(),contexts.data_ptr(),0,0,rows,local_width,context_width])
        return local,self.bias_gradient(contexts)

    def trajectory_normalize(self, trajectories, rank):
        scales=self.empty((2,));logs=self.empty((2,));normalized=self.empty(trajectories.shape)
        self.append([14,trajectories.data_ptr(),scales.data_ptr(),logs.data_ptr(),0,0,trajectories.numel(),rank])
        self.append([15,trajectories.data_ptr(),scales.data_ptr(),normalized.data_ptr(),0,0,trajectories.numel(),rank])
        return normalized,logs

    def layer_norm(self, x, gamma, beta, eps):
        rows,cols=x.shape;output=self.empty(x.shape);stats=self.empty((rows,2))
        self.append([6,x.data_ptr(),gamma.data_ptr(),beta.data_ptr(),output.data_ptr(),stats.data_ptr(),
                     rows,cols,float_bits(eps)])
        return output,stats

    def layer_norm_vjp(self, x, dy, gamma, stats):
        rows,cols=x.shape;dx=self.empty(x.shape);contributions=self.empty(x.shape)
        self.append([7,x.data_ptr(),dy.data_ptr(),gamma.data_ptr(),dx.data_ptr(),stats.data_ptr(),rows,cols,0,0,
                     contributions.data_ptr()])
        return dx,self.bias_gradient(contributions),self.bias_gradient(dy)

    def head_layout(self, source, heads, tokens, width, mode, *, batch=1):
        if batch!=1 and mode<2:raise ValueError('batched combined QKV layout is unsupported')
        shape={0:(3,heads,tokens,width),1:(tokens,3*heads*width),
               2:(batch*tokens,heads*width),3:(batch*heads,tokens,width)}[mode]
        output=self.empty(shape)
        self.append([9,source.data_ptr(),output.data_ptr(),0,0,0,heads,tokens,width,mode,batch])
        return output

    def attention(self, q, k, v, *, causal=False):
        heads,queries,width=q.shape;keys=k.shape[1]
        scores=self.gemm(q,k,tb=True,factor=width**-.5)
        probabilities=self.empty(scores.shape)
        self.append([4,scores.data_ptr(),0,probabilities.data_ptr(),0,0,heads*queries,keys,int(causal),queries])
        output=self.gemm(probabilities,v)
        return output,probabilities

    def attention_vjp(self, q, k, v, probabilities, dy):
        heads,queries,width=q.shape;keys=k.shape[1]
        dv=self.gemm(probabilities,dy,ta=True)
        dp=self.gemm(dy,v,tb=True);ds=self.empty(dp.shape)
        self.append([5,probabilities.data_ptr(),dp.data_ptr(),ds.data_ptr(),0,0,heads*queries,keys,0,queries])
        dq=self.gemm(ds,k,factor=width**-.5);dk=self.gemm(ds,q,ta=True,factor=width**-.5)
        return dq,dk,dv

    def concatenate_head_gradients(self, values):
        # The three VJPs use distinct output allocations; a simple contiguous
        # copy is represented as an ADD with an immutable zero buffer.
        heads,tokens,width=values[0].shape
        result=self.empty((3,heads,tokens,width))
        for index,value in enumerate(values):
            self.copy_into(value,result[index])
        return result

    def adam(self, parameters, gradients, *, lr=2e-4, betas=(.9,.99), eps=1e-8, state=None):
        parameters=tuple(parameters);gradients=tuple(gradients)
        if len(parameters)!=len(gradients):raise ValueError('Adam parameter/gradient count differs')
        rates=tuple(lr) if isinstance(lr,(tuple,list)) else (lr,)*len(parameters)
        if len(rates)!=len(parameters) or any(not 0<float(rate)<float('inf') for rate in rates):
            raise ValueError('one positive finite Adam learning rate per parameter required')
        if state is not None:
            if len(state['exp_avg'])!=len(parameters) or len(state['exp_avg_sq'])!=len(parameters):
                raise ValueError('persistent Adam state parameter count differs')
            self._restored_step=state['step']
        self.moments=[];self.variances=[]
        for index,(parameter,gradient) in enumerate(zip(parameters,gradients)):
            moment=torch.zeros_like(parameter) if state is None else state['exp_avg'][index]
            variance=torch.zeros_like(parameter) if state is None else state['exp_avg_sq'][index]
            if any(v.shape!=parameter.shape or v.dtype!=parameter.dtype or v.device!=parameter.device for v in (moment,variance)):
                raise ValueError('persistent Adam state shape/dtype/device differs')
            self.moments.append(moment);self.variances.append(variance)
            self.append([3,parameter.data_ptr(),gradient.data_ptr(),moment.data_ptr(),variance.data_ptr(),0,
                         parameter.numel(),float_bits(rates[index]),float_bits(betas[0]),float_bits(betas[1]),float_bits(eps)])

    def finalize(self):
        self.kernel=self.kernel or CooperativeTrainingKernel()
        self.stages=torch.tensor(self.instructions,dtype=torch.int64,device=self.device)
        self.cursor=torch.zeros((),dtype=torch.int32,device=self.device)
        self.step_number=getattr(self,'_restored_step',None)
        if self.step_number is None:self.step_number=torch.zeros((),dtype=torch.int64,device=self.device)
        self.clocks=torch.empty(len(self.instructions)+1,dtype=torch.int64,device=self.device)
        return self

    def step(self):
        self.kernel.launch(self.stages,self.cursor,self.step_number,self.clocks)

    def optimizer_state(self):
        """Shared Adam buffers for a new checkpoint plan; never run concurrently."""
        return dict(step=self.step_number,exp_avg=tuple(self.moments),exp_avg_sq=tuple(self.variances))
