"""Optional local thin-student AOTAutograd boundary, without CUDA graph pools.

No actor, norm reward, primal-dual coefficient, collective, optimizer or teacher
cache is owned here. Callers sum raw weighted KL/cotangents globally and perform
one actor VJP/update. Teacher inputs/probabilities are explicit transient args.
This helper is experimental and is not installed into ReplayMaintenance.
"""
import torch
from .factorized_fp32 import factorized_logits
from .function_preservation import chunked_teacher_kl
from .online_preservation import _stored_parameters


class CompiledThinStudent:
    def __init__(self,engine,editable,*,global_tokens,embedding_names,compile=True,compiler_backend=None,compile_kl=True):
        if global_tokens<1:raise ValueError('positive global token count required')
        if engine.gpt_precision!='fp32' or engine.kl_dtype!=torch.float64:
            raise ValueError('strict FP32 GPT and FP64 teacher arithmetic required')
        self.engine=engine;self.editable=frozenset(editable);self.global_tokens=global_tokens
        self.embedding_names=frozenset(embedding_names)
        self.base_layout={n:(v.data_ptr(),v.shape,v.stride(),v.dtype,v.device) for n,v in engine.parameters.items()}
        options=dict(fullgraph=True,dynamic=False)
        if compiler_backend is None:options['options']={'triton.cudagraphs':False}
        else:options['backend']=compiler_backend
        self.compile_kl=compile_kl
        target=self._loss if compile_kl else self._logits
        self.compiled=torch.compile(target,**options) if compile else target

    def _logits(self,action,scale,inputs):
        with torch.no_grad():
            stored=_stored_parameters(self.engine.parameters,action,scale,self.editable,self.embedding_names)
        return factorized_logits(self.engine.view.gpt,self.engine.parameters,self.engine.buffers,inputs,
            action,scale,realized_parameters=stored,linear_backend='ieee',dx_backend='ieee')

    def _loss(self,action,scale,inputs,teacher_log_probs):
        logits=self._logits(action,scale,inputs)
        return chunked_teacher_kl(logits,teacher_log_probs,self.engine.chunk_tokens)*(inputs.numel()/self.global_tokens)

    def step(self,wire,teachers,editable,*,backward=True):
        if frozenset(editable)!=self.editable:raise ValueError('editable structure changed; create a separate compiled student')
        current={n:(v.data_ptr(),v.shape,v.stride(),v.dtype,v.device) for n,v in self.engine.parameters.items()}
        if current!=self.base_layout:raise ValueError('base model storage/schema changed')
        action,scale,leaves,_=wire
        if any(not leaf.is_leaf for leaf in leaves):raise ValueError('independent broadcast factor leaves required')
        if backward:
            for leaf in leaves:leaf.grad=None
        total=torch.zeros((),dtype=torch.float64,device=self.engine.device)
        for teacher in teachers:
            if teacher.log_probs.device!=self.engine.device or teacher.log_probs.dtype!=torch.float64:
                raise ValueError('resident exact FP64 teacher probabilities required')
            with torch.set_grad_enabled(backward):
                inputs=teacher.inputs.to(self.engine.device)
                if self.compile_kl:loss=self.compiled(action,scale,inputs,teacher.log_probs)
                else:
                    logits=self.compiled(action,scale,inputs)
                    loss=chunked_teacher_kl(logits,teacher.log_probs,self.engine.chunk_tokens)*(inputs.numel()/self.global_tokens)
                total+=loss.detach()
                if backward:loss.backward()
            self.engine.compute.student_forwards+=1;self.engine.compute.student_tokens+=teacher.tokens
            if backward:
                self.engine.compute.student_backwards+=1;self.engine.compute.student_backward_tokens+=teacher.tokens
        gradients=[torch.zeros_like(leaf) if leaf.grad is None else leaf.grad.detach() for leaf in leaves] if backward else []
        # One host synchronization instead of a synchronization per thin leaf;
        # the predicate is the same finite conjunction as the eager method.
        finite=torch.stack([torch.isfinite(total),*[torch.isfinite(g).all() for g in gradients]]).all()
        if not bool(finite):raise FloatingPointError('nonfinite local thin student loss/cotangent')
        return total,gradients
