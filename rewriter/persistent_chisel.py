"""Literal persistent-plan adapter for the current NanoGPT-block Chisel.

Only immutable observation serialization happens during construction. Every
parameter-dependent operation, gradient and optimizer update is a device stage.
"""
import torch
from .algebra import Factors, RawAction
from .persistent_controller import _staged_snapshot
from .persistent_gpt import append_gpt_block, append_rms_norm, rounded
from .persistent_plan import PersistentTrainingPlan


def append_transformer_chisel(plan, actor, snapshot):
    if actor.architecture != 'nanogpt_blocks_patch_io_v1':
        raise ValueError('expected current TransformerChisel architecture')
    parameters=dict(actor.named_parameters());buffers=dict(actor.named_buffers())
    patch=actor.patch_size;rank=actor.rank;matrix_parts=[];matrix_sizes=[];matrix_kinds=[]
    for obs in snapshot.matrices.values():
        for kind,values in ((0,obs.rows),(1,obs.columns)):
            value=actor._patches(values).detach().contiguous()
            matrix_parts.append(value);matrix_sizes.append((value.shape[0],values.shape[0]))
            matrix_kinds.extend([kind]*value.shape[0])
    matrix_input=torch.cat(matrix_parts);plan.storage.append(matrix_input)
    plan.chisel_observation_inputs={'matrix':matrix_input}
    matrix,_=plan.linear(matrix_input,actor.matrix_input.weight,actor.matrix_input.bias)
    # Consecutive token type runs keep the forward embedding lookup and its
    # reverse sum inside the literal kernel without a dense one-hot allocation.
    def typed_tokens(value,kinds):
        pieces=[];start=0
        for stop in range(1,len(kinds)+1):
            if stop==len(kinds) or kinds[stop]!=kinds[start]:
                pieces.append(plan.add(value[start:stop],plan.broadcast(actor.token_type.weight[kinds[start]],stop-start)))
                start=stop
        return plan.concatenate(pieces)
    reads=[typed_tokens(matrix,matrix_kinds)];writes=[typed_tokens(matrix,[k+4 for k in matrix_kinds])]
    vector_input=None;vector_sizes=[]
    if snapshot.vectors:
        vector_parts=[]
        for obs in snapshot.vectors.values():
            value=actor._patches(obs.coordinates).detach().contiguous()
            vector_parts.append(value);vector_sizes.append((value.shape[0],obs.coordinates.shape[0]))
        vector_input=torch.cat(vector_parts);plan.storage.append(vector_input)
        plan.chisel_observation_inputs['vector']=vector_input
        vector,_=plan.linear(vector_input,actor.vector_input.weight,actor.vector_input.bias)
        reads.append(typed_tokens(vector,[2]*len(vector)));writes.append(typed_tokens(vector,[6]*len(vector)))
    normalized,logs=plan.trajectory_normalize(snapshot.trajectories,rank)
    trajectory_input=plan.context_concat(normalized,logs)
    trajectory,_=plan.linear(trajectory_input,actor.trajectory_input.weight,actor.trajectory_input.bias)
    reads.append(typed_tokens(trajectory,[3]*len(trajectory)))
    read=plan.concatenate(reads);write=plan.concatenate(writes);read_count=len(read);matrix_count=len(matrix)
    x=rounded(plan,plan.concatenate((read,write)))
    x,initial_back=append_rms_norm(plan,x,actor.norm1.gains,train_gain=True)
    backwards=[]
    for index in range(len(actor.blocks)):
        x,back=append_gpt_block(plan,parameters,buffers,index,x,train_norms=True);backwards.append(back)
    x,final_back=append_rms_norm(plan,x[read_count:],actor.norm2.gains,train_gain=True)
    matrix_output,_=plan.linear(x[:matrix_count],actor.matrix_output.weight,actor.matrix_output.bias)
    offset=0;factors=[]
    for count,length in matrix_sizes:
        factors.append(matrix_output[offset:offset+count].reshape(-1,rank)[:length]);offset+=count
    matrices={}
    for index,(name,obs) in enumerate(snapshot.matrices.items()):
        p,q=factors[2*index:2*index+2];a=obs.anchors.A
        p=plan.add(p,plan.gemm(a,plan.gemm(a,p,ta=True)),beta=-1.)
        matrices[name]=Factors(p,q,obs.anchors)
    vectors={}
    if vector_input is not None:
        vector_output,_=plan.linear(x[matrix_count:],actor.vector_output.weight,actor.vector_output.bias)
        offset=0
        for (name,_),(count,length) in zip(snapshot.vectors.items(),vector_sizes):
            vectors[name]=vector_output[offset:offset+count].flatten()[:length];offset+=count
    output=RawAction(matrices,vectors)
    def zero(shape):
        z=torch.zeros(shape,device=plan.device,dtype=torch.float32);plan.storage.append(z);return z
    def padded(values,shape):
        # Every padding element is reset on each update; no stale scratch.
        result=plan.empty(shape);plan.copy_into(zero(shape),result)
        plan.copy_into(values.flatten(),result.flatten()[:values.numel()]);return result
    def projection_parameter_vjp(features,dy):
        return plan.gemm(dy,features,ta=True,split=True),plan.bias_gradient(dy)
    def backward(cotangents):
        gradients={};pieces=[]
        for index,(name,obs) in enumerate(snapshot.matrices.items()):
            gp=cotangents.matrices[name].P;gq=cotangents.matrices[name].Q;a=obs.anchors.A
            gp=plan.add(gp,plan.gemm(a,plan.gemm(a,gp,ta=True)),beta=-1.)
            for value,(count,_) in zip((gp,gq),matrix_sizes[2*index:2*index+2]):
                pieces.append(padded(value,(count,patch*rank)))
        dm=plan.concatenate(pieces)
        dxm,gradients['matrix_output.weight'],gradients['matrix_output.bias']=plan.linear_vjp(x[:matrix_count],actor.matrix_output.weight,dm)
        dxpieces=[dxm]
        if vector_input is not None:
            dv=plan.concatenate([padded(cotangents.vectors[name],(count,patch))
                for (name,_),(count,_) in zip(snapshot.vectors.items(),vector_sizes)])
            dxv,gradients['vector_output.weight'],gradients['vector_output.bias']=plan.linear_vjp(x[matrix_count:],actor.vector_output.weight,dv)
            dxpieces.append(dxv)
        dx=rounded(plan,plan.concatenate(dxpieces))
        dx,gradients['norm2.gains']=final_back(dx)
        dx=plan.concatenate((zero((read_count,actor.model_dim)),dx))
        for back in reversed(backwards):
            dx,grads=back(dx);gradients.update(grads)
        dx,gradients['norm1.gains']=initial_back(dx)
        dm=plan.add(dx[:matrix_count],dx[read_count:read_count+matrix_count])
        gradients['matrix_input.weight'],gradients['matrix_input.bias']=projection_parameter_vjp(matrix_input,dm)
        type_gradients=[None]*7
        for kind in (0,1):
            runs=[];start=0
            for stop in range(1,len(matrix_kinds)+1):
                if stop==len(matrix_kinds) or matrix_kinds[stop]!=matrix_kinds[start]:
                    if matrix_kinds[start]==kind:runs.append((start,stop))
                    start=stop
            for shifted in (False,True):
                base=read_count if shifted else 0
                selected=plan.concatenate([dx[base+start:base+stop] for start,stop in runs])
                type_gradients[kind+4*shifted]=plan.bias_gradient(selected).view(1,-1)
        vector_count=sum(n for n,_ in vector_sizes)
        if vector_input is not None:
            dr=dx[matrix_count:matrix_count+vector_count]
            dw=dx[read_count+matrix_count:]
            gradients['vector_input.weight'],gradients['vector_input.bias']=projection_parameter_vjp(vector_input,plan.add(dr,dw))
            type_gradients[2]=plan.bias_gradient(dr).view(1,-1);type_gradients[6]=plan.bias_gradient(dw).view(1,-1)
        else:
            for name in ('vector_input.weight','vector_input.bias','vector_output.weight','vector_output.bias'):
                gradients[name]=zero(parameters[name].shape)
            type_gradients[2]=zero((1,actor.model_dim));type_gradients[6]=zero((1,actor.model_dim))
        dt=dx[matrix_count+vector_count:read_count]
        gradients['trajectory_input.weight'],gradients['trajectory_input.bias']=projection_parameter_vjp(trajectory_input,dt)
        type_gradients[3]=plan.bias_gradient(dt).view(1,-1)
        gradients['token_type.weight']=plan.concatenate(type_gradients)
        if gradients.keys()!=parameters.keys():raise ValueError('gradient coverage differs: '+str(gradients.keys()^parameters.keys()))
        return {name:gradients[name] for name in parameters}
    return output,backward


class PersistentChiselTrainer:
    def __init__(self,actor,snapshot,cotangents,*,lr=2e-4,betas=(.9,.99),eps=1e-8,kernel=None):
        self.actor=actor;self.snapshot=_staged_snapshot(snapshot)
        self.plan=PersistentTrainingPlan(next(actor.parameters()).device,kernel=kernel)
        self.output,backward=append_transformer_chisel(self.plan,actor,self.snapshot)
        self.gradients=backward(cotangents)
        self.plan.adam(tuple(actor.parameters()),tuple(self.gradients.values()),lr=lr,betas=betas,eps=eps)
        self.plan.finalize()
    def step(self):self.plan.step()
