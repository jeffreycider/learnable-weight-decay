"""Full Chisel actor forward/VJP/Adam as one persistent cooperative kernel.

The external RawAction cotangent is an explicit input for this milestone.
Student GPT and the offline objective are not yet part of this program.
"""
import dataclasses
import torch

from .algebra import Anchors,Factors,RawAction
from .persistent_plan import PersistentTrainingPlan
from .persistent_transformer import append_transformer


def _append_mlp(plan, module, inputs):
    hidden,pre=plan.linear(inputs,module[0].weight,module[0].bias,gelu=True)
    output,_=plan.linear(hidden,module[2].weight,module[2].bias)
    def vjp(dy):
        dh,dw2,db2=plan.linear_vjp(hidden,module[2].weight,dy,input_gelu_preactivation=pre)
        dx,dw1,db1=plan.linear_vjp(inputs,module[0].weight,dh)
        return dx,{'0.weight':dw1,'0.bias':db1,'2.weight':dw2,'2.bias':db2}
    return output,vjp


def _append_mha(plan, module, query_input, kv_input):
    width=module.embed_dim;heads=module.num_heads;dim=width//heads
    qn,kn=query_input.shape[0],kv_input.shape[0]
    weights=module.in_proj_weight.split(width);biases=module.in_proj_bias.split(width)
    qprojected,_=plan.linear(query_input,weights[0],biases[0])
    kprojected,_=plan.linear(kv_input,weights[1],biases[1])
    vprojected,_=plan.linear(kv_input,weights[2],biases[2])
    q=plan.head_layout(qprojected,heads,qn,dim,3)
    k=plan.head_layout(kprojected,heads,kn,dim,3)
    v=plan.head_layout(vprojected,heads,kn,dim,3)
    attention,probabilities=plan.attention(q,k,v)
    merged=plan.head_layout(attention,heads,qn,dim,2)
    output,_=plan.linear(merged,module.out_proj.weight,module.out_proj.bias)
    def vjp(dy):
        dm,dwo,dbo=plan.linear_vjp(merged,module.out_proj.weight,dy)
        da=plan.head_layout(dm,heads,qn,dim,3)
        dq,dk,dv=plan.attention_vjp(q,k,v,probabilities,da)
        dq=plan.head_layout(dq,heads,qn,dim,2)
        dk=plan.head_layout(dk,heads,kn,dim,2)
        dv=plan.head_layout(dv,heads,kn,dim,2)
        dqi,dwq,dbq=plan.linear_vjp(query_input,weights[0],dq)
        dki,dwk,dbk=plan.linear_vjp(kv_input,weights[1],dk)
        dvi,dwv,dbv=plan.linear_vjp(kv_input,weights[2],dv)
        return dqi,plan.add(dki,dvi),{'in_proj_weight':plan.concatenate((dwq,dwk,dwv)),
            'in_proj_bias':plan.concatenate((dbq,dbk,dbv)),'out_proj.weight':dwo,'out_proj.bias':dbo}
    return output,vjp


def _append_pool(plan, actor, local):
    width=local.shape[1];queries=actor.pool_queries
    scores=plan.gemm(queries,local,tb=True,factor=width**-.5)
    probabilities=plan.empty(scores.shape)
    plan.append([4,scores.data_ptr(),0,probabilities.data_ptr(),0,0,scores.shape[0],scores.shape[1],0,scores.shape[0]])
    pooled=plan.gemm(probabilities,local,split=True)
    output,_=plan.linear(pooled,actor.pool_projection.weight,actor.pool_projection.bias)
    def vjp(dy):
        dp,dw,db=plan.linear_vjp(pooled,actor.pool_projection.weight,dy)
        dprob=plan.gemm(dp,local,tb=True)
        ds=plan.empty(scores.shape)
        plan.append([5,probabilities.data_ptr(),dprob.data_ptr(),ds.data_ptr(),0,0,scores.shape[0],scores.shape[1],0,scores.shape[0]])
        dqueries=plan.gemm(ds,local,factor=width**-.5,split=True)
        dh_value=plan.gemm(probabilities,dp,ta=True)
        dh_score=plan.gemm(ds,queries,ta=True,factor=width**-.5)
        return plan.add(dh_value,dh_score),{'pool_queries':dqueries,'pool_projection.weight':dw,'pool_projection.bias':db}
    return output,vjp


def _staged_snapshot(snapshot,*,copy=False):
    """Input serialization only: make immutable feature/anchor buffers row-major."""
    detach=lambda value:value.detach().contiguous().clone() if copy else value.detach().contiguous()
    matrices={name:dataclasses.replace(obs,rows=detach(obs.rows),columns=detach(obs.columns),
        anchors=Anchors(detach(obs.anchors.A),detach(obs.anchors.B))) for name,obs in snapshot.matrices.items()}
    vectors={name:dataclasses.replace(obs,coordinates=detach(obs.coordinates)) for name,obs in snapshot.vectors.items()}
    trajectories=detach(snapshot.trajectories).reshape(snapshot.trajectories.shape[0],-1)
    return dataclasses.replace(snapshot,matrices=matrices,vectors=vectors,trajectories=trajectories)


def append_global_rewriter(plan, actor, snapshot):
    """Append full actor forward, returning RawAction and a reverse-plan closure."""
    local_records=[];tokens=[];queries=[]
    for name,obs in snapshot.matrices.items():
        rows,row_encoder_vjp=_append_mlp(plan,actor.coordinate_encoder,obs.rows)
        columns,column_encoder_vjp=_append_mlp(plan,actor.coordinate_encoder,obs.columns)
        local=plan.concatenate((rows,columns))
        projected,pool_vjp=_append_pool(plan,actor,local)
        token=projected
        query=plan.reduce(token,token.shape[0],token.shape[1],factor=1/token.shape[0]).view(1,-1)
        local_records.append(dict(name=name,kind='matrix',rows=rows,columns=columns,
            row_encoder_vjp=row_encoder_vjp,column_encoder_vjp=column_encoder_vjp,pool_vjp=pool_vjp,
            token=token,anchor=obs.anchors))
        tokens.append(token);queries.append(query)
    for name,obs in snapshot.vectors.items():
        local,encoder_vjp=_append_mlp(plan,actor.vector_encoder,obs.coordinates)
        projected,pool_vjp=_append_pool(plan,actor,local)
        token=plan.reduce(projected,projected.shape[0],projected.shape[1],factor=1/projected.shape[0]).view(1,-1)
        local_records.append(dict(name=name,kind='vector',local=local,encoder_vjp=encoder_vjp,
                                  pool_vjp=pool_vjp,token=token))
        tokens.append(token);queries.append(token)
    normalized,logs=plan.trajectory_normalize(snapshot.trajectories,actor.rank)
    trajectory,trajectory_vjp=_append_mlp(plan,actor.trajectory_encoder,normalized)
    log_input=logs.view(1,2)
    scale_encoding,_=plan.linear(log_input,actor.trajectory_scale_encoder.weight,actor.trajectory_scale_encoder.bias)
    trajectory=plan.add(trajectory,plan.broadcast(scale_encoding.flatten(),trajectory.shape[0]))
    all_tokens=plan.concatenate((*tokens,trajectory))
    query_tokens=plan.concatenate(queries)
    crossed,cross_vjp=_append_mha(plan,actor.cross,actor.latents,all_tokens)
    residual=plan.add(actor.latents,crossed)
    latent,cross_stats=plan.layer_norm(residual,actor.cross_norm.weight,actor.cross_norm.bias,actor.cross_norm.eps)
    core_vjps=[]
    for block in actor.core:
        latent,block_vjp=append_transformer(plan,block,latent);core_vjps.append(block_vjp)
    contexts,readout_vjp=_append_mha(plan,actor.readout,query_tokens,latent)
    matrices={};vectors={}
    for index,record in enumerate(local_records):
        context=contexts[index]
        if record['kind']=='matrix':
            row_input=plan.context_concat(record['rows'],context)
            col_input=plan.context_concat(record['columns'],context)
            p,row_vjp=_append_mlp(plan,actor.row_decoder,row_input)
            q,col_vjp=_append_mlp(plan,actor.column_decoder,col_input)
            anchor=record['anchor']
            projected=plan.gemm(anchor.A,plan.gemm(anchor.A,p,ta=True))
            p=plan.add(p,projected,beta=-1.)
            matrices[record['name']]=Factors(p,q,anchor)
            record.update(row_decoder_vjp=row_vjp,column_decoder_vjp=col_vjp)
        else:
            decoder_input=plan.context_concat(record['local'],context)
            output,decoder_vjp=_append_mlp(plan,actor.vector_decoder,decoder_input)
            vectors[record['name']]=output.flatten()
            record['decoder_vjp']=decoder_vjp
    output=RawAction(matrices,vectors)
    def vjp(cotangents):
        gradients={}
        def accumulate(values,prefix=''):
            for name,value in values.items():
                name=prefix+name
                gradients[name]=plan.add(gradients[name],value) if name in gradients else value
        context_gradients=[]
        for record in local_records:
            name=record['name']
            if record['kind']=='matrix':
                gp=cotangents.matrices[name].P;gq=cotangents.matrices[name].Q;anchor=record['anchor']
                projected=plan.gemm(anchor.A,plan.gemm(anchor.A,gp,ta=True))
                gp=plan.add(gp,projected,beta=-1.)
                drows,grads=record['row_decoder_vjp'](gp);accumulate(grads,'row_decoder.')
                dcols,grads=record['column_decoder_vjp'](gq);accumulate(grads,'column_decoder.')
                dr,cr=plan.context_vjp(drows,record['rows'].shape[1])
                dc,cc=plan.context_vjp(dcols,record['columns'].shape[1])
                record.update(row_local_gradient=dr,column_local_gradient=dc)
                context_gradients.append(plan.add(cr,cc).view(1,-1))
            else:
                dy=cotangents.vectors[name].view(-1,1)
                dlocal,grads=record['decoder_vjp'](dy);accumulate(grads,'vector_decoder.')
                dl,context_gradient=plan.context_vjp(dlocal,record['local'].shape[1])
                record['local_gradient']=dl;context_gradients.append(context_gradient.view(1,-1))
        dq,dl,grads=readout_vjp(plan.concatenate(context_gradients));accumulate(grads,'readout.')
        for index in reversed(range(len(core_vjps))):
            dl,grads=core_vjps[index](dl);accumulate(grads,f'core.{index}.')
        dz,dgamma,dbeta=plan.layer_norm_vjp(residual,dl,actor.cross_norm.weight,cross_stats)
        accumulate({'weight':dgamma,'bias':dbeta},'cross_norm.')
        dlatents,dall,grads=cross_vjp(dz);accumulate(grads,'cross.')
        accumulate({'latents':plan.add(dz,dlatents)})
        offset=0
        for index,record in enumerate(local_records):
            count=record['token'].shape[0]
            dtoken=dall[offset:offset+count];offset+=count
            if record['kind']=='matrix':
                projected_gradient=plan.add(dtoken,plan.broadcast(dq[index],count,factor=1/count))
            else:
                projected_gradient=plan.broadcast(plan.add(dtoken.flatten(),dq[index]),
                    actor.pool_queries.shape[0],factor=1/actor.pool_queries.shape[0])
            dh,grads=record['pool_vjp'](projected_gradient);accumulate(grads)
            if record['kind']=='matrix':
                rows=record['rows'].shape[0]
                dr=plan.add(dh[:rows],record['row_local_gradient'])
                dc=plan.add(dh[rows:],record['column_local_gradient'])
                _,grads=record['row_encoder_vjp'](dr);accumulate(grads,'coordinate_encoder.')
                _,grads=record['column_encoder_vjp'](dc);accumulate(grads,'coordinate_encoder.')
            else:
                dh=plan.add(dh,record['local_gradient'])
                _,grads=record['encoder_vjp'](dh);accumulate(grads,'vector_encoder.')
        dtrajectory=dall[offset:]
        _,grads=trajectory_vjp(dtrajectory);accumulate(grads,'trajectory_encoder.')
        scale_gradient=plan.reduce(dtrajectory,dtrajectory.shape[0],dtrajectory.shape[1]).view(1,-1)
        _,dw,db=plan.linear_vjp(log_input,actor.trajectory_scale_encoder.weight,scale_gradient)
        accumulate({'weight':dw,'bias':db},'trajectory_scale_encoder.')
        expected=dict(actor.named_parameters())
        if gradients.keys()!=expected.keys():
            raise ValueError('persistent actor gradient coverage differs: '+str(set(expected)^set(gradients)))
        return {name:gradients[name] for name in expected}
    return output,vjp


class PersistentGlobalRewriterTrainer:
    def __init__(self, actor, snapshot, cotangents, *, lr=2e-4, betas=(.9,.99), eps=1e-8, kernel=None):
        self.actor=actor;self.snapshot=_staged_snapshot(snapshot)
        self.cotangents=RawAction({name:Factors(value.P.detach().contiguous(),value.Q.detach().contiguous(),value.anchors)
            for name,value in cotangents.matrices.items()},
            {name:value.detach().contiguous() for name,value in cotangents.vectors.items()})
        device=next(actor.parameters()).device
        plan=self.plan=PersistentTrainingPlan(device,kernel=kernel)
        self.output,backward=append_global_rewriter(plan,actor,self.snapshot)
        self.gradients=backward(self.cotangents)
        self.parameters=dict(actor.named_parameters())
        plan.adam(tuple(self.parameters.values()),tuple(self.gradients.values()),lr=lr,betas=betas,eps=eps)
        plan.finalize()

    def step(self):self.plan.step()
