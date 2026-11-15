"""Read-only FP32/FP64 evaluation of the same saved Chisel proposals.

Production remains FP32. This diagnostic changes arithmetic only: it retains
the stored RoPE frequencies, FP32-computed sin/cos tables and RMS epsilon.
The FP64 variants respectively retain or remove the final edited-weight cast.
Each precision has its own unedited teacher; comparisons against the original
FP32 teacher are also reported, because precision can itself change ranks.
No optimizer updates, scalar search, validation data, or training occurs.
"""
import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from baseline.model import GPT
from rewriter.adaptive_preservation import build_adaptive_proposal
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.function_preservation import FunctionalPreservation
from rewriter.gpt_precision import RMS_EPSILON
from rewriter.grouped_coordinate_chisel import GroupedCoordinateChisel
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.prefix_cache import unpack_record
from rewriter.rank_preservation import topk_preservation_flags
from rewriter.relative_ranking_loss import relative_tie_aware_topk_margin_loss
from scripts.diagnose_cpu_norm_preservation import load_bound_state, teacher_cache_class, tensor_sha, pin_sources
from scripts.fit_cpu_edit_teacher import sha
from scripts.profile_cpu_joint_gradients import fitted_inputs


@torch.no_grad()
def diagnostic_logits(parameters, buffers, inputs, *, layers, dtype):
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('FP32 or FP64 diagnostic only')
    def norm(x, name=None):
        weight = parameters[name].to(dtype) if name else None
        return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=RMS_EPSILON)
    def linear(x, name):
        return F.linear(x, parameters[name+'.weight'].to(dtype), parameters[name+'.bias'].to(dtype))
    def rotary(x, frequency):
        # Keep the same angle tables as the production path. Do not invoke
        # baseline Rotary, which would cast FP64 q/k operands back to FP32.
        pos = torch.arange(x.shape[1], dtype=torch.float32, device=x.device)
        theta = torch.outer(pos, frequency.float())[None, :, None, :]
        co, si = theta.cos().to(dtype), theta.sin().to(dtype)
        x1, x2 = x.chunk(2, -1)
        return torch.cat((x1*co+x2*si, x1*(-si)+x2*co), 3)
    x = norm(F.embedding(inputs, parameters['embed.weight'].to(dtype)), 'norm1.gains')
    batch, length, width = x.shape
    heads = width // 128
    for i in range(layers):
        prefix = f'blocks.{i}.'
        u = norm(x, prefix+'norm1.gains')
        q, k, v = [linear(u, prefix+'attn.'+n).view(batch, length, heads, 128) for n in ('q','k','v')]
        frequency = buffers[prefix+'attn.rotary.angular_freq']
        q, k = rotary(norm(q), frequency), rotary(norm(k), frequency)
        with sdpa_kernel(SDPBackend.MATH):
            y = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
                                              scale=.12, is_causal=True).transpose(1,2)
        x = x + linear(y.contiguous().view(batch, length, width), prefix+'attn.proj')
        u = linear(norm(x, prefix+'norm2.gains'), prefix+'mlp.fc').relu().square()
        x = x + linear(u, prefix+'mlp.proj')
    logits = linear(norm(x, 'norm2.gains'), 'proj')
    return 15*logits*(logits.square()+225).rsqrt()


class UnroundedProposal(Mapping):
    """Materialize one FP64 tensor at a time from the exact captured factors."""
    def __init__(self, base, action, scale, editable, transpose):
        self.base, self.action, self.scale = base, action, scale
        self.editable, self.transpose = editable, transpose
    def __len__(self): return len(self.base)
    def __iter__(self): return iter(self.base)
    def __getitem__(self, name):
        value = self.base[name].double()
        if name not in self.editable: return value
        delta = self.action.matrices[name].dense() if value.ndim == 2 else self.action.vectors[name]
        if value.ndim == 2 and self.transpose.get(name, False): delta = delta.T
        return value + self.scale*delta


@torch.no_grad()
def compare_logits(teacher, student, k=100):
    teacher = teacher.reshape(-1,teacher.shape[-1])
    student = student.reshape_as(teacher)
    counts = {}
    kl_sum = rank_sum = 0.
    max_kl = -float('inf')
    max_delta = 0.
    changed_positions, gaps = [], []
    for start in range(0,len(teacher),128):
        t, s = teacher[start:start+128].double(), student[start:start+128].double()
        lp, lq = t.log_softmax(-1), s.log_softmax(-1)
        kl = (lp.exp()*(lp-lq)).sum(-1)
        flags = topk_preservation_flags(lp, s, k=k)
        for name, value in flags.items(): counts[name] = counts.get(name,0)+int(value.sum())
        changed_positions.extend((flags['strict_topk_changed'].nonzero().flatten()+start).tolist())
        ranking = relative_tie_aware_topk_margin_loss(lp,s,k=k,margin_fraction=.5)
        kl_sum += float(kl.sum()); rank_sum += float(ranking.sum())
        max_kl = max(max_kl,float(kl.max()))
        max_delta = max(max_delta,float((s-t).abs().max()))
        tv = t.topk(k+1,dim=-1).values
        g = tv[:,:-1]-tv[:,1:]
        gaps.append(g[g>0])
    gaps = torch.cat(gaps)
    return dict(tokens=len(teacher), mean_kl=kl_sum/len(teacher), max_token_kl=max_kl,
                ranking_penalty=100*rank_sum/len(teacher), counts=counts,
                changed_positions=changed_positions, max_abs_logit_change=max_delta,
                positive_teacher_adjacent_gap_quantiles={str(q):float(torch.quantile(gaps,q)) for q in (0.,.001,.01,.5)} if gaps.numel() else {})


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--state-id',type=int,default=6)
    parser.add_argument('--sequences',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if not 1<=args.sequences<=2: raise ValueError('Bounded diagnostic: one or two original fitting sequences')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    args.output.mkdir(parents=True,exist_ok=False)
    sources = pin_sources(args.output)
    started = time.perf_counter()
    input_hashes = {str(p):sha(p) for p in (args.before,args.after)}
    before = torch.load(args.before,weights_only=False,mmap=True,map_location='cpu')
    after = torch.load(args.after,weights_only=False,mmap=True,map_location='cpu')
    if after['completed_updates']!=before['completed_updates']+1: raise ValueError('Adjacent saved updates required')
    meta=before['metadata']
    if before['metadata']!=after['metadata']: raise ValueError('Training configurations differ')
    case=next(c for c in meta['cases'] if c['state_id']==args.state_id)
    for key,path in case['paths'].items():
        if sha(path)!=case['sha256'][key]: raise ValueError('Case file changed: '+key)
    state=torch.load(case['paths']['state'],weights_only=False,mmap=True,map_location='cpu')
    panel=torch.load(case['paths']['panel'],weights_only=False,mmap=True,map_location='cpu')
    inputs=fitted_inputs(panel,case); del panel
    snapshot=unpack_record(state['observation'],device='cpu').snapshot
    base=state['model']
    layers=len({n.split('.')[1] for n in base if n.startswith('blocks.')})
    model=GPT(base['embed.weight'].shape[0],layers,base['embed.weight'].shape[1]).float().eval().requires_grad_(False)
    engine=FunctionalPreservation(model,microbatch_sequences=1,kl_dtype=torch.float64,gpt_precision='fp32')
    load_bound_state(engine,base,case['sha256']['state'])
    grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
    actor=GroupedCoordinateChisel(rank=8,trajectory_dim=400,model_dim=256,num_layers=2,latent_tokens=8,
        fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32').eval().requires_grad_(False)
    private=actor.make_private(snapshot,state_key=('cpu-teaching',0,args.state_id,case['sha256']['state']),seed=0)
    cache=next(c for c in meta['teacher_cache_provenance'] if c['state_id']==args.state_id)
    if sha(Path(cache['path'])/'manifest.json')!=cache['manifest_sha256']:raise ValueError('Teacher manifest changed')
    provider=teacher_cache_class(cache.get('format','fp64_log_probs'))(cache['path'],engine,inputs,case['sequence_ids'],case['sha256']['state'])
    rows=[]; counts=dict(teacher_forwards=0,student_forwards=0,gpt_backwards=0,optimizer_updates=0)
    # Save just these two diagnostic teachers to avoid retaining large logits.
    for seq in range(args.sequences):
        teachers={}
        for dtype in (torch.float32,torch.float64):
            teachers[str(dtype)]=diagnostic_logits(engine.parameters,engine.buffers,inputs[seq:seq+1],layers=layers,dtype=dtype)
            counts['teacher_forwards']+=1
        cached=provider(seq,seq+1)
        if not torch.equal(teachers['torch.float32'].double().log_softmax(-1),cached.log_probs):
            raise RuntimeError('Diagnostic FP32 teacher differs from original cached teacher')
        del cached
        torch.save(teachers,args.output/f'teachers_{seq}.pt')
        row=dict(kind='precision_only_unedited_teacher',sequence=seq,sequence_id=case['sequence_ids'][seq],
                 result=compare_logits(teachers['torch.float32'],teachers['torch.float64']))
        rows.append(row); print(json.dumps(row),flush=True)
        del teachers
    for label,saved in [('before',before),('after',after)]:
        actor.load_state_dict(saved['actor'],strict=True)
        private.load_state_dict(saved['private'][args.state_id],strict=True)
        captured=[]
        def propose(scoped):
            cores=query_weight_cores(engine.parameters,private,embedding_names=tuple(n for n,t in engine.transpose_storage.items() if t))
            return actor(scoped,private,weight_cores=cores)
        parameters,norm,stats,editable,_=build_adaptive_proposal(propose,snapshot,engine,grouped,radius_fraction=.01,
            detach_credit=False,execution_capture=lambda action,scale:captured.append((action,scale)))
        action,scale=captured[0]
        unrounded=UnroundedProposal(engine.parameters,action,scale,editable,engine.transpose_storage)
        for name in parameters:
            if not torch.equal(unrounded[name].float(),parameters[name]):raise RuntimeError('Captured proposal differs from actual FP32 edit: '+name)
        for seq in range(args.sequences):
            teachers=torch.load(args.output/f'teachers_{seq}.pt',weights_only=True,mmap=True)
            for mode,weights,dtype in [('fp32_realized',parameters,torch.float32),
                                      ('fp64_realized',parameters,torch.float64),
                                      ('fp64_unrounded',unrounded,torch.float64)]:
                logits=diagnostic_logits(weights,engine.buffers,inputs[seq:seq+1],layers=layers,dtype=dtype)
                counts['student_forwards']+=1
                row=dict(kind='edited',checkpoint=label,completed_updates=saved['completed_updates'],mode=mode,
                    sequence=seq,sequence_id=case['sequence_ids'][seq],norm_objective=float(norm),
                    matched_teacher=compare_logits(teachers[str(dtype)],logits),
                    original_fp32_teacher=compare_logits(teachers['torch.float32'],logits) if dtype==torch.float64 else None)
                rows.append(row);print(json.dumps(row),flush=True)
                del logits
            del teachers
        del parameters,unrounded,captured,action,scale
    if any(sha(p)!=h for p,h in input_hashes.items()) or any(sha(p)!=h for p,h in sources.items()):
        raise RuntimeError('Saved checkpoint or diagnostic source changed')
    result=dict(scope=__doc__,state_id=args.state_id,sequences=args.sequences,input_sha256=tensor_sha(inputs[:args.sequences]),
                checkpoint_sha256=input_hashes,source_sha256=sources,rows=rows,compute=counts,elapsed_seconds=time.perf_counter()-started,
                limitations=['FP64 is a diagnostic model; its teacher rankings can differ from the production FP32 teacher.',
                             'Two fitting sequences from one early GPT checkpoint, not held-out or online evidence.',
                             'RoPE sin/cos constants and RMS epsilon are fixed to the FP32 model values.'])
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
