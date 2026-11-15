"""Bounded GPU logit-only gate, no GPT/actor/optimizer evaluation."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baseline.execution import enable_deterministic_execution
from rewriter.gpt_precision import configure_precision
from rewriter.tokenwise_reference import TokenwiseConfig,logit_terms_and_cotangent
from rewriter.relative_rank_tail import maximum_relative_rank_loss
from rewriter.token_preservation import token_preservation_terms


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
 enable_deterministic_execution();configure_precision('fp32');torch.cuda.set_device(0);torch.manual_seed(91721)
 cfg=TokenwiseConfig(1e-10,1e-10,1e-8,.01,ranking_weight=100.,ranking_k=100,ranking_tie_policy='exact_teacher_ties',ranking_loss_mode='relative_gap_micro_max',loss_backend='reference_micro_max_v1')
 started=time.perf_counter();rows=[]
 for vocab in (10,50304):
  logits=torch.randn(128,vocab,device='cuda');teacher=logits.double().log_softmax(-1);student=logits+torch.randn_like(logits)*1e-4;del logits
  actual,sums,kl=logit_terms_and_cotangent(student,teacher,config=cfg,multiplier=.3,panel_tokens=512,chunk_tokens=32)
  s=student.detach().clone().requires_grad_();lp=s.double().log_softmax(-1);raw=(teacher.exp()*(teacher-lp)).sum(-1)
  terms=token_preservation_terms(raw,multiplier=.3,mean_tolerance=cfg.mean_tolerance,tail_tolerance=cfg.tail_tolerance,scale=cfg.scale,rho=cfg.rho)
  maximum=maximum_relative_rank_loss(teacher,s,k=min(100,vocab)).max(dim=0).values
  objective=terms['total'].sum()/512+maximum*100/4
  expected=torch.autograd.grad(objective,s)[0]
  relative=float((actual.double()-expected.double()).norm()/expected.double().norm().clamp_min(1e-30));rank_difference=float((sums[2]-maximum.detach()*100*128).abs());kl_difference=float((kl.to(raw.device)-raw.detach()).abs().max())
  assert relative<=1e-6 and rank_difference<=1e-10*max(1.,float(maximum)) and kl_difference<=3e-15
  rows.append(dict(vocabulary=vocab,contexts=128,gradient_relative=relative,ranking_sum_difference=rank_difference,max_kl_difference=kl_difference))
  del actual,sums,kl,s,lp,raw,terms,maximum,objective,expected,teacher,student
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 logits=torch.randn(16384,50304,device='cuda');teacher=logits.double().log_softmax(-1);student=logits+torch.randn_like(logits)*1e-4;del logits
 diagnostics={};a,sa,ka=logit_terms_and_cotangent(student,teacher,config=cfg,multiplier=.3,panel_tokens=65536,chunk_tokens=128,diagnostics=diagnostics)
 b,sb,kb=logit_terms_and_cotangent(student,teacher,config=cfg,multiplier=.3,panel_tokens=65536,chunk_tokens=256)
 assert torch.equal(a,b) and torch.equal(ka,kb) and torch.equal(sa[2:],sb[2:])
 assert torch.isfinite(a).all() and torch.isfinite(sa).all()
 summary=dict(all_gates_pass=True,dense_oracles=rows,fullshape=dict(contexts=16384,vocabulary=50304,chunk_invariant_gradient_bitexact=True,diagnostics=diagnostics,peak_allocated_bytes=torch.cuda.max_memory_allocated()),elapsed_seconds=time.perf_counter()-started,compute=dict(GPT_forwards=0,GPT_backwards=0,actor_updates=0),scope='Logit VJP gate; no full-model finite-Adam equivalence claim')
 output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
