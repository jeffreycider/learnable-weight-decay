from types import SimpleNamespace
from dataclasses import asdict
import pytest
import torch
from rewriter.offline_tokenwise_graph import TokenwiseGraphAccumulator,validate_training_options
from rewriter.offline_graph_backend import OfflineGraphAccumulator
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from tests.test_offline_graph_backend import fake_adapter


def test_tokenwise_metrics_feed_actual_trainer_aggregation(monkeypatch):
 from scripts.train_multistate_chisel import tokenwise_aggregate
 cfg=TokenwiseConfig(1e-10,1e-10,1e-8,.01);dual=TokenwiseDual([6],config=cfg,dual_lr=.01)
 a=object.__new__(TokenwiseGraphAccumulator);a.graph=SimpleNamespace(inputs=torch.ones(2,3,dtype=torch.long))
 monkeypatch.setattr(OfflineGraphAccumulator,'_metrics',lambda self,out,dual,state:({'mean_kl':float(out['token_kl'].mean())},{'norm_term':-.2}))
 rows=torch.tensor([-1e-16,0.,1e-11,1e-10,2e-10,3e-9],dtype=torch.float64)
 out=dict(token_kl=rows,token_loss_components=torch.tensor([.1,.2,.3],dtype=torch.float64),rank_change_counts=torch.tensor([1,2,1]),regularizer=torch.tensor(-.2))
 measured,stats=a._metrics(out,dual,6);result=tokenwise_aggregate([measured])
 assert result['token_count']==6 and result['sequence_count']==2
 assert result['token_tail_exceedances']==2 and result['negative_token_kl_count']==1
 assert result['argmax_changes']==1 and result['top10_order_changes']==2
 assert measured['kl_penalty']==pytest.approx(.6) and stats['offline_backend']=='tokenwise_graph'
 assert measured['token_kl_quantiles']['max']==3e-9


def test_new_backend_keeps_complete_accumulation_and_audit_close():
 base=fake_adapter();a=TokenwiseGraphAccumulator(base.actor,base.forward,base.engine,None,base.constants,radius_fraction=.01,shrinkage_coefficient=1.)
 a.begin_update([6,999],global_checkpoint_count=200)
 with pytest.raises(RuntimeError):a.finish_update()
 a._seen={6,999}
 before=[p.clone() for p in a.parameters];a.finish_update()
 assert all(torch.equal(p,b) for p,b in zip(a.parameters,before))
 a.close();assert a.graph is None and a._expected is None
 a.begin_update([6],global_checkpoint_count=200);a.close()


def test_graph_scope_is_explicit_and_unit_reward_only():
 options=dict(gpt_precision='fp32',norm_backend='fused',objective='tokenwise-v1',actor_backend='transformer',device='cuda',shrinkage_coefficient=1.,sequences_per_checkpoint=64,gpt_microbatch=16,checkpoint_batch=200,teacher_cache='cache',offline_systems_cache=True,eager_chisel=False,compile_gpt=False,gradient_diagnostics=False,no_component_timing=True)
 validate_training_options(SimpleNamespace(**options))
 for key,value in [('objective','primal-dual'),('shrinkage_coefficient',0.),('gpt_microbatch',32),('checkpoint_batch',4)]:
  with pytest.raises(ValueError):validate_training_options(SimpleNamespace(**dict(options,**{key:value})))
