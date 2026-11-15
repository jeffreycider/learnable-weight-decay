"""Optional reference reuse must preserve arithmetic and reject stale bases."""
import pytest
import torch
from tests.test_transformer_action import fixture,FactorActor
from tests.test_online_norm_lifecycle import make,AUX
from rewriter.transformer_action import build_action_reference_cache,prepare_transformer_action


@pytest.mark.parametrize('radius',[.0001,10.])
def test_cache_matches_uncached_action_and_actor_gradient_exactly(radius):
    base,snapshot=fixture(torch.float64);actor=FactorActor()
    cache=build_action_reference_cache(base)
    old=prepare_transformer_action(actor,snapshot,base,radius_fraction=radius)
    new=prepare_transformer_action(actor,snapshot,base,radius_fraction=radius,reference_cache=cache)
    a,b=old.materialize(),new.materialize()
    for name in a:torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
    ga,=torch.autograd.grad(sum(v.square().sum() for v in a.values()),actor.x)
    gb,=torch.autograd.grad(sum(v.square().sum() for v in b.values()),actor.x)
    torch.testing.assert_close(ga,gb,rtol=0,atol=0)
    # The cache belongs to bases, not an actor version: next candidate reuses it.
    with torch.no_grad():actor.x.mul_(1.01)
    prepare_transformer_action(actor,snapshot,base,reference_cache=cache)


def test_cache_rejects_replaced_or_mutated_base_and_constants():
    base,snapshot=fixture();cache=build_action_reference_cache(base);actor=FactorActor().float()
    replaced={n:v.clone() for n,v in base.items()}
    with pytest.raises(ValueError,match='different base'):prepare_transformer_action(actor,snapshot,replaced,reference_cache=cache)
    cache.norm_squared.add_(1)
    with pytest.raises(RuntimeError,match='immutable'):prepare_transformer_action(actor,snapshot,base,reference_cache=cache)
    cache=build_action_reference_cache(base);base['proj.weight'].add_(1)
    with pytest.raises(RuntimeError,match='immutable'):prepare_transformer_action(actor,snapshot,base,reference_cache=cache)


def test_one_reference_construction_and_one_radius_scan_per_auxiliary_commit(monkeypatch):
    import rewriter.transformer_action as module
    real_group=module.GroupedNormObjective;real_norm=module.editable_weight_norm_sq;calls={'group':0,'radius':0}
    def group(*args,**kwargs):calls['group']+=1;return real_group(*args,**kwargs)
    def norm(*args,**kwargs):calls['radius']+=1;return real_norm(*args,**kwargs)
    monkeypatch.setattr(module,'GroupedNormObjective',group);monkeypatch.setattr(module,'editable_weight_norm_sq',norm)
    life,live,snapshot=make(AUX);life.commit(snapshot,live,action_step=1000)
    assert calls=={'group':1,'radius':1}


def test_all_frozen_zero_groups_need_no_radius_scan():
    base,snapshot=fixture(torch.float64)
    for p in base.values():p.zero_()
    cache=build_action_reference_cache(base)
    assert not cache.editable and cache.norm_squared==0
    actor=FactorActor()
    proposal=prepare_transformer_action(actor,snapshot,base,reference_cache=cache).materialize()
    for name in base:torch.testing.assert_close(proposal[name],base[name],rtol=0,atol=0)
