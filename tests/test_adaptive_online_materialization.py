"""Dense execution is detached; the thin action/guard still owns delayed credit."""
from types import SimpleNamespace
import pytest
import torch
from rewriter.adaptive_preservation import build_adaptive_proposal
from rewriter.grouped_norm import GroupedNormObjective
from tests.test_adaptive_online import fixture


@pytest.mark.parametrize('radius', [1e-8, 100.])
def test_detached_materialization_has_exact_values_and_fewer_saved_tensor_bytes(radius):
    life, live, snapshot, _, _ = fixture(radius)
    base = {n:p.detach().clone() for n,p in live.items()}
    engine = SimpleNamespace(parameters=base, transpose_storage={n:n=='embed.weight' for n in base})
    outputs = []
    saved_sizes = []
    for enabled in (True, False):
        captured = []
        storages = {}
        def pack(tensor):
            storage = tensor.untyped_storage()
            storages[(str(tensor.device),storage.data_ptr())] = storage.nbytes()
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t:t):
            result = build_adaptive_proposal(
                lambda s:life.binding(s,base,embedding_names=('embed.weight',)), snapshot,
                engine, GroupedNormObjective(base,zero_reference_policy='freeze'),
                radius_fraction=radius, detach_credit=False,
                execution_capture=lambda action,scale:captured.append((action,scale)),
                materialization_grad=enabled)
        outputs.append(result)
        saved_sizes.append(sum(storages.values()))
        action, scale = captured[0]
        assert scale.requires_grad and scale.grad_fn is not None
        assert any(f.C.requires_grad for f in action.matrices.values())
    dense, thin = outputs
    assert dense[2] == thin[2]  # measured group norms, guard and realized edits
    assert torch.equal(dense[1],thin[1])
    assert dense[1].requires_grad and not thin[1].requires_grad
    for name,value in thin[0].items():
        assert torch.equal(value,dense[0][name])
        assert not value.requires_grad and value.grad_fn is None
    assert saved_sizes[1] < saved_sizes[0]


def test_pending_executed_storage_cannot_alias_live_gpt():
    life,live,snapshot,_,_ = fixture()
    life.commit(snapshot,live,action_step=0,base_step_completed=0)
    expected = {n:p.clone() for n,p in life.pending['executed'].items()}
    assert all(not p.requires_grad and p.grad_fn is None for p in life.pending['executed'].values())
    for name,parameter in live.items():
        assert parameter.untyped_storage().data_ptr() != life.pending['executed'][name].untyped_storage().data_ptr()
    with torch.no_grad():
        next(iter(live.values())).add_(1.)
    assert all(torch.equal(p,expected[n]) for n,p in life.pending['executed'].items())
    with pytest.raises(RuntimeError,match='mutated'):
        life.state_dict()


@pytest.mark.parametrize('detach,capture',[(True,None),(True,lambda *args:None),(False,None)])
def test_detached_dense_path_requires_explicit_thin_credit(detach,capture):
    with pytest.raises(ValueError,match='thin execution capture'):
        build_adaptive_proposal(None,None,None,None,radius_fraction=.01,
                                detach_credit=detach,execution_capture=capture,
                                materialization_grad=False)
