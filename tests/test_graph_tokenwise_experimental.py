import sys
from types import SimpleNamespace
import pytest
import torch
from rewriter.graph_tokenwise import TokenwiseThinGradientGraph
from rewriter.graph_tokenwise_experimental import CompensatedDxTokenwiseGraph


def test_default_strict_options_are_unchanged():
    graph = object.__new__(TokenwiseThinGradientGraph)
    assert graph._student_options({'unused': torch.ones(1)}) == {}


def test_dx_decomposes_only_eligible_current_weights(monkeypatch):
    calls = []
    def split(value):
        assert not value.requires_grad and value.is_contiguous()
        calls.append(value.clone())
        return value.clone(), torch.zeros_like(value)
    monkeypatch.setitem(sys.modules, 'rewriter.kernels.compensated_pack_triton', SimpleNamespace(split=split))
    graph = object.__new__(CompensatedDxTokenwiseGraph)
    weights = {'proj.weight': torch.randn(3072, 8, requires_grad=True),
               'embed.weight': torch.randn(4096, 8),
               'blocks.0.attn.q.weight': torch.randn(8, 8),
               'proj.bias': torch.randn(3072)}
    first = graph._student_options(weights)
    assert set(first) == {'dx_split_weights'}
    assert set(first['dx_split_weights']) == {'proj.weight'}
    assert len(calls) == 1
    with torch.no_grad(): weights['proj.weight'].add_(1)
    second = graph._student_options(weights)
    assert len(calls) == 2
    torch.testing.assert_close(second['dx_split_weights']['proj.weight'][0], weights['proj.weight'].T)
    assert not torch.equal(first['dx_split_weights']['proj.weight'][0], second['dx_split_weights']['proj.weight'][0])


def test_combined_target_variant_rejected_before_initialization(monkeypatch):
    monkeypatch.delenv('NVIDIA_TF32_OVERRIDE', raising=False)
    with pytest.raises(ValueError, match='independently'):
        CompensatedDxTokenwiseGraph(teacher_representation='logits')


def test_uses_existing_graph_context_and_restores_strict_default(monkeypatch):
    from rewriter.compensated_dx import current_dx_backend
    monkeypatch.delenv('NVIDIA_TF32_OVERRIDE', raising=False)
    monkeypatch.setattr(TokenwiseThinGradientGraph, '__init__', lambda self: None)
    graph = CompensatedDxTokenwiseGraph()
    graph.graph = None
    graph.actor = SimpleNamespace(zero_grad=lambda **kwargs: None)
    graph._body = current_dx_backend
    assert graph.eager() == 'cublas3_cached'
    assert current_dx_backend() == 'ieee'


def test_global_tensor_core_disable_fails_before_expensive_oracles(monkeypatch):
    from rewriter.graph_tokenwise_experimental import require_compensated_dx_environment
    monkeypatch.setenv('NVIDIA_TF32_OVERRIDE', '0')
    with pytest.raises(ValueError, match='disables'):
        require_compensated_dx_environment()
    monkeypatch.delenv('NVIDIA_TF32_OVERRIDE')
    require_compensated_dx_environment()
