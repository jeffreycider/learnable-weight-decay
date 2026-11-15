"""Independent CPU AOT audit; allocator residency still needs the owner GPU test."""
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from rewriter.compiled_thin_student import CompiledThinStudent
from rewriter.online_preservation import ReplayMaintenance
from rewriter.qualification_metrics import tensor_comparison
from tests.test_compiled_thin_student import fixture


@pytest.mark.parametrize('compile_kl', [False, True])
def test_raw_cotangents_rebind_and_transient_teachers_against_actual_eager(compile_kl):
    torch.set_num_threads(1)
    engine, wire, old_teachers, editable = fixture()
    inputs = torch.cat([*(t.inputs for t in old_teachers), old_teachers[0].inputs[:1]])
    del old_teachers
    # Three local sequences split2+1, global24: each microbatch contributes
    # its actual token count /120, not an unweighted mean of microbatch means.
    assert inputs.shape == (3, 5)
    engine.microbatch_sequences = 2
    replay = SimpleNamespace(engine=engine, model=engine.view.gpt,
                             device=engine.device, embedding_names={'embed.weight'},
                             settings={'sequences':24}, seq_len=5)
    helper = CompiledThinStudent(engine, editable, global_tokens=120,
                                 embedding_names={'embed.weight'},
                                 compiler_backend='aot_eager',compile_kl=compile_kl)
    addresses = {n: p.data_ptr() for n, p in engine.parameters.items()}
    results = []
    for binding in range(2):
        if binding:
            state = {n: p.clone() for n, p in engine.parameters.items()}
            state['proj.bias'].add_(torch.linspace(-.2,.2,len(state['proj.bias'])))
            engine.load_teacher_state(state, engine.buffers)
        teachers = engine.cache_teacher(inputs, data_role='past_training')
        assert [t.tokens for t in teachers] == [10,5]
        for leaf in wire[2]:
            leaf.grad = None
        expected, grads = ReplayMaintenance._student(replay,wire,teachers,editable,backward=True)
        saved = {str(i): g.clone() for i,g in enumerate(grads)}
        actual, got = helper.step(wire,teachers,editable)
        metric = tensor_comparison(saved,{str(i):g for i,g in enumerate(got)})
        assert metric['finite'] and metric['relative_error'] < 3e-4
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=1e-12)
        final, no_grad = helper.step(wire,teachers,editable,backward=False)
        assert no_grad == [] and not final.requires_grad
        torch.testing.assert_close(final,actual,rtol=2e-5,atol=1e-12)
        # The helper and compiled callable remain alive. No-grad final readout
        # must not retain these Python teacher objects across maintenance blocks.
        refs = [weakref.ref(t.log_probs) for t in teachers]
        results.append(float(actual))
        del teachers, expected, actual, final, grads, got
        gc.collect()
        assert all(ref() is None for ref in refs)
    assert addresses == {n:p.data_ptr() for n,p in engine.parameters.items()}
    assert results[0] != results[1]  # actual new base was consumed, no stale output
    assert all(p.grad is None and not p.requires_grad for p in engine.parameters.values())
