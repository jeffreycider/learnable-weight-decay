"""CPU architecture qualification, not Torch2.11 CUDA or throughput evidence."""
import copy
from dataclasses import fields, is_dataclass

import pytest
import torch

from baseline.model import GPT, Linear, Rotary
from baseline.fp32_execution import enable_fp32_execution
from rewriter.sensors import SensorCollector
from rewriter.sensor_compile_boundaries import compiled_sensor_leaves
from tests.test_sensors import anchors_for


def tensors(value, prefix=''):
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().clone()}
    if is_dataclass(value):
        value = {f.name: getattr(value, f.name) for f in fields(value)}
    if isinstance(value, (tuple, list)):
        value = dict(enumerate(value))
    result = {}
    if isinstance(value, dict):
        for k, v in value.items():
            result.update(tensors(v, prefix+'/'+str(k)))
    return result


@pytest.mark.parametrize('layers', [1, 12])
@pytest.mark.parametrize('compile_loss', [False, True])
def test_leaf_fullgraphs_preserve_two_step_sensor_and_adam_response(layers, compile_loss):
    torch.manual_seed(219)
    base = GPT(32, layers, 128).float()
    anchors = anchors_for(base)
    inputs = [torch.randint(32, (2, 3)) for _ in range(2)]
    targets = [torch.randint(32, (2, 3)) for _ in range(2)]

    def run(compiled):
        from contextlib import nullcontext
        model = copy.deepcopy(base)
        collector = SensorCollector(model, anchors, 2, 3, train_probes=1,
                                    audit_probes=1, trajectory_positions=6)
        enable_fp32_execution(model, sensors=collector)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        # Nonempty matched Adam state without an extra GPT call.
        for p in model.parameters():
            optimizer.state[p] = dict(step=torch.tensor(7.),
                                     exp_avg=torch.full_like(p, .001),
                                     exp_avg_sq=torch.full_like(p, .01))
        ids = {n: id(p) for n, p in model.named_parameters()}
        result = []
        compile_counts = []
        if compiled:
            # An unchanged second-step counter alone could also describe an
            # entirely eager execution. Require real first-step graph capture.
            torch._dynamo.reset()
        initial_graphs = torch._dynamo.utils.counters['stats']['unique_graphs']
        original_loss = model._fp32_loss
        context = compiled_sensor_leaves(model, backend='aot_eager', compile_loss=compile_loss) if compiled else nullcontext()
        rng = torch.get_rng_state().clone()
        try:
            with context:
                if not compiled or not compile_loss:
                    assert model._fp32_loss is original_loss
                else:
                    assert model._fp32_loss is not original_loss
                for step, (x, y) in enumerate(zip(inputs, targets)):
                    collector.reset(anchors, step=step)
                    collector.set_microbatch(torch.arange(2), x)
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(x, y)
                    loss.backward()
                    snapshot = collector.finalize()
                    grads = {n: p.grad.clone() for n, p in model.named_parameters()}
                    optimizer.step()
                    compile_counts.append(dict(torch._dynamo.utils.counters['stats']))
                    result.append(tensors(dict(loss=loss, sensors=snapshot, grads=grads,
                                               parameters=model.state_dict(), adam=optimizer.state_dict())))
            assert torch.equal(torch.get_rng_state(), rng)
            if compiled:
                assert compile_counts[0]['unique_graphs'] > initial_graphs
                assert compile_counts[0] == compile_counts[1]
            assert ids == {n: id(p) for n, p in model.named_parameters()}
            assert all('forward' not in m.__dict__ for m in model.modules()
                       if type(m) in (Linear, Rotary))
        finally:
            collector.close()
        return result

    reference, candidate = run(False), run(True)
    for a, b in zip(reference, candidate):
        assert a.keys() == b.keys()
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=2e-5, atol=2e-6, msg=key)


def test_leaf_context_restores_on_exception_and_rejects_nested_override():
    model = GPT(32, 1, 128).float()
    with pytest.raises(ValueError, match='FP32 GPT'):
        with compiled_sensor_leaves(model):
            pass
    enable_fp32_execution(model)
    loss = model._fp32_loss
    with pytest.raises(RuntimeError, match='sentinel'):
        with compiled_sensor_leaves(model, backend='aot_eager'):
            with pytest.raises(ValueError, match='nested'):
                with compiled_sensor_leaves(model):
                    pass
            raise RuntimeError('sentinel')
    assert model._fp32_loss is loss
    assert all('forward' not in m.__dict__ for m in model.modules()
               if type(m) in (Linear, Rotary))
