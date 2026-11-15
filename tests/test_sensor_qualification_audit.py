"""Real observer metric preflight, including intentional zero-probe channels."""
import pytest
import torch

from baseline.fp32_execution import enable_fp32_execution
from baseline.model import GPT
from rewriter.qualification_metrics import tensor_comparison
from rewriter.state_capture import DistributedStateObserver
from scripts.qualify_compiled_training_response import cpu_copy, state_tensors
from scripts.qualify_training_gemm import comparison as owner_comparison


@pytest.mark.parametrize('probes', [0, 1])
def test_actual_observer_sensor_and_actor_input_inventories(probes):
    torch.set_num_threads(1)
    torch.manual_seed(281)
    model = enable_fp32_execution(GPT(32, 1, 128).float())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    observer = DistributedStateObserver(model, [optimizer],
        dict(batch_tokens=6, seq_len=3, max_steps=10),
        sensor_config=dict(probe_rank=2, trajectory_positions=4,
                           train_sequence_probes=probes, audit_sequence_probes=probes))
    model._fp32_sensors = observer.collector
    try:
        inputs = torch.randint(32, (2, 3))
        observer.before_step(0, model, [optimizer])
        observer.before_microbatch(0, torch.arange(2), inputs)
        model(inputs, inputs.roll(-1, 1)).backward()
        observer.after_backward(0, model)
        sensors = cpu_copy(state_tensors(observer.sensors))
        # Flattening a dataclass to {} would manufacture a zero-error result.
        assert len(sensors) > 20
        empty = [name for name, value in sensors.items() if not value.numel()]
        assert bool(empty) == (probes == 0)
        optimizer.step()
        snapshot = observer.finish_snapshot(0)
        actor_inputs = cpu_copy(state_tensors(snapshot))
        assert len(actor_inputs) > 20
        assert not observer._in_step and not hasattr(observer, 'sensors')
        for inventory in (sensors, actor_inputs):
            result = tensor_comparison(inventory, inventory)
            assert result['finite'] and result['bit_identical']
            assert result['reference_norm'] > 0
            assert result['max_absolute_error'] == 0
            native = owner_comparison(inventory, inventory)
            assert native['absolute_l2_error'] == 0
            assert set(native['per_parameter']) == set(inventory)
            # A present-but-empty channel must not mask real feature changes.
            changed = cpu_copy(inventory)
            key = next(n for n, t in changed.items() if t.is_floating_point() and t.numel())
            # Observation views may be noncontiguous; reshape can copy and
            # would then mutate a temporary instead of the actual candidate.
            changed[key].add_(1)
            metric = tensor_comparison(inventory, changed)
            assert metric['finite'] and not metric['bit_identical']
            assert metric['absolute_l2_error'] > 0
            native = owner_comparison(changed, inventory)
            assert native['absolute_l2_error'] == pytest.approx(metric['absolute_l2_error'])
            assert native['reference_norm'] == pytest.approx(metric['reference_norm'])
        if empty:
            malformed = dict(sensors)
            malformed[empty[0]] = torch.empty(1, dtype=sensors[empty[0]].dtype)
            with pytest.raises(ValueError, match='shape/dtype'):
                tensor_comparison(sensors, malformed)
            missing = dict(sensors)
            del missing[empty[0]]
            with pytest.raises(ValueError, match='inventories'):
                tensor_comparison(sensors, missing)
    finally:
        observer.collector.close()
