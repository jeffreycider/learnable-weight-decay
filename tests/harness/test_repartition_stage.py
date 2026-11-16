from copy import deepcopy
from pathlib import Path
import random
import numpy as np
import pytest
import torch
import yaml
from chisel.harness import RunState
from chisel.offline import ProbeStore
from chisel.preservation import CheckpointDuals, MultiplicativeDual
from chisel.repartition_stage import initialize
from chisel.storage import sha256


def fixture(tmp_path):
    manifest = tmp_path / 'data.yaml'
    manifest.write_text(yaml.safe_dump(dict(states=[
        dict(domain='text', path=f'{i}.pt', sha256=f'{i:064x}') for i in range(4)])))
    config = dict(data=dict(_target_='chisel.offline_data.Catalog', manifest=str(manifest),
                           batch_counts=dict(text=2), seed=1),
                  private_probes={}, model=dict(width=2), output_dir=str(tmp_path/'old'),
                  runtime=dict(world_size=2), train_steps=10)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    dual = MultiplicativeDual(initial=1, learning_rate=.1, tolerance=1e-10, maximum=1e8)
    rows, expected = [], {}
    for rank in range(2):
        pointers = {}
        for i in range(4):
            path = tmp_path / f'private_{i}_rank{rank}.pt'
            torch.save(dict(owner=rank, identity=i), path)
            pointers[f'{i:064x}'] = str(path)
            if i % 2 == rank:
                expected[f'{i:064x}'] = str(path)
        saved = dict(config=config, step=4, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     torch_rng=torch.get_rng_state(), cuda_rng=[], python_rng=random.getstate(),
                     numpy_rng=np.random.get_state(), extras=dict(dual=dual.state_dict(),
                     private_probes=dict(latest=pointers),
                     checkpoint_duals={f'{i:064x}': dict(dual.state_dict(), value=rank + i)
                                       for i in range(4)}))
        path = tmp_path / f'checkpoint_000005_rank{rank}.pt'
        torch.save(saved, path)
        rows.append(dict(path=path.name, sha256=sha256(path)))
    generation = tmp_path / 'generation_000005.yaml'
    generation.write_text(yaml.safe_dump(dict(step=5, ranks=rows)))
    new = deepcopy(config)
    new['data'].update(_target_='chisel.balanced_catalog.BalancedCatalog', owner_count=2)
    new['output_dir'] = str(tmp_path/'new')
    return new, generation, model, optimizer, expected, dual


def test_reshards_private_state_without_losing_optimizer_or_duals(tmp_path):
    config, generation, model, optimizer, expected, dual = fixture(tmp_path)
    for rank in range(2):
        actual = torch.nn.Linear(2, 2)
        state = RunState(model=actual, optimizer=torch.optim.Adam(actual.parameters(), lr=.01),
                         extras=dict(rank=rank, world_size=2,
                         private_probes=ProbeStore(tmp_path/'new', {}, 'cpu'),
                         dual=deepcopy(dual), checkpoint_duals=CheckpointDuals(deepcopy(dual))))
        initialize(str(generation), sha256(generation))(config, state)
        torch.testing.assert_close(actual.state_dict(), model.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(state.optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0)
        assert state.extras['private_probes'].latest == expected
        assert state.step == 4
        for identity, value in state.extras['checkpoint_duals'].states.items():
            assert value['value'] == int(identity, 16) + int(identity, 16) % 2


def test_rejects_changed_training_settings_or_damaged_parent(tmp_path):
    config, generation, *_ = fixture(tmp_path)
    state = RunState(extras=dict(rank=0, world_size=2))
    with pytest.raises(ValueError, match='generation SHA256'):
        initialize(str(generation), 'wrong')(config, state)
    config['model']['width'] = 4
    with pytest.raises(ValueError, match='training settings'):
        initialize(str(generation), sha256(generation))(config, state)
    config['model']['width'] = 2
    (tmp_path / 'checkpoint_000005_rank1.pt').write_bytes(b'damaged')
    with pytest.raises(ValueError, match='rank checkpoint SHA256'):
        initialize(str(generation), sha256(generation))(config, state)
    initialize('missing', 'missing')({'resume': 'existing'}, state)
