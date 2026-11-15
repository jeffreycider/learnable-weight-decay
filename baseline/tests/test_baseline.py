import ast
from pathlib import Path
import torch
from baseline.model import GPT, initialize_parameters_like_record
from baseline.train import validate_config, set_hparams


def upstream_model():
    source = (Path(__file__).parents[1]/'vendor/pr323_upstream_train.py').read_text()
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name != 'Muon']
    namespace = dict(torch=torch, Tensor=torch.Tensor, nn=torch.nn, F=torch.nn.functional)
    exec(compile(ast.Module(body=classes, type_ignores=[]), '<upstream_model>', 'exec'), namespace)
    return namespace['GPT']


def test_model_and_gradient_match_upstream():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = GPT(64, 1, 128)
    initialize_parameters_like_record(model)
    reference = upstream_model()(64, 1, 128)
    reference.load_state_dict(model.state_dict())
    # Test away from all-zero output projection to exercise all model gradients.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'proj' in name and parameter.ndim == 2:
                parameter.normal_(std=0.02)
    reference.load_state_dict(model.state_dict())
    x, y = torch.randint(64, (2, 8)), torch.randint(64, (2, 8))
    actual, expected = model(x, y), reference(x, y)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward(); expected.backward()
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_inventory():
    with torch.device('meta'):
        model = GPT(50304, 12, 768)
    matrices = [p for p in model.parameters() if p.ndim == 2]
    assert len(matrices) == 74
    assert sum(p.numel() for p in matrices) == 162201600
    assert sum(sum(p.shape) for p in matrices) == 268032
    assert sum(p.numel() for p in model.parameters()) == 162354816


def test_protocol_fail_closed():
    import json, pytest
    config = json.loads((Path(__file__).parents[1]/'config.json').read_text())
    validate_config(config)
    with pytest.raises(ValueError):
        validate_config(dict(config, max_steps=0))
    with pytest.raises(ValueError):
        validate_config(dict(config, batch_tokens=1024))
    validate_config(dict(config, batch_tokens=1024, unscored=True))


def test_schedule():
    class Opt:
        param_groups = [dict(initial_lr=0.025)]
    opt = Opt()
    set_hparams([opt], 0, 3000)
    assert opt.param_groups[0]['lr'] == 0.025
    set_hparams([opt], 1500, 3000)
    assert abs(opt.param_groups[0]['lr'] - 0.025*0.5/0.7) < 1e-12


def test_dataset_manifest_shard_tail_rule(tmp_path):
    import struct
    from baseline.verify_data import data_manifest
    # 10 tokens, batch4: positions 0 and4 are usable; token9 is held out target.
    for name in ['fineweb_train_000001.bin', 'fineweb_train_000002.bin', 'fineweb_val_000000.bin']:
        header = struct.pack('<256i', 20240520, 1, 10, *([0]*253))
        (tmp_path/name).write_bytes(header+struct.pack('<10H', *range(10)))
    config = dict(max_steps=3, batch_tokens=4, val_tokens=4,
                  train_data=str(tmp_path/'fineweb_train_*.bin'), val_data=str(tmp_path/'fineweb_val_*.bin'))
    rows = data_manifest(config, verify=False)
    assert [r['consumed_batches'] for r in rows] == [2, 1, 1]
    assert [Path(r['path']).name for r in rows] == ['fineweb_train_000001.bin', 'fineweb_train_000002.bin', 'fineweb_val_000000.bin']
