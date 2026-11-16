from chisel.harness import resolve


def test_text_treatments_preserve_architecture_data_and_ordinary_optimizer():
    baseline = resolve('recipes/nanogpt_muonh_40gb.yaml')
    for recipe in ('nanogpt_direct_transfer.yaml', 'nanogpt_direct_zero.yaml'):
        config = resolve('recipes/' + recipe)
        for key in ('model', 'data', 'optimizers', 'evaluation_data', 'seeds', 'train_steps', 'microbatch_size'):
            assert config[key] == baseline[key]
        assert config['data']['batch_size'] == 524288
        assert config['data']['batch_size'] // (8 * 1024 * config['microbatch_size']) == 8


def test_smoke_keeps_real_schedule_and_cifar_keeps_qk_scale():
    smoke = resolve('recipes/nanogpt_direct_smoke.yaml')
    assert smoke['train_steps'] == 3325 and smoke['stop_step'] == 20
    cifar = resolve('recipes/cifar_direct_transfer.yaml')
    assert cifar['model']['learned_qk_scale']
    assert not cifar['model']['causal']
