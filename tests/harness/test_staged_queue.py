import yaml
from chisel.staged_queue import staged_recipe


def test_staging_preserves_training_contract_and_changes_only_storage_and_microbatch(tmp_path):
    original = dict(output_dir='/shared/seed7', microbatch_size=32,
        pre_forward=[], post_optimizer=[dict(name='chisel.evaluation.evaluate', hyperparams=dict(every=125, microbatch_size=32))],
        data=dict(batch_size=524288, seq_len=1024), optimizers=dict(hidden=dict(lr=.018)),
        seeds=dict(initialization=7), train_steps=3325)
    path=tmp_path/'source.yaml';path.write_text(yaml.safe_dump(original))
    staged=staged_recipe(path, tmp_path/'local', 8)
    for key in ('data','optimizers','seeds','train_steps'):
        assert staged[key]==original[key]
    assert staged['microbatch_size']==8
    assert staged['output_dir']==str(tmp_path/'local/seed7-local-stage')
    assert yaml.safe_load(path.read_text())==original
