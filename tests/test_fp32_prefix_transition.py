import copy
import pytest
import torch
from baseline.fp32_transition import promote_bundle,training_forward_count
from baseline.checkpoint import validate_resume_config
from scripts.bundle_baseline_checkpoint import bundle_checkpoints,extract_rank,_record_equal
from tests.test_merge_baseline_checkpoint import MergeBaselineTests


@pytest.fixture
def bundle():
    fixture=MergeBaselineTests(methodName='runTest');fixture.setUp()
    try:
        path=fixture.root/'bundle.pt';bundle_checkpoints(fixture.paths,path)
        yield torch.load(path,weights_only=False)
    finally:fixture.doCleanups()


def test_fp32_transition_preserves_all_rank_state_and_only_promotes_precision(bundle):
    before=copy.deepcopy(bundle);rng=torch.get_rng_state().clone()
    result=promote_bundle(bundle,source_sha256='a'*64,microbatch_sequences=1,max_steps=100)
    _record_equal(bundle,before,'original remains unchanged')
    assert torch.equal(rng,torch.get_rng_state())
    for rank in range(8):
        original=extract_rank(bundle,rank);changed=extract_rank(result,rank)
        for key in ['rng','data_cursor','counters','step','elapsed_seconds']:
            _record_equal(original[key],changed[key],key)
        for name,value in original['model'].items():
            torch.testing.assert_close(changed['model'][name],value.float(),rtol=0,atol=0)
        assert changed['optimizers'][0]['state'][0]['exp_avg'].dtype==torch.float32
        assert original['optimizers'][0]['state'][0]['exp_avg'].dtype==torch.bfloat16
        _record_equal(original['optimizers'][1],changed['optimizers'][1],'Muon state')
        for index in range(2):
            _record_equal(original['optimizers'][index]['param_groups'],changed['optimizers'][index]['param_groups'],'optimizer groups')
    cfg=result['shared']['config']
    assert cfg['max_steps']==100 and cfg['schedule_steps']==3250
    assert training_forward_count(cfg,25)==25
    assert training_forward_count(cfg,28)==31 # prefix25*1 + suffix3*2
    validate_resume_config(cfg,copy.deepcopy(cfg))
    with pytest.raises(ValueError,match='research configuration'):validate_resume_config(bundle['shared']['config'],cfg)
    with pytest.raises(ValueError,match='already'):promote_bundle(result,source_sha256='a'*64)


@pytest.mark.parametrize('mutation,match',[
    (lambda b:b['shared']['counters'].update(forward=26),'counters'),
    (lambda b:b['ranks'][3]['data_cursor'].update(batches=24),'cursor'),
    (lambda b:b['ranks'].pop(7),'lacks'),
    (lambda b:b['shared'].update(controller={}), 'baseline'),
])
def test_transition_rejects_incomplete_or_inconsistent_prefix(bundle,mutation,match):
    mutation(bundle)
    with pytest.raises(ValueError,match=match):promote_bundle(bundle,source_sha256='a'*64,microbatch_sequences=1)


def test_transition_cannot_extend_schedule_or_change_effective_batch(bundle):
    with pytest.raises(ValueError,match='schedule'):promote_bundle(bundle,source_sha256='a'*64,max_steps=3251)
    with pytest.raises(ValueError,match='global batch'):promote_bundle(bundle,source_sha256='a'*64,microbatch_sequences=3)
    with pytest.raises(ValueError,match='SHA256'):promote_bundle(bundle,source_sha256='bad',microbatch_sequences=1)


def test_training_extension_is_explicit_pinned_and_only_appends(bundle):
    bundle['shared']['config']['max_steps']=6_000_000
    with pytest.raises(ValueError,match='explicit extension'):
        promote_bundle(bundle,source_sha256='a'*64,microbatch_sequences=1)
    result=promote_bundle(bundle,source_sha256='a'*64,microbatch_sequences=1,training_shards=4)
    for rank in range(8):
        old=bundle['ranks'][rank]['data_cursor'];new=result['ranks'][rank]['data_cursor']
        assert new['file_names'][:len(old['file_names'])]==old['file_names']
        for key in old.keys()-{'file_names'}:assert old[key]==new[key]
    extension=result['precision_transition']['training_shard_extension']
    assert len(extension['extended'])==4 and len(extension['sha256'])==4
    with pytest.raises(ValueError,match='prefix order'):
        promote_bundle(bundle,source_sha256='a'*64,microbatch_sequences=1,training_shards=0)


@pytest.mark.parametrize('field,value', [('rms_epsilon',2.0**-7),
                                       ('rms_epsilon',None),
                                       ('tf32',True)])
def test_old_or_mismatched_precision_transition_is_rejected(bundle,field,value):
    converted=promote_bundle(bundle,source_sha256='a'*64,microbatch_sequences=1,max_steps=100)
    config=converted['shared']['config']
    config['fp32_prefix_transition'][field]=value
    with pytest.raises(ValueError,match='normalization epsilon'):
        training_forward_count(config,25)
