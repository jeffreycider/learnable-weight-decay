import copy

import pytest
import torch

from baseline.chisel_transition import online_rank_checkpoint,validate_online_config,validate_policy,load_policy_checkpoint,sha256
from baseline.fp32_transition import promote_bundle
from baseline.checkpoint import validate_resume_config
from scripts.bundle_baseline_checkpoint import extract_rank,_record_equal
from tests.test_fp32_prefix_transition import bundle


def settings():
    return dict(prefix_steps=25,bootstrap='first_ordinary_step_unedited',online_auxiliary='none',
                task_lr=1e-6,representation_lr_multiplier=10.,gradient_clip=1.,radius_fraction=.01,
                policy_provenance=dict(prefix_bundle_sha256='a'*64))


def test_online_transition_preserves_rank_state_and_requires_explicit_mode_change(bundle):
    promoted=promote_bundle(bundle,source_sha256='a'*64,microbatch_sequences=1,max_steps=100,compile_model=False)
    promoted['shared']['config']['unscored']=True
    original=copy.deepcopy(promoted)
    for rank in range(8):
        base=extract_rank(promoted,rank)
        for mode in ('shadow','learned'):
            changed=online_rank_checkpoint(base,settings(),mode=mode)
            for key in ('model','optimizers','data_cursor','rng','step','counters','elapsed_seconds'):
                _record_equal(changed[key],base[key],key)
            assert changed['controller']['bootstrap']
            assert changed['controller']['rank']==rank
            assert changed['config']['schedule_steps']==3250
            validate_resume_config(changed['config'],copy.deepcopy(changed['config']))
            with pytest.raises(ValueError,match='research configuration'):
                validate_resume_config(base['config'],changed['config'])
    _record_equal(promoted,original,'source unchanged')


def test_online_config_rejects_changed_prefix_precision_batch_and_compilation(bundle):
    p=promote_bundle(bundle,source_sha256='a'*64,microbatch_sequences=1,max_steps=100,compile_model=False)
    p['shared']['config']['unscored']=True
    cfg=online_rank_checkpoint(extract_rank(p,0),settings(),mode='shadow')['config']
    bad=copy.deepcopy(cfg);bad['compile']=True
    with pytest.raises(ValueError,match='compilation'):validate_online_config(bad)
    bad=copy.deepcopy(cfg);bad['transformer_chisel']['bootstrap']='precommit'
    with pytest.raises(ValueError,match='bootstrap'):validate_online_config(bad)
    bad=copy.deepcopy(cfg);bad['fp32_prefix_transition']['rms_epsilon']=2**-7
    with pytest.raises(ValueError,match='corrected'):validate_online_config(bad)
    bad=copy.deepcopy(cfg);bad['transformer_chisel']['policy_provenance']['prefix_bundle_sha256']='b'*64
    with pytest.raises(ValueError,match='prefix source'):validate_online_config(bad)
    bad=copy.deepcopy(cfg);bad['unscored']=False
    with pytest.raises(ValueError,match='production online'):validate_online_config(bad)


def test_actor_provenance_epsilon_and_qualification_are_checked(tmp_path):
    identity=dict(selected_states=1000,manifest_sha256='b'*64,teacher_cache_manifest_sha256='c'*64)
    policy=dict(controller={'x':torch.tensor(1.)},metadata=dict(architecture={'name':'fixture'},state_bank_identity=identity,
        gpt_precision=dict(mode='fp32',rms_epsilon=2**-23,tf32=False,attention_backend='math'),diagnostic_prefix_only=False))
    path=tmp_path/'policy.pt';torch.save(policy,path)
    c=dict(prefix_steps=1000,architecture={'name':'fixture'},policy_checkpoint=str(path),
           policy_provenance=dict(state_bank_identity=identity,policy_checkpoint_sha256=sha256(path)))
    load_policy_checkpoint(c,device='cpu')
    with pytest.raises(ValueError,match='qualification'):load_policy_checkpoint(c,device='cpu',require_qualification=True)
    bad=copy.deepcopy(policy);bad['metadata']['gpt_precision']['rms_epsilon']=2**-7
    with pytest.raises(ValueError,match='corrected'):validate_policy(bad,c)
    bad=copy.deepcopy(c);bad['policy_provenance']['state_bank_identity']['teacher_cache_manifest_sha256']='d'*64
    with pytest.raises(ValueError,match='different state bank'):validate_policy(policy,bad)


@pytest.mark.parametrize('precision,mode,modern,expected',[
    ('baseline','baseline',False,'ordinary_original_precision_gpt'),
    ('fp32','baseline',False,'ordinary_fp32_gpt'),
    ('fp32','shadow',True,'fp32_observer_identity'),
    ('fp32','learned',True,'fp32_chisel_task_lag_one'),
])
def test_whole_iteration_workload_taxonomy_and_actual_pass_denominator(precision,mode,modern,expected):
    from baseline.train import iteration_workload
    cfg=dict(gpt_precision=precision,mode=mode,world_size=8,batch_tokens=524288,seq_len=1024,microbatch_sequences=16)
    if modern:cfg['transformer_chisel']={'enabled':True}
    row=iteration_workload(cfg)
    assert row['training_procedure']==expected
    assert row['global_sequences_per_iteration']==512
    assert row['expected_ordinary_microbatches_per_rank']==4
    assert row['global_tokens_per_iteration']==524288
    assert 'excludes validation' in row['iteration_timing_scope']


def test_first_live_action_retention_is_explicit_and_preserves_pending_clock(bundle):
    p=promote_bundle(bundle,source_sha256='a'*64,microbatch_sequences=1,max_steps=100,compile_model=False)
    p['shared']['config']['unscored']=True
    s={**settings(),'preserve_first_action_checkpoint':True}
    learned=online_rank_checkpoint(extract_rank(p,0),s,mode='learned')
    assert learned['config']['preserve_checkpoint_steps']==[26]
    assert learned['step']==25 and learned['config']['max_steps']==100
    shadow=online_rank_checkpoint(extract_rank(p,0),s,mode='shadow')
    assert 'preserve_checkpoint_steps' not in shadow['config']
    short=copy.deepcopy(extract_rank(p,0));short['config']['max_steps']=26
    with pytest.raises(ValueError,match='later ordinary step'):online_rank_checkpoint(short,s,mode='learned')


def test_new_transition_can_pin_pure_loss_compilation_but_ordinary_resume_cannot_toggle(bundle):
    p=promote_bundle(bundle,source_sha256='a'*64,microbatch_sequences=1,max_steps=100,compile_model=False)
    p['shared']['config']['unscored']=True
    before=copy.deepcopy(p)
    p['shared']['config']['compile_loss']=True
    for rank in range(8):
        base=extract_rank(p,rank);c=online_rank_checkpoint(base,settings(),mode='learned')
        assert c['config']['compile_loss'] is True and c['config']['compile'] is False
        for key in ('model','optimizers','data_cursor','rng','step','counters'):
            _record_equal(c[key],extract_rank(before,rank)[key],key)
        changed=copy.deepcopy(c['config']);changed['compile_loss']=False
        with pytest.raises(ValueError,match='research configuration'):validate_resume_config(c['config'],changed)
