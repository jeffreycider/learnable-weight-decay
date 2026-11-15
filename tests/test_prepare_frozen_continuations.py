import copy
import hashlib
import json
import random
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from baseline.checkpoint import build_checkpoint, restore_checkpoint, validate_resume_config
from baseline.data import FinewebStream
from baseline.optim import Muon
from scripts.prepare_frozen_continuations import (
    continuation_config, reshard_checkpoint, check_endpoint_statistics,
    validate_completed_run, write_arm, family_edit_statistics, ROOT, sha256,
    LEGACY_SCOPED_SHA256,
)


class TinyParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(i+2,3)) for i in range(11)])
        self.bias = nn.Parameter(torch.randn(3))
        self.register_buffer('unchanged_buffer',torch.arange(4))


def optimizers(model):
    adam=torch.optim.AdamW([model.bias],lr=.1,betas=(.8,.95),eps=1e-10,weight_decay=.001)
    muon=Muon(list(model.weights),lr=.025,weight_decay=.05)
    for opt in (adam,muon):
        for g in opt.param_groups:g['initial_lr']=g['lr']
    return [adam,muon]


def fixture(tmp_path):
    for i,n in enumerate((100,140)):
        (tmp_path/f'fineweb_train_{i}.bin').write_bytes(
            struct.pack('<256i',20240520,1,n,*([0]*253))+np.arange(i*1000,i*1000+n,dtype='<u2').tobytes())
    pattern=str(tmp_path/'*.bin');model=TinyParameters();opts=optimizers(model)
    for opt in opts:
        for group in opt.param_groups:
            group['lr'] *= .7  # current scheduled LR is preserved, initial LR unchanged
            for p in group['params']:
                opt.state[p] = (dict(momentum=torch.randn_like(p)) if 'mu' in group else
                                dict(step=torch.tensor(3.),exp_avg=torch.randn_like(p),exp_avg_sq=torch.rand_like(p)))
    loader=FinewebStream(pattern,32,2,rank=0,world_size=1,device='cpu')
    for _ in range(3):next(loader)
    cfg=dict(mode='baseline',world_size=1,max_steps=8,schedule_steps=8,seed=0,
             batch_tokens=32,seq_len=2,microbatch_sequences=2,compile=False,compile_loss=True,
             deterministic=True,train_data=pattern,val_data='never_read',model={'tiny':True},rewriter={})
    state=build_checkpoint(model,opts,None,loader,cfg,3,dict(forward=24,backward=24),123.45)
    target=continuation_config(cfg,label='identity',provenance=dict(source_sha256='a'*64))
    return state,target,loader


def same_tree(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same_tree(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same_tree(x,y)
    else:assert a==b


def test_reshard_preserves_state_and_partitions_all_momenta_once(tmp_path):
    source,target,_=fixture(tmp_path);original=copy.deepcopy(source)
    shards=[reshard_checkpoint(source,target,rank) for rank in range(8)]
    ids=source['optimizers'][1]['param_groups'][0]['params'];owned=[]
    for rank,out in enumerate(shards):
        same_tree(out['model'],source['model']);same_tree(out['optimizers'][0],source['optimizers'][0])
        same_tree(out['optimizers'][1]['param_groups'],source['optimizers'][1]['param_groups'])
        assert set(out['optimizers'][1]['state'])==set(ids[rank::8])
        for pid,moment in out['optimizers'][1]['state'].items():same_tree(moment,source['optimizers'][1]['state'][pid])
        owned.extend(out['optimizers'][1]['state'])
        same_tree(out['rng'],source['rng'])
        assert out['counters']==dict(forward=3,backward=3)
        assert out['elapsed_seconds']==0 and out['fork_history']['source_elapsed_seconds']==123.45
        assert out['fork_history']['actual_prefix_global_forward']==24
        assert out['controller'] is None
    assert sorted(owned)==sorted(ids) and len(set(owned))==len(owned)
    assert sum(s['counters']['forward'] for s in shards)==source['counters']['forward']
    shards[0]['model']['bias'].zero_();shards[0]['optimizers'][0]['state'][0]['exp_avg'].zero_()
    same_tree(source,original)
    same_tree(shards[1]['model'],source['model'])


def test_modulo_ownership_restarts_at_each_group(tmp_path):
    source,target,_=fixture(tmp_path);muon=source['optimizers'][1]
    first=muon['param_groups'][0];second=copy.deepcopy(first)
    ids=first['params'];first['params']=ids[:3];second['params']=ids[3:]
    muon['param_groups'].append(second)
    rank0=reshard_checkpoint(source,target,0)
    assert set(rank0['optimizers'][1]['state'])=={ids[0],ids[3]}
    rank7=reshard_checkpoint(source,target,7)
    assert set(rank7['optimizers'][1]['state'])=={ids[10]}


def test_actual_strict_resume_and_global_next_batch_across_shard_boundary(tmp_path):
    source,target,world1=fixture(tmp_path)
    expected_x,expected_y=next(world1);all_x=[];all_y=[]
    for rank in range(8):
        fork=reshard_checkpoint(source,target,rank)
        model=TinyParameters();opts=optimizers(model)
        loader=FinewebStream(source['config']['train_data'],32,2,rank=rank,world_size=8,device='cpu')
        step,counters,elapsed=restore_checkpoint(fork,model,opts,None,loader,target)
        assert (step,counters,elapsed)==(3,dict(forward=3,backward=3),0.)
        x,y=next(loader);all_x.append(x);all_y.append(y)
        assert loader.file_index==1 and loader.pos==32 and loader.batches==4
        same_tree(model.state_dict(),source['model'])
    torch.testing.assert_close(torch.cat(all_x),expected_x,rtol=0,atol=0)
    torch.testing.assert_close(torch.cat(all_y),expected_y,rtol=0,atol=0)
    with pytest.raises(ValueError,match='research configuration'):
        validate_resume_config(source['config'],target)


def test_weight_fork_changes_only_requested_model_values(tmp_path):
    source,target,_=fixture(tmp_path);edited=copy.deepcopy(source['model']);edited['weights.0']+=.125
    out=reshard_checkpoint(source,target,2,model_state=edited)
    assert torch.equal(out['model']['weights.0'],edited['weights.0'])
    same_tree(out['model']['unchanged_buffer'],source['model']['unchanged_buffer'])
    same_tree(out['optimizers'][0],source['optimizers'][0])
    for pid,st in out['optimizers'][1]['state'].items():same_tree(st,source['optimizers'][1]['state'][pid])


@pytest.mark.parametrize('case',['step','counters','cursor','world','missing_momentum','extra_moment','algorithm','shape','dtype'])
def test_invalid_source_or_target_fails_closed(tmp_path,case):
    source,target,_=fixture(tmp_path);override=None
    if case=='step':source['step']=0
    if case=='counters':source['counters']['forward']=3
    if case=='cursor':source['data_cursor']['batches']=2
    if case=='world':source['config']['world_size']=2
    if case=='missing_momentum':source['optimizers'][1]['state'].pop(0)
    if case=='extra_moment':source['optimizers'][1]['state'][0]['unknown']=torch.ones(1)
    if case=='algorithm':target['schedule_steps']=10
    if case in ('shape','dtype'):
        override=dict(source['model']);override['bias']=torch.zeros(4 if case=='shape' else 3,dtype=torch.float64)
    with pytest.raises(ValueError):reshard_checkpoint(source,target,0,model_state=override)


def test_rng_and_no_model_passes_during_preparation(tmp_path,monkeypatch):
    source,target,_=fixture(tmp_path)
    def forbidden(*a,**kw):raise AssertionError('GPT evaluation forbidden')
    from baseline.model import GPT
    monkeypatch.setattr(GPT,'forward',forbidden)
    before=torch.random.get_rng_state().clone();py=random.getstate();np_rng=np.random.get_state()
    out=reshard_checkpoint(source,target,0)
    torch.testing.assert_close(torch.random.get_rng_state(),before,rtol=0,atol=0)
    same_tree(random.getstate(),py);same_tree(np.random.get_state(),np_rng)
    assert out['fork_history']['preparation_gpt_forwards']==0


def test_write_rank_files_configs_and_hashes(tmp_path):
    source,target,_=fixture(tmp_path)
    manifest=write_arm(source,source['model'],target,tmp_path/'fork',provenance={'test':True})
    assert not manifest['training_launched'] and not manifest['benchmark_compliant']
    for name,row in manifest['files'].items():assert sha256(tmp_path/'fork'/name)==row['sha256']
    for rank in range(8):
        saved=torch.load(tmp_path/'fork'/f'latest_rank{rank}.pt',weights_only=False)
        assert saved['data_cursor']['rank']==rank and saved['config']==target


def test_endpoint_tolerances_reject_wrong_action_but_allow_small_numerical_error():
    record=dict(audit=dict(norm_relative_change=-.003,realized_edit_norm=901.),
                train=dict(raw_action_norm=1000.,global_scale=.9))
    actual=dict(norm_relative_change=-.003*(1+1e-7),realized_edit_norm=901.,raw_action_norm=1000.,global_scale=.9)
    check_endpoint_statistics(actual,record)
    for key in actual:
        bad=dict(actual);bad[key]*=1.00001
        with pytest.raises(ValueError,match=key):check_endpoint_statistics(bad,record)


def test_grouped_endpoint_rejects_compensating_group_errors_even_with_same_mean():
    from rewriter.grouped_norm import GROUP_NAMES
    groups={name:float(i) for i,name in enumerate(GROUP_NAMES)}
    record=dict(config=dict(norm_objective='grouped_l2'),
        audit=dict(norm_relative_change=-.07,realized_edit_norm=10.,group_l2_percentage_reduction=groups),
        train=dict(raw_action_norm=20.,global_scale=.5))
    actual=dict(norm_relative_change=-.07,realized_edit_norm=10.,raw_action_norm=20.,global_scale=.5,
                group_l2_percentage_reduction=dict(groups))
    good=check_endpoint_statistics(actual,record)
    assert good['reported']['group_l2_percentage_reduction']==groups
    # Equal/opposite errors leave the aggregate exactly unchanged.
    bad=copy.deepcopy(actual)
    bad['group_l2_percentage_reduction'][GROUP_NAMES[0]]+=.1
    bad['group_l2_percentage_reduction'][GROUP_NAMES[1]]-=.1
    assert sum(bad['group_l2_percentage_reduction'].values())==sum(groups.values())
    with pytest.raises(ValueError,match='group percentage mismatch'):
        check_endpoint_statistics(bad,record)
    for altered in (dict(groups,extra=0.),{k:v for k,v in groups.items() if k!=GROUP_NAMES[0]}):
        with pytest.raises(ValueError,match='group names'):
            check_endpoint_statistics(dict(actual,group_l2_percentage_reduction=altered),record)
    close=copy.deepcopy(actual)
    close['group_l2_percentage_reduction'][GROUP_NAMES[0]]=5e-9
    close['group_l2_percentage_reduction'][GROUP_NAMES[2]]*=1+1e-7
    check_endpoint_statistics(close,record)


def completed_run(tmp_path):
    p=tmp_path/'run';p.mkdir();cfg={'mode':'baseline','rewriter':{'probe_rank':8}}
    frozen=dict(checkpoint_sha256='a'*64,checkpoint_step=1750,base_config=cfg,steps=1000,
                controller_state_transferred=False,action_units='relative',optimizer_mode='adam',
                preservation_penalty='deadband',kl_arithmetic='float64 from float32 GPT logits',source_readout='live')
    (p/'config.json').write_text(json.dumps(frozen))
    torch.save(dict(step=1000,config=frozen,controller={}),p/'controller.pt')
    (p/'observation.pt').write_bytes(b'not-loaded-during-preflight')
    (p/'replay_provenance.json').write_text(json.dumps(dict(metadata=dict(completed_steps=1750),
        train=[dict(step=1,id=0)],audit=[dict(step=2,id=1)])))
    rows=[dict(event='train',step=1000),dict(event='evaluation',step=1000,label='learned'),
          dict(event='finished',controller_steps=1000,benchmark_compliant=False)]
    (p/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    names=['rewriter/algebra.py','rewriter/controller.py','rewriter/function_preservation.py','scripts/train_frozen_preservation.py']
    (p/'source_record.txt').write_text(''.join(f'===== {n} SHA256 {sha256(ROOT/n)} =====\n' for n in names))
    return p,cfg


def test_completed_run_checks_source_recipe_and_unique_final_audit_before_cuda(tmp_path):
    run,cfg=completed_run(tmp_path)
    r=validate_completed_run(run,cfg,'a'*64,1750)
    assert r['config']['steps']==1000 and len(r['files'])==6
    with pytest.raises(ValueError,match='source checkpoint'):validate_completed_run(run,cfg,'b'*64,1750)
    (run/'metrics.jsonl').write_text(json.dumps(dict(event='train',step=900))+'\n')
    with pytest.raises(ValueError,match='not complete'):validate_completed_run(run,cfg,'a'*64,1750)


def test_changed_reconstruction_source_fails_closed(tmp_path):
    run,cfg=completed_run(tmp_path)
    (run/'source_record.txt').write_text('wrong source')
    with pytest.raises(ValueError,match='source differs'):validate_completed_run(run,cfg,'a'*64,1750)


def test_grouped_preflight_requires_matching_objective_percentage_units_and_source(tmp_path):
    run,cfg=completed_run(tmp_path)
    frozen=json.loads((run/'config.json').read_text())
    frozen.update(editable_scope='grouped',norm_objective='grouped_l2',norm_target_fraction=.01)
    def save_config():
        (run/'config.json').write_text(json.dumps(frozen))
        torch.save(dict(step=1000,config=frozen,controller={}),run/'controller.pt')
    save_config()
    with (run/'source_record.txt').open('a') as stream:
        for name in ('rewriter/edit_scope.py','rewriter/grouped_norm.py'):
            stream.write(f'===== {name} SHA256 {sha256(ROOT/name)} =====\n')
    assert validate_completed_run(run,cfg,'a'*64,1750)['config']['norm_objective']=='grouped_l2'
    frozen['norm_objective']='global_squared';save_config()
    with pytest.raises(ValueError,match='do not match'):validate_completed_run(run,cfg,'a'*64,1750)
    frozen['norm_objective']='grouped_l2';frozen['norm_target_fraction']=1e-4;save_config()
    with pytest.raises(ValueError,match='percentage units'):validate_completed_run(run,cfg,'a'*64,1750)


def test_known_eb67fa2_hidden_sources_remain_replayable_without_accepting_unknown_source(tmp_path):
    run,cfg=completed_run(tmp_path)
    frozen=json.loads((run/'config.json').read_text());frozen['editable_scope']='hidden'
    (run/'config.json').write_text(json.dumps(frozen))
    torch.save(dict(step=1000,config=frozen,controller={}),run/'controller.pt')
    text=(run/'source_record.txt').read_text().replace(sha256(ROOT/'scripts/train_frozen_preservation.py'),
         LEGACY_SCOPED_SHA256['scripts/train_frozen_preservation.py'])
    text+='===== rewriter/edit_scope.py SHA256 '+LEGACY_SCOPED_SHA256['rewriter/edit_scope.py']+' =====\n'
    (run/'source_record.txt').write_text(text)
    assert validate_completed_run(run,cfg,'a'*64,1750)['config']['editable_scope']=='hidden'
    (run/'source_record.txt').write_text(text.replace(LEGACY_SCOPED_SHA256['rewriter/edit_scope.py'],'b'*64))
    with pytest.raises(ValueError,match='source differs'):validate_completed_run(run,cfg,'a'*64,1750)


def test_family_norm_contractions_and_negative_weight_cosine():
    base={'embed.weight':torch.tensor([[2.,3.]]),'proj.weight':torch.tensor([[4.,2.]]),
          'blocks.0.w':torch.tensor([[1.,2.]]),'proj.bias':torch.tensor([5.])}
    edited={n:.75*v for n,v in base.items()};result=family_edit_statistics(base,edited)
    assert set(result)=={'embedding','head','hidden_matrices','one_dimensional'}
    for row in result.values():
        assert row['cosine_toward_negative_weight']==pytest.approx(1.)
        assert row['norm_sq_change']==pytest.approx((.75**2-1)*row['base_norm_sq'])
        assert row['edit_norm_sq']==pytest.approx(.25**2*row['base_norm_sq'])
    identity=family_edit_statistics(base,base)
    assert all(r['cosine_toward_negative_weight'] is None for r in identity.values())


@pytest.mark.parametrize('case',['stale_controller','future_replay'])
def test_stale_controller_or_future_replay_rejected_before_cuda(tmp_path,case):
    run,cfg=completed_run(tmp_path)
    if case=='stale_controller':
        saved=torch.load(run/'controller.pt',weights_only=False);saved['step']=999;torch.save(saved,run/'controller.pt')
    else:
        replay=json.loads((run/'replay_provenance.json').read_text());replay['train'][0]['step']=1750
        (run/'replay_provenance.json').write_text(json.dumps(replay))
    with pytest.raises(ValueError):validate_completed_run(run,cfg,'a'*64,1750)
