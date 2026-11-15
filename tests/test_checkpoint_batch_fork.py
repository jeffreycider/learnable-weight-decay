import copy
import hashlib
import io
import json
import tarfile

import pytest
import torch

from scripts import fork_chisel_checkpoint_batch as f


@pytest.fixture
def fixture(tmp_path):
    def archive(name, files):
        path = tmp_path / name
        with tarfile.open(path, 'w:gz') as tar:
            for key, text in files.items():
                raw = text.encode(); item = tarfile.TarInfo(key); item.size = len(raw)
                tar.addfile(item, io.BytesIO(raw))
        return path
    files = {'scripts/train_multistate_chisel.py': 'unchanged',
             'rewriter/offline_tokenwise_graph.py': 'only200'}
    old = archive('old.tgz', files)
    new = archive('new.tgz', dict(files, **{'rewriter/offline_tokenwise_graph.py': '40and200'}))
    objective = {'mean_tolerance': 1e-10, 'loss_backend': 'tiled_fp64_v1'}
    bank = {'teacher_cache_manifest_sha256': 'same-cache'}
    state = dict(metadata=dict(world_size=8, options=dict(objective='tokenwise-v1',
        offline_backend='tokenwise_graph', checkpoint_batch=200, lr=1e-6,
        epochs=160, minimum_epochs=30, full_training_audit_final=True),
        tokenwise_objective=objective, state_bank_identity=bank,
        implementation_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()}),
        controller={'w': torch.randn(3,2)}, optimizer=dict(state={0:dict(step=torch.tensor(60.),
            exp_avg=torch.randn(3,2), exp_avg_sq=torch.rand(3,2))}, param_groups=[
            dict(role='action_heads', lr=1e-6, lr_multiplier=1., params=[0]),
            dict(role='representation', lr=1e-6*10, lr_multiplier=10., params=[1])]),
        primal_dual={'multipliers': {1:.12, 999:.15}, 'updates':12000},
        fit_schedule={'best':129., 'stale_checks':0, 'reductions':0},
        per_rank_rng=[{'torch':torch.get_rng_state()} for _ in range(8)],
        per_rank_compute=[{'student_tokens':1234} for _ in range(8)],
        next_epoch=15, next_batch=0, updates=60, clipped_updates=60, completed=False)
    source=tmp_path/'controller_epoch_0015.pt';torch.save(state,source)
    evidence=[]
    for lr in (1e-6,2e-7):
        path=tmp_path/f'gate-{lr}.json';path.write_text(json.dumps({'passed':True,'head_lr':lr,'checkpoints':40}))
        evidence.append(dict(head_lr=lr,path=path.name,sha256=f.sha(path),
            required_values=[[['passed'],True],[['head_lr'],lr],[['checkpoints'],40]]))
    contract=dict(schema='qualified_checkpoint_batch_fork_v1',qualified=True,
        input_checkpoint_sha256=f.sha(source),input_updates=60,input_next_epoch=15,input_head_lr=1e-6,
        target_checkpoint_batch=40,head_learning_rates=[1e-6,2e-7],first_review_completed_epoch=20,
        old_archive_sha256=f.sha(old),new_archive_sha256=f.sha(new),
        archive_changed_paths=['rewriter/offline_tokenwise_graph.py'],qualification_evidence=evidence,
        tokenwise_objective=objective,state_bank_identity=bank)
    path=tmp_path/'contract.json';path.write_text(json.dumps(contract))
    return dict(source=source,destination=tmp_path/'fork.pt',contract_path=path,
        old_archive=old,new_archive=new,evidence_root=tmp_path,head_lr=1e-6),state,contract


@pytest.mark.parametrize('lr',[1e-6,2e-7])
def test_preserves_all_scientific_state_except_declared_lr(fixture,lr):
    args,state,c=fixture;args['head_lr']=lr;before=f.sha(args['source'])
    result=f.fork(**args);got=torch.load(args['destination'],weights_only=False)
    assert f.sha(args['source'])==before
    assert f.protected_digest(got)==f.protected_digest(state)==result['protected_state_sha256']
    assert got['metadata']['options']==dict(state['metadata']['options'],checkpoint_batch=40,lr=lr)
    assert got['metadata']['tokenwise_objective']==state['metadata']['tokenwise_objective']
    assert [g['lr'] for g in got['optimizer']['param_groups']]==[lr,lr*10]
    assert got['next_epoch']==15 and got['updates']==60 and got['next_batch']==0
    assert not result['first_review_is_stopping_boundary']
    assert got['metadata']['options']['epochs']==160 and not got['completed']


@pytest.mark.parametrize('failure',['unqualified','missing_lr_gate','bad_gate','changed_input',
    'mid_epoch','wrong_clock','completed','wrong_group_lr','wrong_source','wrong_objective',
    'extra_archive_change','rolling','existing_destination','unapproved_lr'])
def test_fail_closed(fixture,failure):
    args,state,c=fixture
    if failure=='unqualified':c['qualified']=False
    elif failure=='missing_lr_gate':c['qualification_evidence']=c['qualification_evidence'][:1]
    elif failure=='bad_gate':
        p=args['evidence_root']/c['qualification_evidence'][0]['path'];p.write_text('{}')
    elif failure=='changed_input':c['input_checkpoint_sha256']='wrong'
    elif failure=='mid_epoch':state['next_batch']=1
    elif failure=='wrong_clock':state['updates']=61
    elif failure=='completed':state['completed']=True
    elif failure=='wrong_group_lr':state['optimizer']['param_groups'][1]['lr']=9e-6
    elif failure=='wrong_source':state['metadata']['implementation_sha256']['scripts/train_multistate_chisel.py']='wrong'
    elif failure=='wrong_objective':state['metadata']['tokenwise_objective']={'mean_tolerance':1e-8}
    elif failure=='extra_archive_change':c['archive_changed_paths']=[]
    elif failure=='rolling':args['source']=args['source'].with_name('controller_latest.pt')
    elif failure=='existing_destination':args['destination'].write_bytes(b'preserve')
    elif failure=='unapproved_lr':args['head_lr']=1e-5
    torch.save(state,args['source'])
    if failure!='changed_input':c['input_checkpoint_sha256']=f.sha(args['source'])
    args['contract_path'].write_text(json.dumps(c))
    with pytest.raises(ValueError):f.fork(**args)
    if failure=='existing_destination':assert args['destination'].read_bytes()==b'preserve'
    else:assert not args['destination'].exists()


def test_digest_detects_moments_duals_rng_and_schedule_changes(fixture):
    _,state,_=fixture;before=f.protected_digest(state)
    changes=[lambda s:s['optimizer']['state'][0]['exp_avg'].add_(1),
             lambda s:s['primal_dual']['multipliers'].update({1:0}),
             lambda s:s['per_rank_rng'][0]['torch'].zero_(),
             lambda s:s['fit_schedule'].update(best=0),
             lambda s:s.update(next_epoch=0)]
    for change in changes:
        altered=copy.deepcopy(state);change(altered);assert f.protected_digest(altered)!=before
