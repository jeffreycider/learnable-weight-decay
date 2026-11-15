import hashlib
import io
import json
import tarfile

import pytest
import torch

from scripts import migrate_tokenwise_graph_checkpoint as m


@pytest.fixture
def case(tmp_path):
    def archive(name, files):
        p = tmp_path / name
        with tarfile.open(p, 'w:gz') as t:
            for key, value in files.items():
                data = value.encode(); info = tarfile.TarInfo(key); info.size = len(data)
                t.addfile(info, io.BytesIO(data))
        return p
    old_files = {'scripts/train_multistate_chisel.py': 'old', 'rewriter/tokenwise_reference.py': 'same'}
    new_files = dict(old_files, **{'scripts/train_multistate_chisel.py': 'new', 'rewriter/tokenwise_graph_backend.py': 'graph'})
    old = archive('old.tar.gz', old_files); new = archive('new.tar.gz', new_files)
    objective = {'mean_tolerance': 1e-10, 'dual_update': 'clipped_relative', 'loss_backend': 'tiled_fp64_v1'}
    bank = {'teacher_cache_manifest_sha256': 'corrected-eps23'}
    state = dict(metadata=dict(world_size=8, options={'offline_backend': 'reference', 'objective': 'tokenwise-v1', 'lr': 1e-6},
                 implementation_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k,v in old_files.items()},
                 tokenwise_objective=objective, state_bank_identity=bank, offline_backend={'name': 'reference'}),
                 controller={'weight': torch.randn(2,3)}, optimizer={'state': {0: {'step': torch.tensor(20.), 'exp_avg': torch.randn(2,3)}}},
                 primal_dual={'multipliers': {6: .05, 999: .05}, 'updates': 4000},
                 fit_schedule={'stale_checks': 1, 'best': 288., 'lr_reductions': 0},
                 per_rank_rng=[{'torch': torch.get_rng_state(), 'cuda': torch.arange(16,dtype=torch.uint8)} for _ in range(8)],
                 per_rank_compute=[{'student_backwards': 2000+i} for i in range(8)],
                 next_epoch=5, next_batch=0, updates=20, clipped_updates=20, completed=False)
    source = tmp_path/'controller_epoch_0005.pt'; torch.save(state,source)
    evidence=[]
    for role in ('full200_ddp','trainer_adapter'):
        p=tmp_path/(role+'.json');p.write_text(json.dumps({'passed':True,'source':m.sha(new)}))
        evidence.append({'role':role,'path':p.name,'sha256':m.sha(p),'required_values':[[['passed'],True],[['source'],m.sha(new)]]})
    contract=dict(schema='qualified_tokenwise_graph_migration_v1',qualified=True,old_archive_sha256=m.sha(old),new_archive_sha256=m.sha(new),
                  qualification_evidence=evidence,target_implementation_paths=list(new_files),
                  changed_existing_dependencies=['scripts/train_multistate_chisel.py'],state_bank_identity=bank,
                  tokenwise_objective=objective,target_backend_metadata={'name':'tokenwise_graph'})
    c=tmp_path/'contract.json';c.write_text(json.dumps(contract))
    return dict(source=source,destination=tmp_path/'migrated.pt',contract_path=c,old_archive=old,new_archive=new,evidence_root=tmp_path),state,contract


def test_preserves_all_state_and_only_changes_declared_metadata(case):
    args,state,contract=case; before=m.sha(args['source'])
    record=m.migrate(**args); got=torch.load(args['destination'],weights_only=False)
    assert m.sha(args['source'])==before
    assert m.numerical_digest(got)==m.numerical_digest(state)==record['numerical_state_sha256']
    assert got['metadata']['tokenwise_objective']==state['metadata']['tokenwise_objective']
    assert got['metadata']['options']==dict(state['metadata']['options'],offline_backend='tokenwise_graph')
    assert got['primal_dual']==state['primal_dual']
    assert got['fit_schedule']==state['fit_schedule']


@pytest.mark.parametrize('failure',['rolling','mid_epoch','completed','old_source','objective','bank','missing_evidence','evidence_mutated','unapproved_source','archive_mutated','existing_destination'])
def test_rejects_unqualified_or_unsafe_migration(case,failure):
    args,state,c=case
    if failure=='rolling': args['source']=args['source'].with_name('controller_latest.pt')
    elif failure=='mid_epoch':state['next_batch']=1
    elif failure=='completed':state['completed']=True
    elif failure=='old_source':state['metadata']['implementation_sha256']['rewriter/tokenwise_reference.py']='bad'
    elif failure=='objective':state['metadata']['tokenwise_objective']=dict(state['metadata']['tokenwise_objective'],mean_tolerance=1e-8)
    elif failure=='bank':state['metadata']['state_bank_identity']={'teacher_cache_manifest_sha256':'wrong'}
    elif failure=='missing_evidence':c['qualification_evidence']=c['qualification_evidence'][:1]
    elif failure=='evidence_mutated':(args['evidence_root']/'full200_ddp.json').write_text('{"passed":false}')
    elif failure=='unapproved_source':c['changed_existing_dependencies']=[]
    elif failure=='archive_mutated':args['new_archive'].write_bytes(b'wrong')
    elif failure=='existing_destination':args['destination'].write_bytes(b'preserve')
    torch.save(state,args['source']);args['contract_path'].write_text(json.dumps(c))
    with pytest.raises(ValueError):m.migrate(**args)
    if failure=='existing_destination':assert args['destination'].read_bytes()==b'preserve'
    else:assert not args['destination'].exists()
