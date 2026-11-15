import copy
from types import SimpleNamespace
import pytest
import torch
from scripts.evaluate_cpu_joint_unseen_tokens import validate_endpoint,select_unseen_inputs,validate_seed_identity
from scripts.diagnose_cpu_norm_preservation import tensor_sha


def fixture():
    cases=[dict(state_id=i,sequence_ids=[i*10,i*10+1]) for i in (6,7)]
    checkpoint=dict(completed_updates=20,metadata={'cases':cases},dual={'updates':20})
    rows=[dict(local_id=c['state_id'],context_ids=c['sequence_ids'],strict_topk_changes=0,norm_objective=-1e-6,mean_group_l2_percentage_reduction=1e-6,ranking_k=100) for c in cases]
    result=dict(completed_updates=20,final=dict(completed_updates_before_measure=20,updates_after=20,learning=False,gradient_profile=False,rows=rows))
    return checkpoint,result


def test_endpoint_requires_matching_completed_preserving_shrinking_cases():
    checkpoint,result=fixture();assert set(validate_endpoint(checkpoint,result))=={6,7}
    for key,value in [('strict_topk_changes',1),('norm_objective',0.),('mean_group_l2_percentage_reduction',0.),('ranking_k',10),('context_ids',[99])]:
        wrong=copy.deepcopy(result);wrong['final']['rows'][0][key]=value
        with pytest.raises(ValueError):validate_endpoint(checkpoint,wrong)
    wrong=copy.deepcopy(result);wrong['final']['learning']=True
    with pytest.raises(ValueError,match='endpoint'):validate_endpoint(checkpoint,wrong)


def test_unseen_slice_rejects_overlap_and_changed_fitted_inputs():
    inputs=torch.arange(24).reshape(6,4);panel=dict(inputs=inputs,metadata={'sequence_ids':[10,11,12,13,14,15]})
    case=dict(sequence_ids=[10,11],input_sha256=tensor_sha(inputs[:2]))
    selected,ids=select_unseen_inputs(panel,case,2,2)
    assert ids==[12,13] and torch.equal(selected,inputs[2:4])
    with pytest.raises(ValueError,match='overlap'):select_unseen_inputs(panel,case,1,2)
    wrong=dict(case,input_sha256='wrong')
    with pytest.raises(ValueError,match='hash'):select_unseen_inputs(panel,wrong,2,2)
    with pytest.raises(ValueError,match='slice'):select_unseen_inputs(panel,case,5,2)


def test_mocked_shared_and_private_identity_contract():
    case=dict(sha256={'state':'state'},private_identity={'uid':7});meta=dict(state_sha256='state',architecture={'kind':'grouped'})
    actor=SimpleNamespace(configuration=lambda:{'kind':'grouped'});private=SimpleNamespace(get_extra_state=lambda:{'uid':7})
    validate_seed_identity(case,meta,actor,private)
    with pytest.raises(ValueError,match='private'):validate_seed_identity(dict(case,private_identity={'uid':6}),meta,actor,private)
    with pytest.raises(ValueError,match='architecture'):validate_seed_identity(case,dict(meta,architecture={'kind':'old'}),actor,private)


def test_reserved_six_sequences_exclude_entire_noncontiguous_training_pool():
    inputs=torch.arange(128*4).reshape(128,4)
    ids=list(range(1000,1128))
    positions=list(range(8))+list(range(10,122))
    panel=dict(inputs=inputs,metadata=dict(sequence_ids=ids))
    case=dict(sequence_ids=[ids[i] for i in positions],input_sha256=tensor_sha(inputs[positions]))
    selected,evaluated=select_unseen_inputs(panel,case,122,6)
    assert len(evaluated)==6 and set(evaluated).isdisjoint(case['sequence_ids'])
    assert torch.equal(selected,inputs[122:128])
    with pytest.raises(ValueError,match='overlap'):select_unseen_inputs(panel,case,121,6)
    checkpoint,result=fixture()
    for c,r in zip(checkpoint['metadata']['cases'],result['final']['rows']):
        c['sequence_ids']=case['sequence_ids'];r['context_ids']=case['sequence_ids']
    assert set(validate_endpoint(checkpoint,result))=={6,7}
    result['final']['rows'][0]['context_ids']=case['sequence_ids'][:8]
    with pytest.raises(ValueError,match='token IDs'):validate_endpoint(checkpoint,result)
