import json,struct
import numpy as np
import torch
from tests.test_multistate_chisel import _tiny_state_bank
from scripts.train_multistate_chisel import main


def test_explicit_same_bank_transition_resets_optimization_and_preserves_actor(tmp_path):
    bank=tmp_path/'bank';_tiny_state_bank(bank)
    data=tmp_path/'fineweb_train_000.bin';tokens=(np.arange(1000)%16).astype(np.uint16)
    data.write_bytes(struct.pack('<iii',20240520,1,len(tokens))+bytes(1012)+tokens.tobytes())
    common=['--state-bank',str(bank),'--train-data',str(data),'--expected-gpt-seed','7',
        '--epochs','1','--checkpoint-batch','4','--sequences-per-checkpoint','2','--gpt-microbatch','1',
        '--replay-pool-sequences','32','--audit-states','1','--audit-sequences','1',
        '--eager-chisel','--device','cpu','--no-plots','--chisel-model-dim','128','--chisel-layers','1']
    old=tmp_path/'old';new=tmp_path/'new'
    main([*common,'--output',str(old),'--objective','legacy-deadband'])
    path=old/'controller_latest.pt';before=torch.load(path,weights_only=False)
    main([*common,'--output',str(new),'--objective','primal-dual','--primal-init',str(path),'--initial-audit-only'])
    after=torch.load(new/'controller_latest.pt',weights_only=False)
    for n,value in before['controller'].items():torch.testing.assert_close(value,after['controller'][n],rtol=0,atol=0)
    assert before['optimizer']['state'] and not after['optimizer']['state']
    assert after['updates']==after['primal_dual']['updates']==0
    assert set(after['primal_dual']['multipliers'].values())=={0.}
    assert after['metadata']['objective_transition']['source_updates']==1
    rows=[json.loads(line) for line in (new/'metrics.jsonl').read_text().splitlines()]
    audit=[r for r in rows if r['event']=='audit_state']
    assert audit and all(not r['initial_zero_edit_floor'] for r in audit)
    assert all(r['trust_term']==0 and not r['edit_size_penalty_enabled'] for r in audit)


def test_dual_ascent_rejects_heldouts_duplicates_and_preserves_state_on_failure():
    import pytest
    from rewriter.primal_dual import PrimalDual,feasibility_score
    dual=PrimalDual([1,2],tolerance=.1,rho=.2,dual_lr=.3)
    dual.step([(1,.2),(2,.05)],expected_ids=[1,2])
    assert dual.multipliers[1]==pytest.approx(.3) and dual.multipliers[2]==0
    saved=dual.state_dict()
    for samples,expected in [([(1,.3),(1,.3)],[1,2]), ([(5,.3)],[5])]:
        with pytest.raises(ValueError):dual.step(samples,expected_ids=expected)
        assert dual.state_dict()==saved
    dual.step([(1,0)],expected_ids=[1]);assert dual.multipliers[1]==0
    with pytest.raises(FloatingPointError):feasibility_score([dict(mean_kl=float('nan'),mean_group_l2_percentage_reduction=2)],.1)
    with pytest.raises(ValueError):feasibility_score([dict(mean_kl=0,mean_group_l2_percentage_reduction=2)],0)
    assert feasibility_score([dict(mean_kl=.05,mean_group_l2_percentage_reduction=2)],.1)==-2
    assert feasibility_score([dict(mean_kl=.2,mean_group_l2_percentage_reduction=99)],.1)>1


def test_transition_fit_can_resume_without_reinitializing_actor_or_duals(tmp_path):
    bank=tmp_path/'bank';_tiny_state_bank(bank)
    data=tmp_path/'fineweb_train_000.bin';tokens=(np.arange(1000)%16).astype(np.uint16)
    data.write_bytes(struct.pack('<iii',20240520,1,len(tokens))+bytes(1012)+tokens.tobytes())
    common=['--state-bank',str(bank),'--train-data',str(data),'--expected-gpt-seed','7',
        '--checkpoint-batch','4','--sequences-per-checkpoint','2','--gpt-microbatch','1',
        '--replay-pool-sequences','32','--audit-states','0','--audit-sequences','1',
        '--eager-chisel','--device','cpu','--no-plots','--chisel-model-dim','128','--chisel-layers','1',
        '--convergence','--minimum-epochs','1','--audit-every-epochs','1']
    old=tmp_path/'old';new=tmp_path/'new';direct=tmp_path/'direct'
    main([*common,'--output',str(old),'--epochs','1','--objective','legacy-deadband'])
    initial=old/'controller_latest.pt'
    main([*common,'--output',str(new),'--epochs','1','--primal-init',str(initial)])
    first=torch.load(new/'controller_latest.pt',weights_only=False)
    main([*common,'--output',str(new),'--epochs','2','--resume',str(new/'controller_latest.pt')])
    main([*common,'--output',str(direct),'--epochs','2','--primal-init',str(initial)])
    resumed=torch.load(new/'controller_latest.pt',weights_only=False);full=torch.load(direct/'controller_latest.pt',weights_only=False)
    assert resumed['primal_dual']==full['primal_dual']
    assert resumed['metadata']['objective_transition']==first['metadata']['objective_transition']
    for n,value in full['controller'].items():torch.testing.assert_close(resumed['controller'][n],value,rtol=0,atol=0)
    for i,state in full['optimizer']['state'].items():
        for k,v in state.items():torch.testing.assert_close(resumed['optimizer']['state'][i][k],v,rtol=0,atol=0)
