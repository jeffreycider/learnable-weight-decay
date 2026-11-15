import copy
import torch
from scripts.qualify_coordinate_relative import optimizer_comparison


def test_saved_optimizer_gate_distinguishes_keys_clocks_and_values():
    p=torch.nn.Parameter(torch.tensor([1.,2.]));opt=torch.optim.Adam([p],lr=1e-4,eps=1e-12)
    p.square().sum().backward();opt.step();reference=copy.deepcopy(opt.state_dict())
    assert all(optimizer_comparison(reference,reference)[k] for k in ('state_keys_equal','step_clocks_equal','param_groups_equal'))
    changed=copy.deepcopy(reference);changed['state'][0]['step']+=1
    row=optimizer_comparison(changed,reference)
    assert row['state_keys_equal'] and not row['step_clocks_equal'] and row['moment_relative']==0
    changed=copy.deepcopy(reference);changed['state'][0]['exp_avg']*=2
    assert optimizer_comparison(changed,reference)['moment_relative']==1
    changed=copy.deepcopy(reference);changed['state'][1]=copy.deepcopy(changed['state'][0])
    row=optimizer_comparison(changed,reference)
    assert not row['state_keys_equal'] and row['candidate_only_keys']==[1]
