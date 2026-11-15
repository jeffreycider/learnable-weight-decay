import copy
import torch
from scripts.qualify_gpu_factor_panel import compare


def test_role_gates_reject_gradient_delta_and_exact_loss_changes_without_cuda():
    seeds={'actor':{'matrix_output.h.weight':torch.zeros(2,dtype=torch.float64),'body.weight':torch.zeros(2,dtype=torch.float64)},'private':{'frame':torch.zeros(2,dtype=torch.float64)}}
    values={role:{n:torch.ones_like(t) for n,t in tensors.items()} for role,tensors in seeds.items()}
    reference=dict(loss={'objective':1.},compute={'student_forwards':2},gradients=copy.deepcopy(values),post=copy.deepcopy(values))
    assert compare(reference,copy.deepcopy(reference),seeds)['passed']
    wrong=copy.deepcopy(reference);wrong['gradients']['private']['frame']*=1.0002
    assert not compare(reference,wrong,seeds)['passed']
    wrong=copy.deepcopy(reference);wrong['post']['actor']['body.weight']*=1.002
    assert not compare(reference,wrong,seeds)['passed']
    wrong=copy.deepcopy(reference);wrong['loss']['objective']+=1e-10
    assert not compare(reference,wrong,seeds)['passed']
    wrong=copy.deepcopy(reference);wrong['gradients']['actor']['matrix_output.h.weight']=None
    assert not compare(reference,wrong,seeds)['passed']
