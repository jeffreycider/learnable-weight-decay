import pytest
import torch
from rewriter.persistent_offline import PersistentOfflineTrainer


def controls():
    # Control updates are pure tensor copies and can be checked without CUDA.
    program=object.__new__(PersistentOfflineTrainer)
    program.objective_mode=1
    program.primal_dual_config=dict(kl_budget=1e-6,kl_multiplier=.2,augmented_rho=.01)
    program.kl_budget_buffer=torch.tensor(1e-6,dtype=torch.float64)
    program.objective_controls=torch.tensor([0.,.2,.01,1.],dtype=torch.float64)
    return program


def test_primal_dual_rebind_keeps_device_storage_and_zero_multiplier():
    program=controls();pointers=(program.kl_budget_buffer.data_ptr(),program.objective_controls.data_ptr())
    program.load_primal_dual(kl_multiplier=0.,kl_budget=2e-6,augmented_rho=.03)
    assert pointers==(program.kl_budget_buffer.data_ptr(),program.objective_controls.data_ptr())
    assert program.objective_controls.tolist()==[0.,0.,.03,1.]
    assert program.kl_budget_buffer.item()==2e-6


@pytest.mark.parametrize('kwargs',[dict(kl_multiplier=-1.),dict(kl_multiplier=float('nan')),
    dict(kl_multiplier=.1,kl_budget=0.),dict(kl_multiplier=.1,augmented_rho=float('inf'))])
def test_invalid_rebind_is_atomic(kwargs):
    program=controls();previous=program.objective_controls.clone()
    with pytest.raises(ValueError):program.load_primal_dual(**kwargs)
    torch.testing.assert_close(program.objective_controls,previous,rtol=0,atol=0)
    assert program.kl_budget_buffer.item()==1e-6


def test_checkpoint_cannot_silently_reuse_other_states_multiplier():
    program=controls()
    with pytest.raises(ValueError,match='requires its'):
        program.load_checkpoint(None,None)
