import copy
from unittest.mock import patch
import pytest
import torch
from scripts.frozen_private_frames import FrozenPrivateFrames
from rewriter.adaptive_sensing import query_weight_cores
from tests.test_adaptive_qr_reuse import case


@pytest.mark.parametrize('precision',['fp32','fp64'])
def test_two_shared_steps_exact_outputs_gradients_and_one_qr_per_phase(precision):
    s,a,p,w=case(precision)
    p.requires_grad_(False)
    b=copy.deepcopy(a)
    opts=[torch.optim.Adam(m.parameters(),lr=1e-5) for m in (a,b)]
    original_key={n:(id(t),t._version,t.clone()) for n,t in p.named_parameters()}
    with patch('torch.linalg.qr',wraps=torch.linalg.qr) as qr:
        phase=FrozenPrivateFrames(p)
        assert qr.call_count==2*len(p.matrix_names)+len(p.vector_names)
        for step in range(2):
            actions=[]; grads=[]
            for i,actor in enumerate((a,b)):
                opts[i].zero_grad(set_to_none=True)
                if i==0:
                    action=actor(s,p,weight_cores=query_weight_cores(w,p,embedding_names=()))
                else:
                    before=qr.call_count
                    cores=phase.query_weight_cores(w,embedding_names=())
                    action=phase.evaluate(actor,s,weight_cores=cores)
                    assert qr.call_count==before
                    assert all(not f.U.requires_grad and f.U.grad_fn is None for f in action.matrices.values())
                dense=action.dense()
                loss=sum(t.square().sum() for t in dense.values())
                loss.backward()
                actions.append(dense)
                grads.append({n:None if v.grad is None else v.grad.clone() for n,v in actor.named_parameters()})
            for n in actions[0]:torch.testing.assert_close(actions[0][n],actions[1][n],rtol=0,atol=0)
            for n,g in grads[0].items():
                h=grads[1][n];assert (g is None)==(h is None)
                if g is not None:torch.testing.assert_close(g,h,rtol=0,atol=0)
            assert a.weight_core_input.weight.grad.norm()>0
            for opt in opts:opt.step()
        view=phase.view
        phase.close()
        with pytest.raises(RuntimeError,match='closed'):view.matrix_rows[0].orthonormal_matrix()
    for n,t in p.named_parameters():
        old_id,old_version,old_value=original_key[n]
        assert id(t)==old_id and t._version==old_version and t.grad is None
        torch.testing.assert_close(t,old_value,rtol=0,atol=0)
    assert all('orthonormal_matrix' not in f.__dict__ for f in p.matrix_rows)


@pytest.mark.parametrize('mutation',['raw','trajectory','replace_raw','replace_trajectory','replace_frame','unfreeze','cached_q','cached_q_unfreeze','view_raw','view_trajectory'])
def test_mutations_fail_closed(mutation):
    s,a,p,w=case('fp64');p.requires_grad_(False)
    phase=FrozenPrivateFrames(p);view=phase.view
    with torch.no_grad():
        if mutation=='raw':p.matrix_rows[0].raw.add_(.01)
        elif mutation=='trajectory':p.trajectory_input.weight.add_(.01)
        elif mutation=='replace_raw':p.matrix_rows[0].raw=torch.nn.Parameter(p.matrix_rows[0].raw.clone(),requires_grad=False)
        elif mutation=='replace_trajectory':p.trajectory_input.weight=torch.nn.Parameter(p.trajectory_input.weight.clone(),requires_grad=False)
        elif mutation=='replace_frame':p.matrix_rows[0]=copy.deepcopy(p.matrix_rows[0])
        elif mutation=='unfreeze':p.trajectory_input.weight.requires_grad_(True)
        elif mutation=='cached_q':view.matrix_rows[0].orthonormal_matrix().add_(.01)
        elif mutation=='cached_q_unfreeze':view.matrix_rows[0].orthonormal_matrix().requires_grad_(True)
        elif mutation=='view_raw':view.matrix_rows[0].raw=torch.nn.Parameter(view.matrix_rows[0].raw.clone(),requires_grad=False)
        elif mutation=='view_trajectory':view.trajectory_input.weight=torch.nn.Parameter(view.trajectory_input.weight.clone(),requires_grad=False)
    with pytest.raises(RuntimeError,match='changed'):phase.evaluate(a,s)
    with pytest.raises(RuntimeError,match='closed'):phase.validate()


def test_requires_all_frozen_exception_cleanup_nested_and_new_phase():
    s,a,p,w=case('fp32')
    with pytest.raises(ValueError,match='all private'):FrozenPrivateFrames(p)
    p.requires_grad_(False)
    with FrozenPrivateFrames(p) as outer:
        with FrozenPrivateFrames(p) as inner:
            assert outer.view is not inner.view
            torch.testing.assert_close(outer.view.matrix_rows[0].orthonormal_matrix(),inner.view.matrix_rows[0].orthonormal_matrix(),rtol=0,atol=0)
        outer.validate()
        def fail(*args,**kw):raise ValueError('failed actor')
        with pytest.raises(ValueError,match='failed actor'):outer.evaluate(fail,s)
        assert outer._active==0
        outer.validate()
    with pytest.raises(RuntimeError,match='closed'):outer.validate()
    with torch.no_grad():p.matrix_rows[0].raw.add_(.001)
    with FrozenPrivateFrames(p) as fresh:fresh.validate()
