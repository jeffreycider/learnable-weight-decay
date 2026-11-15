import torch
from scripts.fit_cpu_teacher_spans import projection_error,vector_projection_error


def test_thin_projection_loss_and_raw_frame_gradients_match_dense():
    torch.manual_seed(45);dtype=torch.float64
    ur=torch.randn(7,3,dtype=dtype,requires_grad=True);vr=torch.randn(6,3,dtype=dtype,requires_grad=True)
    left=torch.randn(7,2,dtype=dtype);right=torch.randn(6,2,dtype=dtype)
    u=torch.linalg.qr(ur).Q;v=torch.linalg.qr(vr).Q
    thin=projection_error(u,v,left,right);target=left@right.T
    dense=(target-u@(u.T@target@v)@v.T).square().sum()
    torch.testing.assert_close(thin,dense,atol=1e-12,rtol=1e-12)
    a=torch.autograd.grad(thin,(ur,vr),retain_graph=True);b=torch.autograd.grad(dense,(ur,vr))
    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-11)


def test_exact_spans_zero_targets_and_vector_gradient():
    torch.manual_seed(46);dtype=torch.float64
    u=torch.linalg.qr(torch.randn(7,3,dtype=dtype)).Q;v=torch.linalg.qr(torch.randn(6,3,dtype=dtype)).Q
    assert float(projection_error(u,v,u[:,:2],v[:,:2]))<1e-27
    assert float(projection_error(u,v,torch.zeros(7,2,dtype=dtype),v[:,:2]))==0.
    raw=torch.randn(7,3,dtype=dtype,requires_grad=True);q=torch.linalg.qr(raw).Q;t=torch.randn(7,dtype=dtype)
    thin=vector_projection_error(q,t);dense=((torch.eye(7,dtype=dtype)-q@q.T)@t).square().sum()
    torch.testing.assert_close(thin,dense,atol=1e-12,rtol=1e-12)
    a=torch.autograd.grad(thin,raw,retain_graph=True)[0];b=torch.autograd.grad(dense,raw)[0]
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)


def test_projection_optimizer_polish_retains_moments_and_step():
    from scripts.fit_cpu_teacher_spans import restore_projection_adam
    import copy
    x=torch.nn.Parameter(torch.ones(2,dtype=torch.float64));opt=torch.optim.Adam([x],lr=1e-3,betas=(.9,.99),eps=1e-12)
    x.square().sum().backward();opt.step();saved=copy.deepcopy(opt.state_dict())
    parent=dict(diagnostic_control=True,metadata={'diagnostic_control':'private_span_projection_only'},projection_optimizer=saved,control_updates=1,projection_parameter_names=['frame'])
    other=torch.optim.Adam([x],lr=.1)
    assert restore_projection_adam(other,parent,{'frame':None})==[1]
    other.param_groups[0]['lr']=1e-4
    for key,value in saved['state'][0].items():torch.testing.assert_close(other.state_dict()['state'][0][key],value,rtol=0,atol=0)
    assert other.param_groups[0]['betas']==(.9,.99) and other.param_groups[0]['eps']==1e-12


def test_continuation_schedule_and_inclusive_target_stop_before_update():
    from scripts.fit_cpu_teacher_spans import continuation_schedule,scheduled_rate,reached_training_target
    import pytest
    schedule=continuation_schedule(['500:1e-6','500:1e-7'],1000)
    assert [scheduled_rate(schedule,i) for i in (0,499,500,999,1000)]==[1e-6,1e-6,1e-7,1e-7,1e-7]
    with pytest.raises(ValueError):continuation_schedule(['500:1e-6'],1000)
    assert reached_training_target(1e-22,1e-22,1e-22)
    assert not reached_training_target(1e-22,1.00001e-22,1e-22)
    assert not reached_training_target(1e-30,1e-30,None)
    # The observed stop row must refer to current parameters, with no next step.
    x=torch.nn.Parameter(torch.tensor(1e-11,dtype=torch.float64));opt=torch.optim.Adam([x],lr=1e-6)
    before=x.detach().clone();loss=x.square()
    stop=reached_training_target(float(loss),float(loss),1.000001e-22)
    if not stop:loss.backward();opt.step()
    assert stop and not opt.state and torch.equal(x,before)
