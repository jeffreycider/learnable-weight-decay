"""Diagnostics are optional observers, never inputs to learned decisions."""
import copy
import io
import torch
from baseline.model import GPT, initialize_parameters_like_record
from rewriter.edit_diagnostics import EditDiagnostics, parameter_family
from rewriter.integration import RewriterAdapter


def config(enabled):
    return dict(model=dict(num_layers=1),batch_tokens=12,seq_len=3,max_steps=3,schedule_steps=3,
        rewriter=dict(probe_rank=2,coordinate_width=8,global_width=16,global_latents=4,
                      global_blocks=1,attention_heads=2,ff_width=32,pool_queries=2,
                      auxiliary_steps=1,warmup_steps=0,ramp_steps=1,anchor_refresh_period=1,
                      trajectory_positions=6,diagnostics=enabled))


def make_model():
    torch.set_num_threads(1)
    torch.manual_seed(77)
    model=GPT(32,1,128).float()
    initialize_parameters_like_record(model)
    return model


def batches():
    g=torch.Generator().manual_seed(80)
    return [(torch.randint(32,(4,3),generator=g),torch.randint(32,(4,3),generator=g)) for _ in range(3)]


def train(t,model,opt,adapter,batch):
    x,y=batch
    adapter.before_step(t,model,[opt])
    for i in (0,2):
        adapter.before_microbatch(t,torch.arange(i,i+2),x[i:i+2])
        loss=model(x[i:i+2],y[i:i+2]);loss.backward()
    adapter.after_backward(t,model)
    pre_base={n:p.detach().float().clone() for n,p in model.named_parameters()}
    opt.step()
    base={n:p.detach().float().clone() for n,p in model.named_parameters()}
    adapter.after_step(t,model,[opt]);opt.zero_grad(set_to_none=True)
    return pre_base,base


def test_measured_norm_change_matches_actual_fp64_and_families():
    model=make_model()
    # Include BF16 storage rounding, which separates realized from proposed edits.
    model.embed.bfloat16()
    opt=torch.optim.AdamW(model.parameters(),lr=.002)
    adapter=RewriterAdapter(model,[opt],config(True))
    _,base=train(0,model,opt,adapter,batches()[0])
    expected={}
    for name,p in model.named_parameters():
        family=parameter_family(name,p)
        change=p.detach().double().square().sum()-base[name].double().square().sum()
        expected[family]=expected.get(family,0.)+float(change)
    assert abs(adapter.metrics['realized_squared_norm_change']-sum(expected.values())) < 1e-9
    for family,value in expected.items():
        assert abs(adapter.metrics[f'realized_squared_norm_change_{family}']-value) < 1e-9
    forward_sq=sum((p.detach().to(torch.bfloat16).double()-base[n].to(torch.bfloat16).double()).square().sum()
                   for n,p in model.named_parameters())
    assert adapter.metrics['forward_cast_dtype']=='torch.bfloat16'
    assert abs(adapter.metrics['forward_bfloat16_edit_norm']-float(forward_sq.sqrt())) < 1e-12
    assert adapter.metrics['diagnostics_extra_resident_bytes']==4*sum(p.numel() for p in model.parameters())
    adapter.collector.close()


def test_reversal_manual_and_zero_edit():
    p=torch.nn.Parameter(torch.tensor([[1.,2.],[3.,4.]]))
    observer=EditDiagnostics({'weight':p},'cpu',chunk_elements=2)
    edit=torch.tensor([[.25,-.5],[.125,.25]])
    observer.begin_commit();observer.prepare_parameter('weight',p)
    observer.record_chunk('weight',0,p.detach(),p.detach()+edit,edit);observer.finish_commit()
    observer.begin_reversal();observer.observe_base_update('weight',-edit)
    result=observer.finish_reversal()
    assert result['previous_edit_reversal_fraction']==1.
    assert result['base_update_previous_edit_cosine']==-1.
    assert result['base_update_previous_edit_dot']==-float(edit.double().square().sum())
    observer.begin_commit();observer.prepare_parameter('weight',p)
    observer.record_chunk('weight',0,p.detach(),p.detach(),torch.zeros_like(p))
    assert observer.finish_commit()['realized_squared_norm_change']==0.
    observer.begin_reversal();observer.observe_base_update('weight',torch.ones_like(p))
    result=observer.finish_reversal()
    assert result['previous_edit_measured']
    assert result['previous_edit_reversal_fraction']==result['base_update_previous_edit_cosine']==0.


def test_diagnostics_preserve_trajectory_and_terminal_measures_prior_action():
    off=make_model();on=copy.deepcopy(off)
    opt_off=torch.optim.AdamW(off.parameters(),lr=.002)
    opt_on=torch.optim.AdamW(on.parameters(),lr=.002)
    a=RewriterAdapter(off,[opt_off],config(False));b=RewriterAdapter(on,[opt_on],config(True))
    previous=None
    rng=torch.get_rng_state().clone()
    for t,batch in enumerate(batches()):
        train(t,off,opt_off,a,batch)
        pre,base=train(t,on,opt_on,b,batch)
        for p,q in zip(off.parameters(),on.parameters()):
            torch.testing.assert_close(p,q,atol=0,rtol=0)
        for p,q in zip(a.controller.parameters(),b.controller.parameters()):
            torch.testing.assert_close(p,q,atol=0,rtol=0)
        torch.testing.assert_close(a.online.optimizer.state_dict(),b.online.optimizer.state_dict(),atol=0,rtol=0)
        torch.testing.assert_close(opt_off.state_dict(),opt_on.state_dict(),atol=0,rtol=0)
        if previous is not None:
            dot=sum(((base[n]-pre[n]).double()*previous[n].double()).sum() for n in base)
            norm=sum(x.double().square().sum() for x in previous.values())
            assert abs(b.metrics['base_update_previous_edit_dot']-float(dot)) < 1e-12
            assert abs(b.metrics['previous_edit_reversal_fraction']-float(-dot/norm)) < 1e-9
            assert b.metrics['previous_edit_measured']
        previous={n:p.detach().float()-base[n] for n,p in on.named_parameters()}
    assert b.metrics['terminal_identity']
    assert b.metrics['realized_squared_norm_change']==0.
    assert not b.diagnostics.pending
    assert torch.equal(rng,torch.get_rng_state())
    a.collector.close();b.collector.close()


def test_checkpoint_retains_actual_edit_for_next_base_reversal():
    model=make_model();opt=torch.optim.AdamW(model.parameters(),lr=.002)
    cfg=config(True);adapter=RewriterAdapter(model,[opt],cfg)
    data=batches();train(0,model,opt,adapter,data[0])
    buffer=io.BytesIO();torch.save(dict(model=model.state_dict(),opt=opt.state_dict(),adapter=adapter.state_dict()),buffer)
    buffer.seek(0);saved=torch.load(buffer,weights_only=False)
    resumed=make_model();resumed.load_state_dict(saved['model'])
    opt2=torch.optim.AdamW(resumed.parameters(),lr=.002);opt2.load_state_dict(saved['opt'])
    adapter2=RewriterAdapter(resumed,[opt2],cfg);adapter2.load_state_dict(saved['adapter'])
    assert adapter2.diagnostics.pending
    for name,edit in adapter.diagnostics.previous_edit.items():
        torch.testing.assert_close(edit,adapter2.diagnostics.previous_edit[name],rtol=0,atol=0)
    train(1,model,opt,adapter,data[1]);train(1,resumed,opt2,adapter2,data[1])
    for key in ('base_update_previous_edit_dot','previous_edit_reversal_fraction','realized_squared_norm_change'):
        assert adapter.metrics[key]==adapter2.metrics[key]
    for p,q in zip(model.parameters(),resumed.parameters()):
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    adapter.collector.close();adapter2.collector.close()


def test_small_stored_edit_can_be_invisible_after_forward_cast():
    parameter=torch.nn.Parameter(torch.ones(2))
    observer=EditDiagnostics({'bias':parameter},'cpu',forward_dtype=torch.bfloat16)
    before=parameter.detach().clone()
    after=before+1e-4
    actual=after-before
    observer.begin_commit();observer.prepare_parameter('bias',parameter)
    observer.record_chunk('bias',0,before,after,actual)
    metrics=observer.finish_commit()
    assert metrics['realized_squared_norm_change']>0
    assert metrics['forward_bfloat16_edit_norm']==0.
    assert float(observer.previous_norm_sq)>0
