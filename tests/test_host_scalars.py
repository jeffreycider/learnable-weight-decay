import math
import struct
import pytest
import torch
from rewriter.host_scalars import scalar_dict_to_host


def bits(value):return struct.pack('d',value)


def test_mixed_scalar_types_order_and_signed_zero():
    values={'negative_zero':torch.tensor(-0.),'double':torch.tensor(1.+2**-40,dtype=torch.float64),
            'single':torch.tensor(1.3),'integer':torch.tensor(2**60+19),'one':torch.tensor([1.],dtype=torch.bfloat16),
            'inf':torch.tensor(float('inf')),'nan':torch.tensor(float('nan'))}
    actual=scalar_dict_to_host(values)
    assert list(actual)==list(values)
    for key,value in values.items():
        if key=='nan':assert math.isnan(actual[key])
        else:assert bits(actual[key])==bits(float(value))
    assert scalar_dict_to_host({})=={}
    with pytest.raises(ValueError):scalar_dict_to_host({'wrong':torch.ones(2)})
    with pytest.raises(ValueError):scalar_dict_to_host({'wrong':torch.tensor(1j)})


def test_observer_gradient_norm_and_metadata_values_exact():
    torch.manual_seed(9)
    gradients={f'p{i}':torch.randn(i+1,7) for i in range(20)}
    # Production computes oriented FP32 g / batch_tokens before g.norm().
    values={name:(g.float()/524288).norm() for name,g in gradients.items()}
    expected={name:float(v) for name,v in values.items()}
    actual=scalar_dict_to_host(values)
    assert all(bits(actual[n])==bits(expected[n]) for n in values)
    assert all(bits(math.log(max(actual[n],1e-12)))==bits(math.log(max(expected[n],1e-12))) for n in values)
    # Metadata's FP32 sqrt/clamp/log is read back without changing its math.
    logs={n:g.square().sum().sqrt().clamp_min(1e-12).log() for n,g in gradients.items()}
    assert all(bits(v)==bits(float(logs[n])) for n,v in scalar_dict_to_host(logs).items())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA observer readback qualification')
def test_cuda_scalars_match_sequential_host_reads():
    torch.manual_seed(11)
    values={f'p{i}':(torch.randn(i+1,7,device='cuda')/524288).norm() for i in range(173)}
    actual=scalar_dict_to_host(values)
    assert all(bits(actual[n])==bits(float(v)) for n,v in values.items())


def test_two_step_observer_snapshot_and_gpt_equivalence():
    """Exercise the proposed three-line caller change without editing live source."""
    import copy
    from dataclasses import fields,is_dataclass
    from baseline.model import GPT
    from rewriter.state_capture import DistributedStateObserver
    class Candidate(DistributedStateObserver):
        @torch.no_grad()
        def after_backward(self,t,model):
            if not self.enabled:return
            self.sensors=self.collector.finalize(distributed=self.distributed)
            self.gradient={};self.gradient_norm={};self.before={}
            if self.rank_id==0:
                for n,p in self.params.items():
                    g=self.oriented(n,p.grad).float()/self.config['batch_tokens']
                    self.gradient_norm[n]=g.norm()  # defer the scalar read
                    self.gradient[n]=((g@self.anchors[n].B,g.T@self.anchors[n].A) if p.ndim==2 else g.clone())
                    self.before[n]=p.detach().clone()
                self.gradient_norm=scalar_dict_to_host(self.gradient_norm)
    def identical(a,b):
        if isinstance(a,torch.Tensor):
            assert a.dtype==b.dtype and a.shape==b.shape
            assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
        elif is_dataclass(a):
            for field in fields(a):identical(getattr(a,field.name),getattr(b,field.name))
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for name in a:identical(a[name],b[name])
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b)
            for x,y in zip(a,b):identical(x,y)
        else:assert a==b
    torch.set_num_threads(1);torch.manual_seed(39)
    a=GPT(vocab_size=16,num_layers=1,model_dim=128).float();b=copy.deepcopy(a)
    oa=torch.optim.AdamW(a.parameters(),lr=.001);ob=torch.optim.AdamW(b.parameters(),lr=.001)
    config=dict(batch_tokens=12,seq_len=3,max_steps=2,schedule_steps=3250)
    sensors=dict(probe_rank=2,anchor_refresh_period=1,trajectory_positions=4,
                 train_sequence_probes=0,audit_sequence_probes=0)
    reference=DistributedStateObserver(a,[oa],config,sensor_config=sensors)
    candidate=Candidate(b,[ob],config,sensor_config=sensors)
    inputs=torch.arange(12).reshape(4,3)%16;targets=(inputs+1)%16
    for step in range(2):
        for model,opt,observer in ((a,oa,reference),(b,ob,candidate)):
            observer.before_step(step,model,[opt]);observer.before_microbatch(step,torch.arange(4),inputs)
            model(inputs,targets).backward();observer.after_backward(step,model);opt.step()
        identical(reference.gradient_norm,candidate.gradient_norm)
        identical(reference.finish_snapshot(step),candidate.finish_snapshot(step))
        identical(a.state_dict(),b.state_dict());identical(oa.state_dict(),ob.state_dict())
        a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
