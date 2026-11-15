import torch
from chisel.activation_records import ActivationRecords


def test_samples_keep_unprojected_vectors_and_adjoints_across_microbatches():
    model=torch.nn.Sequential(torch.nn.Linear(4,3,bias=False),torch.nn.Linear(3,2,bias=False))
    positions=torch.tensor([0,2,5])
    recorder=ActivationRecords(model,r'[01]',positions,1);recorder.begin(True)
    x=torch.randn(6,4);reference=[]
    for chunk in x.split(2):
        hidden=model[0](chunk);hidden.retain_grad()
        y=model[1](hidden);y.square().sum().backward()
        reference.append((hidden.detach().clone(),hidden.grad.clone()))
    result=recorder.snapshot(0,1)
    torch.testing.assert_close(result['modules']['0']['input'],x[positions])
    torch.testing.assert_close(result['modules']['0']['output'],torch.cat([v[0] for v in reference])[positions])
    torch.testing.assert_close(result['modules']['0']['adjoint'],torch.cat([v[1] for v in reference])[positions])
    torch.testing.assert_close(result['modules']['1']['input'],result['modules']['0']['output'])
    recorder.begin(False)
    assert result['modules']['0']['input'].abs().sum()>0
    recorder.close()


def test_full_capture_has_every_position():
    model=torch.nn.Sequential(torch.nn.Linear(4,3))
    recorder=ActivationRecords(model,'0',torch.arange(6),2);recorder.begin(True)
    x=torch.randn(3,2,4);model(x).sum().backward()
    result=recorder.snapshot(0,1)
    torch.testing.assert_close(result['modules']['0']['input'],x.flatten(0,1))
    assert result['sample_ids'].tolist()==[[i,j] for i in range(3) for j in range(2)]
    recorder.close()
