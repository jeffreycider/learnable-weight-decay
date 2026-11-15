import pytest
import torch
from scripts.evaluate_adaptive_gauge_precision import reconstruct_cpu


@pytest.mark.parametrize('power',[1,2])
def test_cpu_precision_controls_preserve_binary_definition_and_input_storage(power):
    torch.manual_seed(89);torch.set_num_threads(1)
    model={'first':torch.randn(12,10),'second':torch.randn(9,12),'bias':torch.randn(12)}
    before={n:p.clone() for n,p in model.items()}
    rows=[dict(rank_limit=8,inverse_power=power,selected=[
        dict(first_weight='first',second_weight='second',first_bias='bias',channel=i,exponent=e)
        for i,e in [(1,-1),(9,1)]])]
    got=reconstruct_cpu(model,rows)
    d=torch.ones(12);d[1]=.5;d[9]=2
    expected={'first':model['first']*d[:,None],'second':model['second']/d[None,:].pow(power),'bias':model['bias']*d}
    for n,p in expected.items():
        torch.testing.assert_close(got['direct_binary'][n],p,rtol=0,atol=0)
        torch.testing.assert_close(got['adaptive_fp64_cast'][n],p,rtol=0,atol=0)
        torch.testing.assert_close(got['adaptive_fp32'][n],p,rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(model[n],before[n],rtol=0,atol=0)
