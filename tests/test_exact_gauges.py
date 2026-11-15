import copy
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from baseline.model import MLP, CausalSelfAttention
from rewriter.exact_gauges import (
    GaugeSpec, bounded_optimal_scales, baseline_gauge_specs, norm_optimal_gauge_control,
)


def randomize(module, seed=81):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, dtype=p.dtype, generator=gen))
    return module


@pytest.mark.parametrize('kind', ['relu_squared_mlp', 'attention_value'])
def test_fp64_function_preservation_and_norm_nonincrease_with_bias(kind):
    module = randomize(MLP(8).double() if kind == 'relu_squared_mlp' else
                       CausalSelfAttention(8, head_dim=4).double())
    old = {k:v.detach().clone() for k,v in module.state_dict().items()}
    specs = baseline_gauge_specs(module)
    control = norm_optimal_gauge_control(module.state_dict(), specs)
    transformed = copy.deepcopy(module)
    transformed.load_state_dict({**module.state_dict(), **control.replacements})
    for seed in range(3):
        x = torch.randn(2, 7+seed, 8, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
        torch.testing.assert_close(transformed(x), module(x), rtol=5e-13, atol=3e-12)
    for k,v in module.state_dict().items():
        torch.testing.assert_close(v, old[k], rtol=0, atol=0)
    before = sum(float(v.detach().square().sum()) for v in module.parameters())
    after = sum(float(v.detach().square().sum()) for v in transformed.parameters())
    assert after <= before + 1e-10
    assert after-before == pytest.approx(control.diagnostics['norm_sq_change'], abs=2e-11)
    torch.testing.assert_close(transformed.proj.bias, module.proj.bias, rtol=0, atol=0)
    if kind == 'attention_value':
        for name in old:
            if name.startswith(('q.', 'k.', 'rotary.')):
                torch.testing.assert_close(transformed.state_dict()[name], old[name], rtol=0, atol=0)
    assert not any(v.requires_grad for v in control.replacements.values())
    assert control.diagnostics['diagnostic_only']


@pytest.mark.parametrize('power', [1,2])
def test_closed_form_matches_stationarity_and_bounded_grid(power):
    u = torch.tensor([.01,1.,3.,100.,0.,2.,0.],dtype=torch.float64)
    v = torch.tensor([100.,2.,2.,.01,3.,0.,0.],dtype=torch.float64)
    d = bounded_optimal_scales(u,v,inverse_power=power)
    expected = (power*v[:4]/u[:4]).pow(1/(2*power+2)).clamp(.5,2.)
    torch.testing.assert_close(d[:4], expected)
    torch.testing.assert_close(d[4:],torch.tensor([2.,.5,1.],dtype=d.dtype))
    grid = torch.linspace(.5,2.,1001,dtype=d.dtype)
    energy = u*d.square()+v/d.pow(2*power)
    assert torch.all(energy <= (u[:,None]*grid.square()+v[:,None]/grid.pow(2*power)).min(dim=1).values+1e-12)
    interior=(d>.5)&(d<2)&(u>0)
    torch.testing.assert_close((2*u*d-2*power*v/d.pow(2*power+1))[interior],torch.zeros_like(d[interior]),atol=1e-13,rtol=0)


def test_bias_energy_changes_optimum_and_zero_branches_are_bounded():
    params={'up':torch.zeros(3,2,dtype=torch.float64),
            'bias':torch.tensor([4.,0.,0.],dtype=torch.float64),
            'down':torch.tensor([[1.,2.,0.]],dtype=torch.float64)}
    out=norm_optimal_gauge_control(params,[GaugeSpec('relu_squared_mlp','up','down','bias')])
    torch.testing.assert_close(out.scales['up'], torch.tensor([(.125)**(1/6),2.,1.],dtype=torch.float64))
    assert out.diagnostics['norm_sq_change']<0


def test_all_arbitrary_nested_layers_compose_without_touching_other_weights():
    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.arbitrary = nn.ModuleDict({'one': MLP(8), 'two': CausalSelfAttention(8,4)})
            self.other = nn.Linear(8,8)
        def forward(self,x):
            return self.other(x+self.arbitrary['one'](x)+self.arbitrary['two'](x))
    model=randomize(Stack().double())
    specs=baseline_gauge_specs(model)
    assert len(specs)==2
    assert all(s.first_weight.startswith('arbitrary.') for s in specs)
    control=norm_optimal_gauge_control(model.state_dict(),specs)
    assert not any(k.startswith('other.') for k in control.replacements)
    edited=copy.deepcopy(model)
    edited.load_state_dict({**model.state_dict(),**control.replacements})
    x=torch.randn(2,5,8,dtype=torch.float64)
    torch.testing.assert_close(model(x),edited(x),rtol=2e-12,atol=2e-11)


def test_explicit_names_allow_biasless_non_square_pair():
    g=torch.Generator().manual_seed(95)
    p={'a':torch.randn(7,3,generator=g,dtype=torch.float64),
       'b':torch.randn(2,7,generator=g,dtype=torch.float64)}
    out=norm_optimal_gauge_control(p,[GaugeSpec('relu_squared_mlp','a','b')])
    x=torch.randn(11,3,generator=g,dtype=torch.float64)
    old=F.linear(F.linear(x,p['a']).relu().square(),p['b'])
    new=F.linear(F.linear(x,out.replacements['a']).relu().square(),out.replacements['b'])
    torch.testing.assert_close(new,old,rtol=1e-13,atol=1e-12)


def test_bfloat16_rounding_is_reported_without_exact_machine_claim():
    model=randomize(MLP(8).bfloat16())
    out=norm_optimal_gauge_control(model.state_dict(),baseline_gauge_specs(model))
    new=copy.deepcopy(model);new.load_state_dict({**model.state_dict(),**out.replacements})
    x=torch.randn(3,5,8,generator=torch.Generator().manual_seed(14)).bfloat16()
    assert not torch.equal(model(x),new(x))
    assert all(v.dtype==torch.bfloat16 for v in out.replacements.values())
    assert out.diagnostics['exact_function_claim']=='real arithmetic for the stated operators only'
    assert out.diagnostics['norm_sq_after'] != out.diagnostics['norm_sq_before']+out.diagnostics['ideal_norm_sq_change']


@pytest.mark.parametrize('kwargs', [dict(lower=0),dict(lower=1.1),dict(upper=.9),dict(upper=float('inf')),dict(inverse_power=3)])
def test_invalid_bound_or_power_rejected(kwargs):
    settings=dict(inverse_power=2);settings.update(kwargs)
    with pytest.raises(ValueError):bounded_optimal_scales(torch.ones(2),torch.ones(2),**settings)


@pytest.mark.parametrize('case',['alias','overlap','shape','heads','nan'])
def test_invalid_or_ambiguous_model_mapping_rejected(case):
    p={'a':torch.ones(4,3), 'b':torch.ones(2,4)}
    specs=[GaugeSpec('attention_value','a','b',num_heads=2)]
    if case=='alias':p['elsewhere']=p['a']
    if case=='overlap':specs=specs*2
    if case=='shape':p['b']=torch.ones(2,3)
    if case=='heads':specs=[GaugeSpec('attention_value','a','b',num_heads=3)]
    if case=='nan':p['a'][0,0]=float('nan')
    with pytest.raises(ValueError):norm_optimal_gauge_control(p,specs)


@pytest.mark.parametrize('kind,power',[('relu_squared_mlp',2),('attention_value',1)])
def test_discrete_grid_matches_per_coordinate_minimum_and_contains_identity(kind,power):
    gen=torch.Generator().manual_seed(98)
    params={'a':torch.randn(8,3,generator=gen,dtype=torch.float64),
            'b':torch.randn(2,8,generator=gen,dtype=torch.float64),
            'bias':torch.randn(8,generator=gen,dtype=torch.float64)}
    params['a'][0]=0;params['b'][:,0]=0;params['bias'][0]=0
    specs=[GaugeSpec(kind,'a','b','bias',num_heads=2)]
    discrete=norm_optimal_gauge_control(params,specs,scale_grid=(.5,1.,2.))
    continuous=norm_optimal_gauge_control(params,specs)
    u=params['a'].square().sum(1)+params['bias'].square();v=params['b'].square().sum(0)
    d=discrete.scales['a'];grid=torch.tensor([.5,1.,2.],dtype=d.dtype)
    expected=(u[:,None]*grid.square()+v[:,None]/grid.pow(2*power)).min(1).values
    torch.testing.assert_close(u*d.square()+v/d.pow(2*power),expected)
    assert set(d.tolist()) <= {.5,1.,2.} and d[0]==1
    assert continuous.diagnostics['ideal_norm_sq_change'] <= discrete.diagnostics['ideal_norm_sq_change']+1e-12
    assert discrete.diagnostics['norm_sq_change'] <= 0
    assert discrete.diagnostics['scale_policy']=='discrete_grid_minimum'


@pytest.mark.parametrize('kind',['mlp','attention'])
def test_fixed_bfloat16_fixture_grid_reduces_measured_drift(kind):
    # Fixture comparison only; this does not assert all-input machine invariance.
    model=randomize((MLP(8) if kind=='mlp' else CausalSelfAttention(8,4)).bfloat16())
    x=torch.randn(3,5,8,generator=torch.Generator().manual_seed(14)).bfloat16()
    original=model(x).detach().float()
    errors=[]
    for grid in (None,(.5,1.,2.)):
        out=norm_optimal_gauge_control(model.state_dict(),baseline_gauge_specs(model),scale_grid=grid)
        new=copy.deepcopy(model);new.load_state_dict({**model.state_dict(),**out.replacements})
        errors.append(float((new(x).detach().float()-original).norm()))
    assert errors[0]>0 and errors[1]<=errors[0]


@pytest.mark.parametrize('grid',[(),(.5,2.),(0.,1.),(1.,3.),(1.,float('nan'))])
def test_invalid_discrete_grids_rejected(grid):
    with pytest.raises(ValueError,match='scale_grid'):
        norm_optimal_gauge_control({},[],scale_grid=grid)


@pytest.mark.parametrize('value_scale',[2.**-.5,.5])
def test_uniform_attention_group_control_preserves_function_and_group_objective(value_scale):
    from baseline.model import GPT
    from rewriter.exact_gauges import uniform_attention_group_control
    from rewriter.grouped_norm import GroupedNormObjective
    torch.manual_seed(972)
    model=GPT(32,2,128).double();parameters=dict(model.named_parameters())
    control=uniform_attention_group_control(parameters,model,value_scale=value_scale)
    result=GroupedNormObjective(parameters).evaluate({**parameters,**control.replacements})
    assert result.metrics()['mean_group_l2_percentage_reduction']==pytest.approx(
        control.diagnostics['expected_mean_group_l2_percentage_reduction'],abs=1e-12)
    assert control.diagnostics['outside_anchored_action_family'] and not control.diagnostics['deployable']
    changed=copy.deepcopy(model);changed.load_state_dict({**model.state_dict(),**control.replacements})
    inputs=torch.randint(32,(2,5))
    def logits(m):
        x=m.norm1(m.embed(inputs))
        for block in m.blocks:x=block(x)
        return m.proj(m.norm2(x))
    torch.testing.assert_close(logits(model),logits(changed),rtol=1e-12,atol=1e-12)
    assert all('.attn.v.' in n or n.endswith('.attn.proj.weight') for n in control.replacements)
    assert result.metrics()['group_l2_percentage_reduction']['attention.out.weight']<0
    bad=dict(parameters);bad['blocks.0.attn.v.bias']=torch.zeros_like(bad['blocks.0.attn.v.bias'])
    bad['blocks.1.attn.v.bias']=torch.zeros_like(bad['blocks.1.attn.v.bias'])
    with pytest.raises(ValueError,match='nonzero'):uniform_attention_group_control(bad,model)
    aliased=dict(parameters,alias=parameters['blocks.0.attn.v.weight'])
    # Unrecognized editable names or aliases must fail, not claim invariance.
    with pytest.raises(ValueError):uniform_attention_group_control(aliased,model)
