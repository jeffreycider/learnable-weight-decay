import copy
import pytest
import torch

from baseline.optim import Muon,muon_update
from rewriter.qualification_metrics import muon_snapshot,compare_muon


def saved_shard(rank,world=2):
    named={'small':torch.nn.Parameter(torch.ones(2,2)),
           'wide':torch.nn.Parameter(torch.ones(2,5)),
           'tall':torch.nn.Parameter(torch.ones(5,2))}
    optimizer=Muon(list(named.values()),lr=.02,weight_decay=.05,mu=.95)
    for index,p in enumerate(optimizer.param_groups[0]['params']):
        if index%world==rank:optimizer.state[p]['momentum']=torch.full_like(p,.2+index)
    return named,optimizer


@pytest.mark.parametrize('rank',[0,1])
def test_actual_muon_sorted_shard_snapshot_is_detached_and_read_only(rank):
    named,opt=saved_shard(rank)
    original_count=len(opt.state)
    snapshot=muon_snapshot(opt,named,rank=rank,world_size=2)
    assert len(opt.state)==original_count  # no defaultdict entries for nonowners
    assert snapshot['groups'][0]['parameters']==['tall','wide','small']
    assert snapshot['groups'][0]['owned_parameters']==['tall','wide','small'][rank::2]
    assert compare_muon(snapshot,copy.deepcopy(snapshot))['global_metric']['bit_identical']
    for state in opt.state.values():state['momentum'].zero_()
    assert all(torch.count_nonzero(v)==v.numel() for v in snapshot['momentum'].values())


def test_muon_comparison_rejects_wrong_shard_order_hyperparams_or_missing_saved_state():
    named,opt=saved_shard(0);reference=muon_snapshot(opt,named,rank=0,world_size=2)
    for edit in ('rank','order','lr','missing'):
        candidate=copy.deepcopy(reference)
        if edit=='rank':candidate['rank']=1
        elif edit=='order':candidate['groups'][0]['parameters'].reverse()
        elif edit=='lr':candidate['groups'][0]['hyperparameters']['lr']*=2
        else:candidate['momentum'].pop('tall')
        with pytest.raises(ValueError):compare_muon(reference,candidate)
    with pytest.raises(ValueError,match='another rank|saved Muon'):
        muon_snapshot(opt,named,rank=1,world_size=2)
    opt.state.clear()
    with pytest.raises(ValueError,match='saved Muon'):
        muon_snapshot(opt,named,rank=0,world_size=2)


def test_muon_reports_changed_and_nonfinite_momentum_without_hiding_small_shards():
    named,opt=saved_shard(0);a=muon_snapshot(opt,named,rank=0,world_size=2);b=copy.deepcopy(a)
    b['momentum']['small'][0,0]+=.01
    result=compare_muon(a,b)
    assert result['global_metric']['relative_error']>0
    assert not result['per_parameter']['small']['bit_identical']
    assert result['per_parameter']['tall']['bit_identical']
    b['momentum']['small'][0,0]=float('nan')
    assert not compare_muon(a,b)['global_metric']['finite']


def test_baseline_muon_consumes_gradient_in_place_so_capture_must_precede_step():
    gradient=torch.tensor([[1.,.2],[.3,-.7]])
    momentum=torch.tensor([[.1,.4],[-.2,.6]])
    raw=gradient.clone();saved=momentum.clone()
    # Actual baseline equations; bypass CPU compiler only, no formula replica.
    with torch.no_grad():
        update=muon_update._torchdynamo_orig_callable(gradient,momentum,mu=.95)
    assert not torch.equal(gradient,raw)
    assert not torch.equal(momentum,saved)
    assert torch.isfinite(update).all()
    assert torch.equal(raw,torch.tensor([[1.,.2],[.3,-.7]]))
