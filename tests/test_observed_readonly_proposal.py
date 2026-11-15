import torch
from types import SimpleNamespace
from dataclasses import replace
from tests.test_prefix_cache import toy_record
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.transformer_chisel import TransformerChisel
from scripts.train_multistate_chisel import build_proposal


def test_readonly_observations_remain_visible_but_need_no_editable_scale():
    torch.set_num_threads(1);torch.manual_seed(43)
    snapshot,_=toy_record();w=torch.randn(7,5);b=torch.randn(7)
    matrices=dict(snapshot.matrices);first=matrices['first']
    matrices['first']=replace(first,weight_probes=(w@first.anchors.B,w.T@first.anchors.A))
    vectors=dict(snapshot.vectors);vectors['bias']=replace(vectors['bias'],weight=b)
    snapshot=replace(snapshot,matrices=matrices,vectors=vectors)
    # 'last' is observed, like the fixed CIFAR whitening matrix, but its stored
    # value belongs to the engine's immutable model rather than editable map.
    params={'first':w,'bias':b}
    engine=SimpleNamespace(parameters=params)
    def edited(action,scale):
        assert torch.count_nonzero(action.matrices['last'].P)==0
        assert torch.count_nonzero(action.matrices['last'].Q)==0
        return {'first':w+scale*action.matrices['first'].dense(),'bias':b+scale*action.vectors['bias']}
    engine.edited_parameters=edited
    grouped=GroupedNormObjective(params,groups={'matrix':('first',),'bias':('bias',)})
    actor=TransformerChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,patch_size=4)
    with torch.no_grad():
        for head in actor.output_heads:head.weight.normal_(std=.01)
    result=build_proposal(actor,snapshot,engine,grouped,radius_fraction=.01)
    assert set(result[0])==set(params)
    assert set(snapshot.matrices)=={'first','last'}
    assert torch.isfinite(result[1])
