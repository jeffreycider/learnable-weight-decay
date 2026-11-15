import random
import pytest
import torch
from rewriter.coordinate_fitting_panels import context_slice,split_fitting_panels


def context(domain,total):
    inputs=torch.arange(total)[:,None]
    return dict(domain=domain,inputs=inputs,ids=list(range(total)),target=torch.arange(total),
                teacher=lambda a,b:inputs[a:b]+10000,engine=object(),snapshot=object())


@pytest.mark.parametrize('domain,total,inner,joint',[('gpt',128,16,64),('cifar10',2048,1024,1024)])
def test_disjoint_views_aligned_teacher_and_no_global_rng(domain,total,inner,joint):
    source=context(domain,total);rng=random.getstate()
    support,shared=split_fitting_panels(source,inner_steps=8,inner_size=inner,joint_size=joint,seed=42)
    assert random.getstate()==rng
    assert len(support)==8
    for selected in support+[shared]:
        assert selected['engine'] is source['engine'] and selected['snapshot'] is source['snapshot']
        assert selected['inputs'].flatten().tolist()==selected['ids']==selected['target'].tolist()
        assert torch.equal(selected['teacher'](0,len(selected['ids'])),selected['inputs']+10000)
        assert selected['inputs'].untyped_storage().data_ptr()==source['inputs'].untyped_storage().data_ptr()
    assert all(not set(s['ids'])&set(shared['ids']) for s in support)
    support2,shared2=split_fitting_panels(source,inner_steps=8,inner_size=inner,joint_size=joint,seed=42)
    assert [s['ids'] for s in support]==[s['ids'] for s in support2] and shared['ids']==shared2['ids']
    assert source['ids']==list(range(total))


def test_nested_slice_translates_teacher_once_per_slice():
    source=context('gpt',128)
    selected=context_slice(context_slice(source,16,64),16,16)
    assert selected['source_panel_offset']==32
    assert torch.equal(selected['teacher'](0,16),source['inputs'][32:48]+10000)
    with pytest.raises(ValueError,match='outside'):selected['teacher'](0,17)


def test_disjoint_split_rejects_insufficient_data_and_duplicate_ids():
    source=context('gpt',64)
    with pytest.raises(ValueError,match='not enough'):
        split_fitting_panels(source,inner_steps=8,inner_size=16,joint_size=64,seed=1)
    source=context('gpt',128);source['ids'][1]=source['ids'][0]
    with pytest.raises(ValueError,match='unique'):
        split_fitting_panels(source,inner_steps=8,inner_size=16,joint_size=64,seed=1)
