import json
from chisel.capture import TrajectoryWriter
from chisel.catalog import build_catalog


def test_catalog_requires_all_activation_shards_and_excludes_terminal(tmp_path):
    writer=TrajectoryWriter(tmp_path/'run/trajectory')
    writer.submit(0,dict(step=0,activation_shards=1,has_next_training_step=True))
    writer.submit(1,dict(step=1,activation_shards=0,has_next_training_step=False));writer.close()
    cfg=dict(output=str(tmp_path/'catalog.yaml'),datasets=[dict(root=str(tmp_path),receipts='run/trajectory/**/state.ready.json',domain='cifar')])
    assert build_catalog(cfg)==[]
    activations=TrajectoryWriter(tmp_path/'run/trajectory','activation_rank0000.pt')
    activations.submit(0,dict(sample='vectors'));activations.close()
    rows=build_catalog(cfg)
    assert len(rows)==1 and rows[0]['step']==0
    assert rows[0]['path']=='run/trajectory/000000/state.pt'
