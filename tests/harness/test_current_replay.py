import json
import pytest
import torch
from chisel.harness import run
from chisel.checkpoint_io import CheckpointReader
from chisel.current_replay import CurrentCheckpointReader
from chisel.offline_data import Replay
from tests.harness.test_integration import source_config


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('buffered',[False,True])
def test_selective_current_replay_matches_full_state_without_mutation(tmp_path, device,buffered):
    run(source_config(tmp_path))
    entry=json.loads((tmp_path/'source/trajectory/000003/state.ready.json').read_text())
    reference=Replay(device,CheckpointReader(cache_bytes=1<<28),phase='current',current_state_only=False).load(entry)
    reader=CurrentCheckpointReader(cache_bytes=1<<28,buffered=buffered)
    replay=Replay(device,reader,phase='current')
    current=replay.load(entry)
    cached=reader.read(entry)
    saved={k:v.clone() for k,v in cached['optimizer']['hidden']['state'][0].items() if torch.is_tensor(v)}
    repeated=replay.load(entry)
    for i in (1,2,3,4,5):
        torch.testing.assert_close(current[i],reference[i],rtol=0,atol=0)
        torch.testing.assert_close(repeated[i],reference[i],rtol=0,atol=0)
    for k,v in saved.items():
        torch.testing.assert_close(cached['optimizer']['hidden']['state'][0][k],v,rtol=0,atol=0)
    assert cached['optimizer'].keys()=={'hidden'}
    assert all(n.startswith('blocks.') and n.endswith('.weight') for n in cached['gradients'])
    full=CheckpointReader(cache_bytes=1<<28);full.read(entry)
    assert reader.resident_bytes < full.resident_bytes


def test_projected_records_cannot_be_used_for_next_optimizer_proposal():
    with pytest.raises(ValueError,match='full base step'):
        Replay('cpu',CurrentCheckpointReader(),phase='next')
