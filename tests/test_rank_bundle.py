import torch
import pytest
from scripts.bundle_baseline_checkpoint import bundle_checkpoints,extract_rank,_record_equal
from scripts.merge_baseline_checkpoint import _equal


def test_rank_bundle_exactly_recovers_each_rng_cursor_and_optimizer():
    from tests.test_merge_baseline_checkpoint import MergeBaselineTests
    fixture=MergeBaselineTests(methodName='runTest');fixture.setUp()
    try:
        import numpy as np
        for rank in range(8):fixture.mutate(rank,lambda state:state['rng'].update(numpy=np.random.get_state()))
        output=fixture.root/'bundle.pt';metadata=bundle_checkpoints(fixture.paths[::-1],output)
        bundle=torch.load(output,weights_only=False)
        assert metadata['resumable_world_size']==8 and metadata['step']==25
        assert len(bundle['shared']['optimizers'][1]['state'])==6
        for rank,path in enumerate(fixture.paths):
            actual=extract_rank(bundle,rank)
            expected=torch.load(path,weights_only=False)
            _record_equal(actual,expected,'roundtrip')
            assert actual['rng']['cpu'].item()==rank
        assert output.stat().st_size<sum(p.stat().st_size for p in fixture.paths)/3
        with pytest.raises(ValueError):extract_rank(bundle,8)
        del bundle['ranks'][7]
        with pytest.raises(ValueError,match='lacks'):extract_rank(bundle,0)
    finally:fixture.doCleanups()
