import os
os.environ.setdefault('PROFILE_CHISEL_SKIP_SOURCE_GUARD', '1')
import dataclasses
import pytest
import torch
from rewriter.graph_stager import GraphCheckpointStager, PinnedLeafBuilder, StagerError
from rewriter.static_snapshot_store import _leaves, layout_signature


def _fixtures(tmp_path, gpt_precision='baseline'):
    from tests.chisel_fixtures import tiny_bank_and_cache   # production imports only (owner isolated archives)
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import TeacherCacheReader
    bank, cache_dir = tiny_bank_and_cache(tmp_path, gpt_precision=gpt_precision)
    reader = StateBankReader(bank, verify_once=True, mmap_model=True)
    cache = TeacherCacheReader(cache_dir, verify_once=True, mmap=True)
    fresh = StateBankReader(bank, verify_once=True, mmap_model=True)
    return reader, cache, fresh


def test_staged_record_matches_fresh_read_bitwise_with_exact_layout_and_passes_verification(tmp_path):
    reader, cache, fresh = _fixtures(tmp_path)
    stager = GraphCheckpointStager(reader, cache, pin=False)
    states = [1, 2, 3, 1]
    stager.schedule(states)
    try:
        for state_id in states:
            record, panel, seconds = stager.take(state_id)
            eager = fresh.read(state_id, device='cpu', model_device='cpu')
            # the owner's load_engine calls assert_verified: file stamp + tensor identity of the copies
            assert record.assert_verified() == eager.assert_verified()
            for name, value in eager.model.items():
                assert torch.equal(record.model[name], value) and record.model[name].data_ptr() != value.data_ptr(), name
            for (name, held), (_, want) in zip(_leaves(record.snapshot, record.train), _leaves(eager.snapshot, eager.train)):
                assert torch.equal(held, want), name
                assert held.stride() == want.stride() and held.shape == want.shape and held.dtype == want.dtype, name
                assert held.storage_offset() >= 0
            assert layout_signature(record.snapshot, record.train) == layout_signature(eager.snapshot, eager.train)
            want_panel = cache.read(state_id, 'train')
            assert torch.equal(panel.inputs, want_panel.inputs) and torch.equal(panel.features, want_panel.features)
            assert panel.metadata['state_sha256'] == record.assert_verified()
            assert {'bank_read', 'model_copy', 'observation_build', 'panel_stage', 'wait_for_worker'} <= set(seconds)
        assert stager.consumed == 4 and stager.failed is None
    finally:
        stager.close()


def _synthetic():
    """Tiny snapshot with the production layout kinds the bank fixture lacks:
    column-major anchors, stride-0 broadcast rows/trajectories, strided holes."""
    from rewriter.controller import ControllerSnapshot, MatrixObservation, VectorObservation
    from rewriter.algebra import Anchors, Measurements
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    column_major = base.t().contiguous().t()                 # stride (1, 4)
    broadcast = torch.arange(6, dtype=torch.float32).reshape(1, 6).expand(4, 6)   # stride (0, 1)
    holes = base[:, ::2]                                     # shape (4,3) stride (6, 2): footprint 1+3*6+2*2 = 23 elements
    matrices = {'m': MatrixObservation(Anchors(column_major, holes), broadcast, base, (base.clone(),))}
    vectors = {'v': VectorObservation(torch.arange(5.), torch.ones(5))}
    snapshot = ControllerSnapshot(matrices, vectors, broadcast.clone(), torch.tensor(3.), torch.tensor(0.03))
    train = Measurements({'m': (holes,)}, {'v': torch.zeros(5)})
    return snapshot, train, holes


def test_builder_preserves_stride_zero_broadcasts_column_major_and_holes(monkeypatch):
    # dirty allocator memory: the builder must never rely on fresh zeros (Astra review)
    real_empty = torch.empty
    monkeypatch.setattr(torch, 'empty', lambda *a, **k: real_empty(*a, **k).fill_(17.) if k.get('dtype', torch.float32).is_floating_point else real_empty(*a, **k))
    builder = PinnedLeafBuilder(pin=False)
    snapshot, train, holes = _synthetic()
    built, arenas = builder.build(snapshot, train)
    for name, want in _leaves(snapshot, train):
        held = built[name]
        assert torch.equal(held, want) and held.stride() == want.stride(), name
    # broadcast rows keep stride-0 storage: unique row only
    assert built['m.rows'].stride() == (0, 1) and built['m.rows'].untyped_storage().nbytes() >= 6 * 4
    # holes: gaps INSIDE the 23-element footprint read as zero; element 23 onwards is alignment padding
    # outside every leaf's footprint and is deliberately not inspected (Astra: previous test read 24)
    leaf = built['m.anchors.B']
    footprint = leaf.as_strided((23,), (1,))
    addressed = torch.zeros(23, dtype=torch.bool)
    for r in range(4):
        for c in range(3):
            addressed[r * 6 + c * 2] = True
    assert torch.equal(footprint[~addressed], torch.zeros(int((~addressed).sum())))
    assert torch.equal(footprint[addressed].reshape(4, 3), holes)
    # whole arenas are zeroed explicitly, so alignment padding is defined too (dirty allocator above)
    for dtype in arenas:
        arena = arenas[dtype]
        covered = torch.zeros(arena.numel(), dtype=torch.bool)
        for name, want in _leaves(snapshot, train):
            t = built[name]
            if t.dtype != dtype:
                continue
            start = (t.data_ptr() - arena.data_ptr()) // t.element_size()
            covered[start:start + (1 + sum((d - 1) * s for d, s in zip(t.shape, t.stride())) if t.numel() else 0)] = True
        assert torch.equal(arena[~covered], torch.zeros(int((~covered).sum()), dtype=dtype)), dtype
    # every leaf of a dtype shares one arena storage
    ptrs = {built[n].untyped_storage().data_ptr() for n, t in _leaves(snapshot, train) if t.dtype == torch.float32}
    assert len(ptrs) == 1


def test_layout_change_fails_closed_until_reset(tmp_path):
    reader, cache, fresh = _fixtures(tmp_path)
    stager = GraphCheckpointStager(reader, cache, pin=False)
    good = layout_signature(*[getattr(fresh.read(1, device='cpu', model_device='cpu'), k) for k in ('snapshot', 'train')])
    # a different expected signature rejects the very first state
    stager.signature = tuple(list(good)[:-1] + [('radius', (2,), (1,), torch.float32)])
    stager.schedule([1, 2])
    try:
        with pytest.raises(StagerError, match='layout'):
            stager.take(1)
        assert stager.failed is not None
        with pytest.raises(StagerError, match='reset'):
            stager.take(2)
        with pytest.raises(StagerError, match='reset'):
            stager.schedule([2])
        stager.reset()
        stager.signature = good
        stager.schedule([2])
        record, panel, _ = stager.take(2)
        assert record.completed_steps == 2 and stager.failed is None
    finally:
        stager.close()


def test_order_mismatch_and_worker_error_fail_closed(tmp_path):
    reader, cache, fresh = _fixtures(tmp_path)
    stager = GraphCheckpointStager(reader, cache, pin=False)
    stager.schedule([1, 2])
    try:
        with pytest.raises(StagerError, match='order mismatch'):
            stager.take(2)
        assert stager.failed is not None
        stager.reset()
        stager.schedule([99])                 # outside the committed bank: reader raises in the worker
        with pytest.raises(StagerError, match='staging state 99 failed'):
            stager.take(99)
        assert stager.failed is not None
    finally:
        stager.close()


def test_verification_loss_during_staging_is_detected(tmp_path):
    reader, cache, fresh = _fixtures(tmp_path)
    stager = GraphCheckpointStager(reader, cache, pin=False)
    stager.schedule([1])
    try:
        item = stager._queue.get()
        item.record.model[next(iter(item.record.model))].add_(0.)   # bump _version of a pinned copy -> identity mismatch
        stager._queue.put(item)
        with pytest.raises(StagerError, match='verification'):
            stager.take(1)
    finally:
        stager.close()


def test_scalar_leaves_can_be_placed_on_a_device_without_changing_values(tmp_path):
    reader, cache, fresh = _fixtures(tmp_path)
    stager = GraphCheckpointStager(reader, cache, pin=False, scalar_leaves_device='cpu')
    stager.schedule([3])
    try:
        record, _, _ = stager.take(3)
        eager = fresh.read(3, device='cpu', model_device='cpu')
        assert torch.equal(record.snapshot.weight_norm, eager.snapshot.weight_norm)
        assert torch.equal(record.snapshot.radius, eager.snapshot.radius)
        assert record.snapshot.radius.stride() == eager.snapshot.radius.stride()
    finally:
        stager.close()


def test_schedule_rejects_overlapping_sequences_and_close_verifies_worker_exit(tmp_path):
    reader, cache, fresh = _fixtures(tmp_path)
    stager = GraphCheckpointStager(reader, cache, pin=False)
    stager.schedule([1, 2, 3])
    with pytest.raises(StagerError, match='still active'):
        stager.schedule([2])                   # one staged/in flight + two not issued
    record, _, _ = stager.take(1)
    with pytest.raises(StagerError, match='still active'):
        stager.schedule([3])                   # state 2 issued, state 3 pending
    stager.take(2); stager.take(3)
    stager.schedule([1, 1])                    # repeats in order are legitimate (1/997/998/999 then 1 style loops)
    assert stager.take(1)[0].completed_steps == 1 and stager.take(1)[0].completed_steps == 1
    stager.schedule([2])
    with pytest.raises(StagerError, match='order mismatch'):   # failed take has already issued nothing further here
        stager.take(3)
    stager.close()                             # drains the staged state 2 and joins
    assert not stager._thread.is_alive() and stager._outstanding == 0


def test_device_record_rebuilds_reference_layout_for_capture(tmp_path):
    from rewriter.graph_stager import device_record
    reader, cache, fresh = _fixtures(tmp_path)
    stager = GraphCheckpointStager(reader, cache, pin=False)
    stager.schedule([2])
    try:
        host, _, _ = stager.take(2)
        dev = device_record(host, 'cpu')           # CPU stands in for the device here; CUDA path below
        eager = fresh.read(2, device='cpu', model_device='cpu')
        assert dev.assert_verified() == eager.assert_verified()
        for (name, held), (_, want) in zip(_leaves(dev.snapshot, dev.train), _leaves(eager.snapshot, eager.train)):
            assert torch.equal(held, want) and held.stride() == want.stride(), name
            assert held.untyped_storage().data_ptr() != _leaves(host.snapshot, host.train)[0][1].untyped_storage().data_ptr() or name != 'trajectories'
        assert dev.model is host.model             # model tensors untouched
    finally:
        stager.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='pinned memory needs CUDA')
def test_cuda_pinned_leaves_feed_device_copies(tmp_path):
    from rewriter.graph_stager import device_record
    reader, cache, fresh = _fixtures(tmp_path)
    stager = GraphCheckpointStager(reader, cache)
    stager.schedule([1, 2])
    try:
        for state_id in (1, 2):
            record, panel, _ = stager.take(state_id)
            for name, held in _leaves(record.snapshot, record.train):
                assert held.is_pinned(), name
            assert all(v.is_pinned() for v in record.model.values()) and panel.features.is_pinned()
            eager = fresh.read(state_id, device='cpu', model_device='cpu')
            # capture-style device rebuild (unique-axis copies; stride-0 destinations reject full copy_)
            dev = device_record(record, 'cuda', non_blocking=True)
            torch.cuda.synchronize()
            for (name, held), (_, want) in zip(_leaves(dev.snapshot, dev.train), _leaves(eager.snapshot, eager.train)):
                assert held.is_cuda and held.stride() == want.stride() and torch.equal(held.cpu(), want), name
            # rebind-style: the owner's load_checkpoint copies host leaves into its own device buffers (default blocking)
            for (name, held), (_, want) in zip(_leaves(record.snapshot, record.train), _leaves(eager.snapshot, eager.train)):
                owned = torch.zeros_like(want, device='cuda')
                owned.copy_(held)
                assert torch.equal(owned.cpu(), want), name
    finally:
        stager.close()
    # the bank fixture has no stride-0 snapshot leaves (Astra): exercise broadcast / column-major / holes explicitly
    from rewriter.graph_stager import _unique_slice
    snapshot, train, _ = _synthetic()
    host, _ = PinnedLeafBuilder(pin=True).build(snapshot, train)
    dev, _ = PinnedLeafBuilder(False, 'cuda').build(snapshot, train, non_blocking=True)
    torch.cuda.synchronize()
    for name, want in _leaves(snapshot, train):
        assert host[name].is_pinned() and host[name].stride() == want.stride(), name
        assert dev[name].is_cuda and dev[name].stride() == want.stride() and torch.equal(dev[name].cpu(), want), name
        # owner-style rebind into an independently owned device buffer of the same layout: unique-axis copy only
        owned = torch.zeros(1 + sum((d - 1) * s for d, s in zip(want.shape, want.stride())) if want.numel() else 0,
                            dtype=want.dtype, device='cuda').as_strided(want.shape, want.stride())
        _unique_slice(owned).copy_(_unique_slice(host[name]), non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(owned.cpu(), want), name


def test_copy_arenas_refills_captured_views_with_one_copy_per_dtype(tmp_path):
    """Owner follow-up (05:59): ~900 blocking per-leaf copies per bind -> one arena copy per dtype.
    The device arena (CPU stands in) built from record A keeps its views (the 'captured'
    addresses); copying record B's staged host arenas into it must make every view equal B."""
    from rewriter.graph_stager import device_record, copy_arenas
    reader, cache, fresh = _fixtures(tmp_path)
    stager = GraphCheckpointStager(reader, cache, pin=False)
    stager.schedule([1, 2, 3])
    try:
        first, _, _, host_first = stager.take(1, return_arenas=True)
        captured, dev_arenas = device_record(first, 'cpu', return_arenas=True)
        views = dict(_leaves(captured.snapshot, captured.train))
        addresses = {n: (t.data_ptr(), t.stride()) for n, t in views.items()}
        assert set(dev_arenas) == set(host_first) and all(dev_arenas[d].numel() == host_first[d].numel() for d in dev_arenas)
        for state_id in (2, 3):
            record, _, _, host = stager.take(state_id, return_arenas=True)
            copy_arenas(dev_arenas, host)
            eager = fresh.read(state_id, device='cpu', model_device='cpu')
            for (name, want) in _leaves(eager.snapshot, eager.train):
                held = views[name]
                assert torch.equal(held, want) and held.stride() == want.stride(), name
                assert (held.data_ptr(), held.stride()) == addresses[name], name   # captured addresses unchanged
        # ALL metadata is validated before ANY copy (Astra: a valid first dtype must not be written
        # when a later dtype is invalid): same signature but one arena the wrong size -> nothing moves
        from rewriter.graph_stager import ArenaSet
        before = {d: dev_arenas[d].clone() for d in dev_arenas}
        wrong = dict(host.arenas)
        last = sorted(wrong, key=str)[-1]
        wrong[last] = torch.full((host.numel(last) + 1,), 7., dtype=last)
        for d in wrong:
            if d != last:
                wrong[d] = torch.full_like(wrong[d], 7.)      # would be visible if copied first
        with pytest.raises(ValueError, match='mismatch'):
            copy_arenas(dev_arenas, ArenaSet(wrong, host.signature, host.device))
        assert all(torch.equal(dev_arenas[d], before[d]) for d in dev_arenas)
        # a different layout signature is rejected even when sizes agree
        other = list(host.signature); other[-1] = ('radius', (2,), (1,), other[-1][3])
        with pytest.raises(ValueError, match='signature'):
            copy_arenas(dev_arenas, ArenaSet(host.arenas, tuple(other), host.device))
        with pytest.raises(TypeError):
            copy_arenas(dev_arenas, {})
        assert all(torch.equal(dev_arenas[d], before[d]) for d in dev_arenas)
    finally:
        stager.close()


def test_worker_fp32_upcast_is_exact_and_accepted_by_an_fp32_engine(tmp_path):
    """Owner bind profile 06:10: the bank embedding is bfloat16 (baseline/model.py:102) and
    load_teacher_state upcasts it per state on the consumer; fp32_parameter_names moves that
    exact promotion into the worker through Astra's stage_fp32_teacher_model (9793cb8).
    The engine load must accept the record unchanged and buffers must keep their dtype."""
    from rewriter.teacher_cache import TeacherCacheReader
    from rewriter.function_preservation import FunctionalPreservation
    from baseline.model import GPT
    reader, cache, fresh = _fixtures(tmp_path, gpt_precision='fp32')   # cache written in production precision mode
    cfg = reader.manifest['provenance']['model_config']
    # production engines are gpt_precision='fp32' (adapter guard); that mode upcasts on load itself
    engine = FunctionalPreservation(GPT(**cfg).requires_grad_(False), microbatch_sequences=1, kl_dtype=torch.float64, gpt_precision='fp32')
    reference = FunctionalPreservation(GPT(**cfg).requires_grad_(False), microbatch_sequences=1, kl_dtype=torch.float64, gpt_precision='fp32')
    baseline_engine = FunctionalPreservation(GPT(**cfg).requires_grad_(False), microbatch_sequences=1, kl_dtype=torch.float64)
    # exact engine parameter names, as Astra's helper requires (no heuristics over names/ranks)
    stager = GraphCheckpointStager(reader, cache, pin=False, fp32_parameter_names=engine.parameters.keys())
    with pytest.raises(ValueError, match='not empty'):
        GraphCheckpointStager(reader, cache, pin=False, fp32_parameter_names=())
    stager.schedule([1, 2])
    try:
        for state_id in (1, 2):
            record, panel, _ = stager.take(state_id)
            eager = fresh.read(state_id, device='cpu', model_device='cpu')
            low = [n for n, v in eager.model.items() if v.is_floating_point() and v.dtype != torch.float32]
            assert low, 'fixture bank is expected to hold a bfloat16 embedding'
            for name, value in eager.model.items():
                held = record.model[name]
                if name in engine.parameters:
                    assert held.dtype == torch.float32 and torch.equal(held, value.float()), name   # exact promotion
                else:
                    assert held.dtype == value.dtype and torch.equal(held, value), name           # buffers keep dtype
                assert held.data_ptr() != value.data_ptr(), name                                    # independent storage
            assert record.assert_verified() == eager.assert_verified()
            cache.load_engine(engine, record)              # same path the owner's load_checkpoint uses
            cache.load_engine(reference, eager)
            for name in reference.parameters:
                assert torch.equal(engine.parameters[name], reference.parameters[name]), name
            for name in reference.buffers:
                assert torch.equal(engine.buffers[name], reference.buffers[name]), name
        # a baseline-precision engine keeps the bf16 embedding; the upcast record (or this fp32 cache) is rejected
        with pytest.raises(ValueError, match='precision differ|dtype differs'):
            cache.load_engine(baseline_engine, record)
    finally:
        stager.close()
