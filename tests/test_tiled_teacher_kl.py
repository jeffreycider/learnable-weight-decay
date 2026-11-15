import os
os.environ.setdefault('PROFILE_CHISEL_SKIP_SOURCE_GUARD', '1')  # shared tree is edited concurrently by other agents
import pytest
import torch
import torch.nn.functional as F
from rewriter.function_preservation import chunked_teacher_kl
from rewriter.tiled_teacher_kl import tiled_teacher_kl


def _case(vocab, magnitude, device='cpu', rows=(2, 3)):
    torch.manual_seed(918)
    raw = torch.randn(*rows, vocab, device=device) * 4
    teacher = 15 * raw * (raw.square() + 225).rsqrt()
    student = (teacher + torch.randn_like(raw) * magnitude).requires_grad_()
    return student, teacher


@pytest.mark.parametrize('vocab', [17, 2049, 50304])
@pytest.mark.parametrize('magnitude', [0., 1e-6, 1e-3, .2])
@pytest.mark.parametrize('tile', [8, 2048])
def test_tiled_kl_matches_chunked_reference_on_cpu(vocab, magnitude, tile):
    torch.set_num_threads(1)
    student, teacher = _case(vocab, magnitude)
    logq = F.log_softmax(teacher.double(), -1)
    actual = tiled_teacher_kl(student, teacher, tile)
    expected = chunked_teacher_kl(student, logq, 2)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, atol=5e-14, rtol=1e-8)
    if magnitude == 0.:
        assert abs(float(actual)) < 1e-13
    ga, = torch.autograd.grad(actual * 71.3, student, retain_graph=True)
    ge, = torch.autograd.grad(expected * 71.3, student)
    torch.testing.assert_close(ga, ge, atol=3e-11, rtol=1e-5)


def test_tiled_kl_rejects_bad_inputs():
    student, teacher = _case(33, 1e-3)
    with pytest.raises(ValueError, match='FP32'):
        tiled_teacher_kl(student, teacher.double())
    with pytest.raises(ValueError, match='tile'):
        tiled_teacher_kl(student, teacher, 3)
    with pytest.raises(ValueError, match='shape'):
        tiled_teacher_kl(student, teacher[..., :-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA Triton kernels')
@pytest.mark.parametrize('vocab', [17, 50304])
@pytest.mark.parametrize('magnitude', [0., 1e-6, 1e-3, .2])
def test_tiled_kl_triton_matches_reference_on_cuda(vocab, magnitude):
    student, teacher = _case(vocab, magnitude, 'cuda', rows=(2, 5))
    logq = F.log_softmax(teacher.double(), -1)
    actual = tiled_teacher_kl(student, teacher)
    expected = chunked_teacher_kl(student, logq, 128)
    torch.testing.assert_close(actual, expected, atol=5e-14, rtol=1e-8)
    ga, = torch.autograd.grad(actual * 71.3, student, retain_graph=True)
    ge, = torch.autograd.grad(expected * 71.3, student)
    torch.testing.assert_close(ga, ge, atol=3e-11, rtol=1e-5)
    # Equivalent torch emulation on the same device must agree with the kernel.
    from rewriter import tiled_teacher_kl as module
    saved = module.triton
    module.triton = None
    try:
        emulated = tiled_teacher_kl(student, teacher)
        ge2, = torch.autograd.grad(emulated * 71.3, student)
    finally:
        module.triton = saved
    torch.testing.assert_close(actual, emulated, atol=1e-15, rtol=1e-12)
    torch.testing.assert_close(ga, ge2, atol=1e-12, rtol=1e-6)


def test_profiler_harness_tiled_arm_matches_reference_within_fp64_tolerance(tmp_path):
    from tests.test_step_timing import _tiny_bank_and_cache
    from scripts.profile_chisel_update import main
    bank, cache = _tiny_bank_and_cache(tmp_path)
    common = ['--state-bank', str(bank), '--teacher-cache', str(cache), '--device', 'cpu', '--gpt-precision', 'baseline',
              '--state-ids', '1', '2', '--epoch', '0', '--sequences', '2', '--microbatch', '1', '--checkpoint-batch', '4',
              '--repeats', '1', '--chisel-model-dim', '128', '--chisel-layers', '1', '--chisel-patch-size', '8']
    reference = main([*common, '--output', str(tmp_path / 'ref'), '--timing', 'sync'])
    for backend in ('tiled', 'tiled-softcap'):
        tiled = main([*common, '--output', str(tmp_path / backend), '--timing', 'sync', '--kl-backend', backend])
        for r, t in zip(reference['passes'][0]['mean_kl'], tiled['passes'][0]['mean_kl']):
            assert abs(r - t) <= 5e-14 + 1e-8 * abs(r), backend
        assert tiled['arm']['kl_backend'] == backend
        assert tiled['passes'][0]['gradient_norm'] == pytest.approx(reference['passes'][0]['gradient_norm'], rel=1e-5), backend
    # the arm must not leak its forward/KL monkeypatches
    import rewriter.function_preservation as preservation
    again = main([*common, '--output', str(tmp_path / 'again'), '--timing', 'sync'])
    assert again['gradient_sha256'] == reference['gradient_sha256']


def test_tiled_arm_panel_validation_rejects_wrong_precision_binding_and_head(tmp_path):
    from baseline.model import GPT
    from rewriter.function_preservation import FunctionalPreservation
    from rewriter.teacher_cache import write_panel, TeacherCacheReader, bind_engine
    from rewriter.tiled_teacher_kl import validate_panel_for_tiled
    torch.manual_seed(3)
    model = GPT(vocab_size=32, num_layers=1, model_dim=128).float()
    engine = FunctionalPreservation(model, kl_dtype=torch.float64, microbatch_sequences=2)
    inputs = torch.randint(32, (2, 4))
    write_panel(tmp_path, 1, 'train', engine, inputs, [1, 2], state_sha256='a' * 64, execution={'mode': 'eager'})
    panel = TeacherCacheReader(tmp_path, require_complete=False).read(1, 'train')
    bind_engine(engine, 'b' * 64)  # bound to another checkpoint
    with pytest.raises(ValueError, match='bound'):
        validate_panel_for_tiled(panel, engine)
    bind_engine(engine, 'a' * 64)
    validate_panel_for_tiled(panel, engine)
    bad = dict(panel.metadata, gpt_precision='fp32')
    from rewriter.teacher_cache import TeacherPanel
    with pytest.raises(ValueError, match='precision'):
        validate_panel_for_tiled(TeacherPanel(panel.inputs, panel.features, panel.sequence_ids, bad), engine)
    with torch.no_grad():
        engine.parameters['proj.weight'].add_(.01)
    with pytest.raises(ValueError, match='head'):
        validate_panel_for_tiled(panel, engine)


@pytest.mark.parametrize('vocab', [17, 50304])
@pytest.mark.parametrize('magnitude', [0., 1e-6, 1e-3, .2])
def test_softcap_inside_kernel_matches_autograd_softcap_within_fp32_tolerance(vocab, magnitude):
    """Phase 2 arm: raw head outputs in, softcap applied inside. Forward uses the
    module's FP32 operation order; backward uses the closed-form derivative, so
    gradients agree to FP32 rounding, not bit-level."""
    torch.set_num_threads(1)
    torch.manual_seed(7)
    raw_t = torch.randn(2, 3, vocab) * 4
    raw_s = (raw_t + torch.randn_like(raw_t) * magnitude).requires_grad_()
    cap = lambda z: 15 * z * (z.square() + 225).rsqrt()
    expected = chunked_teacher_kl(cap(raw_s), F.log_softmax(cap(raw_t).double(), -1), 2)
    actual = tiled_teacher_kl(raw_s, raw_t, 8, softcap=True)
    torch.testing.assert_close(actual, expected, atol=5e-14, rtol=1e-8)
    ge, = torch.autograd.grad(expected * 71.3, raw_s, retain_graph=True)
    ga, = torch.autograd.grad(actual * 71.3, raw_s)
    torch.testing.assert_close(ga, ge, atol=3e-11, rtol=1e-5)
    if magnitude == 0.:
        assert abs(float(actual.detach())) < 1e-13


def test_full_stack_arm_tiled_plus_constants_cache_plus_panel_prefetch(tmp_path):
    """The arms the owner would stack: tiled KL + SHA constants cache + panel-only prefetch + event timing."""
    from tests.test_step_timing import _tiny_bank_and_cache
    from scripts.profile_chisel_update import main
    bank, cache = _tiny_bank_and_cache(tmp_path)
    common = ['--state-bank', str(bank), '--teacher-cache', str(cache), '--device', 'cpu', '--gpt-precision', 'baseline',
              '--state-ids', '1', '2', '3', '--epoch', '1', '--sequences', '2', '--microbatch', '1', '--checkpoint-batch', '4',
              '--repeats', '2', '--chisel-model-dim', '128', '--chisel-layers', '1', '--chisel-patch-size', '8']
    reference = main([*common, '--output', str(tmp_path / 'ref'), '--timing', 'sync'])
    stacked = main([*common, '--output', str(tmp_path / 'stack'), '--timing', 'event', '--deferred-kl', '--prefetch-panel-only',
                    '--constants-cache', '--kl-backend', 'tiled'])
    assert stacked['all_passes_same_update_all_ranks']  # cache hits on passes 2/3 reproduce pass 1 bits
    for r, t in zip(reference['passes'][0]['mean_kl'], stacked['passes'][0]['mean_kl']):
        assert abs(r - t) <= 5e-14 + 1e-8 * abs(r)
    assert stacked['passes'][0]['gradient_norm'] == pytest.approx(reference['passes'][0]['gradient_norm'], rel=1e-5)


def test_harness_undoes_global_patches_on_exception(tmp_path):
    """Astra review: cleanup must be finally-equivalent, and the failure must be injected AFTER the
    patches are installed. A teacher panel file deleted from a copied cache makes cache.read raise
    inside the first checkpoint of the pass (patches, timing installer and prefetch thread are live)."""
    import shutil
    import rewriter.function_preservation as preservation
    import scripts.train_multistate_chisel as train
    from tests.test_step_timing import _tiny_bank_and_cache
    from scripts.profile_chisel_update import main
    bank, cache = _tiny_bank_and_cache(tmp_path)
    broken = tmp_path / 'targets_broken'
    shutil.copytree(cache, broken)
    (broken / 'state_0002_train.pt').unlink()  # state 1 loads fine, state 2 fails mid-pass
    before = (preservation.chunked_teacher_kl, preservation._LogitsView.forward, train.timed)
    common = ['--state-bank', str(bank), '--teacher-cache', str(broken), '--device', 'cpu', '--gpt-precision', 'baseline',
              '--state-ids', '1', '2', '--epoch', '0', '--sequences', '2', '--microbatch', '1', '--checkpoint-batch', '4',
              '--repeats', '1', '--chisel-model-dim', '128', '--chisel-layers', '1', '--chisel-patch-size', '8']
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        main([*common, '--output', str(tmp_path / 'bad'), '--timing', 'event', '--kl-backend', 'tiled-softcap', '--prefetch-panel-only'])
    assert (preservation.chunked_teacher_kl, preservation._LogitsView.forward, train.timed) == before
    # and a same-process reference run afterwards is unaffected
    good = main([*common[:3], str(cache), *common[4:], '--output', str(tmp_path / 'good'), '--timing', 'sync'])
    assert good['all_measured_passes_bit_identical']
