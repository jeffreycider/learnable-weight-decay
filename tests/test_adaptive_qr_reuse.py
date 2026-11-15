import copy
from unittest.mock import patch
import pytest
import torch
from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.adaptive_qr_reuse import evaluate_adaptive_with_qr_reuse
from tests.test_controller import make_snapshot


def case(precision):
    torch.set_num_threads(1)
    torch.manual_seed(451)
    snapshot = make_snapshot()
    actor = AdaptiveCoordinateChisel(rank=2, trajectory_dim=12, model_dim=128,
        num_layers=1, latent_tokens=8, fresh_weight_core=True, factor_precision=precision)
    private = actor.make_private(snapshot, state_key=('qr-test', 6), seed=19)
    with torch.no_grad():
        actor.matrix_output.weight.normal_(std=.01)
        actor.vector_output.weight.normal_(std=.01)
        actor.blocks[0].attn.proj.weight.normal_(std=.01)
    weights = {n: torch.randn(o.rows.shape[0], o.columns.shape[0], requires_grad=True)
               for n, o in snapshot.matrices.items()}
    return snapshot, actor, private, weights


def evaluate(actor, snapshot, private, weights, reuse):
    if reuse:
        return evaluate_adaptive_with_qr_reuse(actor, snapshot, private, weights, embedding_names=())
    return actor(snapshot, private, weight_cores=query_weight_cores(weights, private, embedding_names=()))


@pytest.mark.parametrize('precision', ['fp32', 'fp64'])
def test_nonzero_output_all_gradient_parity_two_adam_steps_and_counts(precision):
    s, a, p, w = case(precision)
    b, q = copy.deepcopy(a), copy.deepcopy(p)
    optimizers = [torch.optim.Adam(list(x.parameters()) + list(y.parameters()), lr=1e-5)
                  for x, y in [(a,p),(b,q)]]
    targets = None
    for step in range(2):
        results, gradients, counts = [], [], []
        for i,(actor, private) in enumerate([(a,p),(b,q)]):
            optimizers[i].zero_grad(set_to_none=True)
            with patch('torch.linalg.qr', wraps=torch.linalg.qr) as qr:
                action = evaluate(actor,s,private,w,bool(i))
            counts.append(qr.call_count)
            dense = action.dense()
            if targets is None:
                targets = {n:torch.randn_like(t) for n,t in dense.items()}
            loss = sum((t*targets[n]).sum() for n,t in dense.items())
            loss.backward()
            results.append(dense)
            gradients.append({prefix+n: None if v.grad is None else v.grad.clone()
                for prefix,m in [('actor.',actor),('private.',private)] for n,v in m.named_parameters()})
        assert counts == [6*len(p.matrix_names)+2*len(p.vector_names),
                          2*len(p.matrix_names)+len(p.vector_names)]
        for n in results[0]:
            torch.testing.assert_close(results[0][n],results[1][n],
                rtol=0 if step==0 else 2e-5,atol=0 if step==0 else 2e-6)
        for n,g in gradients[0].items():
            h=gradients[1][n]
            assert (g is None)==(h is None), n
            if g is not None:
                strict=precision=='fp64' and step==0
                torch.testing.assert_close(g,h,rtol=1e-9 if strict else 3e-4,
                                           atol=1e-11 if strict else 2e-5,msg=n)
        assert p.matrix_rows[0].raw.grad.norm()>0
        assert a.weight_core_input.weight.grad.norm()>0
        assert all(v.grad is None for v in w.values())
        for opt in optimizers: opt.step()
        assert all("orthonormal_matrix" not in f.__dict__ for f in (*p.matrix_rows,*p.matrix_columns,*p.vector_frames))


def test_nested_exception_cleanup_and_mutation_guard():
    s,a,p,w = case('fp32')
    def nested(snapshot,private,**kwargs):
        assert private is not p
        inner=evaluate(a,s,p,w,True)
        assert all("orthonormal_matrix" not in f.__dict__ for f in p.matrix_rows)
        def fail(*args,**kw): raise ValueError('inner failure')
        with pytest.raises(ValueError,match='inner failure'):
            evaluate_adaptive_with_qr_reuse(fail,s,p,w,embedding_names=())
        assert all("orthonormal_matrix" not in f.__dict__ for f in p.matrix_rows)
        return a(snapshot,private,**kwargs)
    expected=evaluate(a,s,p,w,True).dense()
    actual=evaluate_adaptive_with_qr_reuse(nested,s,p,w,embedding_names=()).dense()
    for n in expected: torch.testing.assert_close(expected[n],actual[n],rtol=0,atol=0)
    def broken(*args,**kwargs): raise ValueError('test failure')
    with pytest.raises(ValueError,match='test failure'):
        evaluate_adaptive_with_qr_reuse(broken,s,p,w,embedding_names=())
    assert all("orthonormal_matrix" not in f.__dict__ for f in p.matrix_rows)
    def mutated(snapshot,private,**kwargs):
        with torch.no_grad(): private.matrix_rows[0].raw.add_(.001)
        return a(snapshot,private,**kwargs)
    with pytest.raises(RuntimeError,match='frame or autograd'):
        evaluate_adaptive_with_qr_reuse(mutated,s,p,w,embedding_names=())
    assert all("orthonormal_matrix" not in f.__dict__ for f in p.matrix_rows)
    evaluate(a,s,p,w,True).norm_sq().backward()


def test_parameter_replacement_and_grad_mode_guard():
    s,a,p,w = case('fp32')
    def replaced(snapshot,private,**kwargs):
        private.matrix_rows[0].raw=torch.nn.Parameter(private.matrix_rows[0].raw.detach().clone())
        return a(snapshot,private,**kwargs)
    with pytest.raises(RuntimeError,match='frame or autograd'):
        evaluate_adaptive_with_qr_reuse(replaced,s,p,w,embedding_names=())
    def mode(snapshot,private,**kwargs):
        with torch.no_grad(): return a(snapshot,private,**kwargs)
    with pytest.raises(RuntimeError,match='frame or autograd'):
        evaluate_adaptive_with_qr_reuse(mode,s,p,w,embedding_names=())


def test_default_frame_fullgraph_compile_unchanged_and_reuse_rejects_compiled_actor():
    s,a,p,w=case('fp32')
    frame=p.matrix_rows[0]
    compiled=torch.compile(frame.orthonormal_matrix,backend='eager',fullgraph=True)
    torch.testing.assert_close(compiled(),frame.orthonormal_matrix(),rtol=0,atol=0)
    with pytest.raises(RuntimeError,match='eager actors only'):
        evaluate_adaptive_with_qr_reuse(torch.compile(a,backend='eager'),s,p,w,embedding_names=())
    with pytest.raises(RuntimeError,match='eager actors only'):
        evaluate_adaptive_with_qr_reuse(torch.compile(a.forward,backend='eager'),s,p,w,embedding_names=())


def test_temporary_views_release_immediately_and_frozen_frames_recompute():
    import weakref
    s,a,p,w=case('fp64')
    for frame in (*p.matrix_rows,*p.matrix_columns,*p.vector_frames):
        frame.raw.requires_grad_(False)
    refs=[]
    def inspect(snapshot,private,**kwargs):
        refs.extend(weakref.ref(f) for f in (*private.matrix_rows,*private.matrix_columns,*private.vector_frames))
        refs.append(weakref.ref(private))
        return a(snapshot,private,**kwargs)
    for _ in range(2):
        with patch('torch.linalg.qr',wraps=torch.linalg.qr) as qr:
            out=evaluate_adaptive_with_qr_reuse(inspect,s,p,w,embedding_names=())
        assert qr.call_count==2*len(p.matrix_names)+len(p.vector_names)
        assert all(ref() is None for ref in refs)
        out.norm_sq().backward()
    assert all(f.raw.grad is None for f in p.matrix_rows)
    assert a.matrix_output.weight.grad.norm()>0
