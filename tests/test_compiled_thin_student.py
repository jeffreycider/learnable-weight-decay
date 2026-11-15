import importlib.util
from pathlib import Path
import pytest
import torch
from rewriter.compiled_thin_student import CompiledThinStudent
from rewriter.function_preservation import TeacherBatch


def fixture():
    path=Path(__file__).with_name('test_factorized_fp32.py')
    spec=importlib.util.spec_from_file_location('compiled_student_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    engine,action,leaves,inputs=module.setup()
    from rewriter.algebra import RawAction,Factors
    action=RawAction({n:Factors(f.P.detach().clone().requires_grad_(),f.Q.detach().clone().requires_grad_(),f.anchors)
                      for n,f in action.matrices.items()},
                     {n:v.detach().clone().requires_grad_() for n,v in action.vectors.items()})
    leaves=[v for f in action.matrices.values() for v in (f.P,f.Q)]+list(action.vectors.values())
    scale=torch.tensor(.1,requires_grad=True);leaves.append(scale)
    teachers=engine.cache_teacher(inputs,data_role='past_training')
    return engine,(action,scale,leaves,[]),teachers,set(engine.parameters)


@pytest.mark.parametrize('compile_kl',[False,True])
def test_compiled_boundary_preserves_raw_all_factor_cotangents_and_global_weight(compile_kl):
    torch.set_num_threads(1)
    engine,wire,teachers,editable=fixture()
    eager=CompiledThinStudent(engine,editable,global_tokens=80,embedding_names={'embed.weight'},compile=False)
    compiled=CompiledThinStudent(engine,editable,global_tokens=80,embedding_names={'embed.weight'},compiler_backend='eager',compile_kl=compile_kl)
    a,ga=eager.step(wire,teachers,editable);ga=[v.clone() for v in ga]
    b,gb=compiled.step(wire,teachers,editable)
    assert torch.equal(a,b)
    for got,want in zip(gb,ga):torch.testing.assert_close(got,want,rtol=0,atol=0)
    local=CompiledThinStudent(engine,editable,global_tokens=10,embedding_names={'embed.weight'},compile=False)
    c,gc=local.step(wire,teachers,editable)
    torch.testing.assert_close(c,8*a,rtol=0,atol=0)
    for got,want in zip(gc,ga):torch.testing.assert_close(got,8*want,rtol=0,atol=0)
    assert all(not p.requires_grad and p.grad is None for p in engine.parameters.values())


def test_changed_edit_mask_rejected_before_student_execution():
    engine,wire,teachers,editable=fixture()
    student=CompiledThinStudent(engine,editable,global_tokens=10,embedding_names={'embed.weight'},compile=False)
    with pytest.raises(ValueError,match='editable structure'):
        student.step(wire,teachers,editable-{'proj.weight'})
