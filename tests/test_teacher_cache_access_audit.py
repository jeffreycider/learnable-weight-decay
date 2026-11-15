"""Count logical teacher-cache reads; CPU tests do not measure PCIe traffic."""
import pytest
import torch

from rewriter.function_preservation import chunked_teacher_kl


@pytest.mark.parametrize('steps',[1,4])
def test_maintenance_uses_two_teacher_reads_per_backward_and_one_final_read(monkeypatch,steps):
    torch.manual_seed(41)
    teacher=torch.randn(2,3,7,dtype=torch.float64).log_softmax(-1)
    source=teacher.untyped_storage().data_ptr()
    reads=[];original=torch.Tensor.to
    def observed_to(value,*args,**kwargs):
        if value.untyped_storage().data_ptr()==source:
            reads.append(value.numel()*value.element_size())
        return original(value,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',observed_to)
    for _ in range(steps):
        student=torch.randn(2,3,7,requires_grad=True)
        chunked_teacher_kl(student,teacher,chunk_tokens=2).backward()
        assert student.grad is not None
    training_bytes=sum(reads)
    with torch.no_grad():
        final=chunked_teacher_kl(torch.randn(2,3,7),teacher,chunk_tokens=2)
    assert final.isfinite()
    cache_bytes=teacher.numel()*teacher.element_size()
    assert training_bytes==2*steps*cache_bytes
    assert sum(reads)-training_bytes==cache_bytes
    assert sum(reads)==(2*steps+1)*cache_bytes
