import hashlib,io
import pytest
import torch
from rewriter.streaming_save import HashingWriter


def test_torch_save_byte_identity_shared_strided_and_mixed_tensors():
    x=torch.arange(120,dtype=torch.float32).reshape(10,12)
    record=dict(model={'x':x,'shared':x.T,'slice':x[:,::2],'bf16':x.to(torch.bfloat16)},
                observation={'banks':[x.T.clone(),torch.zeros(0,8)],'scalars':[1,False,None]})
    reference=io.BytesIO();torch.save(record,reference)
    output=io.BytesIO();writer=HashingWriter(output);torch.save(record,writer);writer.flush()
    assert output.getvalue()==reference.getvalue()
    assert writer.hexdigest()==hashlib.sha256(reference.getvalue()).hexdigest()
    restored=torch.load(io.BytesIO(output.getvalue()),weights_only=False)
    assert restored['model']['shared'].stride()==x.T.stride()
    assert restored['model']['shared'].untyped_storage().data_ptr()==restored['model']['x'].untyped_storage().data_ptr()


def test_fail_closed_on_seeks_short_writes_and_external_position_changes():
    output=io.BytesIO();writer=HashingWriter(output);writer.write(b'abc')
    with pytest.raises(OSError):writer.seek(0)
    output.seek(0)
    with pytest.raises(OSError):writer.write(b'd')
    output.seek(2)
    with pytest.raises(ValueError):HashingWriter(output)
    class Short(io.BytesIO):
        def write(self,data):return super().write(data[:-1])
    with pytest.raises(OSError):HashingWriter(Short()).write(b'abc')
