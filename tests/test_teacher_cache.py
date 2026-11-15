import pytest
import torch
from baseline.model import GPT
from rewriter.function_preservation import FunctionalPreservation,chunked_teacher_kl
from rewriter.teacher_cache import write_panel,TeacherCacheReader,bind_engine,head_inputs,head_logits


def test_block_selection_keeps_targets_and_sequence_identity_aligned(tmp_path):
    torch.manual_seed(93)
    model=GPT(vocab_size=32,num_layers=1,model_dim=128).float()
    engine=FunctionalPreservation(model,kl_dtype=torch.float64,microbatch_sequences=2)
    inputs=torch.randint(32,(6,4))
    write_panel(tmp_path,1,'train',engine,inputs,[10,11,20,21,30,31],state_sha256='c'*64,execution={'mode':'eager'})
    panel=TeacherCacheReader(tmp_path,require_complete=False).read(1,'train')
    selected=panel.select_blocks([2,0])
    assert selected.sequence_ids==(30,31,10,11)
    assert torch.equal(selected.inputs,torch.cat([inputs[4:6],inputs[:2]]))
    assert torch.equal(selected.teacher_batch(engine,0,2).log_probs,panel.teacher_batch(engine,4,6).log_probs)
    with pytest.raises(ValueError,match='unique'):panel.select_blocks([1,1])


def test_exact_targets_gradient_and_checkpoint_binding(tmp_path):
    torch.manual_seed(81)
    model=GPT(vocab_size=32,num_layers=1,model_dim=128).float()
    engine=FunctionalPreservation(model,kl_dtype=torch.float64,microbatch_sequences=1)
    inputs=torch.randint(32,(2,4));sha='a'*64
    write_panel(tmp_path,1,'train',engine,inputs,[3,7],state_sha256=sha,execution={'mode':'eager'})
    panel=TeacherCacheReader(tmp_path,require_complete=False).read(1,'train')
    for i in range(2):
        actual=panel.teacher_batch(engine,i,i+1)
        expected=engine.cache_teacher(inputs[i:i+1],data_role='past_training')[0]
        assert torch.equal(actual.log_probs,expected.log_probs)
        logits=engine._forward(engine.parameters,inputs[i:i+1]).detach()
        assert chunked_teacher_kl(logits,actual.log_probs).item()==0
        left=(logits+.01*torch.randn_like(logits)).requires_grad_();right=left.detach().clone().requires_grad_()
        l=chunked_teacher_kl(left,actual.log_probs);r=chunked_teacher_kl(right,expected.log_probs)
        l.backward();r.backward();assert torch.equal(left.grad,right.grad)
    engine.load_teacher_state(engine.parameters,engine.buffers)
    with pytest.raises(ValueError,match='bound'):panel.teacher_batch(engine,0,1)
    bind_engine(engine,'b'*64)
    with pytest.raises(ValueError,match='bound'):panel.teacher_batch(engine,0,1)


def test_bfloat16_head_inputs_are_not_requantized():
    model=GPT(vocab_size=32,num_layers=1,model_dim=128)
    model.embed.to(torch.bfloat16)
    engine=FunctionalPreservation(model,kl_dtype=torch.float64)
    inputs=torch.randint(32,(1,4));features=head_inputs(engine,inputs)
    assert features.dtype==torch.bfloat16
    assert torch.equal(head_logits(engine,features),engine._forward(engine.parameters,inputs))


def test_corrupt_panel_and_changed_head_rejected(tmp_path):
    model=GPT(vocab_size=32,num_layers=1,model_dim=128).float()
    engine=FunctionalPreservation(model,kl_dtype=torch.float64)
    inputs=torch.randint(32,(1,4));sha='a'*64
    write_panel(tmp_path,1,'audit',engine,inputs,[9],state_sha256=sha,execution={'mode':'eager'})
    with pytest.raises(ValueError,match='incomplete'):TeacherCacheReader(tmp_path)
    panel=TeacherCacheReader(tmp_path,require_complete=False).read(1,'audit')
    with torch.no_grad():engine.parameters['proj.weight'].add_(.01)
    with pytest.raises(ValueError,match='head differs'):panel.teacher_batch(engine,0,1)
    path=tmp_path/'state_0001_audit.pt'
    with path.open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(ValueError,match='hash'):TeacherCacheReader(tmp_path,require_complete=False).read(1,'audit')


def test_complete_manifest_pins_panel_sidecar(tmp_path):
    import json
    model=GPT(vocab_size=32,num_layers=1,model_dim=128).float()
    engine=FunctionalPreservation(model,kl_dtype=torch.float64)
    inputs=torch.randint(32,(1,4))
    rows=[write_panel(tmp_path,1,role,engine,inputs,[n],state_sha256='a'*64,execution={'mode':'eager'})
          for n,role in enumerate(('train','audit'))]
    (tmp_path/'manifest.json').write_text(json.dumps(dict(complete=True,args={'states':1},panels=rows)))
    reader=TeacherCacheReader(tmp_path)
    reader.read(1,'train')
    path=tmp_path/'state_0001_train.json';info=json.loads(path.read_text());info['state_sha256']='b'*64
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError,match='provenance'):reader.read(1,'train')


def test_owned_verified_load_skips_repeated_head_hash_and_detects_mutation(tmp_path,monkeypatch):
    import json
    import rewriter.teacher_cache as cache_module
    from rewriter.state_bank import StateBankWriter,StateBankReader
    from tests.test_prefix_cache import toy_record
    from tests.test_state_bank import provenance
    model=GPT(vocab_size=32,num_layers=1,model_dim=128).float()
    snapshot,train=toy_record()
    for old,new in [('first','embed.weight'),('last','proj.weight')]:
        snapshot.matrices[new]=snapshot.matrices.pop(old);train.matrices[new]=train.matrices.pop(old)
    snapshot.vectors['proj.bias']=snapshot.vectors.pop('bias');train.vectors['proj.bias']=train.vectors.pop('bias')
    bank=tmp_path/'bank'
    with StateBankWriter(bank,provenance(),expected_steps=1) as writer:
        writer.append(1,model.state_dict(),snapshot,train,data_cursor_batches=1,identity_action=True)
    reader=StateBankReader(bank,verify_once=True,mmap_model=True);record=reader.read(1)
    engine=FunctionalPreservation(model,kl_dtype=torch.float64)
    inputs=torch.randint(32,(1,4));directory=tmp_path/'targets'
    rows=[write_panel(directory,1,role,engine,inputs,[n],state_sha256=record.assert_verified(),execution={'mode':'eager'}) for n,role in enumerate(('train','audit'))]
    (directory/'manifest.json').write_text(json.dumps(dict(complete=True,args={'states':1},panels=rows)))
    cached=TeacherCacheReader(directory,verify_once=True,mmap=True)
    panel=cached.read(1,'train');cached.load_engine(engine,record)
    def forbidden(*args):raise AssertionError('redundant immutable hash')
    monkeypatch.setattr(cache_module,'tensor_hash',forbidden);monkeypatch.setattr(cache_module,'file_hash',forbidden)
    panel=cached.read(1,'train');actual=panel.teacher_batch(engine,0,1)
    expected=engine.cache_teacher(inputs,data_role='past_training')[0]
    assert torch.equal(actual.log_probs,expected.log_probs)
    with torch.no_grad():engine.parameters['proj.weight'].add_(1)
    with pytest.raises(ValueError,match='identity/version'):panel.teacher_batch(engine,0,1)
    record.model['proj.weight'].add_(1)
    with pytest.raises(ValueError,match='mutated'):cached.load_engine(engine,record)
