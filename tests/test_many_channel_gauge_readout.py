from types import SimpleNamespace
import torch
from scripts.evaluate_many_channel_gauge import reconstruct,paired_panel
from rewriter.function_preservation import PreservationCompute


def test_reconstruct_uses_full_binary_gauge_without_interpolation():
    p={'v':torch.arange(12.).reshape(3,4),'b':torch.ones(3),'o':torch.ones(2,3),'untouched':torch.tensor(9.)}
    row=dict(family='mlp',inverse_power=2,channels=[dict(first_weight='v',first_bias='b',second_weight='o',channel=i) for i in range(3)],channel_exponents=[-1,0,1])
    q=reconstruct(p,[row]);d=torch.tensor([.5,1.,2.])
    assert torch.equal(q['v'],p['v']*d[:,None]);assert torch.equal(q['o'],p['o']/d[None,:]**2)
    assert q['untouched'] is p['untouched'];assert torch.equal(p['b'],torch.ones(3))


def test_paired_metrics_reuse_exact_two_forwards_and_detect_changed_logits():
    torch.set_num_threads(1);torch.manual_seed(42)
    class Engine:
        device=torch.device('cpu');microbatch_sequences=2;chunk_tokens=3;teacher_id='fixed'
        def __init__(self):self.parameters={'w':torch.randn(7,105)};self.compute=PreservationCompute();self.calls=0
        def _validate_replacements(self,p):assert p['w'].shape==self.parameters['w'].shape
        def _forward(self,p,x):self.calls+=1;return p['w'][x]
    engine=Engine();inputs=torch.tensor([[0,1,2],[3,4,5],[6,0,1],[2,3,4]])
    panel=SimpleNamespace(inputs=inputs,sequence_ids=[1,2,3,4],teacher_batch=lambda e,a,b:SimpleNamespace(teacher_id=e.teacher_id,inputs=inputs[a:b],log_probs=e.parameters['w'][inputs[a:b]].double().log_softmax(-1)))
    same=paired_panel(engine,engine.parameters,panel,4,backend='reference')
    assert same['direct_logit_comparison']['equal'];assert same['readouts']['gauge']['strict_topk_changes']==0
    assert engine.calls==engine.compute.student_forwards==4
    altered={'w':engine.parameters['w'].clone()};altered['w'][:,0]+=20
    changed=paired_panel(engine,altered,panel,4,backend='reference')
    assert not changed['direct_logit_comparison']['equal']
    assert changed['direct_logit_comparison']['different_elements']==inputs.numel()
    assert changed['direct_logit_comparison']['max_abs_difference']==20
    assert changed['readouts']['gauge']['strict_topk_changes']>0
    assert changed['readouts']['gauge']['max_token_kl']>0
    assert engine.compute.student_backwards==0 and engine.calls==8
