"""Tiny real-GPT pending-credit lifecycle with an independent eager reference.

This is CPU AOT coverage, not the actual Transformer actor/GPU adoption gate.
"""
import copy
import json
from types import MethodType

import torch

from rewriter.compiled_thin_student import CompiledThinStudent
from rewriter.qualification_metrics import parameter_delta, tensor_comparison
from tests.test_online_preservation import make
from tests.test_online_norm_lifecycle import assert_tree
from tests.test_transformer_online import _ordinary_step


def test_compiled_student_lifecycle_clocks_and_response_diagnostics(tmp_path,record_property):
    model, optimizer, adapter = make(tmp_path)
    _ordinary_step(model, optimizer, adapter, 2)
    saved = copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                               adapter=adapter.state_dict()))
    untouched = copy.deepcopy(saved)
    assert adapter.lifecycle.action is not None
    assert int(adapter.lifecycle.optimizer.state[adapter.lifecycle.actor.x]['step']) == 1
    adapter.close()

    def run(compiled):
        model, optimizer, adapter = make(tmp_path)
        model.load_state_dict(saved['model'])
        # CPU load_state_dict can retain tensor storage from its input. Each
        # arm must own its moments; eager must not mutate the saved baseline.
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        adapter.load_state_dict(copy.deepcopy(saved['adapter']))
        helper = None
        if compiled:
            def student(replay,wire,teachers,editable,*,backward):
                nonlocal helper
                if helper is None:
                    helper = CompiledThinStudent(replay.engine,editable,
                        global_tokens=replay.settings['sequences']*replay.seq_len,
                        embedding_names=replay.embedding_names,compiler_backend='aot_eager',
                        compile_kl=False)
                return helper.step(wire,teachers,editable,backward=backward)
            adapter.replay._student = MethodType(student,adapter.replay)
        rows=[]
        try:
            for step in (3,4):
                before={n:p.detach().clone() for n,p in adapter.lifecycle.actor.named_parameters()}
                _ordinary_step(model,optimizer,adapter,step)
                actor=dict(adapter.lifecycle.actor.named_parameters())
                delta=parameter_delta(actor,before)
                moments=copy.deepcopy(adapter.lifecycle.optimizer.state_dict())
                assert int(adapter.lifecycle.optimizer.state[adapter.lifecycle.actor.x]['step']) == 1+2*(step-2)
                assert adapter.lifecycle.meta_updates == step-2
                assert adapter.replay.updates == step-1
                assert adapter.replay.dual.updates == step-1
                assert adapter.lifecycle.action is not None
                assert not adapter.collector.active and adapter.collector.ctx is None
                assert adapter.metrics['chisel_extra_gpt_forwards']==6
                assert adapter.metrics['chisel_extra_gpt_backwards']==2
                rows.append(dict(delta=delta,moments=moments,
                    dual=copy.deepcopy(adapter.replay.dual.state_dict()),
                    final_kl=adapter.metrics['chisel_replay_final_kl'],
                    panel=copy.deepcopy(adapter.replay.last_panel)))
        finally:
            adapter.close()
        assert_tree(saved,untouched)
        return rows

    eager, compiled = run(False),run(True)
    diagnostics=[]
    for index,(a,b) in enumerate(zip(eager,compiled)):
        response=tensor_comparison(a['delta'],b['delta'])
        assert response['finite'] and response['reference_norm']>0
        assert response['bit_identical']
        assert a['moments']['param_groups']==b['moments']['param_groups']
        moment_metrics={}
        for key,state in a['moments']['state'].items():
            for name,value in state.items():
                other=b['moments']['state'][key][name]
                if name=='step':
                    torch.testing.assert_close(value,other,rtol=0,atol=0)
                else:
                    metric=tensor_comparison({'value':value},{'value':other})
                    assert metric['finite']
                    assert metric['bit_identical']
                    moment_metrics[str(key)+'/'+name]=metric
        assert a['dual']['updates']==b['dual']['updates']
        assert a['dual']==b['dual']
        assert a['final_kl']==b['final_kl']
        assert a['panel']['id_sha256']==b['panel']['id_sha256']
        assert a['panel']['input_sha256']==b['panel']['input_sha256']
        diagnostics.append(dict(ordinary_step=index+3,actor_update_delta=response,
            adam_moments=moment_metrics,dual_reference=a['dual']['multipliers'][0],
            dual_candidate=b['dual']['multipliers'][0],final_kl_reference=a['final_kl'],
            final_kl_candidate=b['final_kl']))
    # Exactness is specific to this CPU AOT fixture; it is not a GPU gate.
    record_property('numerical_response_diagnostics',json.dumps(diagnostics))
