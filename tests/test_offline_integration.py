import json,struct
import numpy as np
import torch
from baseline.model import GPT
from tests.test_multistate_chisel import _tiny_state_bank,state_replay
def test_integrated_offline_flags_exact_two_epoch_adam(tmp_path):
    from scripts.train_multistate_chisel import main,select_sequences,stable_seed
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import write_panel
    from rewriter.function_preservation import FunctionalPreservation
    bank=tmp_path/'bank';cfg=_tiny_state_bank(bank)
    data=tmp_path/'fineweb_train_000001.bin';tokens=(np.arange(1000)%16).astype(np.uint16)
    data.write_bytes(struct.pack('<iii',20240520,1,len(tokens))+bytes(1012)+tokens.tobytes())
    reader=StateBankReader(bank);cache=tmp_path/'targets';engine=FunctionalPreservation(GPT(**cfg['model']),microbatch_sequences=1,kl_dtype=torch.float64,gpt_precision='fp32')
    names=set(engine.parameters);panel_rows=[]
    for state_id in range(1,6):
        record=reader.read(state_id)
        engine.load_teacher_state({n:v for n,v in record.model.items() if n in names},{n:v for n,v in record.model.items() if n not in names})
        pools=state_replay([str(data)],state_id,cfg,pool_sequences=32,seed=400001,epoch=0,split_seed=500001)
        for role in ('train','audit'):
            inputs,ids=select_sequences(pools[role],2,seed=stable_seed('fixed-panel-test',state_id,role))
            panel_rows.append(write_panel(cache,state_id,role,engine,inputs,ids,state_sha256=reader.entries[state_id-1]['sha256'],execution={'test':True}))
    (cache/'manifest.json').write_text(json.dumps({'complete':True,'args':{'states':5},'panels':panel_rows,'execution':{'gpt_precision':'fp32'},'purpose':'tiny exact cache integration test'}))
    common=['--state-bank',str(bank),'--teacher-cache',str(cache),'--train-data',str(data),
        '--epochs','2','--checkpoint-batch','2','--sequences-per-checkpoint','1','--gpt-microbatch','1',
        '--replay-pool-sequences','32','--lr','.00002','--kl-budget','.1','--expected-gpt-seed','7',
        '--audit-states','0','--training-audit-states','1','--audit-sequences','1','--audit-every-epochs','1','--checkpoint-every','1',
        '--full-training-audit','--gradient-diagnostics','--verify-state-once','--mmap-state',
        '--eager-chisel','--device','cpu','--no-plots','--chisel-model-dim','128','--chisel-layers','1',
        '--gpt-precision','fp32','--representation-lr-multiplier','10']
    outputs=[]
    for mode,flags in [('baseline',[]),('candidate',['--offline-systems-cache','--no-component-timing'])]:
        output=tmp_path/mode
        main([*common,'--output',str(output),*flags])
        outputs.append(torch.load(output/'controller_latest.pt',weights_only=False))
    def exact(a,b):
        if isinstance(a,torch.Tensor):assert torch.equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:exact(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):exact(x,y)
        else:assert a==b
    for key in ('controller','optimizer','primal_dual','updates','next_epoch','next_batch'):
        exact(outputs[0][key],outputs[1][key])
    assert outputs[1]['updates']==4
    assert 'rewriter/offline_systems.py' in outputs[1]['metadata']['implementation_sha256']
    a,b=([json.loads(x) for x in (tmp_path/m/'metrics.jsonl').read_text().splitlines()] for m in ('baseline','candidate'))
    for event in ('train_state','audit_state'):
        left=[x for x in a if x['event']==event];right=[x for x in b if x['event']==event]
        assert len(left)==len(right)
        for x,y in zip(left,right):
            for key in ('mean_kl','objective','norm_term','hard_radius','global_scale','sequence_ids'):
                assert x[key]==y[key]
            assert y['component_seconds'] is None

    for rows in (a,b):
        summaries=[r for r in rows if r['event']=='audit_summary']
        assert all(len(r['state_ids'])==1 for r in summaries if r['role']=='heldout_sequences')
        assert all(len(r['state_ids'])==1 for r in summaries if r['role']=='heldout_states')
        assert all(len(r['state_ids'])==4 for r in summaries if r['role']=='fixed_training')
