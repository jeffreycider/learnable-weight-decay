"""End-to-end namespace test: two GPT seeds, real caches and actor updates."""
import json
import struct
from pathlib import Path
import numpy as np
import torch
from baseline.model import GPT
from rewriter.function_preservation import FunctionalPreservation
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import write_panel
from rewriter.multibank import file_sha,SCHEMA
from scripts.train_multistate_chisel import main,heldout_sequence


def test_composite_trainer_keeps_duals_and_readouts_in_seed_namespace(tmp_path):
    from test_multistate_chisel import _tiny_state_bank
    torch.set_num_threads(1)
    data=tmp_path/'train.bin';tokens=(np.arange(1000)%16).astype(np.uint16)
    data.write_bytes(struct.pack('<iii',20240520,1,len(tokens))+bytes(1012)+tokens.tobytes())
    banks=[]
    for seed in (0,1):
        bank=tmp_path/f'bank{seed}';_tiny_state_bank(bank)
        manifest=bank/'manifest.json';m=json.loads(manifest.read_text());m['provenance']['seed']=seed;manifest.write_text(json.dumps(m))
        reader=StateBankReader(bank);cache=tmp_path/f'cache{seed}';cache.mkdir();rows=[]
        engine=FunctionalPreservation(GPT(16,1,128).float().requires_grad_(False),microbatch_sequences=1,kl_dtype=torch.float64,gpt_precision='fp32')
        for local in range(1,6):
            record=reader.read(local)
            engine.load_teacher_state({n:record.model[n] for n in engine.parameters},{n:record.model[n] for n in engine.buffers})
            for role in ('train','audit'):
                x=torch.tensor([[1,2,3,4],[4,3,2,1]])
                ids=[i for i in range(100) if heldout_sequence(i,500001)==(role=='audit')][:2]
                rows.append(write_panel(cache,local,role,engine,x,ids,state_sha256=record.assert_verified(),execution={'gpt_precision':'fp32'}))
        (cache/'manifest.json').write_text(json.dumps(dict(complete=True,args={'states':5},execution={'gpt_precision':'fp32'},panels=rows)))
        banks.append(dict(seed=seed,state_bank=str(bank),teacher_cache=str(cache),state_manifest_sha256=file_sha(manifest),state_index_sha256=file_sha(bank/'index.jsonl'),teacher_manifest_sha256=file_sha(cache/'manifest.json')))
    index=tmp_path/'index.json';index.write_text(json.dumps(dict(schema=SCHEMA,state_id_stride=5,banks=banks,heldout_online_seeds=[2])))
    out=tmp_path/'fit'
    args=['--multibank-index',str(index),'--output',str(out),'--train-data',str(data),'--epochs','1','--checkpoint-batch','4','--sequences-per-checkpoint','2','--gpt-microbatch','1','--replay-pool-sequences','32','--lr','2e-5','--audit-states','0','--audit-sequences','1','--audit-every-epochs','1','--convergence','--minimum-epochs','1','--eager-chisel','--device','cpu','--no-plots','--chisel-model-dim','128','--chisel-layers','1','--gpt-precision','fp32','--objective','tokenwise-v1','--kl-tolerance','.001','--token-tail-tolerance','.002','--token-normalization-scale','.01','--augmented-rho','.03','--ranking-weight','.4','--ranking-k','100','--ranking-tie-policy','exact_teacher_ties']
    main(args)
    state=torch.load(out/'controller_latest.pt',weights_only=False)
    assert state['updates']==2 and state['primal_dual']['updates']==2
    assert set(state['primal_dual']['multipliers'])=={1,2,3,4,6,7,8,9}
    assert state['metadata']['controller_pretraining'] and not state['metadata']['same_run_state_replay']
    rows=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
    train=[r for r in rows if r['event']=='train_state']
    assert {r['bank_seed'] for r in train}=={0,1}
    assert all(r['gpt_completed_steps']==(r['state_id']-1)%5+1 for r in train)
    assert all(r['ranking_k']==16 and 'strict_topk_changes' in r and 'top10_order_changes' not in r for r in train)
    heldout=[r for r in rows if r['event']=='audit_state' and r['role']=='heldout_states']
    assert {r['state_id'] for r in heldout}=={5,10}
    assert all(r['dual_multiplier']==0 for r in heldout)
    summaries=[r for r in rows if r['event']=='audit_summary']
    assert all(set(r['per_seed'])=={'0','1'} for r in summaries)
