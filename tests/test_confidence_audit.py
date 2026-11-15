import json
import struct
import numpy as np
import pytest
import torch
from baseline.model import GPT
from rewriter.function_preservation import FunctionalPreservation
from rewriter.confidence_audit import token_kl_and_ce,sequence_summary,bootstrap_mean_interval,population_summary,pooled_quantiles,POPULATIONS


def _engine():
    torch.manual_seed(5)
    model=GPT(16,1,128)
    return FunctionalPreservation(model,microbatch_sequences=2,kl_dtype=torch.float64)


def test_token_kl_mean_matches_production_chunked_kl_and_ce_readout():
    engine=_engine()
    inputs=torch.randint(0,16,(2,6))
    teacher=engine.cache_teacher(inputs,data_role='past_training')[0]
    edited={n:(v+.05*torch.randn_like(v)).to(v.dtype) if n=='proj.weight' else v for n,v in engine.parameters.items()}
    measured=token_kl_and_ce(engine,edited,teacher)
    with torch.no_grad():
        loss,ce=engine.student_parameter_loss(edited,teacher,return_ce=True)
    assert measured['kl'].shape==(2,6) and measured['student_nll'].shape==(2,5)
    assert measured['parity_abs_difference']<1e-12
    assert float(measured['kl'].mean())==pytest.approx(float(loss),abs=1e-12)
    assert measured['production_mean_kl']==pytest.approx(float(loss),abs=1e-15)
    assert float(measured['student_nll'].mean())==pytest.approx(ce['student_ce'],abs=1e-12)
    assert float(measured['teacher_nll'].mean())==pytest.approx(ce['teacher_ce'],abs=1e-12)
    assert (measured['kl']>=-1e-15).all()
    identity=token_kl_and_ce(engine,dict(engine.parameters),teacher)
    assert float(identity['kl'].abs().max())<1e-12
    assert engine.compute.student_backwards==0


def test_token_kl_rejects_foreign_teacher_and_fp32_cache():
    engine=_engine();other=_engine()
    inputs=torch.randint(0,16,(1,4))
    foreign=other.cache_teacher(inputs,data_role='past_training')[0]
    with pytest.raises(ValueError,match='different frozen checkpoint'):token_kl_and_ce(engine,dict(engine.parameters),foreign)
    fp32=FunctionalPreservation(GPT(16,1,128),microbatch_sequences=1,kl_dtype=torch.float32)
    teacher=fp32.cache_teacher(inputs,data_role='past_training')[0]
    with pytest.raises(ValueError,match='FP64'):token_kl_and_ce(fp32,dict(fp32.parameters),teacher)


def test_sequence_summary_reports_tails_and_deterministic_bootstrap():
    kl=torch.tensor([[1e-7]*4,[2e-7]*4,[3e-6]*4],dtype=torch.float64)
    nll_s=torch.ones(3,3,dtype=torch.float64);nll_t=nll_s-torch.tensor([[.1],[0.],[-.1]],dtype=torch.float64)
    summary=sequence_summary(kl,nll_s,nll_t,tolerance=1e-6,bootstrap_seed=1)
    assert summary['sequences']==3 and summary['tokens']==12 and summary['ce_tokens']==9
    assert 'first L-1 positions' in summary['ce_scope']
    assert summary['mean_kl']==pytest.approx((1e-7+2e-7+3e-6)/3)
    assert summary['sequence_kl_quantiles']['p100']==pytest.approx(3e-6)
    assert summary['fraction_sequences_above_tolerance']==pytest.approx(1/3)
    assert summary['fraction_tokens_above_tolerance']==pytest.approx(1/3)
    assert summary['fraction_sequences_ce_worse']==pytest.approx(1/3)
    assert summary['ce_change']==pytest.approx(0.,abs=1e-12)
    again=sequence_summary(kl,nll_s,nll_t,tolerance=1e-6,bootstrap_seed=1)
    assert again['bootstrap_mean_kl_95']==summary['bootstrap_mean_kl_95']
    interval=bootstrap_mean_interval(torch.tensor([1.,1.,1.,1.]),seed=0)
    assert interval['low']==interval['high']==1.
    with pytest.raises(ValueError):bootstrap_mean_interval(torch.tensor([1.]))


def test_population_summary_is_descriptive_only():
    def row(state,kl,shrink):
        return dict(state_id=state,population='fitting_withheld',mean_group_l2_percentage_reduction=shrink,
                    summary=dict(mean_kl=kl,per_sequence_kl=[kl*.5,kl*1.5],sequences=2,tokens=8,ce_change=0.,
                                 sequence_kl_quantiles={'p99':kl*1.5},token_kl_quantiles={'p99.9':kl*3}))
    summary=population_summary([row(6,4e-6,.4),row(40,1e-6,.3)],tolerance=1e-6)
    assert summary['states']==2 and summary['states_with_mean_kl_at_or_below_tolerance']==1
    assert summary['pooled_token_kl_quantiles'] is None and summary['maximum_state_token_kl_quantiles']=={'p99.9':1.2e-5}
    pooled=np.array([1e-7]*15+[5e-6],dtype=np.float32)
    with_tokens=population_summary([row(6,4e-6,.4),row(40,1e-6,.3)],tolerance=1e-6,pooled_tokens=pooled)
    assert with_tokens['pooled_token_kl_quantiles']['p100']==pytest.approx(5e-6) and with_tokens['pooled_token_kl_quantiles']['p50']==pytest.approx(1e-7)
    with pytest.raises(ValueError,match='pooled token count'):population_summary([row(6,4e-6,.4)],tolerance=1e-6,pooled_tokens=pooled)
    assert summary['early_states']==[dict(state_id=6,mean_kl=4e-6)]
    assert summary['worst_states'][0]['state_id']==6
    assert 'pass' not in json.dumps(summary).lower()
    assert set(POPULATIONS)=={'fitting_seen','fitting_withheld','checkpoint_heldout_development','fresh_next_batch'}


def test_pooled_quantiles_handle_more_than_torch_quantile_limit_exactly():
    n=2**24+5  # above the documented torch.quantile input limit on some builds/devices; production pools are ~1e8 tokens
    values=np.linspace(0.,1.,n,dtype=np.float32)
    q=pooled_quantiles(values)
    assert q['p50']==pytest.approx(.5,abs=1e-6) and q['p100']==pytest.approx(1.) and q['p99.9']==pytest.approx(.999,abs=1e-6)
    assert pooled_quantiles(torch.tensor([3.,1.,2.]))['p50']==2.
    with pytest.raises(ValueError):pooled_quantiles(np.empty(0))


def test_compare_edits_separates_attenuation_from_direction_change():
    from rewriter.confidence_audit import compare_edits
    base={'a.weight':torch.zeros(2,2),'a.bias':torch.zeros(2),'frozen':torch.ones(2)}
    reference={'a.weight':torch.tensor([[1.,0.],[0.,0.]]),'a.bias':torch.tensor([0.,2.]),'frozen':torch.ones(2)}
    attenuated={'a.weight':.5*reference['a.weight'],'a.bias':.5*reference['a.bias'],'frozen':torch.ones(2)}
    groups={'w':('a.weight',),'b':('a.bias',),'f':('frozen',)};editable={'a.weight','a.bias'}
    out=compare_edits(attenuated,reference,base,groups,editable)
    assert out['overall_cosine']==pytest.approx(1.) and out['overall_amplitude_ratio']==pytest.approx(.5)
    assert out['groups']['w']['amplitude_ratio']==pytest.approx(.5) and out['groups']['f']['cosine'] is None
    rotated={'a.weight':torch.tensor([[0.,1.],[0.,0.]]),'a.bias':torch.tensor([2.,0.]),'frozen':torch.ones(2)}
    out=compare_edits(rotated,reference,base,groups,editable)
    assert out['overall_cosine']==pytest.approx(0.,abs=1e-12) and out['overall_amplitude_ratio']==pytest.approx(1.)


def _tiny_fit(tmp_path):
    from test_multistate_chisel import _tiny_state_bank
    from scripts.train_multistate_chisel import main,select_sequences,stable_seed,state_replay
    from rewriter.state_bank import StateBankReader
    from rewriter.teacher_cache import write_panel
    bank=tmp_path/'bank';cfg=_tiny_state_bank(bank)
    data=tmp_path/'fineweb_train_000001.bin';tokens=(np.arange(1000)%16).astype(np.uint16)
    data.write_bytes(struct.pack('<iii',20240520,1,len(tokens))+bytes(1012)+tokens.tobytes())
    reader=StateBankReader(bank);cache=tmp_path/'targets';engine=FunctionalPreservation(GPT(**cfg['model']),microbatch_sequences=1,kl_dtype=torch.float64)
    names=set(engine.parameters);panel_rows=[]
    for state_id in range(1,6):
        record=reader.read(state_id)
        engine.load_teacher_state({n:v for n,v in record.model.items() if n in names},{n:v for n,v in record.model.items() if n not in names})
        pools=state_replay([str(data)],state_id,cfg,pool_sequences=32,seed=400001,epoch=0,split_seed=500001)
        for role,count in (('train',4),('audit',2)):
            inputs,ids=select_sequences(pools[role],count,seed=stable_seed('confidence-test',state_id,role))
            panel_rows.append(write_panel(cache,state_id,role,engine,inputs,ids,state_sha256=reader.entries[state_id-1]['sha256'],execution={'test':True}))
    (cache/'manifest.json').write_text(json.dumps({'complete':True,'args':{'states':5},'panels':panel_rows,'purpose':'tiny confidence audit test'}))
    output=tmp_path/'run'
    main(['--state-bank',str(bank),'--teacher-cache',str(cache),'--output',str(output),'--train-data',str(data),
          '--epochs','1','--checkpoint-batch','2','--sequences-per-checkpoint','2','--gpt-microbatch','1',
          '--replay-pool-sequences','32','--lr','.0001','--kl-budget','.1','--kl-tolerance','1e-4','--expected-gpt-seed','7',
          '--audit-states','1','--audit-sequences','2','--audit-every','999','--checkpoint-every','1',
          '--eager-chisel','--device','cpu','--no-plots'])
    return bank,cache,data,output/'controller_latest.pt'


def test_phase_one_all_fitting_seen_only_and_sha_pins(tmp_path):
    import hashlib
    from scripts.evaluate_chisel_confidence import main
    bank,cache,data,checkpoint=_tiny_fit(tmp_path)
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output=tmp_path/'phase1'
    main(['--checkpoint',str(checkpoint),'--checkpoint-sha256',sha.upper(),'--state-bank',str(bank),'--teacher-cache',str(cache),'--train-data',str(data),
          '--output',str(output),'--all-fitting-states','--no-fitting-withheld','--heldout-states','--device','cpu','--eager-chisel'])
    rows=[json.loads(l) for l in (output/'states.jsonl').read_text().splitlines()]
    assert {r['population'] for r in rows}=={'fitting_seen'} and sorted(r['state_id'] for r in rows)==[1,2,3,4]
    summary=json.loads((output/'summary.json').read_text())
    assert set(summary['populations'])=={'fitting_seen'} and summary['populations']['fitting_seen']['states']==4
    assert summary['config']['checkpoint_sha256_pinned'] is True and summary['config']['all_fitting_states'] is True
    assert summary['compute']['student_tokens']==sum(r['summary']['tokens'] for r in rows)==4*4*4 and summary['compute']['student_forwards']==16
    assert summary['compute']['cached_teacher_head_forwards']==16 and summary['compute']['teacher_forwards']==0
    assert summary['config']['fitting_withheld'] is False and summary['config']['heldout_states']==[] and summary['config']['fresh_state'] is None
    with pytest.raises(ValueError,match='SHA256'):
        main(['--checkpoint',str(checkpoint),'--checkpoint-sha256','00'*32,'--state-bank',str(bank),'--teacher-cache',str(cache),'--train-data',str(data),
              '--output',str(tmp_path/'phase1b'),'--all-fitting-states','--no-fitting-withheld','--heldout-states','--device','cpu','--eager-chisel'])
    with pytest.raises(ValueError,match='no audit items'):
        main(['--checkpoint',str(checkpoint),'--state-bank',str(bank),'--teacher-cache',str(cache),'--train-data',str(data),
              '--output',str(tmp_path/'phase1c'),'--fitting-states','--heldout-states','--device','cpu','--eager-chisel'])


def test_fresh_batches_audit_consecutive_unseen_batches(tmp_path):
    from scripts.evaluate_chisel_confidence import main
    bank,cache,data,checkpoint=_tiny_fit(tmp_path)
    output=tmp_path/'fresh2'
    main(['--checkpoint',str(checkpoint),'--state-bank',str(bank),'--teacher-cache',str(cache),'--train-data',str(data),
          '--output',str(output),'--fitting-states','--heldout-states','--fresh-state','5','--fresh-batches','2','--fresh-sequences','4','--device','cpu','--eager-chisel'])
    rows=[json.loads(l) for l in (output/'states.jsonl').read_text().splitlines()]
    assert [r['fresh_batch_offset'] for r in rows]==[0,1] and all(r['population']=='fresh_next_batch' and r['state_id']==5 for r in rows)
    first,second=rows
    assert all(5*32<=i<6*32 for i in first['sequence_ids']) and all(6*32<=i<7*32 for i in second['sequence_ids'])
    summary=json.loads((output/'summary.json').read_text())
    assert summary['populations']['fresh_next_batch']['states']==2 and summary['config']['fresh_batches']==2
    assert summary['compute']['teacher_forwards']==8 and summary['compute']['student_tokens']==2*4*4


def test_confidence_audit_end_to_end_on_tiny_bank(tmp_path):
    from scripts.evaluate_chisel_confidence import main
    from scripts.train_multistate_chisel import heldout_sequence
    bank,cache,data,checkpoint=_tiny_fit(tmp_path)
    output=tmp_path/'audit'
    main(['--checkpoint',str(checkpoint),'--state-bank',str(bank),'--teacher-cache',str(cache),'--train-data',str(data),
          '--output',str(output),'--fitting-states','1','2','--heldout-states','5','--fresh-state','5','--fresh-sequences','4',
          '--device','cpu','--eager-chisel','--reference-checkpoint',str(checkpoint)])
    rows=[json.loads(l) for l in (output/'states.jsonl').read_text().splitlines()]
    summary=json.loads((output/'summary.json').read_text())
    assert summary['compute']['student_backwards']==0 and summary['compute']['student_forwards']>0
    assert len(rows)==2*2+1+1
    # accounting accumulates across checkpoint loads: every row's tokens are counted exactly once
    assert summary['compute']['student_tokens']==sum(r['summary']['tokens'] for r in rows)
    assert summary['compute']['student_forwards']==sum(-(-r['summary']['sequences']//1) for r in rows)  # microbatch 1 in the tiny cache
    bypop=summary['compute_by_population']
    assert set(bypop)=={'fitting_seen','fitting_withheld','checkpoint_heldout_development','fresh_next_batch'}
    assert bypop['fresh_next_batch']['teacher_forwards']==4 and bypop['fresh_next_batch']['cached_teacher_head_forwards']==0
    assert all(bypop[p]['teacher_forwards']==0 and bypop[p]['cached_teacher_head_forwards']==bypop[p]['student_forwards']>0 for p in ('fitting_seen','fitting_withheld','checkpoint_heldout_development'))
    assert summary['compute']['teacher_forwards']==4 and summary['compute']['cached_teacher_head_tokens']+summary['compute']['teacher_tokens']==summary['compute']['student_tokens']
    by=lambda population:[r for r in rows if r['population']==population]
    assert {r['state_id'] for r in by('fitting_seen')}=={1,2} and all(len(r['sequence_ids'])==4 for r in by('fitting_seen'))
    assert all(not heldout_sequence(i,500001) for r in by('fitting_seen') for i in r['sequence_ids'])
    assert all(heldout_sequence(i,500001) for p in ('fitting_withheld','checkpoint_heldout_development') for r in by(p) for i in r['sequence_ids'])
    fresh=by('fresh_next_batch')[0]
    assert fresh['state_id']==5 and len(fresh['sequence_ids'])==4 and all(i>=5*32 for i in fresh['sequence_ids'])
    assert fresh['sequence_roles']==['audit' if heldout_sequence(i,500001) else 'train' for i in fresh['sequence_ids']]
    for r in rows:
        assert r['parity_abs_difference']<1e-9 and r['read_only_unchanged']
        assert len(r['summary']['per_sequence_kl'])==r['summary']['sequences']
        assert r['population_definition']==POPULATIONS[r['population']]
        assert 'scale' not in r
        if r['realized_edit_norm']>0:
            assert r['reference_comparison']['overall_cosine']==pytest.approx(1.) and r['reference_comparison']['overall_amplitude_ratio']==pytest.approx(1.)
    assert any(r['summary']['mean_kl']>0 for r in rows)
    assert set(summary['populations'])=={'fitting_seen','fitting_withheld','checkpoint_heldout_development','fresh_next_batch'}
    assert summary['populations']['checkpoint_heldout_development']['states']==1
    for p in summary['populations'].values():
        assert p['ce_tokens']==p['sequences']*3 and p['tokens']==p['sequences']*4 and 'KL over all L positions' in p['ce_scope']
    assert summary['populations']['fitting_seen']['reference_comparison']['overall_cosine']['min']==pytest.approx(1.)
    assert summary['fresh_sequence_roles']['train']+summary['fresh_sequence_roles']['audit']==4
    assert json.loads((output/'config.json').read_text())['reference_checkpoint']['update']==2
    for population,p in summary['populations'].items():
        pool=np.load(output/'token_pools'/f'{population}_rank0.npy')
        assert pool.dtype==np.float32 and pool.size==p['tokens']
        manifest=json.loads((output/'token_pools'/'manifest_rank0.json').read_text())
        assert manifest['world_size']==1 and manifest['counts'][population]==p['tokens']
        assert p['pooled_token_kl_quantiles']['p50']==pytest.approx(float(np.quantile(pool.astype(np.float64),.5)))
        tokens=torch.tensor([v for r in by(population) for v in r['summary']['per_sequence_kl']])
        assert p['pooled_token_kl_quantiles']['p100']>=float(tokens.max()) and p['pooled_token_kl_quantiles']['p100']==pytest.approx(p['maximum_state_token_kl_quantiles']['p100'],rel=1e-6)
    with pytest.raises(ValueError,match='final bank checkpoint'):
        main(['--checkpoint',str(checkpoint),'--state-bank',str(bank),'--teacher-cache',str(cache),'--train-data',str(data),
              '--output',str(tmp_path/'audit2'),'--fitting-states','1','--heldout-states','5','--fresh-state','4','--device','cpu','--eager-chisel'])
    text=(output/'summary.md').read_text()
    assert 'development' in text and 'pass' not in text.lower()
    assert json.loads((output/'config.json').read_text())['policy_selection'] is False


def test_stage_comparison_restricts_to_the_stage_panel_set():
    from scripts.compare_confidence_with_stage_audit import compare
    rows=[dict(state_id=s,population='fitting_seen',mean_group_l2_percentage_reduction=.1,summary=dict(mean_kl=k)) for s,k in [(1,1e-7),(33,3e-7),(66,5e-7),(99,9e-7)]]
    out=compare(rows,[1,33,66],3e-7,5e-7)
    assert out['states']==3 and out['confidence_mean_of_means']==pytest.approx(3e-7) and out['mean_relative_difference']==pytest.approx(0.)
    assert out['confidence_max']==pytest.approx(5e-7) and out['worst_state']==66
    with pytest.raises(ValueError,match='lack states'):compare(rows,[1,2],1e-7,1e-7)
