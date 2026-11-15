"""Frozen full-rank gauge control, outside Chisel's action space; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline.model import GPT
from rewriter.edit_scope import editable_names, editable_weight_norm_sq
from rewriter.exact_gauges import uniform_attention_group_control,single_channel_binary_attention_control
from rewriter.algebra import Factors
from rewriter.function_preservation import FunctionalPreservation
from rewriter.gpt_precision import precision_metadata
from rewriter.grouped_norm import GroupedNormObjective
from rewriter.state_bank import StateBankReader
from rewriter.teacher_cache import TeacherCacheReader
from scripts.train_multistate_chisel import evaluate_sequences
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual,tokenwise_panel


@torch.no_grad()
def rank_readout(engine,replacements,panel,sequence_count,*,backend='tiled_tie_fp64_v1'):
    """Actual finite logits, cached teacher head, no backward or tolerance edits."""
    if sequence_count<1 or sequence_count>len(panel.inputs) or sequence_count%engine.microbatch_sequences:
        raise ValueError('requested sequences must be complete original teacher microbatches')
    config=TokenwiseConfig(mean_tolerance=0.,tail_tolerance=0.,scale=1.,rho=1.,
        ranking_k=100,ranking_weight=0.,ranking_tie_policy='exact_teacher_ties',loss_backend=backend)
    dual=TokenwiseDual([0],config=config,dual_lr=.01)
    result=tokenwise_panel(engine,replacements,torch.zeros((),device=engine.device),panel.inputs[:sequence_count],
        config=config,dual=dual,state_id=0,teacher_provider=lambda a,b:panel.teacher_batch(engine,a,b),backward=False,return_ce=True)
    return dict(sequence_ids=list(panel.sequence_ids[:sequence_count]),mean_kl=result['mean_kl'],
        max_token_kl=result['token_kl_quantiles']['max'],min_token_kl=result['token_kl_quantiles']['min'],
        tokens=result['token_count'],ranking_k=result['ranking_k'],
        literal_argmax_changes=result['argmax_changes'],literal_top100_order_changes=result['topk_order_changes'],
        literal_top100_membership_changes=result['topk_membership_changes'],
        **{name:result[name] for name in ('tie_aware_argmax_changes','strict_order_changes','strict_membership_changes','strict_topk_changes')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-bank',type=Path,required=True)
    p.add_argument('--teacher-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--state-id',type=int,default=999)
    p.add_argument('--value-scale',type=float,default=2.**-.5)
    p.add_argument('--rank-metrics',action='store_true')
    p.add_argument('--train-sequences',type=int,default=16)
    p.add_argument('--audit-sequences',type=int,default=64)
    p.add_argument('--wandb',action='store_true')
    p.add_argument('--single-channel-binary',action='store_true')
    p.add_argument('--include-projection',action='store_true',
                   help='Also evaluate its Frobenius projection into saved anchored action space; no fitting')
    args=p.parse_args();torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();device=torch.device('cuda')
    reader=StateBankReader(args.state_bank,verify_once=True,mmap_model=True)
    cache=TeacherCacheReader(args.teacher_cache,verify_once=True,mmap=True)
    record=reader.read(args.state_id,model_device='cpu',device=device)
    model=GPT(**reader.manifest['provenance']['model_config']).to(device).requires_grad_(False)
    engine=FunctionalPreservation(model,microbatch_sequences=16,teacher_cache_device=device,
                                  kl_dtype=torch.float64,gpt_precision='fp32')
    cache.load_engine(engine,record)
    control=(single_channel_binary_attention_control(engine.parameters,model) if args.single_channel_binary
             else uniform_attention_group_control(engine.parameters,model,value_scale=args.value_scale))
    replacements={**engine.parameters,**control.replacements}
    grouped=GroupedNormObjective(engine.parameters,zero_reference_policy='freeze')
    norm=grouped.evaluate(replacements)
    norm_metrics=norm.metrics()
    weight_norm=editable_weight_norm_sq(engine.parameters,editable_names(engine.parameters,'grouped')).sqrt()
    sample_name=next(n for n in control.replacements if n.endswith('attn.v.weight'))
    sample_rank=int(torch.linalg.matrix_rank(control.replacements[sample_name]-engine.parameters[sample_name]))
    probe_rank=next(iter(record.snapshot.matrices.values())).anchors.A.shape[1]
    metadata=dict(state_id=args.state_id,state_sha256=record.assert_verified(),
                  precision=precision_metadata('fp32'),control=control.diagnostics,
                  norm_metrics=norm_metrics,relative_whole_model_edit_norm=float(norm.edit_norm_sq.sqrt()/weight_norm),
                  no_parameter_updates=True,no_policy_selection=True,
                  rank_witness=dict(parameter=sample_name,edit_rank=sample_rank,
                                    anchored_rank_bound=2*probe_rank,
                                    exceeds_anchored_rank_bound=sample_rank>2*probe_rank),
                  teacher_manifest_sha256=hashlib.sha256((args.teacher_cache/'manifest.json').read_bytes()).hexdigest())
    root=Path(__file__).resolve().parents[1]
    metadata['source_sha256']={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [Path(__file__).resolve(),root/'rewriter/exact_gauges.py',root/'rewriter/gpt_precision.py',
                  root/'rewriter/function_preservation.py',root/'rewriter/grouped_norm.py']}
    if args.rank_metrics:
        metadata['source_sha256'].update({str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [root/'rewriter/tokenwise_reference.py',root/'rewriter/tokenwise_tiled_tie.py',root/'rewriter/rank_preservation.py',root/'rewriter/teacher_cache.py']})
        metadata['readout_protocol']=dict(train_sequences=args.train_sequences,audit_sequences=args.audit_sequences,
            baseline='exact zero edit evaluated on each identical panel',loss_backend='tiled_tie_fp64_v1',
            no_fitting=True,no_query_feedback=True,strict_teacher_ties='exact; no tolerance added')
    (args.output/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    rows=[]
    for role in ('train','audit'):
        panel=cache.read(args.state_id,role)
        if args.rank_metrics:
            count=args.train_sequences if role=='train' else args.audit_sequences
            for name,parameters in [('zero_edit',engine.parameters),('single_channel_binary_gauge' if args.single_channel_binary else 'uniform_attention_gauge',replacements)]:
                rows.append(dict(role=role,arm=name,**rank_readout(engine,parameters,panel,count)))
            continue
        if role=='train':
            zero=engine.student_parameter_loss(engine.parameters,panel.teacher_batch(engine,0,16))
            if float(zero)!=0:raise AssertionError('matching FP32 teacher identity must have zero KL')
        kl,ce=evaluate_sequences(engine,replacements,panel.inputs,
            teacher_provider=lambda start,stop:panel.teacher_batch(engine,start,stop),return_ce=True)
        rows.append(dict(role=role,sequence_ids=panel.sequence_ids,mean_kl=kl,**ce))
    counts=engine.compute.as_dict();assert counts['student_backwards']==0
    counts['cached_teacher_head_forwards']=getattr(engine,'cached_teacher_head_forwards',0)
    summary=dict(**metadata,readouts=rows,compute=counts,elapsed_seconds=time.perf_counter()-started)
    if args.include_projection:
        projected = dict(engine.parameters)
        squared_error = squared_original = 0.
        for name, edited in control.replacements.items():
            base = engine.parameters[name]
            delta = edited - base
            if base.ndim == 2:
                anchors = record.snapshot.matrices[name].anchors
                projected_delta = Factors.project_dense(delta, anchors).dense()
            else:
                projected_delta = delta  # All these biases are editable coordinates.
            projected[name] = base + projected_delta
            squared_error += float((delta.double() - projected_delta.double()).square().sum())
            squared_original += float(delta.double().square().sum())
        projected_rows = []
        for role in ('train', 'audit'):
            panel = cache.read(args.state_id, role)
            kl, ce = evaluate_sequences(engine, projected, panel.inputs,
                teacher_provider=lambda start, stop: panel.teacher_batch(engine, start, stop), return_ce=True)
            projected_rows.append(dict(role=role, sequence_ids=panel.sequence_ids, mean_kl=kl, **ce))
        summary['projected_control'] = dict(
            norm_metrics=grouped.evaluate(projected).metrics(), readouts=projected_rows,
            relative_projection_error=(squared_error / squared_original)**.5,
            interpretation='One projected gauge, not an optimum or a bound on learned anchored edits',
            no_fitting=True, vector_edits='Retained exactly; matrices projected independently')
        summary['compute'] = engine.compute.as_dict()
        summary['compute']['cached_teacher_head_forwards'] = getattr(engine, 'cached_teacher_head_forwards', 0)
        summary['elapsed_seconds'] = time.perf_counter() - started
        summary['source_sha256']['rewriter/algebra.py'] = hashlib.sha256((root/'rewriter/algebra.py').read_bytes()).hexdigest()
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if args.wandb:
        import wandb
        w=wandb.init(project='chisel',mode='offline',dir=str(args.output),name=args.output.name,config=metadata)
        for row in rows:w.log({f"{row['role']}/{row.get('arm','gauge')}/{k}":v for k,v in row.items() if isinstance(v,(int,float))})
        (args.output/'wandb_path.txt').write_text(str(Path(w.dir).parent)+'\n');w.finish()
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
