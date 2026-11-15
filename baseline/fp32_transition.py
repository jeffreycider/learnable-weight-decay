"""Explicit same-run captured-prefix to FP32 suffix conversion, never ordinary resume."""
import copy
import re
import json
from pathlib import Path
import torch
from rewriter.gpt_precision import RMS_EPSILON


def training_forward_count(config, completed_steps):
    """Count actual prescribed microbatch passes, retaining the prefix's old size."""
    per_step=config['batch_tokens']//(config['seq_len']*config['world_size']*config['microbatch_sequences'])
    transition=config.get('fp32_prefix_transition')
    if transition is None:return completed_steps*per_step
    if (transition.get('rms_epsilon')!=RMS_EPSILON
            or transition.get('suffix_gpt_dtype')!='float32'
            or transition.get('tf32') is not False):
        raise ValueError('FP32 suffix precision or normalization epsilon differs; reconvert the original prefix')
    prefix=transition['completed_steps']
    if config.get('gpt_precision')!='fp32' or completed_steps<prefix:
        raise ValueError('invalid FP32 suffix clock')
    counts=transition['prefix_counters']
    if counts['forward']!=counts['backward']:raise ValueError('prefix forward/backward mismatch')
    return counts['forward']+(completed_steps-prefix)*per_step


def promote_bundle(bundle, *, source_sha256, microbatch_sequences=16, max_steps=None, compile_model=True, training_shards=None):
    from scripts.bundle_baseline_checkpoint import FORMAT,extract_rank
    if bundle.get('format_version')!=FORMAT:raise ValueError('complete resumable rank bundle required')
    if not re.fullmatch('[0-9a-f]{64}',source_sha256):raise ValueError('source SHA256 required')
    original=bundle['shared'];prior=original['config'];step=original['step']
    if prior.get('mode')!='baseline' or original['controller'] is not None:
        raise ValueError('only an unedited baseline prefix can be promoted')
    if prior.get('gpt_precision','baseline')!='baseline' or prior.get('fp32_prefix_transition'):
        raise ValueError('prefix already has a precision transition')
    schedule=prior.get('schedule_steps',prior['max_steps']);end=schedule if max_steps is None else max_steps
    if not step<end<=schedule:raise ValueError('suffix must continue within original schedule')
    if not isinstance(microbatch_sequences,int) or microbatch_sequences<1 or prior['batch_tokens']%(prior['seq_len']*prior['world_size']*microbatch_sequences):
        raise ValueError('microbatch must divide the unchanged global batch')
    expected=training_forward_count(prior,step)
    if original['counters']!={'forward':expected,'backward':expected}:raise ValueError('prefix counters disagree')
    for rank in range(prior['world_size']):
        record=extract_rank(bundle,rank)
        if record['data_cursor']['batches']!=step or record['data_cursor']['rank']!=rank:
            raise ValueError('prefix rank cursor mismatch')
    original_names=bundle['ranks'][0]['data_cursor']['file_names']
    names=original_names if training_shards is None else [f'fineweb_train_{i:06d}.bin' for i in range(1,training_shards+1)]
    if names[:len(original_names)]!=original_names or any(r['data_cursor']['file_names']!=original_names for r in bundle['ranks'].values()):
        raise ValueError('training shard extension must preserve the exact original prefix order')
    pinned={r['path']:r for r in json.loads((Path(__file__).parent/'vendor/dataset_files.json').read_text())}
    if any(n not in pinned for n in names):raise ValueError('training shard not in pinned dataset')
    capacity=sum((((pinned[n]['size']-1024)//2)-2)//prior['batch_tokens'] for n in names)
    if capacity<end:raise ValueError('suffix needs an explicit extension of the pinned training shards')
    result=copy.deepcopy(bundle);shared=result['shared'];config=shared['config']
    config.pop('state_bank',None)
    config.update(gpt_precision='fp32',max_steps=end,schedule_steps=schedule,
                  microbatch_sequences=microbatch_sequences,compile=bool(compile_model),compile_loss=False)
    config['fp32_prefix_transition']=dict(schema=1,source_bundle_sha256=source_sha256,
        completed_steps=step,prefix_counters=copy.deepcopy(original['counters']),
        prefix_config=copy.deepcopy(prior),suffix_gpt_dtype='float32',tf32=False,
        rms_epsilon=RMS_EPSILON,optimizer_algebra='unchanged_PR323_including_BF16_Muon_NS',
        rationale='Explicit matched FP32 continuation of the same captured training trajectory')
    # Promote model values and optimizer moments, retaining integer tensors and
    # scalar Adam counters. No optimizer update, data read, or RNG draw occurs.
    def promote(value):
        if isinstance(value,torch.Tensor):
            return value.float() if value.dtype in (torch.bfloat16,torch.float16) else value
        if isinstance(value,dict):return {k:promote(v) for k,v in value.items()}
        if isinstance(value,list):return [promote(v) for v in value]
        if isinstance(value,tuple):return tuple(promote(v) for v in value)
        return value
    shared['model']=promote(shared['model']);shared['optimizers']=promote(shared['optimizers'])
    if any(v.is_floating_point() and v.dtype!=torch.float32 for v in shared['model'].values()):
        raise ValueError('model contains an unhandled floating dtype')
    if names!=original_names:
        config['fp32_prefix_transition']['training_shard_extension']=dict(original=original_names,extended=names,sha256={n:pinned[n]['lfs']['oid'] for n in names})
        for rank in result['ranks'].values():rank['data_cursor']['file_names']=list(names)
    result['precision_transition']=copy.deepcopy(config['fp32_prefix_transition'])
    return result
