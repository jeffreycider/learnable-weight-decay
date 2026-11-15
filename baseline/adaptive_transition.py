"""Explicit grouped Chisel transition; no precision conversion or GPT calls.

Input is a genuine baseline rank checkpoint (format_version=1), already converted
by the separate FP32 transition. The chosen private bank is bound to exactly its
original teaching state, never transferred to a different prefix by shape alone.
"""
import copy
import hashlib
import math
from pathlib import Path
import torch

FAMILY = 'adaptive_grouped_lag_one_v1'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def assert_activation(actual, original):
    if actual.keys()!=original.keys():raise ValueError('activation model inventory differs')
    for n,v in actual.items():
        expected=original[n].to(device=v.device,dtype=torch.float32) if original[n].is_floating_point() else original[n].to(v.device)
        if v.dtype!=expected.dtype or v.shape!=expected.shape or not torch.equal(v,expected):
            raise ValueError('activation must equal selected teaching state after explicit FP32 conversion: '+n)


def validate_adaptive_config(config):
    c=config['transformer_chisel']
    if c.get('family')!=FAMILY:raise ValueError('explicit adaptive family required')
    world=config.get('world_size')
    if world not in (1,2):raise ValueError('adaptive staging supports single-rank or two-rank CPU/gloo only')
    if world==2 and c.get('distributed_qualification')!='cpu_gloo_two_rank_staging':
        raise ValueError('two-rank CPU/gloo staging must be explicitly selected')
    if config.get('gpt_precision')!='fp32' or config.get('compile',True) or config.get('state_bank'):
        raise ValueError('adaptive adapter requires eager FP32 GPT without capture writer')
    if c.get('parameter_view')!='unwrapped_stored_fp32' or c.get('actor_execution')!={'backend':'eager'}:
        raise ValueError('unwrapped FP32/eager policy required; lookahead unsupported')
    if c.get('online_auxiliary')!='none' or c.get('replay_maintenance') or c.get('norm_auxiliary'):
        raise ValueError('adaptive adapter implements only ordinary-next-task credit')
    if c.get('bootstrap')!='teaching_snapshot_at_activation' or type(c.get('prefix_steps')) is not int or not 1<=c['prefix_steps']<config['max_steps']:
        raise ValueError('invalid adaptive activation clock')
    if c.get('case_id')!=c['prefix_steps']:raise ValueError('selected private case cannot transfer to another prefix')
    for key in ('head_lr','body_lr','private_lr','shared_eps','private_eps','shared_clip','private_clip','radius_fraction'):
        if not math.isfinite(c[key]) or c[key]<=0:raise ValueError('positive finite '+key+' required')
    if c.get('betas')!=[.9,.99]:raise ValueError('explicit supported Adam betas required')
    if not isinstance(c.get('run_id'),str) or not c['run_id']:raise ValueError('run identity required')
    # No implicit precision conversion: it must already have happened upstream.
    from rewriter.gpt_precision import RMS_EPSILON
    conversion=config.get('fp32_prefix_transition',{})
    if conversion.get('completed_steps')!=c['prefix_steps'] or conversion.get('rms_epsilon')!=RMS_EPSILON or conversion.get('tf32') is not False:
        raise ValueError('explicit corrected FP32 prefix transition required')


def load_sources(settings,device='cpu'):
    for key in ('policy','teaching_state'):
        if sha256(settings[key+'_path'])!=settings[key+'_sha256']:raise ValueError(key+' SHA256 mismatch')
    policy=torch.load(settings['policy_path'],map_location=device,weights_only=False,mmap=True)
    state=torch.load(settings['teaching_state_path'],map_location=device,weights_only=False,mmap=True)
    cases=[c for c in policy['metadata']['cases'] if c['state_id']==settings['case_id']]
    if len(cases)!=1:raise ValueError('exactly one selected private case required')
    case=cases[0]
    if case['sha256']['state']!=settings['teaching_state_sha256'] or state['completed_steps']!=settings['prefix_steps']:
        raise ValueError('selected private case/state/activation identity mismatch')
    private=policy['private'][settings['case_id']]
    if private['_extra_state']!=case['private_identity']:raise ValueError('private metadata identity mismatch')
    key=tuple(private['_extra_state']['state_key'])
    if key!=('cpu-teaching',0,settings['case_id'],settings['teaching_state_sha256']):
        raise ValueError('private source state key differs')
    if policy['actor']['_extra_state']!=settings['architecture']:raise ValueError('shared architecture differs')
    return policy,state,case


def build_transition(baseline, *, policy_path, teaching_state_path, case_id, settings, mode):
    if mode not in ('shadow','learned'):raise ValueError('shadow or learned required')
    required={'format_version','model','optimizers','controller','data_cursor','config','step','counters','elapsed_seconds','rng'}
    if not required<=baseline.keys() or baseline['format_version']!=1 or not baseline['optimizers']:
        raise ValueError('genuine resumable baseline rank checkpoint required')
    if baseline['controller'] is not None or baseline['config'].get('mode')!='baseline':raise ValueError('baseline-only transition required')
    rank=baseline['data_cursor'].get('rank')
    if baseline['data_cursor'].get('batches')!=baseline['step'] or type(rank) is not int or not 0<=rank<baseline['config'].get('world_size',0):
        raise ValueError('baseline data cursor/clock/rank mismatch')
    c=copy.deepcopy(settings)
    c.update(family=FAMILY,case_id=case_id,prefix_steps=baseline['step'],policy_path=str(Path(policy_path).resolve()),
        teaching_state_path=str(Path(teaching_state_path).resolve()),policy_sha256=sha256(policy_path),teaching_state_sha256=sha256(teaching_state_path))
    policy=torch.load(policy_path,map_location='cpu',weights_only=False,mmap=True)
    c['architecture']=copy.deepcopy(policy['actor']['_extra_state'])
    config=copy.deepcopy(baseline['config']);config.update(mode=mode,transformer_chisel=c)
    validate_adaptive_config(config)
    _,state,_=load_sources(c)
    assert_activation(baseline['model'],state['model'])
    result=dict(baseline);result['config']=config
    result['controller']=dict(schema=FAMILY,bootstrap=True,mode=mode,settings=copy.deepcopy(c),completed_steps=baseline['step'],rank=rank)
    # GPT weights and optimizer/data/RNG remain untouched. Adapter load commits
    # the initial learned action after baseline.restore_checkpoint restores them.
    return result


def main():
    import argparse,json
    from .checkpoint import save_atomic
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('baseline','policy','teaching-state','settings','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--case-id',type=int,required=True);p.add_argument('--mode',choices=('shadow','learned'),required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    baseline=torch.load(a.baseline,map_location='cpu',weights_only=False,mmap=True)
    out=build_transition(baseline,policy_path=a.policy,teaching_state_path=a.teaching_state,case_id=a.case_id,
                         settings=dict(json.loads(a.settings.read_text()),source_baseline_path=str(a.baseline.resolve()),source_baseline_sha256=sha256(a.baseline)),mode=a.mode)
    save_atomic(out,a.output)


if __name__=='__main__':main()
