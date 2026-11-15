"""CPU numerical evidence for matched saved-optimizer systems qualifications.

Keep these copies/reductions outside the timed update. Compare Adam deltas,
not large unchanged parameter baselines. Undefined/nonfinite metrics fail closed.
"""
import math

import torch


def aggregate_recorded_errors(records):
    """Reaggregate disjoint tensor errors without large unrelated denominators.

    Input records contain FP64 reference norms and absolute L2/max errors from
    the qualification export. No tensors or candidate norms are reconstructed;
    in particular these records cannot establish cosine or bit identity.
    """
    if not records:
        raise ValueError('nonempty recorded tensor inventory required')
    norms, errors, maxima = [], [], []
    for name, record in records.items():
        values = [record[k] for k in
                  ('reference_norm', 'absolute_l2_error', 'max_absolute_error')]
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('invalid recorded tensor error: '+name)
        norms.append(values[0]); errors.append(values[1]); maxima.append(values[2])
    reference = math.hypot(*norms)
    error = math.hypot(*errors)
    return dict(tensor_count=len(records), reference_norm=reference,
                absolute_l2_error=error, max_absolute_error=max(maxima),
                relative_error=error/reference if reference else (0. if not error else None))


def tensor_comparison(reference, candidate):
    if set(reference) != set(candidate):
        raise ValueError('tensor inventories differ')
    rr = cc = dd = dot = maximum = 0.
    exact = True
    for name in reference:
        a, b = reference[name], candidate[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('tensor shape/dtype differs: '+name)
        # torch.equal treats +0 and -0 as equal; the bit-identity field must not.
        bits_a = a.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        bits_b = b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        exact = exact and torch.equal(bits_a, bits_b)
        a, b = a.detach().cpu().double(), b.detach().cpu().double()
        if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
            return dict(finite=False, bit_identical=False, relative_error=None,
                        cosine=None, reference_norm=None, candidate_norm=None,
                        absolute_l2_error=None, max_absolute_error=None)
        d = b-a
        rr += float(a.square().sum()); cc += float(b.square().sum())
        dd += float(d.square().sum()); dot += float((a*b).sum())
        maximum = max(maximum, float(d.abs().max()) if d.numel() else 0.)
    rn, cn, dn = math.sqrt(rr), math.sqrt(cc), math.sqrt(dd)
    return dict(finite=True, bit_identical=exact, reference_norm=rn, candidate_norm=cn,
                absolute_l2_error=dn, max_absolute_error=maximum,
                relative_error=dn/rn if rn else (0. if not dn else None),
                cosine=dot/(rn*cn) if rn and cn else (1. if not dn else None))


def metric_within(metric, relative_limit, cosine_min=None):
    values = (metric.get('relative_error'), metric.get('cosine'))
    if not metric.get('finite') or any(v is None or not math.isfinite(v) for v in values):
        return False
    return values[0] <= relative_limit and (cosine_min is None or values[1] >= cosine_min)


def parameter_delta(parameters, initial):
    if set(parameters) != set(initial):
        raise ValueError('initial and updated parameter inventories differ')
    # Subtract in stored precision, matching the qualification's actual stored
    # weight update, then tensor_comparison measures it in FP64.
    return {n: value.detach().cpu()-initial[n].detach().cpu() for n, value in parameters.items()}


def optimizer_snapshot(optimizer, named_parameters):
    names = {id(p): n for n, p in named_parameters.items()}
    groups, states = [], {}
    for group in optimizer.param_groups:
        members = [names[id(p)] for p in group['params']]
        groups.append(members)
        for p, name in zip(group['params'], members):
            states[name] = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                            for k, v in optimizer.state[p].items()}
    return dict(groups=groups, states=states)


def compare_adam(reference, candidate):
    if reference['groups'] != candidate['groups']:
        raise ValueError('Adam groups differ')
    if set(reference['states']) != set(candidate['states']):
        raise ValueError('Adam state inventory differs')
    for name, state in reference['states'].items():
        other = candidate['states'][name]
        if set(state) != set(other) or set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('qualification requires standard Adam moments and step')
        if not torch.equal(torch.as_tensor(state['step']), torch.as_tensor(other['step'])):
            raise ValueError('Adam step counter differs: '+name)
    result = {'step_counters_exact': True}
    for key in ('exp_avg', 'exp_avg_sq'):
        def compare(names):
            return tensor_comparison({n: reference['states'][n][key] for n in names},
                                     {n: candidate['states'][n][key] for n in names})
        result[key] = dict(global_metric=compare(reference['states']),
                           groups=[compare(group) for group in reference['groups']])
    return result


def muon_snapshot(optimizer, named_parameters, *, rank, world_size):
    """Copy the nonempty saved momentum shard used by baseline Muon on this rank.

    Call outside timing. Snapshot RAW gradients separately BEFORE Muon.step:
    baseline muon_update mutates both momentum and .grad. This snapshot contains
    optimizer state, not a certificate for the resulting parameter update.
    """
    from baseline.optim import Muon
    if not isinstance(optimizer,Muon):
        raise TypeError('baseline Muon optimizer required')
    if (type(rank) is not int or type(world_size) is not int
            or world_size<1 or not 0<=rank<world_size):
        raise ValueError('valid rank/world_size required')
    names={id(p):n for n,p in named_parameters.items()}
    if len(names)!=len(named_parameters):
        raise ValueError('named parameters must be unique')
    groups=[];momentum={};all_members=[]
    for group in optimizer.param_groups:
        members=[names[id(p)] for p in group['params']]
        all_members.extend(members)
        # Exactly baseline.optim.Muon.step's base_i + rank schedule, using the
        # optimizer's existing sorted order rather than sorting names again.
        owned=members[rank::world_size]
        hp={k:float(group[k]) for k in ('lr','weight_decay','mu')}
        if not all(math.isfinite(v) for v in hp.values()):
            raise ValueError('nonfinite Muon hyperparameter')
        groups.append(dict(parameters=members,owned_parameters=owned,hyperparameters=hp))
        for index,(p,name) in enumerate(zip(group['params'],members)):
            state=optimizer.state.get(p,{})  # Snapshot must not create empty states.
            if index%world_size!=rank:
                if state:raise ValueError('Muon state belongs to another rank: '+name)
                continue
            if set(state)!={'momentum'}:
                raise ValueError('nonempty saved Muon momentum required: '+name)
            value=state['momentum']
            if not isinstance(value,torch.Tensor) or value.shape!=p.shape or value.dtype!=p.dtype:
                raise ValueError('Muon momentum shape/dtype differs: '+name)
            momentum[name]=value.detach().cpu().clone()
    if len(all_members)!=len(set(all_members)):
        raise ValueError('Muon parameter appears in multiple groups')
    return dict(rank=rank,world_size=world_size,groups=groups,momentum=momentum)


def compare_muon(reference,candidate):
    """Compare matching saved shards; no invented optimizer-response threshold."""
    if any(reference[k]!=candidate[k] for k in ('rank','world_size','groups')):
        raise ValueError('Muon ownership, parameter order, or hyperparameters differ')
    expected=[n for group in reference['groups'] for n in group['owned_parameters']]
    if (len(expected)!=len(set(expected)) or set(reference['momentum'])!=set(expected)
            or set(candidate['momentum'])!=set(expected)):
        raise ValueError('Muon owned momentum inventory differs')
    def compare(names):
        return tensor_comparison({n:reference['momentum'][n] for n in names},
                                 {n:candidate['momentum'][n] for n in names})
    return dict(rank=reference['rank'],world_size=reference['world_size'],
                owned_parameter_count=len(expected),global_metric=compare(expected),
                groups=[compare(g['owned_parameters']) for g in reference['groups']],
                per_parameter={n:compare([n]) for n in expected})


@torch.no_grad()
def logit_response_comparison(before, reference_after, candidate_after, chunk_tokens=128):
    """Compare FP32 finite responses without copying full vocabulary panels to CPU.

    Accumulate diagnostics in FP64 on the tensors' device. The response itself
    is subtracted in stored FP32, as in the original local qualification.
    Absolute logit RMS is reported separately from relative response error.
    """
    if any(v.shape != before.shape or v.dtype != torch.float32 or v.device != before.device
           for v in (before,reference_after,candidate_after)) or before.ndim < 2 or chunk_tokens < 1:
        raise ValueError('matching FP32 logits and positive chunk size required')
    arrays=[v.reshape(-1,v.shape[-1]) for v in (before,reference_after,candidate_after)]
    totals=torch.zeros(7,dtype=torch.float64,device=before.device)
    for start in range(0,len(arrays[0]),chunk_tokens):
        base,a,b=[v[start:start+chunk_tokens] for v in arrays]
        x,y=(a-base).double(),(b-base).double()
        d=y-x;absolute=b.double()-a.double()
        totals[:5].add_(torch.stack((x.square().sum(),y.square().sum(),d.square().sum(),
                                    (x*y).sum(),absolute.square().sum())))
        totals[5]=torch.maximum(totals[5],d.abs().max())
        totals[6]=torch.maximum(totals[6],absolute.abs().max())
    rr,cc,dd,dot,absolute,max_response,max_logit=totals.cpu().tolist()
    if not all(math.isfinite(v) for v in (rr,cc,dd,dot,absolute,max_response,max_logit)):
        return {'finite':False}
    rn,cn,dn=math.sqrt(rr),math.sqrt(cc),math.sqrt(dd)
    return dict(finite=True,reference_response_norm=rn,candidate_response_norm=cn,
                absolute_response_l2_error=dn,response_max_absolute_error=max_response,
                response_relative_error=dn/rn if rn else (0. if not dn else None),
                response_cosine=dot/(rn*cn) if rn and cn else (1. if not dn else None),
                absolute_logit_rms_error=math.sqrt(absolute/before.numel()),
                absolute_logit_max_error=max_logit,logit_elements=before.numel())
