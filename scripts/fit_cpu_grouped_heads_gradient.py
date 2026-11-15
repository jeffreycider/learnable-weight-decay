"""Fit actual grouped linear output heads from zero with frozen-feature gradients.

Fixed checkpoints and supplied preserving edits. This is neither norm/KL training
nor evidence of generalization. Cached linear maps include the physical action
scales; they give the same head derivatives as the frozen transformer path.
No least-squares coefficients, teacher logits, or GPT evaluations are used.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.fit_cpu_edit_teacher import sha


def objective(groups, coefficients):
    return sum(g['weight'] * (g['A'] @ coefficients[n] - g['b']).square().sum()
               for n, g in groups.items())


def merge_designs(datasets):
    """One head bank, equal checkpoint weighting, distinct checkpoint features."""
    if not datasets:
        raise ValueError('at least one design dataset required')
    names=set(datasets[0]['groups'])
    if any(set(d['groups'])!=names for d in datasets):
        raise ValueError('functional group inventories differ')
    merged={}
    for n in datasets[0]['groups']:
        rows=[];targets=[];width=datasets[0]['groups'][n]['A'].shape[1]
        for d in datasets:
            g=d['groups'][n]
            if g['A'].ndim!=2 or g['A'].shape[1]!=width or g['b'].shape!=(g['A'].shape[0],):
                raise ValueError('incompatible shared-head design')
            weight=float(g['weight'])
            if not math.isfinite(weight) or weight<=0:
                raise ValueError('positive finite design weight required')
            scale=math.sqrt(weight/len(datasets))
            rows.append(g['A']*scale);targets.append(g['b']*scale)
        merged[n]=dict(A=torch.cat(rows),b=torch.cat(targets),weight=1.)
    return merged


def verify_shared_bodies(datasets):
    """Reject accidental fitting of shared heads on different transformer bodies."""
    from scripts.fit_cpu_edit_teacher import exact_state_equal
    bodies=[];parents=[];states=[];architectures=[]
    for d in datasets:
        parent=d['checkpoint']
        if sha(parent)!=d['source_sha256'][parent]:raise ValueError('parent checkpoint changed')
        saved=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
        body={n:v for n,v in saved['actor'].items() if n!='_extra_state' and not n.startswith(('matrix_output.','vector_output.'))}
        bodies.append(body);parents.append(parent);states.append(saved['metadata']['state_sha256'])
        architectures.append(saved['metadata']['architecture'])
    if len(set(states))!=len(states):raise ValueError('duplicate checkpoint example')
    if any(not exact_state_equal(bodies[0],b) for b in bodies[1:]):raise ValueError('shared transformer bodies differ')
    # torch.equal alone accepts equal-valued tensors of different dtypes.
    # The frozen feature generator must have the same precision as well as values.
    if any(any(torch.is_tensor(v) and v.dtype!=bodies[0][n].dtype
               for n,v in b.items()) for b in bodies[1:]):
        raise ValueError('shared transformer body dtypes differ')
    if any(not exact_state_equal(architectures[0],a) for a in architectures[1:]):raise ValueError('shared architectures differ')
    return [dict(checkpoint=p,state_sha256=s,parent_sha256=sha(p)) for p,s in zip(parents,states)]


class FrozenFeaturePreconditioner:
    """Damped inverse quadratic curvature for a fixed design and loss weights.

    For loss w*||A*c-b||^2 and A=U*S*V.T, apply
    V diag(1/[2*w*(S^2+damping*S_max^2)]) V.T to its gradient.
    This is a full-curvature diagnostic for frozen linear heads, not a scalable
    whole-transformer optimizer or a replacement for the main norm/KL objective.
    Damping changes the step, not the objective's optimum.
    No target direction b enters the SVD. Objective weights include known-target
    normalization; the frozen features also have target-fitted ancestry.
    """
    def __init__(self, groups, damping):
        if damping <= 0 or not math.isfinite(damping):
            raise ValueError('positive finite relative damping required')
        self.banks={};self.diagnostics={}
        for n,g in groups.items():
            _,s,vh=torch.linalg.svd(g['A'],full_matrices=False)
            if float(s[0])<=0 or float(g['weight'])<=0:
                raise ValueError('nonzero design and positive objective weight required')
            self.banks[n]=(vh,2*g['weight']*(s.square()+damping*s[0].square()))
            self.diagnostics[n]=dict(rows=g['A'].shape[0],columns=g['A'].shape[1],
                largest_singular=float(s[0]),smallest_singular=float(s[-1]),
                condition=float(s[0]/s[-1]) if float(s[-1])>0 else None,
                relative_damping=damping)

    @torch.no_grad()
    def step(self, coefficients, lr):
        for n,c in coefficients.items():
            vh,denominator=self.banks[n]
            c.add_(vh.T@((vh@c.grad)/denominator),alpha=-lr)


def fit(groups, *, algorithm, lr, steps, log_every=100, on_measure=None, final_lr_ratio=1., damping=1e-12):
    if steps < 1 or lr <= 0:
        raise ValueError('positive steps and LR required')
    if not 0 <= final_lr_ratio <= 1 or (algorithm != 'adam' and final_lr_ratio != 1.):
        raise ValueError('LR decay is supported only for Adam, with ratio in [0,1]')
    coefficients = {n: torch.nn.Parameter(g['A'].new_zeros(g['A'].shape[1]))
                    for n, g in groups.items()}
    setup_started=time.perf_counter()
    preconditioner=None
    if algorithm == 'adam':
        optimizer = torch.optim.Adam(list(coefficients.values()), lr=lr,
                                     betas=(.9, .99), eps=1e-8)
    elif algorithm == 'lbfgs':
        optimizer = torch.optim.LBFGS(list(coefficients.values()), lr=lr,
                                     max_iter=1, max_eval=20, history_size=50,
                                     tolerance_grad=0., tolerance_change=0.,
                                     line_search_fn='strong_wolfe')
    elif algorithm == 'preconditioned':
        optimizer=torch.optim.SGD(list(coefficients.values()),lr=lr)
        preconditioner=FrozenFeaturePreconditioner(groups,damping)
    else:
        raise ValueError('unknown optimizer')
    evaluations = 0
    started = time.perf_counter()
    rows = []

    def measure(step):
        with torch.no_grad():
            errors = {n: float((g['A'] @ coefficients[n] - g['b']).square().sum())
                      for n, g in groups.items()}
            row = dict(step=step, objective=sum(float(groups[n]['weight']) * e
                        for n, e in errors.items()), group_core_error_squared=errors,
                       elapsed_seconds=time.perf_counter()-started,
                       setup_seconds=started-setup_started,
                       lr=optimizer.param_groups[0]['lr'],
                       gradient_evaluations=evaluations)
            if step==0 and preconditioner is not None:row['curvature_diagnostics']=preconditioner.diagnostics
            rows.append(row)
            if on_measure is not None:
                on_measure(row)

    def closure():
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        loss = objective(groups, coefficients)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite fitting loss')
        loss.backward()
        evaluations += 1
        return loss

    measure(0)
    for step in range(1, steps + 1):
        if algorithm == 'adam':
            ratio=final_lr_ratio+(1-final_lr_ratio)*.5*(1+math.cos(math.pi*(step-1)/max(1,steps-1)))
            optimizer.param_groups[0]['lr']=lr*ratio
            closure()
            optimizer.step()
        elif algorithm == 'preconditioned':
            closure()
            preconditioner.step(coefficients,lr)
        else:
            optimizer.step(closure)
        if step % log_every == 0 or step == steps:
            measure(step)
    return {n: c.detach().clone() for n, c in coefficients.items()}, optimizer.state_dict(), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--designs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--algorithm', choices=['adam', 'lbfgs', 'preconditioned'], default='adam')
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--final-lr-ratio', type=float, default=1.)
    parser.add_argument('--damping',type=float,default=1e-12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    startup_source_sha=sha(args.output/'source.py')
    torch.set_num_threads(2)
    datasets = [torch.load(path, map_location='cpu', weights_only=False) for path in args.designs]
    # Verify ancestry before fitting. The least-squares outputs are never read.
    cases=verify_shared_bodies(datasets)
    parent = datasets[0]['checkpoint']
    # Preserve the exact single-case arithmetic used by previous diagnostics.
    groups=datasets[0]['groups'] if len(datasets)==1 else merge_designs(datasets)
    with (args.output/'metrics.jsonl').open('w') as log:
        def record(row):
            log.write(json.dumps(row)+'\n')
            log.flush()
        coefficients, optimizer, rows = fit(groups, algorithm=args.algorithm,
                                            lr=args.lr, steps=args.steps, on_measure=record,
                                            final_lr_ratio=args.final_lr_ratio,damping=args.damping,
                                            log_every=1 if args.algorithm=='preconditioned' else 100)
    result = dict(scope=__doc__, algorithm=args.algorithm, lr=args.lr, steps=args.steps,
                  initialization='zero heads', final_lr_ratio=args.final_lr_ratio,
                  preconditioner=('full frozen-feature curvature; no target direction in SVD; known-target normalization and fitted feature ancestry retained' if args.algorithm=='preconditioned' else None),
                  damping=args.damping if args.algorithm=='preconditioned' else None,
                  initial=rows[0], final=rows[-1],
                  shared_training_cases=cases,
                  group_core_error_units=('physical projected SSE' if len(datasets)==1 else 'SSE after sqrt(checkpoint loss weight / case count) scaling; already objective contributions'),
                  per_case_final_objective=[float(objective(d['groups'],coefficients)) for d in datasets],
                  gpt_forwards=0, gpt_backwards=0,
                  source_sha256={**{str(path):sha(path) for path in args.designs}, __file__: startup_source_sha},
                  parent_checkpoint=parent, parent_sha256=sha(parent))
    torch.save(dict(grouped_coefficients=coefficients, optimizer=optimizer,
                    metadata=result, source_sha256={case['checkpoint']:case['parent_sha256'] for case in cases}),
               args.output/'checkpoint.pt')
    (args.output/'complete.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('algorithm', 'lr', 'initial', 'final')}), flush=True)


if __name__ == '__main__':
    main()
