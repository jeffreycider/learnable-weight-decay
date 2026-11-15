"""Predeclared offline preservation curriculum; never responds to validation."""
import math


def interpolate(knots, step):
    if not knots or knots[0]['step'] != 0:
        raise ValueError('curriculum must begin at step zero')
    fields = set(knots[0]) - {'step'}
    if any(set(knot) - {'step'} != fields for knot in knots):
        raise ValueError('every curriculum knot must specify the same fields')
    if any(right['step'] <= left['step'] for left, right in zip(knots, knots[1:])):
        raise ValueError('curriculum steps must increase')
    if any(value <= 0 for knot in knots for key, value in knot.items() if key != 'step'):
        raise ValueError('geometrically interpolated values must be positive')
    left, right = knots[-1], knots[-1]
    for first, second in zip(knots, knots[1:]):
        if step < second['step']:
            left, right = first, second
            break
    fraction = min(1., max(0., (step - left['step']) / max(1, right['step'] - left['step'])))
    return {key: math.exp((1 - fraction) * math.log(left[key]) + fraction * math.log(right[key]))
            for key in fields}


def schedule(knots):
    required = {'preservation_weight', 'kl_tolerance', 'private_lr'}
    if set(interpolate(knots, 0)) != required:
        raise ValueError(f'curriculum requires exactly {sorted(required)}')

    def hook(config, state):
        values = interpolate(knots, state.step)
        state.extras['offline_curriculum'] = values
        state.extras['dual'].tolerance = values['kl_tolerance']
    return hook
