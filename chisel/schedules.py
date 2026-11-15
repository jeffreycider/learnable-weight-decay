"""Schedule functions selected and parameterized in YAML."""
import math


def cosine(step, total, warmup_fraction, minimum_fraction):
    warmup = int(total * warmup_fraction)
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return minimum_fraction + (1 - minimum_fraction) * .5 * (1 + math.cos(math.pi * progress))


def cooldown(step, total, fraction):
    return min(1., (1 - step / total) / fraction)
