"""Muoff's ordered lifecycle, adapted to injected tasks and single-process runs.

Source: muoff@8f3e4a9, records/track_3_optimization/harness/{loop,run_state,hooks}.py.
The task supplies backward(); the loop owns exactly one optimizer step per iteration.
"""
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any
import argparse
import copy
import yaml

SITES = ("setup", "pre_forward", "pre_optimizer", "post_optimizer", "teardown")


@dataclass
class RunState:
    step: int = -1
    model: Any = None
    optimizer: Any = None
    train_batches: Any = None
    backward: Any = None
    extras: dict = field(default_factory=dict)


def resolve(path, stack=()):
    """Resolve relative YAML inheritance; mappings merge, lists replace."""
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"cyclic YAML inheritance: {path}")
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    parent = config.pop("extends", None)
    return merge(resolve(path.parent / parent, (*stack, path)), config) if parent else config


def merge(base, changes):
    result = copy.deepcopy(base)
    for key, value in changes.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def locate(name):
    module, attribute = name.rsplit(".", 1)
    return getattr(import_module(module), attribute)


def build(spec, **injected):
    """Construct an explicitly named dependency, recursively resolving its arguments."""
    if isinstance(spec, list):
        return [build(item) for item in spec]
    if not isinstance(spec, dict):
        return spec
    args = {key: build(value) for key, value in spec.items() if key != "_target_"}
    if "_target_" not in spec:
        return args
    if args.keys() & injected.keys():
        raise ValueError("dependency arguments cannot override injected runtime objects")
    return locate(spec["_target_"])(**args, **injected)


def bind_sites(config):
    return {site: [locate(entry["name"])(**entry.get("hyperparams", {}))
                   for entry in config.get(site, [])] for site in SITES}


def inject(hooks, config, state):
    for hook in hooks:
        hook(config, state)


def run(config, state=None):
    """No experimental branches: task setup and ordered hooks define behavior."""
    if config["train_steps"] <= 0:
        raise ValueError("train_steps must be positive")
    hooks = bind_sites(config)
    state = RunState() if state is None else state
    inject(hooks["setup"], config, state)
    if state.backward is None or state.optimizer is None:
        raise ValueError("setup must inject backward and optimizer")
    stop = min(config["train_steps"], config.get("stop_step", config["train_steps"]))
    try:
        for step in range(state.step + 1, stop):
            state.step = step
            inject(hooks["pre_forward"], config, state)
            state.backward(config, state)
            inject(hooks["pre_optimizer"], config, state)
            state.optimizer.step()
            state.optimizer.zero_grad(set_to_none=True)
            inject(hooks["post_optimizer"], config, state)
    finally:
        inject(hooks["teardown"], config, state)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    run(resolve(args.config))


if __name__ == "__main__":
    main()
