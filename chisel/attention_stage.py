"""Explicit weights-only initialization of the unrestricted attention-augmented model."""
import torch
from chisel.storage import sha256 as digest


def expanded_weights(actor, saved):
    """Preserve the fitted attention action; zero independent R/C output heads."""
    if actor.attention_proj is None or any(name.startswith('attention_proj.') for name in saved):
        raise ValueError('expected an attention-only source and an augmented target')
    target = actor.state_dict()
    for name, value in target.items():
        if name.startswith(('proj.', 'gain_proj.')):
            target[name] = torch.zeros_like(value)
        else:
            source = name.replace('attention_proj.', 'proj.')
            if source not in saved or saved[source].shape != value.shape:
                raise ValueError(f'incompatible attention initialization: {name}')
            target[name] = saved[source]
    if set(saved) != {name.replace('attention_proj.', 'proj.') for name in target
                       if not name.startswith(('proj.', 'gain_proj.'))}:
        raise ValueError('unexpected source model parameters')
    return target


def initialize(path, sha256):
    """Start a declared new stage with fresh shared Adam; private state is injected separately."""
    def hook(config, state):
        if config.get('resume'):
            return
        if digest(path) != sha256:
            raise ValueError('attention initialization checkpoint SHA256 mismatch')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['config'].get('proposal_function') != 'chisel.attention_identity_fit.propose':
            raise ValueError('source was not trained with the attention identity decoder')
        state.model.load_state_dict(expanded_weights(state.model, saved['model']), strict=True)
    return hook


def audit_initial():
    """Audit recorded inputs before the first unrestricted update; no fitted-batch claim."""
    def hook(config, state):
        if not config.get('resume'):
            from chisel.preservation_fit import evaluate
            if state.step != -1:
                raise ValueError('initial audit must precede all training updates')
            evaluate(1)(config | {'training_audit_inputs': 'recorded'}, state)
    return hook
