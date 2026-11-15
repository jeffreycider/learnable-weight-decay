"""Expand a fixed diagnostic objective to reproducibly selected past minibatches."""
import torch
from chisel.preservation_fit import sampled_inputs


def setup(fractions=(0., .33, .66, 1.), examples_per_batch=256):
    if not fractions or any(not 0 <= f <= 1 for f in fractions):
        raise ValueError('history fractions must lie in [0, 1]')
    if examples_per_batch <= 0:
        raise ValueError('each fitting minibatch needs positive example count')

    def hook(config, state):
        fits = state.extras.get('parallel_fits', state.extras.get('diagnostic_fits'))
        if fits is None:
            fits = [state.extras['fit']]
        replay = state.extras['replay']
        for fit in fits:
            record = replay.reader.read(fit['entry'])
            steps = sorted({round(f * record['step']) for f in fractions})
            inputs = []
            for step in steps:
                _, _, _, _, _, batch, _ = replay.load(fit['entry'], record=record, batch_step=step)
                inputs.append(sampled_inputs(batch, config | {'preservation_examples': examples_per_batch}))
            fit['batch'] = {'inputs': torch.cat(inputs)}
            fit['fitting_batch_steps'] = steps
        # Every closure uses all selected examples; no resampling during L-BFGS.
        config['preservation_examples'] = None
    return hook
