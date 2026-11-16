"""Load only optimizer state needed for edits at the recorded forward weights."""
import re
from chisel.checkpoint_io import CheckpointReader, map_tensors
from chisel.harness import build


class CurrentCheckpointReader(CheckpointReader):
    """An explicit current-weight view; the complete on-disk record is unchanged."""
    current_only = True

    def project(self, record):
        pattern = record['config']['optimizers']['hidden']['parameters']
        return record | dict(
            gradients={n:g for n,g in record['gradients'].items() if re.fullmatch(pattern, n)},
            optimizer={'hidden': record['optimizer']['hidden']})


def prepare(model, record, config, device):
    spec = config['optimizers']['hidden']
    parameters = {n:p for n,p in model.named_parameters() if re.fullmatch(spec['parameters'], n)}
    if not parameters:
        raise ValueError('no editable matrices in current-weight replay')
    for name, parameter in model.named_parameters():
        parameter.grad = record['gradients'][name].to(parameter, copy=True) if name in parameters else None
    optimizer = build(spec['optimizer'], params=list(parameters.values()))
    # MuonH updates momentum in place; never alias a cached CPU record.
    saved = map_tensors(record['optimizer']['hidden'], lambda value: value.to(device, copy=True))
    optimizer.load_state_dict(saved)
    optimizer.prepare()
    return optimizer, parameters
