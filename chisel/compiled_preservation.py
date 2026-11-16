"""Optional compilation of the student forward and raw-logit objective together."""
import torch


def raw_logit_loss(model, parameters, inputs, teacher):
    student = torch.func.functional_call(model, parameters, (inputs,))
    difference = student - teacher.detach()
    return difference.square().mean(), difference.detach().abs().amax()


def configure(config, state):
    if not config.get('compile_preservation_loss', False):
        return
    for settings in ({}, *config.get('domain_settings', {}).values()):
        selected = config | settings
        objective = selected['preservation']
        if (objective.get('constraint_metric') != 'logit_mse'
                or objective.get('center_logits', True)
                or objective['ranking_weight'] != 0
                or selected.get('measure_kl', True)
                or selected.get('measure_rankings', True)):
            raise ValueError('compiled preservation requires raw MSE without KL/ranking measurements')
    state.extras['raw_logit_loss'] = torch.compile(raw_logit_loss,
        **config.get('compile_gpt_loss_kwargs', dict(fullgraph=True, dynamic=False)))
