"""Direct prediction-displacement measurements, in logits and probability units."""
import torch


@torch.no_grad()
def logit_errors(student, teacher):
    teacher, student = teacher.flatten(0, -2).double(), student.flatten(0, -2).double()
    delta = student - teacher
    centered = delta - delta.mean(-1, keepdim=True)
    probability_delta = student.softmax(-1) - teacher.softmax(-1)
    return dict(raw_rms=delta.square().mean(-1).sqrt(),
                centered_rms=centered.square().mean(-1).sqrt(),
                raw_max=delta.abs().amax(-1), centered_max=centered.abs().amax(-1),
                probability_max=probability_delta.abs().amax(-1),
                probability_tv=probability_delta.abs().sum(-1) / 2,
                teacher_centered_rms=(teacher - teacher.mean(-1, keepdim=True)).square().mean(-1).sqrt())


def summarize(rows):
    result = {}
    for name in rows[0]:
        values = torch.cat([row[name] for row in rows])
        result.update({name + '_mean': values.mean().item(), name + '_p95': values.quantile(.95).item(),
                       name + '_max': values.max().item()})
    return result
