"""FP32 sensor callbacks preserve the exact forward and ordinary gradient."""
import copy
import math

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from baseline.model import GPT
from rewriter.gpt_precision import fp32_head_inputs, RMS_EPSILON
from rewriter.sensors import SensorCollector
from tests.test_sensors import anchors_for


def explicit_loss(model, inputs, targets, sensors=None):
    logits=model.proj(fp32_head_inputs(model,inputs,sensors=sensors))
    logits=15*logits*(logits.square()+15**2).rsqrt()
    return F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='sum')


@pytest.mark.parametrize('layers',[2,12])
def test_fp32_callbacks_match_hooked_reference_and_all_gradient_projections(monkeypatch,layers):
    torch.manual_seed(991)
    model=GPT(32,layers,128).float()
    reference=copy.deepcopy(model);plain=copy.deepcopy(model)
    inputs=torch.randint(32,(2,3));targets=torch.randint(32,(2,3))
    anchors=anchors_for(model)
    collector=SensorCollector(model,anchors,2,3,trajectory_rank=2,trajectory_positions=6)
    hooked=SensorCollector(reference,anchors,2,3,trajectory_rank=2,trajectory_positions=6)
    collector.set_microbatch(torch.arange(2),inputs)
    loss=explicit_loss(model,inputs,targets,collector);loss.backward()
    actual=collector.finalize()
    plain_loss=explicit_loss(plain,inputs,targets);plain_loss.backward()
    torch.testing.assert_close(loss,plain_loss,rtol=0,atol=0)
    for (name,p),(_,q) in zip(model.named_parameters(),plain.named_parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        mean=p.grad/inputs.numel()
        if name in anchors:
            if name=='embed.weight':mean=mean.T
            a,b=anchors[name]
            torch.testing.assert_close(actual.matrix[name].ones[0],math.sqrt(2)*mean@b,atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(actual.matrix[name].ones[1],math.sqrt(2)*mean.T@a,atol=2e-6,rtol=2e-5)
        else:
            torch.testing.assert_close(actual.vectors[name].ones,math.sqrt(2)*mean,atol=2e-6,rtol=2e-5)
    # Independent unmodified baseline execution: do not force the reference
    # to share the wrapper's epsilon, which would hide a function change.
    hooked.set_microbatch(torch.arange(2),inputs)
    with sdpa_kernel(SDPBackend.MATH):reference_loss=reference(inputs,targets)
    reference_loss.backward();expected=hooked.finalize()
    torch.testing.assert_close(loss,reference_loss,rtol=0,atol=0)
    for name in actual.matrix:
        for bank in ('train','audit','ones'):
            for a,b in zip(getattr(actual.matrix[name],bank),getattr(expected.matrix[name],bank)):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
    for name in actual.vectors:
        torch.testing.assert_close(actual.vectors[name].ones,expected.vectors[name].ones,rtol=0,atol=0)
    torch.testing.assert_close(actual.trajectories,expected.trajectories,rtol=0,atol=0)
    trajectories=actual.trajectories.reshape(6,2*layers+1,2,2)
    assert (trajectories.square().sum((0,3))>0).all()
    collector.close();hooked.close()
