"""Split a loaded fitting context without reloading weights or evaluating GPT.

Support and joint-update examples are disjoint within a visit. Both are training
data: this is not validation. Repeated support blocks are explicit optimization
on cached examples, not new information. No tensor or base model is mutated.
"""
from __future__ import annotations
import random


def context_slice(context,start,count):
    total=len(context['inputs'])
    if type(start) is not int or type(count) is not int or start<0 or count<1 or start+count>total:
        raise ValueError('fitting context slice outside loaded examples')
    if len(context['ids'])!=total:raise ValueError('example identities do not match inputs')
    result=dict(context)
    result['inputs']=context['inputs'][start:start+count]
    result['ids']=context['ids'][start:start+count]
    if 'target' in context:result['target']=context['target'][start:start+count]
    if 'teacher' in context:
        teacher=context['teacher']
        def translated_teacher(a,b):
            if not 0<=a<b<=count:raise ValueError('teacher request outside selected fitting panel')
            return teacher(start+a,start+b)
        result['teacher']=translated_teacher
    result['source_panel_offset']=context.get('source_panel_offset',0)+start
    result['source_panel_examples']=context.get('source_panel_examples',total)
    return result


def split_fitting_panels(context,*,inner_steps,inner_size,joint_size,seed):
    """Deterministic block rotation using a private Python RNG.

    The joint panel is contiguous. Support blocks lie wholly outside it and are
    cycled in seeded shuffled order. All selections preserve teacher microbatch
    sizes (16 sequences for GPT, 128 images for CIFAR).
    """
    unit={'gpt':16,'cifar10':128}[context['domain']]
    total=len(context['inputs'])
    if type(inner_steps) is not int or inner_steps<1:raise ValueError('positive private inner steps required')
    if any(type(n) is not int or n<unit or n%unit for n in (inner_size,joint_size,total)):
        raise ValueError('fitting panel sizes must preserve teacher microbatch shape')
    if len(set(context['ids']))!=total:raise ValueError('loaded fitting identities must be unique')
    if inner_size+joint_size>total:raise ValueError('not enough examples for disjoint support and joint panels')
    candidates=[]
    for joint_start in range(0,total-joint_size+1,unit):
        starts=[i for i in range(0,total-inner_size+1,inner_size)
                if i+inner_size<=joint_start or i>=joint_start+joint_size]
        if starts:candidates.append((joint_start,starts))
    if not candidates:raise ValueError('no disjoint fitting panels at prescribed microbatch shape')
    rng=random.Random(seed)
    joint_start,starts=rng.choice(candidates)
    rng.shuffle(starts)
    inner=[context_slice(context,starts[i%len(starts)],inner_size) for i in range(inner_steps)]
    joint=context_slice(context,joint_start,joint_size)
    return inner,joint
