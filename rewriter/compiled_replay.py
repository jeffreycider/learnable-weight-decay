"""Default-off experimental local replay compilation.

The validated replay settings serialize backend identity. Global SUM, rank-zero
actor VJP, primal-dual terms, clipping, Adam and pending action lifetime remain
owned by ReplayMaintenance. No teacher cache or CUDA graph pool persists here.
"""
from .compiled_thin_student import CompiledThinStudent


def compiled_replay_student(replay,wire,teachers,editable,*,backward):
    if replay.settings.get('student_backend')!='compiled_student_v1_experimental':
        raise ValueError('explicit experimental replay backend required')
    if replay.settings['teacher_cache_device']!='device':
        raise ValueError('compiled replay requires resident teacher cache within the maintenance block')
    helper=getattr(replay,'_experimental_compiled_student',None)
    if helper is None or helper.editable!=frozenset(editable):
        helper=CompiledThinStudent(replay.engine,editable,global_tokens=replay.settings['sequences']*replay.seq_len,
                                   embedding_names=replay.embedding_names,compile_kl=False)
        replay._experimental_compiled_student=helper
        replay._experimental_compiled_student_builds=getattr(replay,'_experimental_compiled_student_builds',0)+1
    return helper.step(wire,teachers,editable,backward=backward)
