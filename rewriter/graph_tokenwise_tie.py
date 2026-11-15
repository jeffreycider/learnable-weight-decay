"""Default-off exact-teacher-tie graph variant; legacy graph remains independent."""
import torch

from .graph_tokenwise import TokenwiseThinGradientGraph
from .offline_tokenwise_graph import TokenwiseGraphAccumulator
from .rank_preservation import topk_preservation_flags
from .tiled_tie_ranking import tiled_tie_aware_topk_margin_loss


class TieAwareTokenwiseGraph(TokenwiseThinGradientGraph):
    def _validate_ranking_scope(self, config):
        if getattr(config,'ranking_loss_mode','absolute')!='absolute':
            raise ValueError('tie-aware graph is qualified only for absolute ranking; relative objectives require a separate backend')
        if config.ranking_tie_policy != 'exact_teacher_ties':
            raise ValueError('tie-aware graph requires exact_teacher_ties')

    def __init__(self, *args, teacher_representation='log_probs', **kwargs):
        if teacher_representation != 'log_probs':
            raise ValueError('tie-aware graph requires exact cached FP64 log probabilities')
        super().__init__(*args, teacher_representation=teacher_representation, **kwargs)

    def _ranking_loss(self, teacher, logits):
        return tiled_tie_aware_topk_margin_loss(teacher, logits,
            k=min(self.token_config.ranking_k, logits.shape[-1]),
            margin_fraction=self.token_config.margin_fraction, return_teacher_topk=True)

    def _ranking_counts(self, teacher, logits, teacher_topk):
        t=teacher.reshape(-1,teacher.shape[-1]);s=logits.reshape_as(t)
        k=min(self.token_config.ranking_k,s.shape[-1])
        flags=topk_preservation_flags(t,s,k=k,teacher_topk=teacher_topk,student_topk=s.topk(k,dim=-1))
        return torch.stack([flags[name].sum() for name in (
            'literal_argmax_changed','literal_order_changed','literal_membership_changed',
            'tie_aware_argmax_changed','strict_order_changed','strict_membership_changed','strict_topk_changed')])


class TieAwareTokenwiseGraphAccumulator(TokenwiseGraphAccumulator):
    graph_type = TieAwareTokenwiseGraph

    def _metrics(self, output, dual, state):
        measured,stats=super()._metrics(output,dual,state)
        k=min(dual.config.ranking_k,self.graph.engine.parameters['proj.weight'].shape[0])
        counts=output['rank_change_counts'].tolist()
        if len(counts)!=7:raise ValueError('literal and strict rank diagnostics required')
        measured.update(ranking_k=k,ranking_tie_policy=dual.config.ranking_tie_policy,
            topk_order_changes=int(counts[1]),topk_membership_changes=int(counts[2]))
        if k!=10:
            measured.pop('top10_order_changes');measured.pop('top10_membership_changes')
        measured.update({name:int(value) for name,value in zip(
            ('tie_aware_argmax_changes','strict_order_changes','strict_membership_changes','strict_topk_changes'),counts[3:])})
        stats['offline_backend']='tokenwise_tie_graph'
        return measured,stats
