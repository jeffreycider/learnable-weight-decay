"""Default-off tokenwise graph accumulation; global optimizer remains trainer-owned."""
import time
from types import SimpleNamespace
import torch
from .offline_graph_backend import OfflineGraphAccumulator,validate_training_options as validate_legacy
from .graph_tokenwise import TokenwiseThinGradientGraph
from .resident_teacher import ResidentTeacherFeatures
from .edit_scope import scoped_snapshot
from .graph_state_identity import validate_graph_panel_identity


def validate_training_options(args):
    if args.objective!='tokenwise-v1':raise ValueError('tokenwise_graph requires tokenwise-v1')
    if args.checkpoint_batch not in (40,200):
        raise ValueError('tokenwise_graph requires checkpoint_batch 40 or 200')
    # Reuse the shape/precision constraints without relaxing the independent
    # legacy backend's objective or fixed-batch guard.
    options=vars(args).copy();options['objective']='primal-dual'
    # The tokenwise batch allowlist is checked above. Only this validation copy
    # uses the legacy value; the trainer/graph retain the actual global count.
    options['checkpoint_batch']=200
    validate_legacy(SimpleNamespace(**options))
    if args.shrinkage_coefficient!=1.:raise ValueError('tokenwise_graph requires unit shrinkage reward')
    if getattr(args,'calibration_balance_only',False):raise ValueError('graph does not perform balance-only diagnostics')


class TokenwiseGraphAccumulator(OfflineGraphAccumulator):
    """Same begin/accumulate/finish/close lifecycle as the existing graph adapter.

    Exact cached FP64 teacher targets and one aggregate rowwise loss per student
    backward. The host retains all dual/state clocks and performs one global
    gradient SUM, clip and Adam. Audits release the private pool via close().
    """
    graph_type = TokenwiseThinGradientGraph

    def _capture(self,record,panel,dual):
        self._release_graph()
        state, _ = validate_graph_panel_identity(record, panel)
        if not hasattr(dual,'config'):raise ValueError('tokenwise dual configuration required')
        self.constants.load(self.cache,record);grouped=self.constants.grouped()
        snapshot=scoped_snapshot(record.snapshot,self.constants.radius_squared(grouped),self.radius_fraction,scope='grouped')
        resident=ResidentTeacherFeatures(panel,self.engine)
        graph=self.graph_type(self.actor,self.forward,snapshot,self.engine,grouped,resident.inputs,None,
            dual,state,state_weight=1/self.global_count,teacher_features=resident.features,
            shrinkage_coefficient=self.shrinkage_coefficient,dynamic_zero_groups=True,retain_metrics=True,
            token_config=dual.config,teacher_representation='log_probs')
        self.graph=graph;started=time.perf_counter();graph.capture(warmups=self.capture_warmups)
        torch.cuda.synchronize(self.engine.device);self.capture_seconds+=time.perf_counter()-started;self.capture_count+=1
        self._count_execution(resident.inputs,self.capture_warmups,setup=True);self._signature=graph.storage_signature()

    def _metrics(self,output,dual,state):
        measured,stats=super()._metrics(output,dual,state)
        # CPU reductions retain the reference's panel quantile/negative-row
        # semantics. No model execution or extra logit backward is introduced.
        rows=output['token_kl'].detach().cpu().double().flatten();count=rows.numel()
        linear,tail,rank=output['token_loss_components'].tolist();counts=output['rank_change_counts'].tolist()
        measured.update(kl_penalty=linear+tail+rank,dual_linear_term=linear,augmented_penalty=tail,
            tail_quadratic_penalty=tail,ranking_penalty=rank,norm_objective=float(output['regularizer']),
            token_count=count,sequence_count=len(self.graph.inputs),
            token_kl_quantiles=dict(zip(('min','median','p90','p99','p999','max'),torch.quantile(rows,torch.tensor([0.,.5,.9,.99,.999,1.],dtype=torch.float64)).tolist())),
            token_tail_exceedances=int((rows>dual.config.tail_tolerance).sum()),negative_token_kl_count=int((rows<0).sum()),
            mean_nonnegative_token_kl=float(rows.clamp_min(0).mean()),argmax_changes=int(counts[0]),
            top10_order_changes=int(counts[1]),top10_membership_changes=int(counts[2]))
        stats['offline_backend']='tokenwise_graph'
        return measured,stats
