"""Plateau decisions for held-out Chisel checkpoint objectives, never GPT validation."""
from dataclasses import dataclass,asdict
import math


@dataclass
class FitSchedule:
    minimum_epochs: int = 30
    plateau_patience: int = 3
    stopping_patience: int = 5
    minimum_reductions: int = 2
    post_reduction_checks: int = 2
    threshold: float = .001
    factor: float = .5
    minimum_lr: float = 1e-8
    best: float = math.inf
    progress_best: float = math.inf
    best_epoch: int = -1
    stale_checks: int = 0
    plateau_checks: int = 0
    checks_since_reduction: int = 0
    reductions: int = 0
    progress_started: bool = True

    def __post_init__(self):
        if min(self.minimum_epochs,self.plateau_patience,self.stopping_patience,self.minimum_reductions,self.post_reduction_checks)<1:
            raise ValueError('positive schedule counts required')
        if not 0<self.factor<1 or not math.isfinite(self.threshold) or self.threshold<0 or self.minimum_lr<=0:
            raise ValueError('invalid plateau scale')

    def observe(self,value,epoch,lr,*,track_progress=True):
        if not math.isfinite(value) or not math.isfinite(lr) or lr<=0:
            raise ValueError('finite validation objective and positive learning rate required')
        new_best=value<self.best
        if new_best:self.best,self.best_epoch=value,epoch
        if not track_progress:
            if epoch!=0 or self.stale_checks or self.plateau_checks or self.reductions:
                raise ValueError('only the initial baseline may be excluded from progress tracking')
            # Preserve identity as the best feasible saved policy, without
            # calling an improving but still infeasible trained policy stalled.
            # Keep a finite placeholder so JSON logs remain standards compliant.
            self.progress_best=value;self.progress_started=False
            return dict(lr=lr,new_best=new_best,significant_improvement=False,
                        lr_reduced=False,stop=False,baseline_only=True,**self.state_dict())
        improved=not self.progress_started or value<self.progress_best-self.threshold
        self.progress_started=True
        self.checks_since_reduction+=1
        if improved:
            self.progress_best=value;self.stale_checks=self.plateau_checks=0
        else:
            self.stale_checks+=1;self.plateau_checks+=1
        reduced=False
        if self.plateau_checks>=self.plateau_patience and lr>self.minimum_lr:
            lr=max(self.minimum_lr,lr*self.factor)
            self.reductions+=1;self.plateau_checks=0;self.checks_since_reduction=0;reduced=True
        stop=(epoch>=self.minimum_epochs and self.reductions>=self.minimum_reductions
              and self.stale_checks>=self.stopping_patience
              and self.checks_since_reduction>=self.post_reduction_checks)
        return dict(lr=lr,new_best=new_best,significant_improvement=improved,lr_reduced=reduced,
                    stop=stop,**self.state_dict())

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state_dict(cls,state):
        return cls(**state)
