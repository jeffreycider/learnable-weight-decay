"""Qualification-only first-visit verification overlap; tensor math unchanged."""
from contextlib import contextmanager
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rewriter.verify_prefetch import VerifyAhead
from scripts.qualify_coordinate_chisel import FixedContexts
import scripts.train_coordinate_chisel as trainer
from scripts.qualify_private_snapshot_training import snapshot_backend


def upcoming_gpt_ids(schedule,visit,start,stop):
    return [visit(schedule,batch,'gpt')[0].local_id for batch in range(start,stop)]


@contextmanager
def verification_lookahead(enabled,max_updates,evidence):
    old_init=FixedContexts.__init__;old_visit=trainer.OwnerBalancedSchedule.visit
    context=None;ahead=None
    def initialize(self,*a,**k):
        nonlocal context
        old_init(self,*a,**k);context=self
    def visit(self,batch,domain):
        nonlocal ahead
        result=old_visit(self,batch,domain)
        if enabled and domain=='gpt':
            if ahead is None:
                if context is None:raise RuntimeError('verification has no bound data readers')
                ids=upcoming_gpt_ids(self,old_visit,batch,max_updates)
                ahead=VerifyAhead(context.reader,context.cache,roles=('train',),depth=1)
                ahead.schedule(ids);evidence['scheduled_global_ids']=ids
            start=time.perf_counter();ahead.wait(result[0].local_id)
            evidence.setdefault('wait_seconds',[]).append(time.perf_counter()-start)
        return result
    FixedContexts.__init__=initialize;trainer.OwnerBalancedSchedule.visit=visit
    try:yield
    finally:
        if ahead is not None:
            ahead.close();evidence['hashed']=dict(ahead.hashed)
        FixedContexts.__init__=old_init;trainer.OwnerBalancedSchedule.visit=old_visit


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--recipe',required=True)
    p.add_argument('--verify-ahead',choices=['off','on'],required=True);p.add_argument('--output',required=True)
    a,rest=p.parse_known_args();recipe=json.loads(Path(a.recipe).read_text())
    if recipe.get('qualification_only') is not True or recipe.get('qualification_verify_ahead')!=a.verify_ahead:
        raise ValueError('explicit qualification verification mode required')
    if recipe.get('qualification_snapshot_backend')!='pinned_d2h_v1':
        raise ValueError('verification comparison requires the same explicit pinned snapshot backend')
    if recipe.get('verification_qualifier_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('verification qualifier source changed')
    evidence=dict(enabled=a.verify_ahead=='on',scope='First-visit SHA/tensor verification only. Wait is inside complete update wall but outside trainer context_load_seconds; tensor loading and all arithmetic unchanged.')
    sys.argv=[sys.argv[0],'--recipe',a.recipe,'--output',a.output,*rest]
    try:
        with snapshot_backend('pinned_d2h_v1'), verification_lookahead(a.verify_ahead=='on',recipe['stopping']['max_updates'],evidence):
            trainer.main()
    finally:
        output=Path(a.output)
        if output.exists():
            import os
            (output/('verification-lookahead-rank'+os.environ.get('RANK','0')+'.json')).write_text(json.dumps(evidence,indent=2))
