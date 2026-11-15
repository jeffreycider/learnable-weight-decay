"""CPU-only metadata preparation; never forks a store or launches training.

Draft mode does not inspect parent CURRENT. Finalization is allowed only after a
successful closed review and verifies the pinned parent metadata/source first.
The separate qualified fork utility still enforces all writer locks.
"""
import argparse,copy,hashlib,json,tarfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

PARENT_SHA='084dc7be7eeeb71dbbbb760027847bf528c663869fbeab280f9b030577e2f4c0'
CHILD_SHA='41c93aecbe3ba22f360a5a8e36cd55b43755f3c06fd540acca55b1793b002f68'
SCRIPT_SOURCES={'scripts/train_coordinate_chisel.py','scripts/qualify_coordinate_chisel.py','scripts/train_multistate_chisel.py','scripts/capture_cifar_chisel_bank.py'}


def identity(metadata):
 return hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def child_recipe(parent):
 child=copy.deepcopy(parent)
 if child.get('private_actor_backend','eager')!='eager':raise ValueError('parent already uses private actor capture')
 if child['stopping']['review_after_full_sweeps']!=[1]:raise ValueError('unexpected parent review schedule')
 child['private_actor_backend']='eager_actor_graph'
 child['stopping']['review_after_full_sweeps']=[1,2,3]
 return child


def metadata_from_archive(archive,recipe,*,world_size=4,expected_sha=None):
 path=Path(archive)
 if expected_sha and hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha:raise ValueError('source archive SHA differs')
 sources={}
 with tarfile.open(path) as stream:
  for member in stream.getmembers():
   name=member.name;p=Path(name)
   if not member.isfile():continue
   selected=(name.startswith(('rewriter/','baseline/')) and p.suffix=='.py') or (name.startswith('baseline/vendor/') and p.suffix!='.py' and '__pycache__' not in p.parts) or name in SCRIPT_SOURCES
   if selected:sources[name]=hashlib.sha256(stream.extractfile(member).read()).hexdigest()
 if not SCRIPT_SOURCES<=sources.keys():raise ValueError('incomplete runtime source archive')
 return dict(recipe=recipe,sources=dict(sorted(sources.items())),world_size=world_size,
  mode='offline_joint_shared_private_fit',private_identity='domain,seed,localstep,sourceSHA,layoutSHA',
  weighting='pervisit 0.5*owner_pool_size/domain_total; fullfittingreadouts uniform percheckpoint')


def finalize_closed_parent(parent_root,destination,*,parent_metadata,child_metadata,expected_update=211):
 root=Path(parent_root);status=json.loads((root/'status.json').read_text())
 if status.get('status')!='completed' or status.get('exit_code')!=0:raise ValueError('parent review process has not closed successfully')
 if status.get('source_sha256')!=PARENT_SHA:raise ValueError('closed parent source differs')
 actual=json.loads((root/'results/metadata.json').read_text())
 if actual!=parent_metadata:raise ValueError('closed parent metadata differs from pinned recipe/source')
 from rewriter.private_probe_store import PrivateProbeStore
 source=root/'results/joint_store'
 with PrivateProbeStore(source/'global',read_only=True) as store:
  _,ref=store._manifest_and_reference();saved=store.restore(manifest_ref=ref)
 header=saved['sampler']
 if header['run_identity']!=identity(parent_metadata) or header['world_size']!=4:raise ValueError('parent identity/world differs')
 rows=header['ranks']
 if sorted(r['rank'] for r in rows)!=list(range(4)):raise ValueError('parent rank set differs')
 for row in rows:
  state=row['sampler']
  if state['next_global_batch']!=expected_update or state['last_full_sweep_update']!=expected_update or state['stop_reason']!='review_boundary':
   raise ValueError('parent is not the closed full-review boundary')
  if state['heldout_completed']:raise ValueError('unexpected heldout execution')
  pools=parent_metadata['recipe']['review_protocol']['owner_pool_sizes']
  expected_states=sum(pools[d][row['rank']] for d in ('gpt','cifar10'))
  ticks=parent_metadata['recipe']['private_inner_steps']+1
  if len(state['visits'])!=expected_states or sum(state['visits'].values())!=expected_update*2*ticks or any(v<ticks or v%ticks for v in state['visits'].values()):
   raise ValueError('parent private coverage/tick accounting differs')
 return dict(schema='chisel_distributed_checkpoint_fork_plan_v1',world_size=4,source=str(source),destination=str(Path(destination)/'joint_store'),
  parent_manifest_ref=ref,source_run_identity=identity(parent_metadata),destination_run_identity=identity(child_metadata),
  provenance=dict(source_code_id=PARENT_SHA,destination_code_id=CHILD_SHA,reason='Explicit post-review211 backend-only fork; preserve all numerical state, enable private actor graphs, review after coverages2 and3.'))


def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-recipe',required=True);p.add_argument('--parent-source',required=True);p.add_argument('--child-source',required=True);p.add_argument('--destination',required=True);p.add_argument('--output',required=True);p.add_argument('--closed-parent-root')
 a=p.parse_args();parent=json.loads(Path(a.parent_recipe).read_text());child=child_recipe(parent)
 pm=metadata_from_archive(a.parent_source,parent,expected_sha=PARENT_SHA);cm=metadata_from_archive(a.child_source,child,expected_sha=CHILD_SHA)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 for name,value in [('recipe.json',child),('metadata.json',cm),('parent_metadata.json',pm)]: (out/name).write_text(json.dumps(value,indent=2)+'\n')
 if a.closed_parent_root:
  plan=finalize_closed_parent(a.closed_parent_root,a.destination,parent_metadata=pm,child_metadata=cm)
  (out/'fork_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 else:
  draft=dict(schema='coordinate_backend_fork_DRAFT_NOT_EXECUTABLE',parent_manifest_ref=None,destination=a.destination,
   source_run_identity=identity(pm),destination_run_identity=identity(cm),next_expected_update=211,
   changes=['private_actor_backend=eager_actor_graph','stopping.review_after_full_sweeps=[1,2,3]'],
   requirements=['Wait for successful closed review process and full TRAIN readout','Finalize pinned CURRENT only after closure','Use same-world4 CPU Gloo fork utility51fb4e0','Install exact child metadata before trainer --resume; no fresh optimizer or dual initialization'])
  (out/'draft.json').write_text(json.dumps(draft,indent=2)+'\n')

if __name__=='__main__':main()
