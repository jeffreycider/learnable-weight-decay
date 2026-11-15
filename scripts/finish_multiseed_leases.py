"""Release only this program's completed, archived and idle capture leases."""
import argparse,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'runs/multiseed-capture'
def cli(*args):return subprocess.run(['strangeloop',*args],check=True,text=True,capture_output=True,timeout=300).stdout

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
 plan=json.loads((D/'plan.json').read_text())
 if not a.execute:print(json.dumps(plan['lanes']));return
 finished=set();deadline=time.time()+7*3600
 while len(finished)<len(plan['lanes']) and time.time()<deadline:
  for lane,assigned in enumerate(plan['lanes'],1):
   if lane in finished:continue
   state=json.loads((D/f'lane{lane}-state.json').read_text())
   if state['status']!='complete':continue
   assert state['gpu']==assigned['gpu'] and state['teacher_completed']==assigned['seeds']
   assert state['completed']==assigned['seeds']
   for seed in assigned['seeds']:
    bank=json.loads((D/f'seed{seed}-archive.json').read_text());cache=json.loads((D/f'seed{seed}-teacher-archive.json').read_text())
    assert bank['states']==cache['states']==1000 and cache['panels']==2000
    assert bank['seed']==cache['seed']==seed
   # An explicit next-job handoff takes precedence over release.
   marker=D/f'hold-lane{lane}.json'
   if marker.exists():
    row=dict(status='handoff_held',gpu=assigned['gpu'],record=json.loads(marker.read_text()),time=time.time())
   else:
    active=cli('gpu','exec',assigned['gpu'],'--','nvidia-smi','--query-compute-apps=pid','--format=csv,noheader').strip()
    if active:raise RuntimeError('completed lane has an unexpected GPU process; preserve for ownership review')
    result=cli('gpu','down',assigned['gpu'],'--json');row=dict(status='released',gpu=assigned['gpu'],result=json.loads(result),time=time.time())
   (D/f'lane{lane}-release.json').write_text(json.dumps(row,indent=2)+'\n');finished.add(lane)
  if len(finished)<len(plan['lanes']):time.sleep(30)
 if len(finished)<len(plan['lanes']):raise TimeoutError('capture completion/release monitor deadline; unfinished work remains running')
if __name__=='__main__':main()
