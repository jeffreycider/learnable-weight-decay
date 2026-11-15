"""Bounded two-lane capture/teacher supervisor. Both numerical sources are pinned independently."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'runs/multiseed-capture'

def cli(*args):return subprocess.run(['strangeloop',*args],check=True,text=True,capture_output=True,timeout=300).stdout

def save(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def teacher_phase(gpu,seed,state,statepath):
 path=DIRECTORY/f'seed{seed}-teacher-launch.json'
 if not path.exists():
  subprocess.run([sys.executable,str(ROOT/'scripts/launch_multiseed_teacher.py'),'--gpu',gpu,'--seed',str(seed),'--execute'],cwd=ROOT,check=True,timeout=360)
 launch=json.loads(path.read_text());state.update(phase='teacher',teacher_launch=launch);save(statepath,state);declared=False
 while time.time()<state['deadline']:
  code='import json,pathlib; p=pathlib.Path('+repr(launch['output'])+'); c=pathlib.Path('+repr(launch['cache'])+'); f=p/"stdout.log"; lines=f.read_text().splitlines() if f.exists() else []; print(json.dumps(dict(last_lines=lines[-3:],finished=(c/"manifest.json").exists(),wandb=[str(x) for x in (p/"wandb"/"wandb").glob("offline-run-*")],alive=pathlib.Path("/proc/'+str(launch['pid'])+'").exists())))'
  result=json.loads(cli('gpu','exec',gpu,'--','/opt/venv211/bin/python','-c',code));state['teacher_progress']=result;save(statepath,state)
  if result['wandb'] and not declared:
   assert len(result['wandb'])==1;cli('gpu','declare',gpu,'--name',launch['name'],'--wandb-path',result['wandb'][0]);declared=True
  if result['finished'] and not result['alive']:break
  if not result['alive'] and not result['finished']:raise RuntimeError('teacher exited without complete manifest')
  time.sleep(30)
 else:raise TimeoutError('six-hour teacher supervisor deadline')
 for name in ('archive_multiseed_capture.py','archive_multiseed_teacher.py'):
  cli('gpu','push',gpu,str(ROOT/'scripts'/name),'/tmp/'+name)
 result=cli('gpu','exec',gpu,'--','/opt/venv211/bin/python','/tmp/archive_multiseed_teacher.py','--launch',launch['output']+'/launch_manifest.json')
 (DIRECTORY/f'seed{seed}-teacher-archive.json').write_text(result);state['teacher_completed'].append(seed);state['phase']='capture';save(statepath,state)

def main():
 p=argparse.ArgumentParser();p.add_argument('--lane',type=int,choices=[1,2],required=True);p.add_argument('--execute',action='store_true');p.add_argument('--resume-supervisor',action='store_true');a=p.parse_args()
 plan=json.loads((DIRECTORY/'plan.json').read_text());lane=plan['lanes'][a.lane-1];gpu=lane['gpu']
 if not a.execute:print(json.dumps(lane));return
 lock=(DIRECTORY/f'lane{a.lane}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 statepath=DIRECTORY/f'lane{a.lane}-state.json'
 if statepath.exists():
  if not a.resume_supervisor:raise RuntimeError('Existing supervisor state requires explicit reviewed adoption')
  state=json.loads(statepath.read_text())
  assert state['completed']==state.get('teacher_completed',[]), 'Mid-archive/teacher recovery requires separate review'
  assert state['completed']==lane['seeds'][:len(state['completed'])]
  for done in state['completed']:
   assert json.loads((DIRECTORY/f'seed{done}-archive.json').read_text())['seed']==done
   assert json.loads((DIRECTORY/f'seed{done}-teacher-archive.json').read_text())['seed']==done
  assert state['gpu']==gpu and state['seeds']==lane['seeds'] and state['status'] in ('running','error')
  if Path('/proc') .joinpath(str(state['pid'])).exists():raise RuntimeError('previous supervisor still exists')
  state.setdefault('adoptions',[]).append(dict(previous_pid=state['pid'],previous_status=state['status'],previous_error=state.pop('error',None),at=time.time()))
  state['pid']=os.getpid();state['status']='running'
 else:state=dict(gpu=gpu,seeds=lane['seeds'],started=time.time(),deadline=time.time()+6*3600,completed=[],status='running',pid=os.getpid())
 state.setdefault('teacher_completed',[]);save(statepath,state)
 try:
  for seed in lane['seeds']:
   if seed in state['teacher_completed']:continue
   path=DIRECTORY/f'seed{seed}-launch.json'
   if not path.exists():
    if (DIRECTORY/f'seed{seed}-intent.json').exists():raise RuntimeError('ambiguous launch intent exists')
    subprocess.run([sys.executable,str(ROOT/'scripts/launch_multiseed_capture.py'),'--gpu',gpu,'--seed',str(seed),'--execute',*(['--reuse-mounted-volume'] if state['completed'] else [])],cwd=ROOT,check=True,timeout=360)
   launch=json.loads(path.read_text());state.update(seed=seed,launch=launch);save(statepath,state)
   declared=False
   while time.time()<state['deadline']:
    code='import json,pathlib,os; p=pathlib.Path('+repr(launch['output'])+'); f=p/"metrics.jsonl"; text=f.read_text() if f.exists() else ""; rows=[json.loads(x) for x in text[:text.rfind(chr(10))+1].splitlines()]; print(json.dumps(dict(last=rows[-1] if rows else None,finished=any(r.get("event")=="finished" for r in rows),wandb=[str(x) for x in (p/"wandb").glob("offline-run-*")],alive=pathlib.Path("/proc/'+str(launch['pid'])+'").exists())))'
    result=json.loads(cli('gpu','exec',gpu,'--','/opt/venv211/bin/python','-c',code));state['progress']=result;save(statepath,state)
    if result['wandb'] and not declared:
     assert len(result['wandb'])==1;cli('gpu','declare',gpu,'--name',launch['name'],'--wandb-path',result['wandb'][0]);declared=True
    if result['finished'] and not result['alive']:break
    if not result['alive'] and not result['finished']:raise RuntimeError('capture exited without terminal record')
    time.sleep(30)
   else:raise TimeoutError('six-hour capture supervisor deadline')
   remote=f'/tmp/archive_multiseed_capture_seed{seed}.py'
   cli('gpu','push',gpu,str(ROOT/'scripts/archive_multiseed_capture.py'),remote)
   result=cli('gpu','exec',gpu,'--','/opt/venv211/bin/python',remote,'--launch',launch['output']+'/launch_manifest.json')
   (DIRECTORY/f'seed{seed}-archive.json').write_text(result);state['completed'].append(seed);save(statepath,state)
   teacher_phase(gpu,seed,state,statepath)
  state.update(status='complete',finished=time.time());save(statepath,state)
 except BaseException as exc:
  state.update(status='error',error=repr(exc),failed_at=time.time());save(statepath,state);raise
if __name__=='__main__':main()
