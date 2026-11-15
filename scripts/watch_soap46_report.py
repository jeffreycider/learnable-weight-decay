"""Refresh the local SOAP report while the two bounded follow-up lanes run."""
import fcntl,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
lock=(ROOT/'runs/soap46-v1/report-watch.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
deadline=time.time()+8*3600
while time.time()<deadline:
    subprocess.run([sys.executable,'scripts/report_soap46_overnight.py'],cwd=ROOT,check=True,timeout=120)
    states=[]
    for lane in 'AB':
        path=ROOT/f'runs/soap46-v1/followup-{lane}-v1.json'
        states.append(json.loads(path.read_text()) if path.exists() else {})
    if all(state.get('finished') or state.get('error') for state in states):break
    time.sleep(180)
