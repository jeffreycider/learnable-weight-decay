import pathlib,json,subprocess,os,argparse
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);args=p.parse_args()
def atomic(path,text):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(text);os.replace(temp,path)
root=pathlib.Path(f'runs/soap-full-seed{args.seed}-v1')
for arm in ['fixed','learned','low','frozen']:
 launch=root/(arm+'-launch.json')
 if not launch.exists() or (root/(arm+'-archive.json')).exists():continue
 info=json.loads(launch.read_text());gpu=info['gpu'];out=info['output'];local=root/arm;local.mkdir(exist_ok=True)
 if gpu=='999c15e5-51a3-42cd-be18-7427103451e2':continue
 code='''import pathlib,json,sys
p=pathlib.Path(sys.argv[1]);m=p/'metrics.jsonl';w=list((p/'wandb').glob('offline-run-*'));print(json.dumps({'metrics':m.read_text() if m.exists() else '', 'config':(p/'config.json').read_text() if (p/'config.json').exists() else '', 'finished':(p/'finished.json').exists(),'stdout_tail':(p/'stdout.log').read_text()[-3000:],'wandb':str(w[0]) if w else None}))'''
 r=subprocess.run(['strangeloop','gpu','exec',gpu,'--','/opt/venv211/bin/python','-c',code,out],text=True,capture_output=True,timeout=60)
 if r.returncode:print(arm,'poll error',r.stderr[-500:]);continue
 data=json.loads(r.stdout)
 metrics=data['metrics'];deferred=False
 if metrics and not metrics.endswith('\n'):
  if data['finished']:raise ValueError('completed metrics lack final newline')
  metrics=metrics.rsplit('\n',1)[0]+'\n' if '\n' in metrics else '';deferred=True
 for line in metrics.splitlines():json.loads(line)
 data['metrics']=metrics
 atomic(local/'metrics.jsonl',metrics);atomic(local/'stdout_tail.txt',data['stdout_tail'])
 atomic(local/'poll_status.json',json.dumps({'finished':data['finished'],'deferred_unfinished_last_line':deferred}))
 if data['config']:atomic(local/'config.json',data['config'])
 if data['wandb'] and (not (local/'wandb-registered').exists() or (local/'wandb-registered').read_text()!=data['wandb']):
  subprocess.run(['strangeloop','gpu','declare',gpu,'--name',pathlib.Path(out).name,'--wandb-path',data['wandb']],stdout=subprocess.DEVNULL,check=True)
  (local/'wandb-registered').write_text(data['wandb'])
 rows=[json.loads(x) for x in data['metrics'].splitlines()];train=[x for x in rows if x['event']=='train']
 print(json.dumps({'arm':arm,'finished':data['finished'],'last':train[-1] if train else None,'validation':[x for x in rows if x['event']=='validation'][-3:],'tail':data['stdout_tail'][-800:] if not train else None}))
