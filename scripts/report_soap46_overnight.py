"""Build an atomic, self-contained report from closed, verified SOAP46 runs."""
from __future__ import annotations
import argparse,base64,hashlib,html,io,json,os,statistics,zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
MATCH=('variant','seed','baseline_commit','source_sha256','implementation_sha256','steps','microbatch','compile_map','meta_betas','meta_clip','state_policy','basis_refresh_every')
LABEL={'inner-outer':'A: input and return transforms','outer-only':'B: return transforms only'}


def read_runs():
    runs=[]
    for d in sorted((ROOT/'runs/soap46-v1').glob('soap46-*')):
        if not (d/'archive_result.json').exists() or not (d/'config.json').exists():continue
        c=json.loads((d/'config.json').read_text())
        if c.get('steps')!=2900 or c.get('meta_ramp_steps')!=500:continue
        rows=[json.loads(line) for line in (d/'metrics.jsonl').read_text().splitlines()]
        val={r['step']:r for r in rows if r.get('event')=='validation'}
        if 2900 not in val:raise RuntimeError(f'Closed full run lacks endpoint: {d}')
        runs.append({'name':d.name,'config':c,'rows':rows,'val':val,'path':d,'frozen':c['warmup_steps']>=2900})
    return runs


def comparisons(runs):
    result=[]
    for r in runs:
        if r['frozen']:continue
        controls=[c for c in runs if c['frozen'] and c['config']['base_lr']==.0375 and c['config']['cwd']==.025
                  and all(c['config'].get(k)==r['config'].get(k) for k in MATCH)]
        if len(controls)>1:raise RuntimeError('Ambiguous matched control')
        if controls:result.append((r,controls[0]))
    return result


def same_base_comparisons(runs):
    result=[]
    for r in runs:
        if r['frozen'] or r['config']['seed'] not in (1,2):continue
        controls=[c for c in runs if c['frozen'] and
                  all(c['config'].get(k)==r['config'].get(k) for k in (*MATCH,'base_lr','cwd'))]
        if len(controls)>1:raise RuntimeError('Ambiguous same-base control')
        if controls:result.append((r,controls[0]))
    return result


def workflow_status():
    rows=[];directory=ROOT/'runs/soap46-v1'
    for lane in 'AB':
        state_path=directory/f'followup-{lane}-v1.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get('error'):rows.append(f"{lane}: stopped for review: {state['error']}");continue
        if state.get('finished'):rows.append(f'{lane}: bounded experiments and archival complete.');continue
        for suffix,label in [('attribution','same-base attribution'),('confirmation','paired-seed confirmation'),('neighbors','local base-setting tests'),('ramp-decay','residual learning-rate and decay tests')]:
            plan_path=Path(state['recovery_plan']) if suffix=='confirmation' and state.get('recovery_plan') else directory/f'panel-{lane}-{suffix}-v1.json'
            progress_path=plan_path.with_suffix('.progress.json')
            if not progress_path.exists():continue
            progress=json.loads(progress_path.read_text());active=progress.get('active')
            if not active:rows.append(f'{lane}: completing {label} and preserving records.');break
            plan=json.loads(plan_path.read_text());trial=next(t for t in plan['trials'] if t['name']==active)
            status_path=plan_path.with_suffix('.status.json');status=json.loads(status_path.read_text()) if status_path.exists() else {}
            step=(status.get('latest_train') or {}).get('step') if Path(status.get('output','')).name==active else None
            detail=f'last reported iteration {step}/2900' if step is not None else 'awaiting the first progress snapshot'
            rows.append(f"{lane}: {label}, seed {trial.get('seed',0)}, base LR {trial.get('base_lr',.0375):g}, CWD {trial.get('cwd',.025):g}; {detail}.");break
    return rows


def picture(fig):
    buffer=io.BytesIO();fig.savefig(buffer,format='svg',bbox_inches='tight',metadata={'Date':None});plt.close(fig)
    return 'data:image/svg+xml;base64,'+base64.b64encode(buffer.getvalue()).decode()


def endpoint_caption(pairs,variant,seeds,subset=None):
    selected=[(r,c) for r,c in pairs if r['config']['variant']==variant and r['config']['seed'] in seeds]
    if subset=='residual':selected=[(r,c) for r,c in selected if 'neighbor' not in r['name']]
    if subset=='neighbor':selected=[(r,c) for r,c in selected if 'neighbor' in r['name']]
    lower=sum(r['val'][2900]['val_loss']<c['val'][2900]['val_loss'] for r,c in selected)
    unit='completed confirmation seeds' if seeds==(1,2) else 'tested seed-0 settings'
    return f"{LABEL[variant]} ended below its frozen reference in {lower} of {len(selected)} {unit}."


def chart(pairs,variant,seeds,subset=None):
    selected=[(r,c) for r,c in pairs if r['config']['variant']==variant and r['config']['seed'] in seeds]
    if subset=='residual':selected=[(r,c) for r,c in selected if 'neighbor' not in r['name']]
    if subset=='neighbor':selected=[(r,c) for r,c in selected if 'neighbor' in r['name']]
    if not selected:return None
    fig,ax=plt.subplots(figsize=(5,3.8))
    for r,c in selected:
        steps=sorted(set(r['val'])&set(c['val']));cfg=r['config']
        label=(f"seed {cfg['seed']}" if cfg['seed'] else (f"base LR {cfg['base_lr']:g}, CWD {cfg['cwd']:g}" if subset=='neighbor' else f"meta LR {cfg['meta_lr']:.0e}, decay {cfg['residual_decay']:g}"))
        ax.plot(steps,[r['val'][s]['val_loss']-c['val'][s]['val_loss'] for s in steps],label=label,lw=1.5)
    ax.axhline(0,color='black',lw=.8);ax.set(xlabel='GPT training iteration',ylabel='Validation loss difference\nlearned minus frozen (nats/token)')
    ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=9,loc='best');fig.tight_layout()
    return picture(fig)


def training_chart(pairs,variant):
    selected=[(r,c) for r,c in pairs if r['config']['variant']==variant and r['config']['seed'] in(1,2)]
    if not selected:return None
    fig,ax=plt.subplots(figsize=(5,3.8))
    for r,c in selected:
        a={x['step']:x['train_loss'] for x in r['rows'] if x.get('event')=='train'}
        b={x['step']:x['train_loss'] for x in c['rows'] if x.get('event')=='train'}
        steps=sorted(a.keys()&b.keys());d=[a[t]-b[t] for t in steps]
        smooth=[statistics.mean(d[max(0,i-15):i+1]) for i in range(len(d))]
        line=ax.plot(steps,smooth,label=f"seed {r['config']['seed']}, trailing 16-step mean",lw=1.3)[0]
        ax.plot(steps,d,color=line.get_color(),alpha=.12,lw=.5)
    ax.axhline(0,color='black',lw=.8);ax.set(xlabel='GPT training iteration',ylabel='Training loss difference\nlearned minus frozen (nats/token)');ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8);fig.tight_layout();return picture(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'reports/soap46-overnight');args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    runs=read_runs();pairs=comparisons(runs);same_base=same_base_comparisons(runs);confirmed=[(r,c) for r,c in pairs if r['config']['seed'] in(1,2)]
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Learned SOAP basis residuals</title><style>body{font:17px/1.55 system-ui,sans-serif;max-width:900px;margin:auto;padding:24px;color:#18212a;background:#fafafa}h1{font-size:2rem;line-height:1.15}h2{font-size:1.4rem;margin-top:2rem}img{width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:15px;margin-top:8px}.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:14px;width:100%}th,td{padding:9px;text-align:right;border-bottom:1px solid #ccd1d5}th:first-child,td:first-child{text-align:left}.note{background:#edf2f6;padding:16px;border-radius:8px}code{overflow-wrap:anywhere}a{color:#005b94}</style><h1>Do learned SOAP basis residuals improve GPT training?</h1>']
    if len(confirmed)==4:
        for variant in LABEL:
            subset=[(r,c) for r,c in confirmed if r['config']['variant']==variant]
            lower=sum(r['val'][2900]['val_loss']<c['val'][2900]['val_loss'] for r,c in subset)
            finding=('lower in both independent seeds' if lower==2 else 'higher in both independent seeds' if lower==0 else 'lower in one independent seed and higher in the other')
            parts.append(f'<p><strong>{LABEL[variant]} produced validation loss {finding}</strong> than its matched zero-residual reference at the selected settings. Both comparisons use the fixed 2900-iteration endpoint and the published tail-EMA readout.</p>')
    else:
        parts.append(f'<p>{len(confirmed)} independent-seed comparisons are complete; the declared confirmation experiment is still running. All comparisons use the same 2900-iteration endpoint and the published tail-EMA validation readout.</p>')
    parts.append(f'<p>Settings were selected using {len([r for r in runs if not r["frozen"] and r["config"]["seed"]==0])} seed-0 tuning runs. The confirmation seeds were excluded from that selection.</p>')
    status=workflow_status()
    if status:parts.append('<ul>'+''.join('<li>'+html.escape(text)+'</li>' for text in status)+'</ul>')
    interruptions=[];interruption_notes=[]
    for record in sorted((ROOT/'runs/soap46-v1/interrupted-attempts').glob('*/*/interruption.json')):
        interruption=json.loads(record.read_text());interruptions.append(interruption)
        interruption_notes.append(f'<p class="note">A SOAP B seed-2 attempt stopped after {interruption["interrupted_train_steps"]} iterations when its lease ended before expiry; no cause was reported. That incomplete attempt is excluded from the endpoint comparison. The identical seed and configuration were restarted on a replacement node; the original attempt remains in the reproduction records.</p>')
    if len(confirmed)<4:parts.append('<p class="note">The declared seed-1 and seed-2 comparisons are still incomplete. Completed paired values below are interim results. Seed 0 chooses settings; its best result is not evidence of a repeatable improvement.</p>')
    if confirmed:
        for variant in LABEL:
            differences=[r['val'][2900]['val_loss']-c['val'][2900]['val_loss'] for r,c in confirmed if r['config']['variant']==variant]
            if differences:parts.append(f'<p>{LABEL[variant]}: mean learned-minus-frozen endpoint difference across {len(differences)} completed confirmation seeds is <strong>{statistics.mean(differences):+.4f}</strong> nats/token. Negative values favor the learned method. Individual paired values appear below; this small seed count does not establish statistical significance.</p>')
    for variant in LABEL:
        isolated=[(r,c) for r,c in same_base if r['config']['variant']==variant and (r['config']['base_lr']!=.0375 or r['config']['cwd']!=.025)]
        if isolated:
            delta=statistics.mean(r['val'][2900]['val_loss']-c['val'][2900]['val_loss'] for r,c in isolated)
            parts.append(f'<p>At the selected base learning rate and weight decay, {LABEL[variant]} differs from a frozen residual reference by <strong>{delta:+.4f}</strong> nats/token across {len(isolated)} paired seeds. This additional comparison holds the tuned base settings fixed, isolating the residual-learning contribution within the tested implementation.</p>')
    parts.append('<p>The reference is accepted Track 3 record 46: SOAP followed by its original finite Muon polar iteration and original postprocessing. For each hidden matrix, dense residual matrices start at zero and modify the left and right SOAP bases. Method A modifies the input and return transforms; method B modifies only the return transforms. SOAP covariance, basis refresh, and elementwise variance updates remain unmodified. Embedding and output-head optimizers remain unchanged. Frozen controls keep residuals at zero for the entire run while retaining the same optimizer derivative computations and zero-LR Adam moment updates; their measured runtime includes that instrumentation.</p><p>The next ordinary GPT gradient supplies direct one-step credit through the preceding optimizer map, including normalization, finite polar iterations, and the actual lookahead wrapper. There are no extra GPT derivative evaluations. Using the zero-based optimizer index, the residual Adam learning rate is zero through t=500, ramps to its peak at t=1000, and then stays fixed. Adam moments accumulate while residuals are zero. Active updates also multiply residuals by one minus the selected decay, independently of the learning rate.</p>')
    figure=0
    for seeds,title,subset in [((1,2),'Independent-seed confirmation',None),((0,),'Seed-0 residual tuning','residual'),((0,),'Seed-0 base-setting neighbors','neighbor')]:
        parts.append(f'<h2>{title}</h2>')
        for variant in LABEL:
            src=chart(pairs,variant,seeds,subset)
            if src:
                figure+=1;parts.append(f'<figure><img src="{src}" alt="{html.escape(LABEL[variant])} validation differences"><figcaption>Figure {figure}. {endpoint_caption(pairs,variant,seeds,subset)} Each curve uses the fixed validation readout; zero means equal loss. Seed-0 curves are tuning evidence only.</figcaption></figure>')
    isolated=[(r,c) for r,c in same_base if r['config']['base_lr']!=.0375 or r['config']['cwd']!=.025]
    if isolated:
        parts.append('<h2>Holding the selected base settings fixed</h2>')
        for variant in LABEL:
            src=chart(isolated,variant,(1,2))
            if src:
                figure+=1;parts.append(f'<figure><img src="{src}" alt="Same-base validation comparison"><figcaption>Figure {figure}. {LABEL[variant]} versus zero residuals at the same selected base learning rate and CWD. Each curve is an independent paired seed; negative values favor residual learning.</figcaption></figure>')
    for variant in LABEL:
        src=training_chart(pairs,variant)
        if src:
            figure+=1;parts.append(f'<figure><img src="{src}" alt="Training loss difference"><figcaption>Figure {figure}. {LABEL[variant]} on the same prescribed training batches as its frozen reference. Faint curves show each ordinary training loss difference; solid curves show a trailing 16-step mean. Smoothing does not supply additional independent evidence.</figcaption></figure>')
    parts.append('<h2>Fixed endpoint results</h2><p>Controls always use the published base learning rate 0.0375 and CWD 0.025. A learned method with different base settings is a jointly tuned method compared with the published baseline; that comparison does not isolate the effect of residual learning.</p>')
    if confirmed:
        parts.append('<div class="scroll"><table><tr><th>Method / seed</th><th>Learned loss</th><th>Frozen loss</th><th>Difference</th></tr>')
        for r,c in sorted(confirmed,key=lambda pair:(pair[0]['config']['variant'],pair[0]['config']['seed'])):
            learned=r['val'][2900]['val_loss'];frozen=c['val'][2900]['val_loss']
            label='A' if r['config']['variant']=='inner-outer' else 'B'
            parts.append(f'<tr><td>{label} / {r["config"]["seed"]}</td><td>{learned:.5f}</td><td>{frozen:.5f}</td><td>{learned-frozen:+.5f}</td></tr>')
        parts.append('</table></div>')
    parts.append('<details><summary>All tuning and confirmation configurations and endpoints</summary><div class="scroll"><table><tr><th>Method / seed</th><th>Meta LR</th><th>Residual decay</th><th>Base LR</th><th>CWD</th><th>Validation loss</th><th>Difference from frozen</th></tr>')
    pairmap={r['name']:c for r,c in pairs};samebase_map={r['name']:c for r,c in same_base};summary=[]
    for r in runs:
        c=r['config'];v=r['val'][2900]['val_loss'];control=pairmap.get(r['name']);delta=v-control['val'][2900]['val_loss'] if control else None
        parts.append(f'<tr><td>{"A" if c["variant"]=="inner-outer" else "B"}{" frozen" if r["frozen"] else ""} / {c["seed"]}</td><td>{c["meta_lr"]:.0e}</td><td>{c["residual_decay"]:g}</td><td>{c["base_lr"]:g}</td><td>{c["cwd"]:g}</td><td>{v:.5f}</td><td>{f"{delta:+.5f}" if delta is not None else "—"}</td></tr>')
        timed=[x for x in r['rows'] if x.get('event')=='train' and 1100<=x['step']<=1900]
        timing={key:statistics.median(x[key] for x in timed) for key in ('wall_seconds','gpt_forward_backward_seconds','base_optimizer_seconds','credit_seconds')} if timed else {}
        active=[x for x in r['rows'] if x.get('event')=='train' and x.get('meta_effective_lr',0)>0 and 'hypergradient_norm' in x]
        gradient_health={'active_credit_count':len(active),'fraction_clipped':sum(x.get('hypergradient_clip_factor',1)<1 for x in active)/len(active),'median_preclip_norm':statistics.median(x['hypergradient_norm'] for x in active),'max_preclip_norm':max(x['hypergradient_norm'] for x in active)} if active else {}
        summary.append({'name':r['name'],'config':c,'steady_step_median_seconds':timing,'hypergradient_health':gradient_health,'validation_2900':r['val'][2900],'matched_control':control['name'] if control else None,'difference':delta,'same_base_control':samebase_map[r['name']]['name'] if r['name'] in samebase_map else None,'same_base_difference':v-samebase_map[r['name']]['val'][2900]['val_loss'] if r['name'] in samebase_map else None,'metrics_sha256':hashlib.sha256((r['path']/'metrics.jsonl').read_bytes()).hexdigest()})
    diagnostic=[]
    for lane in 'AB':
        name=f'soap46-{lane}-ramp500-1000-d10-lr1e-5-seed0-v1'
        item=next((r for r in summary if r['name']==name),None)
        if item:
            health=item['hypergradient_health']
            diagnostic.append(f'{lane}: {100*health["fraction_clipped"]:.1f}% of active updates clipped; median preclip norm {health["median_preclip_norm"]:.4g}')
    parts.append('</table></div></details>')
    if diagnostic:
        parts.append('<h2>Gradient scale and clipping</h2><p>At the original matched settings (peak meta LR 1e-5, residual decay 0.10, published base settings), '+ '; '.join(diagnostic)+'. The global clipping threshold is 1. These are observed hypergradient scales, not estimates of a Jacobian condition number or proof of why validation differs. Full per-run diagnostics and measured GPT/optimizer timings are in the data download.</p>')
    measured=[r for r in summary if r['config']['seed'] in (1,2) and r['config']['warmup_steps']<2900]
    if measured:
        parts.append('<h2>Measured optimizer cost</h2><p>The table shows stable training-iteration medians from iterations 1100–1900, excluding validation and checkpoint pauses. Each value averages the per-run medians of the completed independent-seed learners. It is not total job wall time. Frozen references also run the derivative instrumentation, so this comparison does not measure overhead against an uninstrumented published baseline.</p><div class="scroll"><table><tr><th>Method</th><th>GPT forward/backward</th><th>Base update</th><th>Delayed credit</th><th>Total iteration</th></tr>')
        for variant in LABEL:
            group=[r for r in measured if r['config']['variant']==variant]
            if not group:continue
            values=[statistics.mean(r['steady_step_median_seconds'][key] for r in group) for key in ('gpt_forward_backward_seconds','base_optimizer_seconds','credit_seconds','wall_seconds')]
            parts.append('<tr><td>'+('A' if variant=='inner-outer' else 'B')+'</td>'+''.join(f'<td>{value*1000:.0f} ms</td>' for value in values)+'</tr>')
        parts.append('</table></div>')
    parts.append('<h2>Reproduction and limits</h2><p><a href="reproduction.zip">Download exact configurations, raw metrics, archive manifests, and pinned custom source</a>. <a href="summary.json">Machine-readable endpoint summary</a>.</p><p>SOAP retains the published precision and full model/data/batch schedule. Zero residuals match the upstream optimizer on identical tensors, but repeated full trajectories showed unresolved numerical variation. Small endpoint differences require independent paired seeds; no numerical-variability threshold is assumed. Published baseline hyperparameters were not retuned. All trials finish at their declared endpoint without validation-dependent changes.</p>')
    parts.extend(interruption_notes)
    stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M PT');parts.append(f'<p>Updated {stamp}. Only closed, archived runs enter this page.</p></html>')
    paired_summary={}
    for variant in LABEL:
        differences=[r['val'][2900]['val_loss']-c['val'][2900]['val_loss'] for r,c in confirmed if r['config']['variant']==variant]
        if differences:paired_summary[variant]={'count':len(differences),'mean_difference':statistics.mean(differences),'individual_differences':differences,'selection_seed_excluded':True}
    for name,content in [('index.html',''.join(parts)),('summary.json',json.dumps({'runs':summary,'completed_confirmation_pairs':len(confirmed),'paired_summary':paired_summary,'workflow':status,'interrupted_attempts':interruptions},indent=2)+'\n')]:
        tmp=out/(name+f'.{os.getpid()}.incomplete');tmp.write_text(content);tmp.replace(out/name)
    tmp=out/f'reproduction.zip.{os.getpid()}.incomplete'
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for r in runs:
            for name in ['config.json','metrics.jsonl','archive_result.json','finished.json','complete.json','launch_plan.json','data_manifest.json','stdout.log']:
                f=r['path']/name
                if f.exists():z.write(f,f'runs/{r["name"]}/{name}')
        for pattern in ('panel-*-v1.json','followup-*-v1.json'):
            for f in sorted((ROOT/'runs/soap46-v1').glob(pattern)):
                z.write(f,'selection/'+f.name)
        attempts=ROOT/'runs/soap46-v1/interrupted-attempts'
        for f in sorted(attempts.rglob('*')):
            if f.is_file():z.write(f,'interrupted-attempts/'+str(f.relative_to(attempts)))
        source=ROOT/'runs/soap46-v1/source-0950ad0.tar.gz'
        if source.exists():z.write(source,'source-0950ad0.tar.gz')
        for f in [ROOT/'scripts/run_soap46_followup.py',ROOT/'scripts/run_soap46_panel.py',ROOT/'scripts/recover_soap46_confirmation.py',ROOT/'scripts/report_soap46_overnight.py']:
            z.write(f,'orchestration/'+f.name)
    tmp.replace(out/'reproduction.zip');print(json.dumps({'report':str(out/'index.html'),'closed_runs':len(runs),'confirmation_pairs':len(confirmed)}))


if __name__=='__main__':main()
