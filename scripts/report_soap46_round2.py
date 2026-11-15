"""Atomic progress report for the renewed residual-only SOAP search."""
import argparse, base64, io, json, statistics, zipfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def load_run(name):
    folder=ROOT/'runs/soap46-v1'/name
    path=folder/'metrics.jsonl'
    if not path.exists():path=folder/'metrics-live.jsonl'
    if not path.exists():return None
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    train=[r for r in rows if r.get('event')=='train']
    val={r['step']:r['val_loss'] for r in rows if r.get('event')=='validation'}
    active=[r for r in train if r['step']>501 and 'hypergradient_norm' in r]
    return dict(name=name,train=train,val=val,completed=2900 in val,archived=(folder/'archive_result.json').exists(),
        step=train[-1]['step'] if train else 0,
        clipped_fraction=sum(r.get('hypergradient_clip_factor',1)<1 for r in active)/len(active) if active else None,
        median_seconds={k:statistics.median(r[k] for r in train[-100:] if k in r) for k in ['gpt_forward_backward_seconds','base_optimizer_seconds','credit_seconds'] if any(k in r for r in train[-100:])})


def picture(fig):
    stream=io.BytesIO();fig.savefig(stream,format='svg',bbox_inches='tight');plt.close(fig)
    return 'data:image/svg+xml;base64,'+base64.b64encode(stream.getvalue()).decode()


def conditioning_section(config, workflow, runs):
    """Describe the new source separately; never rank incomplete factorial cells."""
    hold=ROOT/'runs/soap46-conditioning/held_by_user_20261110_0348.json'
    if hold.exists():
        return ('<h2>Unlaunched optimizer changes are on hold</h2><p>The coordinate-frame '
                'and further Adam-memory factorial was held at the user’s request before '
                'any of its training runs started. The optional warmup-moment reset also '
                'has no GPU training result. Their plans and qualification artifacts are '
                'retained. The subsequent tuning focused on learning rate and weight decay '
                'for the original A/B definitions, with Adam betas, clipping, and residual '
                'coordinates fixed. <a href="conditioning-plan.json">Preserved held plan</a>.</p>')
    cells=[]
    for cell in config['cells']:
        for seed in config['development_seeds']:
            name=f'soap46-r3-{cell["residual_frame"]}-b{str(cell["meta_betas"][1]).replace(".","")}-seed{seed}-v1'
            cells.append((cell,seed,name,runs.get(name)))
    finished=sum(bool(r and r['completed']) for _,_,_,r in cells)
    started=any(r for _,_,_,r in cells)
    title='Coordinate-frame comparison is running' if started else 'Next test: retain residuals in weight coordinates'
    body=(f'<h2>{title}</h2><p>The study compares the current additive basis residual X with '
          'a stored residual Z in weight coordinates, writing X = ZQ at each step. '
          'At a fixed orthogonal basis these describe the same possible edits; across '
          'basis refreshes they preserve different residuals and Adam histories. '
          'Coordinatewise Adam also behaves differently in the two coordinates even '
          'with a fixed basis, so this experiment does not isolate basis reordering alone. '
          'The test crosses these choices with residual Adam β₂ of 0.99 and 0.999 '
          'on development seeds 0 and 3. All eight runs use the same qualified source '
          '<code>befd125</code> and unchanged base optimizer. '
          f'{finished} of eight fixed endpoints are available. '
          'Selection waits for all eight archived runs and uses the mean loss across the two seeds. '
          'The selected setting then receives fresh paired confirmation on seeds 6 and 7. '
          '<a href="conditioning-plan.json">Exact study plan</a>.</p>')
    if not started:return body
    body+='<div class=scroll><table><thead><tr><th>Stored residual</th><th>Adam β₂</th><th>Seed</th><th>Status</th><th>Loss at 2900</th></tr></thead><tbody>'
    for cell,seed,name,r in cells:
        status='queued' if not r else 'archived' if r['archived'] else 'preserving' if r['completed'] else f'{r["step"]}/2900'
        loss='—' if not r or not r['completed'] else f'{r["val"][2900]:.5f}'
        body+=f'<tr><td>{cell["residual_frame"]}</td><td>{cell["meta_betas"][1]}</td><td>{seed}</td><td>{status}</td><td>{loss}</td></tr>'
    body+='</tbody></table></div>'
    fig,axes=plt.subplots(1,2,figsize=(10,3.7));found=False
    for ax,seed in zip(axes,config['development_seeds']):
        for beta in (.99,.999):
            a=runs.get(f'soap46-r3-parameter-b{str(beta).replace(".","")}-seed{seed}-v1')
            b=runs.get(f'soap46-r3-basis-b{str(beta).replace(".","")}-seed{seed}-v1')
            if not a or not b:continue
            steps=sorted(set(a['val'])&set(b['val']))
            if not steps:continue
            found=True;ax.plot(steps,[a['val'][s]-b['val'][s] for s in steps],label=f'β₂={beta}')
        ax.axhline(0,color='black',lw=.8);ax.set(title=f'Development seed {seed}',xlabel='GPT iteration',ylabel='Parameter-frame minus basis-frame loss');ax.grid(alpha=.2)
        if ax.lines and len(ax.lines)>1:ax.legend()
    if found:
        body+=f'<figure><img src="{picture(fig)}" alt="Same-source coordinate-frame loss differences by development seed"><figcaption>Each curve compares parameter-frame and basis-frame residuals at the same β₂ and seed. These are development comparisons. The two seed-level contrasts, not four independent replicates, support any eventual frame conclusion.</figcaption></figure>'
    else:plt.close(fig)
    differences=workflow.get('confirmation_differences')
    if differences:
        body+='<p>The fixed coordinate-study finalist differs from its matched frozen controls by '+', '.join(f'{x:+.5f}' for x in differences)+f' nats/token on fresh seeds 6 and 7 (mean {statistics.mean(differences):+.5f}). These seeds were excluded from selection.</p>'
    elif workflow.get('phase')=='confirmation':
        body+='<p>The eight development runs are archived. The selected fixed setting and matched frozen controls are now running on fresh seeds 6 and 7.</p>'
    return body


def lr_cwd_section(config,workflow,runs):
    jobs=[j for phase in workflow.get('phase_history',[]) for j in phase['jobs']]+workflow.get('jobs',[])
    trials=[t for j in jobs for t in j['trials']]
    body=('<h2>Current study: GPT learning rate and weight decay</h2><p>Original methods A and B '
          'each receive four joint neighbors: GPT learning rate 0.03375 or 0.04125, crossed '
          'with cautious weight decay 0.01875 or 0.03125. Each method also gets a new default '
          'center at 0.0375 and 0.025. Earlier, coarser one-variable neighbors were tested; '
          'this screen tests tighter combinations rather than repeating those points. '
          'Here the learning-rate knob scales the hidden-matrix optimizer’s entire schedule, '
          'including its late power-law coefficient; the embedding and output-head Adam '
          'schedules remain unchanged. Cautious decay is the hidden-matrix decay coefficient.</p>'
          '<p>Residual peak learning rate stays 0.00001, direct residual decay 0.10, '
          'residual Adam betas (0.9, 0.99), clipping norm 1, and original basis coordinates. '
          'The zero-until-500/ramp-to-1000 schedule and 2900-step endpoint stay unchanged. '
          'All runs use the original <code>0950ad0</code> numerical source. A lower loss after '
          'changing GPT hyperparameters would describe a jointly tuned learned method, '
          'not an isolated effect of learning residuals.</p>')
    if workflow.get('phase') in ('review','complete'):
        body=body.replace('Current study: GPT learning rate and weight decay','Completed GPT learning-rate and weight-decay study')
    if workflow.get('development_comparisons'):
        differences=workflow['development_comparisons']
        facts=[]
        for row in differences:
            method='A' if row['candidate']['variant']=='inner-outer' else 'B'
            facts.append(f'{method} {row["difference"]:+.5f}')
        result='Development seed-3 loss differences versus the fixed control are '+', '.join(facts)+' nats/token. '
        result+=('Neither selected setting lowered loss; neither proceeds directly to fresh confirmation.'
                 if all(row['difference']>=0 for row in differences)
                 else 'Only a setting with a negative completed difference qualifies for fresh confirmation.')
        body=body.replace('</h2>','</h2><p><strong>'+result+'</strong></p>',1)
    body+='<div class=scroll><table><thead><tr><th>Trial</th><th>GPT LR</th><th>CWD</th><th>Status</th><th>Loss at 2900</th></tr></thead><tbody>'
    for t in trials:
        r=runs.get(t['name']);status='queued' if not r else 'archived' if r['archived'] else 'preserving' if r['completed'] else f'{r["step"]}/2900'
        loss='—' if not r or not r['completed'] else f'{r["val"][2900]:.5f}'
        body+=f'<tr><td>{t["name"].removeprefix("soap46-lrwd-")}</td><td>{t["base_lr"]:g}</td><td>{t["cwd"]:g}</td><td>{status}</td><td>{loss}</td></tr>'
    body+='</tbody></table></div>'
    for method in ('A','B'):
        center=runs.get(f'soap46-lrwd-{method}-cell4-seed0-v1')
        if not center:continue
        fig,ax=plt.subplots(figsize=(8,3.6));found=False
        for i,(lr,cwd) in enumerate(config['neighbors'][:-1]):
            r=runs.get(f'soap46-lrwd-{method}-cell{i}-seed0-v1')
            if not r:continue
            steps=sorted(set(r['val'])&set(center['val']))
            if not steps:continue
            found=True;ax.plot(steps,[r['val'][s]-center['val'][s] for s in steps],label=f'LR {lr:g}, CWD {cwd:g}')
        if found:
            ax.axhline(0,color='black',lw=.8);ax.set(title=f'Original method {method}: joint neighbors versus new center',xlabel='GPT iteration',ylabel='Validation-loss difference (nats/token)');ax.grid(alpha=.2);ax.legend(fontsize=8)
            body+=f'<figure><img src="{picture(fig)}" alt="Learning-rate and decay neighbors versus current center"><figcaption>Seed-0 tuning comparisons use the new same-source center for this method. Fresh paired confirmation follows selection.</figcaption></figure>'
        else:plt.close(fig)
    body+='<p>Selection includes the center, followed by development-seed replication versus the fixed published baseline. Only a promising selected method proceeds to fresh paired seeds 4 and 5; no equal confirmation quota is imposed on A. <a href="lr-cwd-plan.json">Exact bounded plan</a>.</p>'
    if workflow.get('selected'):
        choices=[]
        for t in workflow['selected']:
            method='A' if t['variant']=='inner-outer' else 'B'
            choices.append(f'method {method}: hidden learning rate {t["base_lr"]:g}, cautious decay {t["cwd"]:g}')
        body+='<p>The completed ten-run screen selected '+ '; '.join(choices)+'. The following stage compared each selected setting with its fixed published-baseline control on development seed 3. Selection on seed 0 is not confirmation of a gain.</p>'
    for job in jobs:
        if not job['id'].startswith(('development-', 'extension-development', 'confirmation-')):
            continue
        if 'shared_control' in job:
            pair_trials=[job['trials'][0],job['shared_control']]
        elif len(job['trials'])==2:
            pair_trials=job['trials']
        else:
            continue
        treatment,control=(runs.get(t['name']) for t in pair_trials)
        if not treatment or not control:
            continue
        steps=sorted(set(treatment['val'])&set(control['val']))
        if not steps:
            continue
        trial=job['trials'][0];method='A' if trial['variant']=='inner-outer' else 'B'
        stage='Fresh confirmation' if job['id'].startswith('confirmation-') else 'Development'
        fig,ax=plt.subplots(figsize=(8,3.6))
        ax.plot(steps,[treatment['val'][s]-control['val'][s] for s in steps],color='#176b45')
        ax.axhline(0,color='black',lw=.8)
        ax.set(title=f'{stage}: method {method}, seed {trial["seed"]}',xlabel='GPT iteration',ylabel='Learned method minus frozen control (nats/token)')
        ax.grid(alpha=.2)
        if treatment['archived'] and control['archived']:
            difference=treatment['val'][2900]-control['val'][2900]
            caption=f'The completed pair differs by {difference:+.5f} nats/token at iteration 2900. '
        else:
            caption='This pair is still running; its fixed-endpoint comparison is pending. '
        caption+='Both runs use the same seed and declared readout. Intermediate validation does not change either run’s optimizer or stopping step.'
        if 'shared_control' in job:
            caption+=' This is the same archived zero-residual control used for method A, not an independent second control run.'
        body+=f'<figure><img src="{picture(fig)}" alt="Same-seed learned SOAP versus fixed control validation loss difference"><figcaption>{caption}</figcaption></figure>'
    if any('shared_control' in j for j in jobs):
        body+='<p>The seed-3 development comparisons share one actual frozen control. With residuals held exactly zero, A and B have the same GPT weight update and baseline optimizer state updates; only their unused residual-gradient calculations differ. The conditional zero-residual map was verified against the pinned upstream optimizer for both variants. Sharing this control does not make the two differences independent, and does not assert bit-identical whole-run numerical trajectories.</p>'
    if workflow.get('extension_attempted'):
        body+='<p>After both initial development comparisons failed, a bounded A-only extension tests three previously untested lower-LR/lower-decay combinations: (0.030,0.01875), (0.03375,0.0125), and (0.030,0.0125). This follows A’s seed-0 boundary minimum, without treating its failed seed-3 result as a gain. All residual settings stay fixed. Only a new seed-0 winner receives one further development-seed comparison; fresh confirmation still requires a lower completed loss than the shared fixed control. B receives no new grid.</p>'
        selected=workflow.get('extension_selected')
        if selected:
            body+=f'<p>The completed extension selects hidden LR {selected["base_lr"]:g} / CWD {selected["cwd"]:g}, including the prior incumbent in the comparison. '
            comparison=workflow.get('extension_development_comparison')
            if comparison:
                body+=f'Its completed development-seed difference is {comparison["difference"]:+.5f} nats/token versus the same fixed control. '
            elif selected['name']==workflow['extension_incumbent']['name']:
                body+='The incumbent remains best, so its already-failed development comparison is not repeated. '
            else:
                body+='Its additional development-seed comparison is pending. '
            body+='These selected development results are separate from any fresh confirmation.</p>'
        incumbent=runs.get(workflow['extension_incumbent']['name'])
        if incumbent:
            fig,ax=plt.subplots(figsize=(8,3.6));found=False
            for t in trials:
                if t['seed']!=0 or not t['name'].startswith('soap46-lrwd-A-extension'):
                    continue
                r=runs.get(t['name'])
                if not r:continue
                steps=sorted(set(r['val'])&set(incumbent['val']))
                if not steps:continue
                found=True
                ax.plot(steps,[r['val'][s]-incumbent['val'][s] for s in steps],label=f'LR {t["base_lr"]:g}, CWD {t["cwd"]:g}')
            if found:
                ax.axhline(0,color='black',lw=.8)
                ax.set(title='A extension versus prior seed-0 boundary minimum',xlabel='GPT iteration',ylabel='Validation-loss difference (nats/token)')
                ax.grid(alpha=.2);ax.legend(fontsize=8)
                body+=f'<figure><img src="{picture(fig)}" alt="Three lower-learning-rate lower-decay development cells against the prior A incumbent"><figcaption>These are seed-0 tuning curves, not fresh confirmation. The reference is the earlier LR 0.03375 / CWD 0.01875 run. Only completed 2900-step endpoints enter selection; no intermediate validation changes a running trial.</figcaption></figure>'
            else:plt.close(fig)
    if workflow.get('confirmation_differences'):
        ds=workflow['confirmation_differences'];body+='<p>Fresh paired differences: '+', '.join(f'{d:+.5f}' for d in ds)+f' nats/token (mean {statistics.mean(ds):+.5f}).</p>'
    if workflow.get('phase')=='review':body+='<p>'+workflow.get('reason','Neither selected setting improved its development comparison.').replace('fresh seeds remain untouched','this selected A setting did not proceed to fresh confirmation')+' Automatic confirmation from that selection was withheld.</p>'
    return body


def effect_variability_section(output):
    path=ROOT/'runs/soap46-lr-cwd/effect_variability.json'
    if not path.exists():return ''
    evidence=json.loads(path.read_text())
    temporary=output/'effect-variability.json.incomplete'
    temporary.write_text(json.dumps(evidence,indent=2));temporary.replace(output/'effect-variability.json')
    body=('<h2>How the assigned comparisons vary across seeds</h2>'
          '<p>The original B center improved both historical confirmation seeds, but its two '
          'later seed-3 development repetitions did not. The original A center lost on both '
          'historical confirmation seeds. These stages remain separate: seed 0 selected settings, '
          'and seed 3 has been used adaptively for development. Two repetitions of seed 3 are '
          'not two independent seeds. This table covers the original recipes and initial '
          'LR/CWD selections, before the later extension.</p>'
          '<div class=scroll><table><thead><tr><th>Recipe</th><th>Seed</th><th>Stage</th>'
          '<th>Learned minus assigned control</th><th>Same base LR/CWD?</th></tr></thead><tbody>')
    for row in evidence['rows']:
        body+=f'<tr><td>{row["recipe"]}</td><td>{row["seed"]}</td><td>{row["stage"]}</td><td>{row["difference"]:+.5f}</td><td>{"yes" if row["same_base_hyperparameters"] else "no"}</td></tr>'
    body+='</tbody></table></div>'
    fig,ax=plt.subplots(figsize=(8,3.8))
    positions={(0,'historical tuning'):0,(1,'historical confirmation'):1,
               (2,'historical confirmation'):2,(3,'earlier development'):3,
               (3,'current development'):4}
    for recipe,marker,color in [('A original center','o','#b34c36'),
                                ('B original center','s','#176b45'),
                                ('A selected lower-LR/lower-CWD','x','#6b4a9f')]:
        selected=[row for row in evidence['rows'] if row['recipe']==recipe]
        ax.scatter([positions[row['seed'],row['stage']] for row in selected],
                   [row['difference'] for row in selected],marker=marker,color=color,label=recipe,s=50)
    ax.axhline(0,color='black',lw=.8)
    ax.axvspan(.5,2.5,color='#708090',alpha=.08)
    ax.set_xticks(range(5),['Seed 0\ntuning','Seed 1\nhistorical','Seed 2\nhistorical','Seed 3\ndev repeat 1','Seed 3\ndev repeat 2'])
    ax.set(ylabel='Learned minus assigned control (nats/token)',title='Assigned effects change across seeds and repetitions')
    ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    body+=f'<figure><img src="{picture(fig)}" alt="Actual assigned SOAP effects across tuning, historical confirmation, and development seeds"><figcaption>Only seeds 1 and 2, shaded, belong to the earlier confirmation stage. Seed 0 is tuning, and the two seed-3 repetitions are development observations of the same seed. The tuned-A point also changes base LR/CWD. No missing pair is filled in.</figcaption></figure>'
    body+='<p>Across historical confirmation seeds 1 and 2 only, '
    for method in ('A','B'):
        summary=evidence['historical_confirmation_only'][method]
        body+=f'{method} has mean difference {summary["mean"]:+.5f} and sample SD {summary["sample_sd"]:.5f} nats/token (two pairs). '
    body+=('These are descriptive summaries, not a pooled confirmation estimate using tuning '
           'and development runs. The current tuned-A comparison changes base hyperparameters '
           'as well as the learning method; a frozen control at those same hyperparameters '
           'is missing. The new seed-0 center repetitions have no newly assigned frozen '
           'controls, so none are invented. Current A and B share one explicitly qualified '
           'seed-3 control. <a href="effect-variability.json">Exact run identities, missing '
           'pairs, and provenance</a>.</p>')
    return body


def uncertainty_section(output,study,runs):
    if not study:return ''
    temporary=output/'fixed-b-uncertainty-plan.json.incomplete'
    temporary.write_text(json.dumps(study['config'],indent=2));temporary.replace(output/'fixed-b-uncertainty-plan.json')
    body=('<h2>Separate fixed-B uncertainty study</h2><p>This predeclared study asks whether '
          'the unchanged original B recipe has a repeatable effect on two unused seeds, '
          'given its historical gains on seeds 1 and 2 and losses in seed-3 development '
          'repetitions. It is not winner confirmation of the failed LR/CWD selection. '
          'Base LR 0.0375, CWD 0.025, and every residual setting stay fixed. Seeds 4 and 5 '
          'each receive a treatment and fixed published-baseline control; all four endpoints '
          'are reported regardless of the first pair. The primary statistic is the mean '
          'of the two treatment-minus-control differences, with individual differences '
          'shown separately and historical outcomes excluded from that mean.</p>')
    if study.get('deferred'):
        return body+'<p>This study has not launched: qualifying A confirmation has priority, and B scheduling is deferred.</p>'
    if not study.get('activation'):
        return body+'<p>No GPU run has launched for this study. It activates only if the bounded A extension yields no qualifying candidate. <a href="fixed-b-uncertainty-plan.json">Frozen plan</a>.</p>'
    body+='<div class=scroll><table><thead><tr><th>Seed</th><th>Run</th><th>Status</th><th>Loss at 2900</th></tr></thead><tbody>'
    for seed in (4,5):
        for frozen in (False,True):
            name=f'soap46-B-uncertainty-{"frozen-" if frozen else ""}seed{seed}-v1';r=runs.get(name)
            status='queued' if not r else 'archived' if r['archived'] else 'preserving' if r['completed'] else f'{r["step"]}/2900'
            loss='—' if not r or not r['completed'] else f'{r["val"][2900]:.5f}'
            body+=f'<tr><td>{seed}</td><td>{"control" if frozen else "B"}</td><td>{status}</td><td>{loss}</td></tr>'
    body+='</tbody></table></div>'
    fig,ax=plt.subplots(figsize=(7,3.4));found=False
    for seed in (4,5):
        treatment=runs.get(f'soap46-B-uncertainty-seed{seed}-v1')
        control=runs.get(f'soap46-B-uncertainty-frozen-seed{seed}-v1')
        if not treatment or not control:continue
        steps=sorted(set(treatment['val'])&set(control['val']))
        if not steps:continue
        found=True
        ax.plot(steps,[treatment['val'][step]-control['val'][step] for step in steps],label=f'Fresh seed {seed}')
    if found:
        ax.axhline(0,color='#555',lw=.8)
        ax.set(xlabel='GPT iteration',ylabel='B minus its paired control (nats/token)')
        ax.grid(alpha=.15);ax.legend()
        body+=f'<figure><img src="{picture(fig)}" alt="Fixed original B validation differences against its own frozen control on fresh seeds four and five"><figcaption>Each fresh seed is compared only with its assigned control at the same iteration. Negative values favor B. The predeclared primary result uses both completed 2900-step differences, not intermediate minima.</figcaption></figure>'
    else:plt.close(fig)
    prefixes=[]
    for seed in (4,5):
        check=ROOT/'runs/soap46-b-uncertainty'/f'prefix_checks_seed{seed}.json'
        if check.exists():
            prefix=json.loads(check.read_text())
            difference=prefix.get('validation_treatment_minus_control',{}).get('500')
            if difference is not None:prefixes.append(f'seed {seed}: {difference:+.5f}')
    if prefixes:
        body+='<p>The recorded zero-action prefixes differ despite matching first-three training losses and X remaining zero through iteration 500. Validation differences at iteration 500 are '+', '.join(prefixes)+' nats/token. This repeat variability limits attribution of small endpoint differences to learned residuals. No prefix-gap subtraction or within-run adjustment is applied.</p>'
    result=study.get('summary')
    if result:
        body+='<p>Individual fresh-seed differences: '+', '.join(f'seed {row["seed"]}: {row["difference"]:+.5f}' for row in result['pairs'])+f' nats/token. The predeclared mean is {result["primary_mean_difference"]:+.5f}.</p>'
    else:body+='<p>The primary two-seed result remains pending until all four archived endpoints are available.</p>'
    return body



def plot_xlr_screen(output,study):
    state=study.get('workflow',{})
    if not state.get('screen_results'):return
    rows=sorted(state['screen_results'],key=lambda r:(r['trial']['meta_lr'],r['trial']['residual_decay']))
    threshold=study['config']['selection']['minimum_improvement']
    fig,ax=plt.subplots(figsize=(5.6,3.7),layout='constrained')
    ax.axhline(0,color='#66717b',lw=1)
    ax.axhline(-threshold,color='#82629b',lw=1,ls='--',label=f'Practical screen: −{threshold:g}')
    values=[r['difference'] for r in rows]
    ax.scatter(range(len(rows)),values,s=65,color='#28678b',zorder=3)
    for i,r in enumerate(rows):
        below=-threshold<r['difference']<0
        ax.annotate(f'{r["difference"]:+.6f}',(i,r['difference']),xytext=(12,0) if below else (0,9),
                    textcoords='offset points',ha='left' if below else 'center',
                    va='center' if below else 'baseline',fontsize=10,
                    bbox=dict(facecolor='white',edgecolor='none',pad=1))
    ax.set_xticks(range(len(rows)),[f'X LR {r["trial"]["meta_lr"]:.0e}\ndecay {r["trial"]["residual_decay"]:.0%}\nlane {r["lane"]}' for r in rows],fontsize=10)
    span=max(max(values+[0])-min(values+[-threshold]),.0005)
    ax.set_ylim(min(values+[-threshold])-.18*span,max(values+[0])+.28*span)
    ax.set_xlim(-.5,len(rows)-.5);ax.set_ylabel('Loss difference (nats/token)')
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    ax.legend(fontsize=9,frameon=False,loc='lower right')
    for ext in ('svg','png'):
        temporary=output/f'residual_lr_decay_evidence.{ext}.incomplete'
        fig.savefig(temporary,format=ext,dpi=180);temporary.replace(output/f'residual_lr_decay_evidence.{ext}')
    plt.close(fig)
    data={'scope':'One development seed; each cell uses its prospectively assigned host control. No significance claim.','rows':rows,'controls':state['controls'],'minimum_improvement':threshold,'selected':state.get('selected'),'units':'tail-EMA validation nats/token at2900 ordinary GPT iterations','numerical_source':study['config']['numeric_source'],'selection':study['config']['selection']}
    temporary=output/'residual_lr_decay_evidence.json.incomplete';temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(output/'residual_lr_decay_evidence.json')
    study['evidence_figure']='residual_lr_decay_evidence.svg'
    pairs=state.get('confirmation_pairs')
    if not pairs:return
    mean=state['primary_confirmation_mean'];values=[p['difference'] for p in pairs]
    fig,ax=plt.subplots(figsize=(5.2,3.5),layout='constrained')
    ax.axhline(0,color='#66717b',lw=1)
    ax.axhline(mean,color='#82629b',lw=1.3,ls='--',label=f'Fresh paired mean: {mean:+.6f}')
    ax.scatter(range(len(pairs)),values,s=75,color='#28678b',zorder=3)
    for i,value in enumerate(values):
        ax.annotate(f'{value:+.6f}',(i,value),xytext=(0,10),textcoords='offset points',
                    ha='center',fontsize=11,bbox=dict(facecolor='white',edgecolor='none',pad=1))
    span=max(max(values+[0])-min(values+[0]),.0005)
    ax.set_ylim(min(values+[0])-.2*span,max(values+[0])+.4*span)
    ax.set_xticks(range(len(pairs)),[f'Fresh seed {p["seed"]}' for p in pairs])
    ax.set_xlim(-.45,len(pairs)-.55);ax.set_ylabel('Loss difference (nats/token)')
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    ax.legend(fontsize=9,frameon=False,loc='lower right')
    for ext in ('svg','png'):
        temporary=output/f'residual_lr_decay_confirmation.{ext}.incomplete'
        fig.savefig(temporary,format=ext,dpi=180);temporary.replace(output/f'residual_lr_decay_confirmation.{ext}')
    plt.close(fig)
    data={'scope':'Predeclared fresh paired confirmation only; development and historical seeds excluded. Both outcomes retained, no significance claim.','pairs':pairs,'primary_mean_difference':mean,'selected_recipe':state['selected']['trial'],'numerical_source':study['config']['numeric_source'],'readout':study['config']['readout']}
    temporary=output/'residual_lr_decay_confirmation.json.incomplete';temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(output/'residual_lr_decay_confirmation.json')
    study['confirmation_figure']='residual_lr_decay_confirmation.svg'


def xlr_decay_section(study,runs):
    if not study:return ''
    body=('<h2>Residual learning rate and direct decay</h2><p>The new six-run '
          'development study keeps GPT learning rate 0.0375 and CWD 0.025 fixed. '
          'It crosses residual Adam peak learning rate 0.00001 or 0.0001 with direct '
          'residual decay 0.01 or 0.03. The former scales learning of X; the latter '
          'multiplies X by 0.99 or 0.97 before each active Adam update, independently '
          'of learning rate. Adam betas, clipping, basis coordinates, and the '
          '500–1000 schedule remain fixed.</p><p>Each of two nodes first runs its '
          'own prospective frozen seed-3 control, then two assigned cells. All six '
          'endpoints are required. A candidate must improve its assigned control by '
          'at least 0.0005 nats/token and finish below both control endpoints. This '
          'is a practical screening rule, not statistical significance. The frozen '
          'ranking selects at most one recipe for fresh paired seeds 6 and 7; both '
          'pairs are reported regardless of the first result. '
          '<a href="residual-lr-decay-plan.json">Frozen plan</a>.</p>')
    state=study.get('workflow',{});jobs=[j for phase in state.get('phase_history',[]) for j in phase['jobs']]+state.get('jobs',[])
    if not jobs:return body+'<p>Prepared; no run has launched for this stage.</p>'
    body+='<div class=scroll><table><thead><tr><th>Stage / lane</th><th>Run</th><th>X peak LR</th><th>X decay</th><th>Status</th><th>Loss at 2900</th></tr></thead><tbody>'
    seen=set()
    for job in jobs:
        for t in job['trials']:
            if t['name'] in seen:continue
            seen.add(t['name']);r=runs.get(t['name'])
            status='queued' if not r else 'archived' if r['archived'] else 'preserving' if r['completed'] else f'{r["step"]}/2900'
            loss='—' if not r or not r['completed'] else f'{r["val"][2900]:.5f}'
            body+=f'<tr><td>{job["id"]}</td><td>{t["name"].removeprefix("soap46-xlrdecay-")}</td><td>{t["meta_lr"]:g}</td><td>{t["residual_decay"]:g}</td><td>{status}</td><td>{loss}</td></tr>'
    body+='</tbody></table></div>'
    if state.get('controls'):
        body+='<p>Prospective control endpoints: '+', '.join(f'lane {r["lane"]}: {r["loss"]:.6f}' for r in state['controls'])+'. Assigned treatment-minus-control differences: '+', '.join(f'{r["trial"]["meta_lr"]:g}/{r["trial"]["residual_decay"]:g}: {r["difference"]:+.6f}' for r in state['screen_results'])+'. All cells are shown, including failures.</p>'
    if study.get('evidence_figure'):
        finding=('One recipe cleared the development screen.' if state.get('selected') else 'No recipe cleared the development screen.')
        body+=f'<figure><img src="{study["evidence_figure"]}" alt="Four residual learning-rate and decay cells compared with their assigned frozen controls"><figcaption>{finding} These are four settings on one development seed, not four independent seeds. Negative is lower loss; the dashed boundary is a practical screen, not significance. <a href="residual_lr_decay_evidence.json">Exact values</a>.</figcaption></figure>'
    if state.get('confirmation_pairs'):
        body+='<p>Fresh paired differences: '+', '.join(f'seed {r["seed"]}: {r["difference"]:+.6f}' for r in state['confirmation_pairs'])+f'; mean {state["primary_confirmation_mean"]:+.6f} nats/token.</p>'
        body+=f'<figure><img src="{study["confirmation_figure"]}" alt="Fresh seed6 and7 validation-loss differences versus their matched frozen controls"><figcaption>The selected recipe is evaluated on two fresh paired seeds. Negative is lower loss. The dashed line averages only these two differences; development and historical results are excluded. This does not establish statistical significance. <a href="residual_lr_decay_confirmation.json">Exact values and recipe</a>.</figcaption></figure>'
    if study.get('method_coverage'):
        body+=('<p>A has also been tested: five residual-rate/decay cells and twelve '
               'distinct GPT learning-rate/CWD cells on development seed 0. Its '
               'original-center confirmation and both selected learning-rate/CWD '
               'replications were worse than their assigned controls. This motivated '
               'B’s current priority; it does not rule out every A setting. '
               'The compact JSON and reproduction archive include exact coverage '
               'and A’s substantial gradient-clipping diagnostics. No fresh A '
               'confirmation was launched after those failed development gates.</p>')
    return body


def build(output):
    plan=json.loads((ROOT/'configs/soap46_tuning_round2.json').read_text())
    trials=plan['lane1_trials']+plan['lane2_trials']
    runs={t['name']:load_run(t['name']) for t in trials}
    workflow_path=ROOT/'runs/soap46-round2/coordinator.json'
    workflow=json.loads(workflow_path.read_text()) if workflow_path.exists() else {}
    related=[j for phase in workflow.get('phase_history',[]) for j in phase['jobs']]+workflow.get('jobs',[])
    for job in related:
        for trial in job['trials']:
            runs.setdefault(trial['name'],load_run(trial['name']))
    conditioning_state_path=ROOT/'runs/soap46-conditioning/coordinator.json'
    conditioning_workflow=json.loads(conditioning_state_path.read_text()) if conditioning_state_path.exists() else {}
    conditioning_jobs=[j for phase in conditioning_workflow.get('phase_history',[]) for j in phase['jobs']]+conditioning_workflow.get('jobs',[])
    for job in conditioning_jobs:
        for trial in job['trials']:runs.setdefault(trial['name'],load_run(trial['name']))
    lr_cwd_state_path=ROOT/'runs/soap46-lr-cwd/coordinator.json'
    lr_cwd_workflow=json.loads(lr_cwd_state_path.read_text()) if lr_cwd_state_path.exists() else {}
    for job in [j for phase in lr_cwd_workflow.get('phase_history',[]) for j in phase['jobs']]+lr_cwd_workflow.get('jobs',[]):
        for trial in job['trials']:runs.setdefault(trial['name'],load_run(trial['name']))
    uncertainty={};uncertainty_config=ROOT/'configs/soap46_b_uncertainty.json'
    if uncertainty_config.exists():
        uncertainty['config']=json.loads(uncertainty_config.read_text())
        for label in ('activation','deferred','summary'):
            path=ROOT/'runs/soap46-b-uncertainty'/f'{label}.json'
            if path.exists():uncertainty[label]=json.loads(path.read_text())
        for seed in (4,5):
            for frozen in (False,True):
                name=f'soap46-B-uncertainty-{"frozen-" if frozen else ""}seed{seed}-v1'
                runs.setdefault(name,load_run(name))
    xlr={};xlr_config=ROOT/'configs/soap46_xlr_decay.json'
    if xlr_config.exists():
        xlr['config']=json.loads(xlr_config.read_text())
        state=ROOT/'runs/soap46-xlr-decay/coordinator.json'
        xlr['workflow']=json.loads(state.read_text()) if state.exists() else {}
        for key in ('method_coverage','prefix_repeat_checks'):
            diagnostic=ROOT/'runs/soap46-xlr-decay'/f'{key}.json'
            if diagnostic.exists():xlr[key]=json.loads(diagnostic.read_text())
        for job in [j for phase in xlr['workflow'].get('phase_history',[]) for j in phase['jobs']]+xlr['workflow'].get('jobs',[]):
            for t in job['trials']:runs.setdefault(t['name'],load_run(t['name']))
    reference_name='soap46-r2-B-frozen-seed0-v1'
    reference=runs[reference_name]
    if reference is None or not reference['completed']:
        reference_name='soap46-B-ramp500-1000-d10-frozen-seed0-v1';reference=load_run(reference_name)
    runs[reference_name]=reference
    historical_control_name='soap46-B-ramp500-1000-d10-frozen-seed0-v1'
    historical_control=load_run(historical_control_name);runs[historical_control_name]=historical_control
    prior_name='soap46-B-ramp500-1000-d10-lr1e-5-seed0-v1'
    prior=load_run(prior_name);runs[prior_name]=prior
    output.mkdir(parents=True,exist_ok=True)
    plot_xlr_screen(output,xlr)
    plots=[]
    for heading,selected in [('Residual learning rate and decay',[t for t in trials if t['variant']=='outer-only' and '-lr' in t['name']]),('Adam memory and gradient clipping',[t for t in trials if 'beta' in t['name'] or 'clip' in t['name']])]:
        fig,ax=plt.subplots(figsize=(8,4));found=False
        for t in selected:
            r=runs[t['name']]
            if not r:continue
            steps=sorted(set(r['val'])&set(reference['val']))
            if not steps:continue
            found=True;ax.plot(steps,[r['val'][s]-reference['val'][s] for s in steps],marker='.',label=t['name'].removeprefix('soap46-r2-').removesuffix('-seed0-v1'))
        if found:
            ax.axhline(0,color='black',lw=.8);ax.set(xlabel='GPT training iteration',ylabel='Validation loss minus frozen control (nats/token)',title=heading);ax.grid(alpha=.2);ax.legend(fontsize=8)
            plots.append(picture(fig))
        else:plt.close(fig)
    table=[]
    for t in trials:
        r=runs[t['name']];loss=r['val'].get(2900) if r else None
        delta=loss-reference['val'][2900] if loss is not None else None
        status='awaiting node' if not r else ('archived' if r['archived'] else 'completed; preserving' if r['completed'] else str(r['step'])+'/2900')
        values=[t['name'].removeprefix('soap46-r2-').removesuffix('-seed0-v1'),status,f"{t['meta_lr']:g}",f"{t['residual_decay']:g}",str(t['meta_betas']),f"{t['meta_clip']:g}",'-' if loss is None else f'{loss:.5f}','-' if delta is None else f'{delta:+.5f}','-' if not r or r['clipped_fraction'] is None else f"{100*r['clipped_fraction']:.1f}%"]
        table.append('<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
    completed=sum(bool(runs[t['name']] and runs[t['name']]['completed']) for t in trials)
    eligible=[runs[t['name']] for t in trials if t['variant']=='outer-only' and t['warmup_steps']==500
              and runs[t['name']] and runs[t['name']]['completed']]
    finding='The new full-length settings are still running.'
    if eligible:
        best=min(eligible,key=lambda r:r['val'][2900])
        finding=(f"The lowest completed new outer-only loss on seed 0 is {best['val'][2900]:.4f}, "
                 f"versus {prior['val'][2900]:.4f} for the previously chosen setting. "
                 'These are tuning measurements; they do not establish a gain on fresh seeds.')
    if workflow.get('confirmation_differences'):
        differences=workflow['confirmation_differences']
        finding=('The fixed finalist has validation-loss differences of '+
                 ' and '.join(f'{x:+.4f}' for x in differences)+
                 f' nats/token on fresh paired seeds 4 and 5 (mean {statistics.mean(differences):+.4f}). '+
                 'These seeds were not used for setting selection; two pairs do not establish statistical significance.')
    now=datetime.now(timezone.utc)
    public_runs={name:({k:v for k,v in run.items() if k!='train'}|{
        'raw_record':f'reproduction.zip:runs/soap46-v1/{name}/',
        'training_rows':len(run['train']),
        'last_train':run['train'][-1] if run['train'] else None}) if run else None
        for name,run in runs.items()}
    summary=dict(schema=2,raw_data='Complete per-step training rows are in reproduction.zip; this publication summary retains validation curves, endpoint/status, timing aggregates, clipping fraction and last training row.',generated_utc=now.isoformat(),generated_pt=now.astimezone(ZoneInfo('America/Los_Angeles')).strftime('%B %d, %H:%M PT'),plan=plan,reference_name=reference_name,runs=public_runs,completed=completed,workflow=workflow,conditioning_workflow=conditioning_workflow,lr_cwd_workflow=lr_cwd_workflow)
    summary['fixed_b_uncertainty_study']=uncertainty
    summary['residual_lr_decay_study']=xlr
    if xlr:
        tmp=output/'residual-lr-decay-plan.json.incomplete';tmp.write_text(json.dumps(xlr['config'],indent=2));tmp.replace(output/'residual-lr-decay-plan.json')
    current_plan_path=ROOT/'configs/soap46_lr_cwd_round3.json'
    if current_plan_path.exists():
        summary['current_plan']=json.loads(current_plan_path.read_text())
        summary['plan_scope']='plan/workflow describe the completed historical residual-optimizer screen; current_plan/lr_cwd_workflow describe the active original-A/B LR/CWD screen. Coordinate-frame and reset plans are held.'
    if xlr.get('workflow'):
        summary['base_lr_cwd_plan']=summary.get('current_plan')
        summary['current_plan']=xlr['config']
        summary['plan_scope']='Current plan is the original-B residual learning-rate/direct-decay stage. Earlier optimizer and GPT LR/CWD studies remain separately identified; all Adam/frame/reset changes remain held.'
    temporary=output/'summary.json.incomplete';temporary.write_text(json.dumps(summary,indent=2));temporary.replace(output/'summary.json')
    figures=''.join(f'<figure><img src="{src}" alt="Validation difference curves"><figcaption>Figure {i+1}. These seed-0 curves are hyperparameter-selection measurements. Negative values are lower than the named frozen control; they are not fresh-seed confirmation.</figcaption></figure>' for i,src in enumerate(plots))
    development=''
    if workflow.get('phase')=='development':
        development='<h2>Seed-3 development comparison is running</h2><p>'
        labels=[]
        for job in workflow['jobs']:
            treatment,control=job['trials'];r=runs.get(treatment['name']);c=runs.get(control['name'])
            treatment_value=(f'{r["val"][2900]:.5f}' if r and 2900 in r['val'] else 'pending')
            control_status=(f'{c["val"][2900]:.5f}' if c and 2900 in c['val'] else f'{c["step"]}/2900 iterations' if c else 'queued')
            labels.append(f'Residual Adam β₂={treatment["meta_betas"][1]}: treatment {treatment_value}; matched frozen control {control_status}')
        development+='; '.join(labels)+'. Selection waits for both complete archived pairs; these are development measurements.</p>'
    if workflow.get('development_comparisons'):
        ending=('The additional optimizer-design study was held after these comparisons; no fresh confirmation followed from it.' if workflow.get('phase')=='held_after_development' else 'This seed selected the finalist and is excluded from fresh confirmation.')
        development='<h2>Earlier residual-Adam development comparison on seed 3</h2><p>'+ '; '.join(f"{r['candidate']['name']}: loss difference {r['difference']:+.5f} versus its matched frozen control" for r in workflow['development_comparisons'])+'. '+ending+'</p>'
    if workflow.get('phase')=='confirmation':
        development+='<p>Fresh paired confirmation on seeds 4 and 5 is now running with the selected settings fixed. Completed pairs will be reported separately from selection.</p>'
    if workflow.get('confirmation_differences'):
        differences=workflow['confirmation_differences']
        development+='<h2>Fresh paired seeds 4 and 5</h2><p>Fixed-finalist validation-loss differences versus matched frozen controls: '+', '.join(f'{x:+.5f}' for x in differences)+f' nats/token; mean {statistics.mean(differences):+.5f}. These two pairs were not used to select the setting. They do not establish statistical significance.</p>'
    page=f"""<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>SOAP residual tuning — round 2</title><style>body{{font:17px/1.55 system-ui;max-width:1050px;margin:auto;padding:24px;color:#17212a}}h1{{line-height:1.15}}img{{width:100%}}figure{{margin:32px 0}}figcaption{{font-size:15px;color:#49545e}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;font-size:14px;white-space:nowrap}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}td:first-child{{text-align:left}}code{{overflow-wrap:anywhere}}a{{color:#006da8}}</style><h1>Can further residual tuning improve outer-only SOAP?</h1><p>{completed} of the 12 earlier full-length screening runs have completed. The previous outer-only method lowered final validation loss on both historical confirmation seeds, by 0.0007 and 0.0013 nats/token. That earlier search tested residual learning rate, direct decay, Adam memory, and clipping while keeping the accepted record-46 optimizer settings fixed.</p><p>Each run uses 2900 ordinary GPT iterations. Residual learning rate is zero through step 500, ramps to its peak at step 1000, then stays constant. Direct residual decay is independent of learning rate. Every setting starts from fresh GPT, residual, and optimizer states.</p><p>Reference for that earlier screen: <code>{reference_name}</code>. That search used seed 0 for tuning and seed 3 for development; it did not launch fresh confirmation after its development results. Subsequent studies are described separately above. Historical seeds 1 and 2 are not relabeled as untouched evidence for later selections.</p>{figures}<h2>Declared settings and fixed endpoints</h2><div class=scroll><table><thead><tr>{''.join('<th>'+x+'</th>' for x in ['Trial','Status','Peak LR','Direct decay','Adam betas','Clip norm','Loss at 2900','Difference','Updates clipped'])}</tr></thead><tbody>{''.join(table)}</tbody></table></div><p>The single inner-and-outer diagnostic uses a smaller rate because the earlier setting clipped roughly 40–47% of hypergradients and lost on both confirmation seeds. Outer-only gradients never clipped under the old norm-1 threshold.</p><h2>Reproduction</h2><p>The numerical trainer remains source <code>0950ad0d323b042c81740b27b9526f820dbd2633</code>; orchestration exposes its existing Adam and clipping options. Model precision, data order, global batch, and published tail-EMA readout remain fixed. The earlier residual-optimizer screen used base learning rate 0.0375 and cautious decay 0.025 throughout; the completed original-A/B screen varied those two hidden-matrix settings as listed above. Frozen controls retain the published defaults. All custom source and all eight final checkpoint shards are preserved per run.</p><p><a href=summary.json>Compact metrics and plans</a> · <a href=reproduction.zip>Available run records</a></p><p>Updated {summary['generated_utc']}.</p></html>"""
    page=page.replace('</h1>','</h1><p><strong>'+finding+'</strong></p>',1)
    if (ROOT/'runs/soap46-conditioning/held_by_user_20261110_0348.json').exists():
        page=page.replace('</h1>','</h1><p><strong>Scope update:</strong> unlaunched coordinate-frame '
            'and moment-reset studies are held. The earlier seed-3 comparisons are complete; '
            'no fresh confirmation followed from that optimizer-design study. Current tuning concerns learning rate '
            'and weight decay for the original A/B methods.</p>',1)
    page=page.replace('Updated '+summary['generated_utc'],'Updated '+summary['generated_pt'])
    if reference_name!=historical_control_name:
        repeat=(f'<p>The new frozen seed-0 control ends at {reference["val"][2900]:.4f}, '
                f'versus {historical_control["val"][2900]:.4f} in the earlier execution with '
                'the same optimizer settings and seed. This observed repeat difference is why '
                'small seed-0 ranking differences need fresh paired confirmation. '
                'The plots now use the new control.</p>')
        page=page.replace('<h2>Declared settings and fixed endpoints</h2>',repeat+'<h2>Declared settings and fixed endpoints</h2>')
    method=('<p>Both methods learn additive corrections to SOAP’s basis matrices. For each of the 72 hidden weight matrices, '
            'method B adds trainable residual matrices to the two SOAP bases only when transforming '
            'the variance-scaled update back into parameter coordinates. Method A also changes the '
            'initial projection. The underlying SOAP covariance and variance updates, Muon polar step, '
            'and remaining baseline operations stay fixed. Residuals learn from the next ordinary '
            'GPT gradient, with no extra GPT derivative evaluations. Embedding and output-head '
            'optimizers stay unchanged. <a href="/soap-results/">The earlier report</a> gives '
            'the implementation and previous paired-seed results.</p>')
    page=page.replace('<p>Each run uses 2900',method+'<p>Each run uses 2900',1)
    memory=('<p><strong>Which Adam memory is being tuned?</strong> The table’s betas belong only '
            'to Adam for the learned residual matrices X. The published SOAP covariance and '
            'projected-square averages still use 0.90, and their bases refresh every step. '
            'Residual Adam β₂=0.999 therefore does not change SOAP’s own β₂. Compared with 0.99, '
            'it averages squared hypergradients over a much longer history (roughly 1000 versus '
            '100 updates), while the residual matrices themselves shrink by the separately '
            'specified direct decay. Whether those different timescales help is an empirical question.</p>'
            '<p>The <a href="https://github.com/nikhilvyas/SOAP">official SOAP example</a> uses '
            'betas (0.95, 0.95). The <a href="https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md">'
            'official Distributed Shampoo guide</a> distinguishes preconditioner, grafting, and '
            'gradient-filtering memories. These are starting points for those optimizers, '
            'not evidence that 0.999 is optimal for our learned-residual Adam.</p>')
    page=page.replace('<h2>Declared settings and fixed endpoints</h2>',memory+'<h2>Declared settings and fixed endpoints</h2>',1)
    conditioning_path=ROOT/'configs/soap46_conditioning_round3.json'
    if conditioning_path.exists():
        conditioning=json.loads(conditioning_path.read_text())
        (output/'conditioning-plan.json.incomplete').write_text(json.dumps(conditioning,indent=2))
        (output/'conditioning-plan.json.incomplete').replace(output/'conditioning-plan.json')
        next_study=conditioning_section(conditioning,conditioning_workflow,runs)
        motion_path=ROOT/'runs/soap46-round2/basis_motion_rank0_step1000_summary.json'
        if motion_path.exists() and not (ROOT/'runs/soap46-conditioning/held_by_user_20261110_0348.json').exists():
            motion=json.loads(motion_path.read_text())
            next_study+=('<p>One saved step-1000 transition motivates the comparison: '
                f'{100*motion["matched_to_different_column_fraction"]:.1f}% of 576 sampled '
                'basis columns matched a different previous column. Holding X fixed changed '
                'its sampled weight-coordinate representation substantially. This was a '
                'CPU-only inspection of one rank and one iteration, not a measurement of '
                'all training steps or evidence that the new parameterization improves loss.</p>')
        page=page.replace('<h2>Reproduction</h2>',next_study+'<h2>Reproduction</h2>',1)
    page=page.replace('<h2>Reproduction</h2>',development+'<h2>Reproduction</h2>')
    lr_cwd_plan_path=ROOT/'configs/soap46_lr_cwd_round3.json'
    if lr_cwd_plan_path.exists():
        lr_cwd_config=json.loads(lr_cwd_plan_path.read_text())
        tmp=output/'lr-cwd-plan.json.incomplete';tmp.write_text(json.dumps(lr_cwd_config,indent=2));tmp.replace(output/'lr-cwd-plan.json')
        page=page.replace('Can further residual tuning improve outer-only SOAP?','SOAP: learning rate and weight decay tuning')
        page=page.replace(method,'',1)
        page=page.replace('</h1>','</h1>'+method+lr_cwd_section(lr_cwd_config,lr_cwd_workflow,runs)+effect_variability_section(output)+uncertainty_section(output,uncertainty,runs)+xlr_decay_section(xlr,runs)+'<h2>Earlier residual-optimizer screen</h2>',1)
    if uncertainty.get('summary'):
        fresh=uncertainty['summary'];lead=(f'<p><strong>Fresh fixed-B checks did not show a benefit:</strong> '
            f'seed 4 differed by {fresh["pairs"][0]["difference"]:+.6f} and seed 5 by '
            f'{fresh["pairs"][1]["difference"]:+.6f} nats/token, mean '
            f'{fresh["primary_mean_difference"]:+.6f}. Positive is worse than each assigned control. '
            'The zero-action prefixes already differed, which limits attribution of small '
            'endpoint changes. No prefix subtraction was applied. The subsequent residual-learning-rate '
            'and direct-decay study is reported separately above; GPT baseline settings stayed fixed.</p>')
        page=page.replace('</h1>','</h1>'+lead,1)
    confirmed=xlr.get('workflow',{}).get('confirmation_pairs')
    if confirmed:
        state=xlr['workflow'];selected=state['selected']['trial']
        lower=sum(p['difference']<0 for p in confirmed)
        finding=('lowered the endpoint on both fresh paired runs' if lower==2 else
                 'had mixed fresh paired results' if lower==1 else
                 'did not lower the endpoint on either fresh paired run')
        latest=(f'<p><strong>The newly selected B recipe {finding}.</strong> '
                f'With residual peak learning rate {selected["meta_lr"]:g} and '
                f'direct decay {selected["residual_decay"]:.0%}, the predeclared '
                f'fresh-seed mean difference is {state["primary_confirmation_mean"]:+.6f} '
                'nats/token versus the matched frozen controls at iteration 2900. '
                'GPT learning rate and CWD stayed at their published defaults. '
                'This is two paired seeds, with all outcomes retained; development '
                'and historical results are not pooled into that mean.</p>')
        page=page.replace('</h1>','</h1>'+latest,1)
    temporary=output/'index.html.incomplete';temporary.write_text(page);temporary.replace(output/'index.html')
    temporary=output/'reproduction.zip.incomplete'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT/'configs/soap46_tuning_round2.json','plan.json')
        if workflow_path.exists():z.write(workflow_path,'coordinator.json')
        if conditioning_state_path.exists():z.write(conditioning_state_path,'conditioning-coordinator.json')
        if lr_cwd_state_path.exists():z.write(lr_cwd_state_path,'lr-cwd-coordinator.json')
        for name,r in runs.items():
            if r:
                for file in ('config.json','metrics.jsonl','metrics-live.jsonl','archive_result.json','launch_plan.json'):
                    path=ROOT/'runs/soap46-v1'/name/file
                    if path.exists():z.write(path,str(path.relative_to(ROOT)))
        for file in ('scripts/launch_soap46.py','scripts/run_soap46_panel.py',
                     'scripts/run_soap46_round2.py','scripts/run_soap46_followup.py',
                     'scripts/poll_soap46.py','scripts/archive_soap46.py',
                     'scripts/report_soap46_round2.py'):
            z.write(ROOT/file,file)
        for file in ('configs/soap46_conditioning_round3.json','scripts/run_soap46_conditioning.py',
                     'configs/soap46_lr_cwd_round3.json','scripts/run_soap46_lr_cwd.py',
                     'tests/test_soap46_lr_cwd_queue.py',
                     'runs/soap46-lr-cwd/shared_control_qualification.json',
                     'runs/soap46-lr-cwd/shared_control_amendment.json',
                     'runs/soap46-lr-cwd/postscreen-resource-amendment.json',
                     'runs/soap46-lr-cwd/handoff_1a20_to_chisel.json',
                     'configs/soap46_lr_cwd_extension.json',
                     'runs/soap46-lr-cwd/extension_activation.json',
                     'runs/soap46-lr-cwd/effect_variability.py',
                     'runs/soap46-lr-cwd/effect_variability_reader.py',
                     'runs/soap46-lr-cwd/effect_variability.json',
                     'configs/soap46_xlr_decay.json',
                     'scripts/run_soap46_xlr_decay.py',
                     'tests/test_soap46_xlr_decay.py',
                     'runs/soap46-xlr-decay/predeclared.json',
                     'runs/soap46-xlr-decay/node_assignment.json',
                     'runs/soap46-xlr-decay/coordinator.json',
                     'runs/soap46-xlr-decay/coverage.md',
                     'runs/soap46-xlr-decay/method_coverage.json',
                     'runs/soap46-xlr-decay/prefix_repeat_checks.json',
                     'runs/soap46-xlr-decay/prefix_repeat_reader.py',
                     'runs/soap46-xlr-decay/a_variance_conditioning_toy.py',
                     'runs/soap46-xlr-decay/a_variance_conditioning_toy.json',
                     'runs/soap46-b-uncertainty/residual_lr_decay_coverage.json',
                     'configs/soap46_b_uncertainty.json',
                     'scripts/run_soap46_b_uncertainty.py',
                     'tests/test_soap46_b_uncertainty.py',
                     'runs/soap46-b-uncertainty/predeclared.json',
                     'runs/soap46-b-uncertainty/activation.json',
                     'runs/soap46-b-uncertainty/deferred.json',
                     'runs/soap46-b-uncertainty/summary.json',
                     'runs/soap46-b-uncertainty/prefix_checks_seed4.json',
                     'runs/soap46-b-uncertainty/prefix_checks_seed5.json',
                     'runs/soap46-round2/basis_motion_rank0_step1000.json',
                     'runs/soap46-round2/basis_motion_actual_source.py',
                     'runs/soap46-round2/basis_motion_provenance.json',
                     'runs/soap46-round2/beta999_trace_comparison.json',
                     'runs/soap46-round2/first_moments_seed3_b999.json',
                     'runs/soap46-round2/temporal_alignment_seed3_b999.json'):
            path=ROOT/file
            if path.exists():z.write(path,file)
        z.write(ROOT/'runs/soap46-v1/source-0950ad0.tar.gz','numerical_source_0950ad0.tar.gz')
    temporary.replace(output/'reproduction.zip')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'reports/soap46-round2');a=p.parse_args()
    print(json.dumps({'completed':build(a.output)['completed']}))
