"""Assemble the current report; detailed measurements remain on linked pages."""
import argparse
import base64
import html
import io
import json
import re
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research'


def save_figure(fig,name):
    fig.tight_layout();buffer=io.BytesIO();fig.savefig(buffer,format='svg');fig.savefig(OUT/(name+'.png'),dpi=170);plt.close(fig)
    return 'data:image/svg+xml;base64,'+base64.b64encode(buffer.getvalue()).decode()


def records(path):
    raw=path.read_bytes()
    if b'\0' in raw or (raw and not raw.endswith(b'\n')):raise ValueError('metrics snapshot is incomplete')
    return [json.loads(line) for line in raw.splitlines()]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fresh-directory',type=Path,default=ROOT/'runs/chisel-transformer-epoch15-fresh')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    chisel=json.loads((ROOT/'reports/chisel/summary.json').read_text())
    detail=(ROOT/'reports/chisel/index.html').read_text()
    loss=re.findall(r'<img src="([^"]+)"',detail)[0]
    latest=chisel['latest_validation'];last=chisel['state1000']
    epochs=chisel['epoch'];updated=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%B %d, %H:%M PT')
    ready='better' if last['objective']<0 else 'worse'
    # This frozen diagnostic was fixed before fresh-test values were observed.
    fresh_dir=args.fresh_directory
    fresh_config=json.loads((fresh_dir/'config.json').read_text())
    fresh_epoch=fresh_config['checkpoint_epoch']
    fresh=records(fresh_dir/'states.jsonl')[0]
    old=next(r for r in records(ROOT/'runs/chisel-transformer-convergence-v1/metrics.jsonl') if r['event']=='audit_state' and r['role']=='heldout_states' and r['update']==fresh_config['checkpoint_update'] and r['state_id']==1000)
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(4.5,3.8))
    decomposed=fresh.get('component_preservation') or records(fresh_dir/'decomposed/states.jsonl')[0]['component_preservation']
    ax.bar(['Past:\nfull edit','Fresh:\nfull edit','Fresh:\nbiases','Fresh:\nmatrices'],[old['mean_kl'],fresh['mean_kl'],decomposed['biases']['mean_kl'],decomposed['matrices']['mean_kl']],color=['#7b929f','#176b9b','#b45f06','#39783d'],width=.6)
    ax.tick_params(axis='x',labelsize=10)
    ax.axhline(1e-4,color='#555',ls='--',lw=1,label='Soft penalty starts')
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.set_ylabel('Prediction KL (nats/token)');ax.legend(fontsize=10);ax.grid(axis='y',alpha=.18)
    preservation=save_figure(fig,'past_fresh')
    # Read completed/pending SOAP comparisons without treating partial runs as endpoints.
    comparisons=[]
    fig,ax=plt.subplots(figsize=(4.5,4))
    for seed in range(3):
        base=ROOT/f'runs/soap-full-seed{seed}-v1'
        if not (base/'fixed/metrics.jsonl').exists():continue
        fixed={r['step']:r for r in records(base/'fixed/metrics.jsonl') if r['event']=='validation'}
        for method,label,color in [('learned','LR 0.0001','#b45f06'),('low','LR 0.00001','#176b9b')]:
            path=base/method/'metrics.jsonl'
            if not path.exists():continue
            values={r['step']:r for r in records(path) if r['event']=='validation'}
            xs=sorted(values.keys()&fixed.keys())
            ax.plot(xs,[values[x]['val_loss']-fixed[x]['val_loss'] for x in xs],color=color,ls=['-','--',':'][seed],label=f'{label}, seed {seed}')
            if 2900 in values and 2900 in fixed:comparisons.append(dict(seed=seed,method=method,loss=values[2900]['val_loss'],difference=values[2900]['val_loss']-fixed[2900]['val_loss']))
    ax.axhline(0,color='#555',ls='--',lw=1);ax.set(xlabel='GPT training iterations',ylabel='Learned − fixed validation loss\n(nats/token)');ax.legend(fontsize=9);ax.grid(alpha=.18)
    soap=save_figure(fig,'soap_difference')
    if comparisons:
        low=[r for r in comparisons if r['method']=='low']
        soap_result=('The smaller SOAP learning rate has not established a consistent improvement.' if low else 'The larger learning rate worsens final validation loss.')
        if len(low)==2 and abs(low[0]['difference'])<1e-4 and low[1]['difference']<0:
            soap_result='Lower-rate SOAP is nearly tied on seed 0 and improves seed 1.'
        soap_numbers=' '.join(f"At learning rate {lr}, the mean final difference is {sum(r['difference'] for r in comparisons if r['method']==method)/sum(r['method']==method for r in comparisons):+.4f} nats/token across {sum(r['method']==method for r in comparisons)} completed paired seed(s)." for method,lr in [('learned','0.0001'),('low','0.00001')] if any(r['method']==method for r in comparisons))
    else:
        soap_result='The full SOAP comparison is still running.'
        soap_numbers='The early advantage at the larger learning rate reversed later in training.'
    kernel_path=ROOT/'runs/megakernel-transformer-v2/summary.json'
    kernel=json.loads(kernel_path.read_text()) if kernel_path.exists() else None
    kernel_text='One CUDA launch now computes Chisel, the edited GPT, finite KL, backward and Adam for one checkpoint and 64 sequences. It remains slower than the compiled-Chisel PyTorch path used by the long fit.'
    kernel_figure=''
    if kernel and kernel.get('matched_timings'):
        timings=kernel['matched_timings']
        fig,ax=plt.subplots(figsize=(4.5,3.8))
        ax.bar(['Current\nproduction','Fused-KL\ncandidate','Literal\nkernel'],[timings['reference_ms']/1000,timings['fused_ms']/1000,timings['literal_ms']/1000],color=['#176b9b','#7b929f','#b45f06'])
        ax.set_ylabel('Warm runtime (seconds)');ax.grid(axis='y',alpha=.18);ax.tick_params(axis='x',labelsize=10)
        picture=save_figure(fig,'kernel_runtime')
        kernel_figure=f'<figure><img src="{picture}" alt="Matched warmed kernel and PyTorch runtime"><figcaption><strong>Figure 4. A single launch has not yet produced a speedup.</strong> Each measurement includes one GPT checkpoint, 64 sequences, gradients and a local Adam update. Both start with resident teacher probabilities. State loading, teacher-head reconstruction, setup and compilation are excluded. The fused-KL candidate is only a small improvement and has not replaced the active fit.</figcaption></figure>'
    fresh_finding='Past and fresh inputs produce similar prediction changes.' if .9<fresh['mean_kl']/old['mean_kl']<1.1 else 'The fresh-data preservation check differs from the past-data check.'
    fresh_outcome=('Both objectives beat making no edit.' if fresh['objective']<0 and old['objective']<0 else 'Both objectives remain worse than making no edit.' if fresh['objective']>=0 and old['objective']>=0 else 'Only one of these objectives beats making no edit.')
    component_finding=('Matrix edits lower KL compared with applying bias edits alone.' if fresh['mean_kl']<.95*decomposed['biases']['mean_kl'] else 'The full edit and bias-only edit have similar KL.' if fresh['mean_kl']<1.05*decomposed['biases']['mean_kl'] else 'Adding matrix edits raises KL relative to bias edits alone.')
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Chisel and learned SOAP: research update</title>
<style>body{{font:18px/1.55 system-ui,sans-serif;color:#18222c;max-width:700px;margin:auto;padding:22px}}h1{{font-size:2rem;line-height:1.15}}h2{{font-size:1.3rem;margin-top:2rem}}figure{{margin:24px 0}}img{{width:100%;height:auto}}figcaption{{font-size:.94rem;color:#354452}}.muted{{color:#586673;font-size:.9rem}}a{{color:#176b9b}}strong{{font-weight:650}}details{{border-top:1px solid #ddd;margin-top:24px;padding-top:16px}}</style><body>
<h1>Chisel is learning, but its intended first edit still needs work</h1>
<p><strong>Chisel learns the offline objective across the checkpoint bank, but its proposed edit at GPT checkpoint 1000 remains unreliable.</strong> After {epochs} epochs, the average objective on all 200 withheld GPT checkpoints is {latest['objective']:.2f}; zero means no edit. At checkpoint 1000, the latest edit is {ready} than no edit. Training continues.</p>
<p>The objective rewards equal percentage norm reduction across parameter types and penalizes changed predictions. Its average improves steadily, while the result at checkpoint 1000 fluctuates. At the latest check, {chisel["validation_states_above_kl_threshold"]} of 200 checkpoints still exceed the soft prediction-preservation threshold. Checkpoint 1000 matters because we intend to begin applying Chisel there during GPT training.</p>
<p class="muted">Updated {updated}. <a href="chisel/index.html">Chisel details</a> · <a href="soap/index.html">SOAP details</a></p>
<figure><img src="{loss}" alt="Chisel training and validation objective across epochs"><figcaption><strong>Figure 1. The transformer is learning the requested offline objective.</strong> Each epoch covers all 800 training checkpoints. The upper panel averages fixed evaluation data; orange includes all 200 withheld checkpoints. The lower panel isolates GPT checkpoint 1000. Lower is better, and zero means no edit. This is Chisel's norm-and-preservation objective, not GPT training loss.</figcaption></figure>
<p><strong>{fresh_finding}</strong> We froze epoch-{fresh_epoch} Chisel and evaluated the same edit at GPT checkpoint 1000 on past sequences and the next fresh batch. {fresh_outcome} Fresh-batch cross-entropy changes by {fresh['ce_change']:+.5f} nats/token.</p>
<figure><img src="{preservation}" alt="Past-data and fresh-data prediction KL for the same frozen Chisel edit"><figcaption><strong>Figure 2. {component_finding}</strong> Each bar averages 64 sequences at GPT checkpoint 1000 using frozen epoch-{fresh_epoch} Chisel. The dashed line marks the soft KL penalty's starting point. No Chisel parameters were updated or selected using this fresh test.</figcaption></figure>
<p><strong>Bias shrinkage still supplies most of the norm reward.</strong> Matrix norms shrink much less, even though the matrices change. The <a href="chisel/index.html">detailed report</a> separates actual edits, norm reduction, loss curves and runtime. No continuing GPT training run has received these Chisel edits.</p>
<h2>Learning SOAP's preconditioner residuals</h2>
<p><strong>{soap_result}</strong> {soap_numbers} These are seed-matched comparisons against the same optimizer with zero learned residuals. The small seed count does not establish a reliable gain. The fixed comparison includes the extra polar transform, so it is not the unmodified published SOAP optimizer.</p>
<figure><img src="{soap}" alt="Validation loss differences between learned SOAP and fixed zero residuals"><figcaption><strong>Figure 3. The early gain does not establish a better endpoint.</strong> Below zero means better validation loss than the fixed comparison at the same GPT iteration. Each curve compares the same seed, data, schedule and tail-EMA readout. The smaller-rate trial on seed 0 followed the larger-rate result; additional seeds keep the smaller rate fixed.</figcaption></figure>
<h2>The megakernel remains under development</h2>
<p><strong>The single-launch implementation is not ready to accelerate training.</strong> {kernel_text} The kernel's gradient-accumulation path now defers clipping and Adam until checkpoint gradients have been combined; the full distributed scientific batch still requires qualification.</p>
{kernel_figure}
<p><strong>Numerical differences still limit the kernel's qualification.</strong> With no weight edit, the kernel introduces KL of about 0.00005, roughly half the preservation penalty's starting threshold. Tested near-threshold gradients differ by less than 1%; a two-checkpoint accumulation test reaches about 1.3%. The full 200-checkpoint, eight-GPU update is not yet qualified.</p>
<p><a href="megakernel/reproduction.zip">Download kernel source, qualification measurements and dependency recipe</a>.</p>
<details><summary>Running experiment, verification and limitations</summary><p>The Chisel fit uses the unchanged NanoGPT residual block, four layers, width 512, and Adam. An update averages 200 GPT checkpoints and 64 sequences per checkpoint across eight H100s. Latest, best and five-epoch Chisel checkpoints are saved. The full bank and cached teacher targets are durable. Chisel validation controls its learning-rate reductions and stopping; official GPT validation does not train Chisel.</p><p>Chisel's finite-KL prototype performs additional GPT passes, so it is outside the original strict Track 3 computation budget. Learned SOAP uses the next ordinary GPT backward for direct one-step hypergradients, with no extra GPT training passes. Neither a small KL nor an offline norm reduction establishes improved future GPT learning.</p><p>Implementation verification passed 351 CPU tests and 11 CUDA kernel tests. The local CPU suite skipped 27 GPU-dependent tests; those skips are not passes.</p><p>Compilation adds startup cost: the measured first compiled production panel took 37 seconds; a later start using the compiler cache took 8 seconds. These are panel timings, not complete process startup. The long fit retains one compiled Chisel graph. Kernel timing excludes loading a checkpoint and reconstructing its cached teacher output.</p></details></body></html>'''
    (OUT/'index.html').write_text(body)
    for name in ['chisel','soap']:
        (OUT/name).mkdir(exist_ok=True)
        shutil.copy2(ROOT/f'reports/{name}/index.html',OUT/name/'index.html')
        archive=ROOT/f'reports/{name}/reproduction.zip'
        if archive.exists():shutil.copy2(archive,OUT/name/'reproduction.zip')
    kernel_out=OUT/'megakernel'
    kernel_out.mkdir(exist_ok=True)
    with zipfile.ZipFile(kernel_out/'reproduction.zip','w',compression=zipfile.ZIP_DEFLATED) as bundle:
        selected={Path(kernel[k]).name for k in ['source_archive','reproduction_recipe','dependency_manifest']} | {'dependencies.sha256','summary.json'}
        for path in sorted((ROOT/'reports/megakernel').glob('*')):
            if path.is_file() and path.name in selected:
                bundle.write(path,'source-and-manifests/'+path.name)
        for path in sorted((ROOT/'runs/megakernel-transformer-v2').glob('*.json')):
            bundle.write(path,'measurements/'+path.name)
    with zipfile.ZipFile(OUT/'chisel/reproduction.zip','a',compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(fresh_dir.glob('*.json*')):
            bundle.write(path,'frozen-chisel-check/'+path.name)
    (OUT/'summary.json').write_text(json.dumps(dict(updated=updated,chisel=chisel,frozen_fresh_check=fresh,frozen_fresh_epoch=fresh_epoch,soap_endpoints=comparisons,kernel=kernel),indent=2))
    print(OUT/'index.html')

if __name__=='__main__':main()
