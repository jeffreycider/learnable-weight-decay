"""Build a self-contained HTML account of the matched full SOAP comparisons."""
import base64,html,io,json,math,zipfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('runs');OUT=Path('reports/soap');OUT.mkdir(parents=True,exist_ok=True)
COLORS=['#176b9b','#b45f06','#39783d']
METHODS=['fixed','learned','low','frozen']
LABELS={'fixed':'Fixed residuals','learned':'Learned, LR 0.0001','low':'Learned, LR 0.00001','frozen':'H frozen after 1000'}
STYLES={'fixed':'--','learned':'-','low':':','frozen':'-.'}


def rows(path):
    if not path.exists():return []
    text=path.read_text()
    if text and not text.endswith('\n'):raise ValueError(f'Incomplete metrics snapshot: {path}')
    return [json.loads(line) for line in text.splitlines()]


def image(fig,name):
    fig.tight_layout()
    for extension in ['png','svg']:
        temporary=OUT/(name+'.'+extension+'.tmp');fig.savefig(temporary,format=extension,dpi=160);temporary.replace(OUT/(name+'.'+extension))
    buf=io.BytesIO();fig.savefig(buf,format='svg');plt.close(fig)
    return 'data:image/svg+xml;base64,'+base64.b64encode(buf.getvalue()).decode()


def smooth(v,k=16):return [np.mean(v[max(0,i-k+1):i+1]) for i in range(len(v))]


def main():
    data={};endpoints=[];state=[]
    for seed in range(3):
        pair={}
        for method in METHODS:
            path=ROOT/f'soap-full-seed{seed}-v1'/method
            events=rows(path/'metrics.jsonl');train=[r for r in events if r['event']=='train']
            val={r['step']:r for r in events if r['event']=='validation'}
            if train:
                walltime=None
                if (path/'manifest.json').exists():walltime=json.loads((path/'manifest.json').read_text()).get('launch_to_finished_seconds')
                if walltime is None and (path/'walltime.json').exists():walltime=json.loads((path/'walltime.json').read_text())['wall_seconds']
                pair[method]={'train':train,'val':val,'wall_seconds':walltime}
            if train:state.append({'seed':seed,'method':method,'completed_training_iterations':train[-1]['step'],'latest_validation_step':max(val) if val else None,'archive_verified':(path/'manifest.json').exists()})
        if pair:data[seed]=pair
        for method in ['learned','low']:
            if all(m in pair and 2900 in pair[m]['val'] for m in ['fixed',method]):
                a=pair['fixed']['val'][2900];b=pair[method]['val'][2900]
                endpoints.append({'seed':seed,'method':method,'meta_lr':1e-4 if method=='learned' else 1e-5,
                                  'fixed':a['val_loss'],'learned':b['val_loss'],'difference':b['val_loss']-a['val_loss'],
                                  'fixed_live':a['live_val_loss'],'learned_live':b['live_val_loss']})
    now=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%B %d, %Y, %H:%M PT')
    summary={};report_title='Historical SOAP experiment with an unintended extra polar'
    if endpoints:
        for method in ['learned','low']:
            subset=[r for r in endpoints if r['method']==method]
            if not subset:continue
            c=float(np.mean([r['fixed'] for r in subset]));l=float(np.mean([r['learned'] for r in subset]));d=l-c;n=len(subset)
            summary[method]={'paired_seeds':n,'mean_fixed_validation':c,'mean_learned_validation':l,'mean_difference':d}
        primary='low' if 'low' in summary else 'learned'
        row=summary[primary];n=row['paired_seeds'];d=row['mean_difference'];lr=1e-5 if primary=='low' else 1e-4
        direction='lowered' if d<0 else 'raised'
        lead=f'At meta learning rate {lr:g}, learning the dense SOAP residual matrices {direction} mean final validation loss by {abs(d):.4f} nats/token after2900 GPT iterations across {n} paired seed'+('s.' if n!=1 else '.')
        lead=lead.replace('after2900','after 2900')
        if n>1:report_title=f'Historical extra-polar SOAP experiment: {n} paired seeds'
        caveat='The comparison uses the same fixed zero-residual control and tail-EMA readout. The control also includes the extra polar transform; it is not the unmodified PR395 record.'
        if primary=='low' and 'learned' in summary:
            high=summary['learned'];caveat+=f' The larger meta learning rate0.0001 raised final loss by {high["mean_difference"]:.4f} on seed0.'
        caveat=caveat.replace('rate0.0001','rate 0.0001').replace('seed0.','seed 0.')
    else:
        lead='The matched 2900-iteration comparison is still running. A full-run improvement has not yet been established.'
        caveat='The completed 256-iteration pilot favored learned residuals on one seed. The full comparison tests whether that difference persists through the original learning-rate schedule.'
    fig,ax=plt.subplots(2,1,figsize=(10,8),sharex=True)
    for seed,pair in data.items():
        for method,record in pair.items():
            v=record['val'];xs=sorted(v)
            ax[0].plot(xs,[v[x]['val_loss'] for x in xs],color=COLORS[seed],ls=STYLES[method],label=f'Seed {seed}: '+LABELS[method])
        for method in ['learned','low','frozen']:
            if all(x in pair for x in ['fixed',method]):
                a=pair['fixed']['val'];b=pair[method]['val'];xs=sorted(a.keys()&b.keys())
                ax[1].plot(xs,[b[x]['val_loss']-a[x]['val_loss'] for x in xs],color=COLORS[seed],ls=STYLES[method],label=f'Seed {seed}: '+LABELS[method])
    ax[0].set_ylabel('Validation loss (nats/token)');ax[0].legend(fontsize=9);ax[0].grid(alpha=.2)
    ax[1].axhline(0,color='black',lw=.8);ax[1].set_ylabel('Learned − fixed validation loss');ax[1].set_xlabel('Ordinary GPT training iteration');
    if ax[1].get_legend_handles_labels()[0]:ax[1].legend()
    ax[1].grid(alpha=.2)
    validation=image(fig,'validation')
    fig,ax=plt.subplots(figsize=(10,4))
    for seed,pair in data.items():
        for method in ['learned','low','frozen']:
            if not all(x in pair for x in ['fixed',method]):continue
            a={r['step']:r['train_loss'] for r in pair['fixed']['train']};b={r['step']:r['train_loss'] for r in pair[method]['train']};xs=sorted(a.keys()&b.keys());ys=[b[x]-a[x] for x in xs]
            ax.plot(xs,ys,color=COLORS[seed],alpha=.12);ax.plot(xs,smooth(ys),color=COLORS[seed],ls=STYLES[method],label=f'Seed {seed}: '+LABELS[method])
    ax.axhline(0,color='black',lw=.8);ax.set_xlabel('Ordinary GPT training iteration');ax.set_ylabel('Learned − fixed training loss (nats/token)');
    if ax.get_legend_handles_labels()[0]:ax.legend()
    ax.grid(alpha=.2)
    training=image(fig,'training')
    endpoint_rows=''.join(f'<tr><td>{r["seed"]}</td><td>{r["meta_lr"]:g}</td><td>{r["fixed"]:.6f}</td><td>{r["learned"]:.6f}</td><td>{r["difference"]:+.6f}</td></tr>' for r in endpoints)
    if not endpoint_rows:endpoint_rows='<tr><td colspan="5">No completed full pair yet.</td></tr>'
    progress_rows=''.join(f'<tr><td>{r["seed"]}</td><td>{LABELS[r["method"]]}</td><td>{r["completed_training_iterations"]}</td><td>{r["latest_validation_step"]}</td><td>{"Verified" if r["archive_verified"] else "Pending"}</td></tr>' for r in state)
    times=[]
    for seed,pair in data.items():
        for method,record in pair.items():
            t=record['train'][5:]
            if not t:continue
            times.append({'seed':seed,'method':method,'wall_seconds':record['wall_seconds'],**{k:float(np.median([r.get(k,0.) for r in t])) for k in ['gpt_forward_backward_seconds','base_optimizer_seconds','credit_seconds']}})
    fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for seed,pair in data.items():
        for method,record in pair.items():
            t=record['train'][5:]
            for pane,key in enumerate(['gpt_forward_backward_seconds','base_optimizer_seconds','credit_seconds']):
                ax[pane].plot([r['step'] for r in t],smooth([1000*r.get(key,0.) for r in t]),color=COLORS[seed],ls=STYLES[method],label=f'Seed {seed}: '+LABELS[method])
    for pane,label in enumerate(['GPT forward/backward (ms)','Base optimizer (ms)','Delayed gradient + Adam (ms)']):
        ax[pane].set_ylabel(label);ax[pane].grid(alpha=.2)
    ax[-1].set_xlabel('Ordinary GPT training iteration')
    if ax[0].get_legend_handles_labels()[0]:ax[0].legend(fontsize=8)
    compute_plot=image(fig,'compute')
    live_rows=''.join(f'<tr><td>{r["seed"]}</td><td>{r["meta_lr"]:g}</td><td>{r["fixed_live"]:.6f}</td><td>{r["learned_live"]:.6f}</td><td>{r["learned_live"]-r["fixed_live"]:+.6f}</td></tr>' for r in endpoints)
    if not live_rows:live_rows='<tr><td colspan="5">No completed full pair yet.</td></tr>'
    timing_rows=''.join(f'<tr><td>{r["seed"]}</td><td>{LABELS[r["method"]]}</td><td>{1000*r["gpt_forward_backward_seconds"]:.1f}</td><td>{1000*r["base_optimizer_seconds"]:.1f}</td><td>{1000*r["credit_seconds"]:.1f}</td></tr>' for r in times)
    wall_rows=''.join(f'<tr><td>{r["seed"]}</td><td>{LABELS[r["method"]]}</td><td>{r["wall_seconds"]/60:.2f}</td></tr>' for r in times if r['wall_seconds'] is not None)
    health=[]
    for seed,pair in data.items():
        for method,record in pair.items():
            train=record['train'];last=train[-1]
            sampled=[r for r in train if r.get('proposal_difference_vs_zero_q_norm',0.)>0 and 'proposal_cosine_vs_zero_q' in r]
            health.append({'seed':seed,'method':method,'step':last['step'],'meta_updates':last.get('meta_steps',0),
                           'hyperparameter_norm':last['hyperparameter_norm'],'mean_residual_to_factor_ratio':last['residual_to_base_factor_mean'],
                           'clipped_updates':sum(r.get('hypergradient_clip_factor',1.)<1 for r in train),
                           'last_proposal_cosine_vs_zero_q':sampled[-1]['proposal_cosine_vs_zero_q'] if sampled else None})
    health_rows=''.join(f'<tr><td>{r["seed"]}</td><td>{LABELS[r["method"]]}</td><td>{r["step"]}</td><td>{r["meta_updates"]}</td><td>{r["hyperparameter_norm"]:.3f}</td><td>{r["mean_residual_to_factor_ratio"]:.4f}</td><td>{r["clipped_updates"]}</td></tr>' for r in health)
    frozen_section=''
    for seed,pair in data.items():
        if 'frozen' not in pair:continue
        f=pair['frozen']['val'].get(2900);h=pair.get('learned',{}).get('val',{}).get(2900)
        result='The continuation is still running.'
        if f and h:
            delta=f['val_loss']-h['val_loss']
            result=f'Final tail-EMA validation loss is {f["val_loss"]:.4f}, compared with {h["val_loss"]:.4f} when learning H continues ({delta:+.4f} nats/token).'
            if 2900 in pair.get('fixed',{}).get('val',{}):
                control=pair['fixed']['val'][2900]['val_loss']
                result+=f' Its difference from the fixed zero-residual control is {f["val_loss"]-control:+.4f} nats/token.'
            summary['frozen_branch']={'seed':seed,'frozen_validation':f['val_loss'],'continued_validation':h['val_loss'],'difference':delta,'inherited_steps':1000,'new_steps':1900}
        frozen_section+=f'<h2>Does continued learning cause the later disadvantage?</h2><p>A diagnostic resumes the high-learning-rate run at iteration 1000 and keeps its dimensionless residual coordinates H fixed for the remaining 1900 iterations. It retains the exact GPT weights, base optimizer state, data position and schedule. A CPU comparison of the ancestor and final checkpoints confirmed all 297,271,296 residual coordinates remained bitwise equal. Q=sH still changes with the ordinary detached Shampoo scale s. This branch tests further learning of H; it does not undo changes made during the first 1000 iterations.</p><p>{result} The branch shares its first 1000 iterations with the high-learning-rate run and is not an independent seed. Dash-dot curves start at the branch checkpoint.</p>'
    validation_finding='The early high-rate gain reverses over the full run; the lower rate is nearly tied with its control on seed0.'
    if any(r['method']=='low' and r['seed']==1 for r in endpoints):validation_finding+=' On seed1, the lower rate ends below its matched control.'
    validation_finding=validation_finding.replace('seed0','seed 0').replace('seed1','seed 1')
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{report_title}</title><style>body{{font:18px/1.55 system-ui,sans-serif;color:#18222c;max-width:1000px;margin:auto;padding:24px}}h1{{font-size:2rem;line-height:1.2}}h2{{font-size:1.35rem;margin-top:2rem}}p{{max-width:850px}}figure{{margin:28px 0}}img{{width:100%;height:auto}}figcaption{{font-size:.95rem;color:#354452}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.95rem}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ccd5dd}}.scroll{{overflow:auto}}.muted{{color:#586673;font-size:.9rem}}code{{font-size:.88em;overflow-wrap:anywhere}}details{{margin-top:24px}}a{{color:#176b9b}}</style><body>
<h1>{report_title}</h1><p><strong>Correction:</strong> We introduced an extra polar transform after incorrectly describing the published PR395 update as already containing one. The user intended the published baseline; their reply did not authorize adding this operation. The paired results below compare two versions of that unintended extra-polar variant. They do not answer whether learned residuals improve the intended published PR395 optimizer. Its corrected implementation is being tested separately; no corrected-baseline training result is available yet.</p><p>{lead}</p><p>{caveat}</p><p class="muted">Updated {now}. Losses are mean natural-log cross-entropy per token.</p>
<div class="scroll"><table><thead><tr><th>Seed</th><th>Meta LR</th><th>Fixed residual validation loss</th><th>Learned residual validation loss</th><th>Learned − fixed</th></tr></thead><tbody>{endpoint_rows}</tbody></table></div>
<figure><img src="{validation}" alt="Validation losses and paired differences over GPT training"><figcaption><b>Figure 1.</b> {validation_finding} Negative differences mean lower validation loss with learned residuals. Each point evaluates the same fixed validation data at one training checkpoint. Dashed curves keep residuals zero; solid curves use meta learning rate 0.0001 and dotted curves use 0.00001. The original tail-EMA readout is matched between the two runs.</figcaption></figure>
<h2>The next ordinary backward trains the optimizer</h2><p style="font:1.15em Georgia,serif;overflow-wrap:anywhere">U = polar((L + s<sub>L</sub>H<sub>L</sub>) S (R + s<sub>R</sub>H<sub>R</sub>))</p><p>S is the current SOAP-processed momentum; L and R are the frozen Shampoo factors. H<sub>L</sub> and H<sub>R</sub> are learned dense residual coordinates. Each detached scale s is the inverse fourth root of the mean damped Gram eigenvalue. The implemented polar map is the fixed finite Newton–Schulz iteration, whose arithmetic is differentiated directly.</p><p>Each hidden weight matrix receives an update obtained by applying the polar transform to its SOAP momentum after left and right Shampoo preconditioning. The learned version adds a dense trainable residual to each preconditioning matrix. These residuals start at zero, so the first update matches the fixed control.</p><p>After the next ordinary GPT backward, its gradient is contracted with the derivative of the previous optimizer proposal. That differentiates the finite polar iterations and all subsequent update transformations, including the actual lookahead and late EMA anchor. Earlier gradients and optimizer statistics stay fixed. This is direct one-step credit, not differentiation through the whole training history.</p><p>Each fresh full run performs 2900 prescribed GPT training iterations: 23,200 rank-local forwards and 23,200 backwards across eight GPUs. The original evaluation schedule adds 33,280 validation forwards across 112 evaluation checkpoints, including separate live and tail-EMA readouts. Both versions use that same schedule. Learning adds optimizer-only matrix algebra and Adam updates; it adds no GPT replay, Hessian products, or validation feedback. Embedding, output-head and one-dimensional parameter optimizers remain unchanged.</p>
<figure><img src="{training}" alt="Paired differences in training loss"><figcaption><b>Figure 2.</b> High-rate training loss crosses above the fixed control; lower-rate differences stay closer to zero. Negative values mean lower loss on the same training batch with learned residuals. Faint lines show every batch; solid lines show trailing 16-iteration averages. Smoothing describes the measured curves and does not establish statistical certainty.</figcaption></figure>
{frozen_section}<h2>Compute overhead</h2><figure><img src="{compute_plot}" alt="Per-iteration compute times for GPT, base optimizer and delayed gradient"><figcaption><b>Figure 3.</b> Delayed credit adds optimizer computation, while measured GPT forward/backward time remains similar. Shown are trailing 16-iteration means of separately timed components. The first five iterations are omitted to separate compilation from repeated execution. Checkpoint and validation time is outside these component timers.</figcaption></figure><div class="scroll"><table><thead><tr><th>Seed</th><th>Optimizer</th><th>GPT forward/backward (ms)</th><th>Base optimizer (ms)</th><th>Delayed gradient + Adam (ms)</th></tr></thead><tbody>{timing_rows}</tbody></table></div><p>These are measured per-iteration medians after the first five iterations, on eight H100 GPUs. Initialization, compilation, validation and checkpoint I/O are excluded from these component timings. Checkpoint pauses remain in the raw wall-time logs.</p>
<p>Completed run wall times, from process launch through final checkpoint and W&amp;B closure, include initialization, compilation, evaluation and checkpoint writes. The frozen branch timing counts only its resumed 1900 iterations; the inherited 1000 were executed in the high-learning-rate run:</p><div class="scroll"><table><thead><tr><th>Seed</th><th>Optimizer</th><th>Total minutes</th></tr></thead><tbody>{wall_rows}</tbody></table></div><details><summary>Settings, progress and limits</summary><div class="scroll"><table><thead><tr><th>Seed</th><th>Optimizer</th><th>Iteration</th><th>Meta updates</th><th>H norm</th><th>Mean residual/base factor norm ratio</th><th>Clipped updates</th></tr></thead><tbody>{health_rows}</tbody></table></div><p>The residual-to-factor ratio compares the Frobenius norm of sH with the corresponding frozen Shampoo factor, averaged over left and right factors of all 72 hidden matrices. These measurements show whether learned residuals materially change the optimizer. Better training must be established by the paired loss comparison.</p><p>The table above uses the prescribed tail-EMA readout for both optimizers. Live-weight losses at the same final checkpoint are shown separately below; they are not selected as an alternative winning readout.</p><div class="scroll"><table><thead><tr><th>Seed</th><th>Meta LR</th><th>Fixed live loss</th><th>Learned live loss</th><th>Learned − fixed</th></tr></thead><tbody>{live_rows}</tbody></table></div><p>The execution stack is PyTorch 2.11.0+cu128 (CUDA 12.8), on eight NVIDIA H100 80GB HBM3 GPUs. Each fresh run uses 2900 training iterations, 524,288 tokens per iteration, sequence length 1024, seed-matched initialization and data order, and the original PR395 schedule. Learned residuals use Adam at the stated fixed learning rate, betas (0.9, 0.99), epsilon 0.00000001 and a model-wide gradient clip of 1. Residual units are set by the frozen damped Shampoo Gram scale. There are 297,271,296 dense residual parameters across 72 hidden matrices, sharded across the eight optimizer owners.</p><p>The control uses the same explicit Shampoo factors and additional finite polar transform as the learned version. It therefore differs from the original published PR395 optimizer. Comparisons here isolate learning the residuals within this modified optimizer. The 0.00001 follow-up was chosen after the 0.0001 full run lost its early advantage. The seed 0 comparison is therefore a tuning result. Seed 1 subsequently tests exactly the same fixed setting with fresh model and optimizer state. Two seeds provide limited evidence about repeatability and do not establish performance across models or datasets.</p><p>The first learned seed 0 attempt was interrupted after 62 iterations, before a checkpoint existed; no provider cause was reported. Its logs are preserved separately. The replacement starts fresh with exactly the same configuration; the interrupted partial run is excluded from paired results.</p><div class="scroll"><table><thead><tr><th>Seed</th><th>Optimizer</th><th>Training iterations observed</th><th>Latest validation checkpoint</th><th>Durable archive</th></tr></thead><tbody>{progress_rows}</tbody></table></div><p>Implementation source: <code>46dcd37</code>. Source archive SHA256: <code>fa81f85494944f2e023a6ee1460100270bcd659cf1cc7f86ba0dcdec18e92ed1</code>. Upstream PR395 commit: <code>20dfc9150e0e99f83881c5a468fc2d8d0106e9b8</code>. Raw records are under <code>runs/soap-full-seed*-v1</code>; verified completed archives contain checkpoints and source in the corresponding StrangeLoop scratch records. Pending archive work is marked in the progress table; it does not change a completed validation measurement.</p><p class="muted">The freeze driver is 51b7b78 layered on the unchanged 46dcd37 training core. Exact archive SHA256: <code>a2239be0544d3e63f463e2bbc166c5f4c465ca45f9f0c5e084d696ac296b09dc</code>.</p></details></body></html>'''
    with zipfile.ZipFile(OUT/'reproduction.zip.tmp','w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for seed,pair in data.items():
            folder=ROOT/f'soap-full-seed{seed}-v1'
            if (folder/'freeze_integrity.json').exists():bundle.write(folder/'freeze_integrity.json',str(folder/'freeze_integrity.json'))
            for method in pair:
                for name in ['metrics.jsonl','config.json','data_manifest.json','manifest.json','finished.json','poll_status.json','walltime.json','hardware.txt','upstream.log']:
                    path=folder/method/name
                    if path.exists():bundle.write(path,str(path))
                for suffix in ['launch.json','archive.json']:
                    path=folder/f'{method}-{suffix}'
                    if path.exists():bundle.write(path,str(path))
        for name in ['soap-full-source.tar.gz','soap-frozen-source.tar.gz']:
            path=Path('/tmp')/name
            if path.exists():bundle.write(path,'source/'+name)
        bundle.write(Path(__file__),'analysis/report_soap_full.py')
        bundle.write('baseline/pr395/README.md','README_soap.md')
        if Path('/tmp/check_frozen_soap.py').exists():bundle.write('/tmp/check_frozen_soap.py','analysis/check_frozen_soap.py')
    (OUT/'reproduction.zip.tmp').replace(OUT/'reproduction.zip')
    body=body.replace('<details><summary>Settings, progress and limits</summary>', '<p><a href="reproduction.zip">Download raw metrics, configurations, data hashes and exact training source archives</a>. Full GPT/optimizer checkpoint files remain in the durable paths recorded in the included manifests.</p><details><summary>Settings, progress and limits</summary>')
    (OUT/'index.html.tmp').write_text(body);(OUT/'index.html.tmp').replace(OUT/'index.html')
    (OUT/'summary.json.tmp').write_text(json.dumps({'summary':summary,'endpoints':endpoints,'progress':state,'timings':times,'health':health},indent=2));(OUT/'summary.json.tmp').replace(OUT/'summary.json')
    print(json.dumps({'report':str(OUT/'index.html'),'summary':summary,'progress':state},indent=2))

if __name__=='__main__':main()
