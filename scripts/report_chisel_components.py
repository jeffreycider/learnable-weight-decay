"""Seen-token fitting diagnosis from fixed, complete logged audit panels."""
import argparse
import base64
import io
import json
import zipfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})

def read_rows(path):
    data=path.read_bytes()
    if b'\0' in data or not data.endswith(b'\n'):raise ValueError('Use a closed, valid snapshot')
    return [json.loads(line) for line in data.splitlines()]

def picture(fig,out,name):
    fig.tight_layout(); b=io.BytesIO();fig.savefig(b,format='svg');fig.savefig(out/(name+'.png'),dpi=170);plt.close(fig)
    return 'data:image/svg+xml;base64,'+base64.b64encode(b.getvalue()).decode()

def equation(text,out,name,height=.8):
    fig,ax=plt.subplots(figsize=(5.3,height));ax.axis('off');ax.text(.5,.5,text,ha='center',va='center',fontsize=15)
    return picture(fig,out,name)

def main():
    p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,default=ROOT/'runs/chisel-transformer-convergence-v1/metrics_compact.jsonl');p.add_argument('--output',type=Path,default=ROOT/'reports/research');a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    rows=read_rows(a.metrics)
    audits=[r for r in rows if r['event']=='audit_summary' and r['role']=='fixed_training']
    derived=[]
    for r in audits:
        states=[s for s in rows if s['event']=='audit_state' and s['role']=='fixed_training' and s['update']==r['update']]
        assert len(states)==32 and len({s['state_id'] for s in states})==32
        assert all(len(s['sequence_ids'])==128 for s in states)
        for s in states:
            assert abs(s['objective']-(s['norm_term']+s['kl_penalty']+s['trust_term']))<1e-8
        avg=lambda key:sum(s[key] for s in states)/len(states)
        for key in ['objective','mean_kl','kl_penalty','mean_group_l2_percentage_reduction']:
            assert abs(avg(key)-r[key])<1e-8,(key,r['update'])
        groups=r['group_l2_percentage_reduction']
        derived.append(dict(epoch=r['completed_epochs'],update=r['update'],mean_kl=r['mean_kl'],norm_term=avg('norm_term'),kl_penalty=avg('kl_penalty'),trust_term=avg('trust_term'),objective=r['objective'],shrink=r['mean_group_l2_percentage_reduction'],matrix_shrink=np.mean([v for k,v in groups.items() if k.endswith('.weight')]),bias_shrink=np.mean([v for k,v in groups.items() if k.endswith('.bias')]),max_state_kl=max(s['mean_kl'] for s in states),min_state_kl=min(s['mean_kl'] for s in states),states_above_budget=sum(s['mean_kl']>1e-4 for s in states)))
    latest=derived[-1];x=[r['epoch'] for r in derived]
    fig,axes=plt.subplots(2,1,figsize=(4.8,6.5),sharex=True)
    axes[0].plot(x,[r['shrink'] for r in derived],'o-',color='#176b9b',markersize=4);axes[0].set_ylabel('Mean group norm reduction (%)');axes[0].axhline(0,color='#555',lw=.8)
    axes[1].fill_between(x,[r['min_state_kl'] for r in derived],[r['max_state_kl'] for r in derived],color='#b45f06',alpha=.15,label='Range across 32 states')
    axes[1].plot(x,[r['mean_kl'] for r in derived],'o-',color='#b45f06',markersize=4,label='Mean seen-token KL')
    axes[1].axhline(1e-4,color='#777',ls='--',label='Current penalty starts');axes[1].axhline(0,color='#333',lw=1,label='Zero-edit result')
    axes[1].set(xlabel='Chisel training epochs',ylabel='KL on fitting tokens (nats/token)');axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axes[1].legend(fontsize=9)
    for ax in axes:ax.grid(alpha=.18)
    seen=picture(fig,out,'seen_fit')
    fig,axes=plt.subplots(2,1,figsize=(4.8,6.0),sharex=True,gridspec_kw={'height_ratios':[1.7,1]})
    for key,label,color in [('norm_term','Norm term (negative reward)','#176b9b'),('kl_penalty','Preservation penalty','#b45f06'),('objective','Total loss','#202a35')]:axes[0].plot(x,[r[key] for r in derived],'o-',label=label,color=color,markersize=3)
    axes[0].axhline(0,color='#777',lw=.8);axes[0].set_ylabel('Additive loss contribution');axes[0].legend(fontsize=9)
    axes[1].plot(x,[r['trust_term'] for r in derived],'o-',color='#39783d',markersize=3);axes[1].set(xlabel='Chisel training epochs',ylabel='Edit-size penalty');axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    for ax in axes:ax.grid(alpha=.18)
    components=picture(fig,out,'seen_loss_components')
    fig,axes=plt.subplots(2,1,figsize=(4.8,5.6),sharex=True)
    for ax,key,label in zip(axes,['bias_shrink','matrix_shrink'],['Bias groups','Matrix groups']):
        ax.plot(x,[r[key] for r in derived],'o-',markersize=4);ax.set_ylabel('Norm reduction (%)');ax.set_title(label,loc='left',fontsize=12);ax.axhline(0,color='#777',lw=.8);ax.grid(alpha=.18)
    axes[1].set_xlabel('Chisel training epochs');axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    families=picture(fig,out,'seen_norm_families')
    fig,ax=plt.subplots(figsize=(4.8,3.8));z=np.linspace(0,2,300)
    ax.plot(z,np.maximum(z-1,0)**2,label='Current penalty');ax.plot(z,10*np.maximum(z-1,0)**2,label='10× current weight');ax.plot(z,z,label='Linear KL: illustrative scale',ls=':');ax.axvline(1,color='#777',lw=.8,ls='--');ax.set(xlabel='KL divided by 0.0001',ylabel='Preservation penalty',ylim=(-.05,3));ax.legend(fontsize=9);ax.grid(alpha=.18)
    shape=picture(fig,out,'preservation_shape')
    loss_eq=equation(r'$\mathcal{L}=-R+\max(K/10^{-4}-1,0)^2+T$',out,'current_loss')
    proposal_eq=equation(r'$\min_{\phi}\;-R(\phi)\quad\mathrm{subject\ to}\quad K_i(\phi)\leq\tau$',out,'proposed_constraint')
    soap_eq=equation(r'$LSR\quad\longrightarrow\quad\mathrm{msign}(LSR)$',out,'soap_extra')
    learned_eq=equation(r'$\mathrm{msign}\!\left[(L+Q_L)S(R+Q_R)\right]$',out,'soap_learned')
    updated=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%B %d, %H:%M PT')
    html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Chisel: shrinkage and preservation on fitting tokens</title><style>body{{font:18px/1.55 system-ui,sans-serif;color:#18222c;max-width:700px;margin:auto;padding:22px}}h1{{font-size:2rem;line-height:1.15}}h2{{font-size:1.3rem;margin-top:2rem}}figure{{margin:25px 0}}img{{width:100%;height:auto}}figcaption{{font-size:.94rem;color:#354452}}.muted{{color:#586673;font-size:.9rem}}a{{color:#176b9b}}details{{border-top:1px solid #ddd;margin-top:24px;padding-top:14px}}strong{{font-weight:650}}code{{overflow-wrap:anywhere;font-size:.9em}}</style><body>
<h1>Chisel shrinks biases, but changes predictions on fitting tokens</h1>
<p><strong>Chisel has not learned the requested function-preserving shrinkage.</strong> At epoch {latest['epoch']}, the fixed fitting-token check shows {latest['shrink']:.2f}% mean group norm reduction, but KL remains about {latest['mean_kl']:.1e} nats/token. The zero-edit network measures KL = 0 on the same check. A falling combined loss did not establish preservation.</p>
<p class="muted">Updated {updated}. Fitting tokens first; unseen-token results are secondary until this test succeeds.</p>
<figure><img src="{seen}" alt="Seen-token norm reduction rises while seen-token KL remains nonzero"><figcaption><strong>Figure 1. Shrinkage improves without KL returning to zero.</strong> Every point evaluates the same 32 training checkpoints and all 128 fitting sequences per checkpoint. These sequences have all participated in fitting after two epochs. Each sequence has 1024 tokens. Shading shows the minimum and maximum checkpoint-mean KL, not per-token extrema. The check is a fixed subset of the 800 training checkpoints.</figcaption></figure>
<p><strong>The combined loss rewards enough shrinkage to pay for changed predictions.</strong> The norm reward keeps growing while the preservation penalty remains positive. At the latest check, {latest['states_above_budget']} of 32 fitting checkpoints exceed even the current soft KL threshold. Figure 2 separates every additive term.</p>
<figure><img src="{components}" alt="Separate norm reward, preservation penalty, total loss and edit-size penalty across training"><figcaption><strong>Figure 2. The norm reward drives the loss downward; preservation has not been solved.</strong> Curves use the same fixed fitting-token check as Figure 1. Total loss is the sum of the negative norm reward, positive preservation penalty, and edit-size penalty. The small edit-size term has its own vertical scale. Penalties are computed per checkpoint before averaging.</figcaption></figure>
<figure><img src="{families}" alt="Separate average norm reduction curves for bias and matrix groups"><figcaption><strong>Figure 3. Biases shrink; matrix norms grow slightly on average.</strong> Each panel averages functional groups of one parameter family on the fixed fitting check. Separate vertical scales expose the much smaller matrix-norm changes. The actual objective weights all 15 groups equally; normalization gains and standalone scalars are excluded.</figcaption></figure>
<h2>The preservation objective needs to change</h2>
<p><strong>The current penalty explicitly permits nonzero KL without any preservation gradient.</strong> In the implemented loss below, R is mean percentage group norm reduction, K is mean token KL for one checkpoint, and T is the edit-size penalty. The complete loss averages checkpoints.</p>
<img src="{loss_eq}" alt="Loss equals negative percentage norm reduction plus squared positive excess of KL over its threshold plus edit-size penalty">
<p><strong>A larger coefficient alone cannot make this penalty target zero KL.</strong> Multiplying the current penalty by ten or a thousand leaves it flat below 0.0001. It can reduce violations above that threshold, but supplies no incentive to improve preservation inside the flat region. The objective I chose therefore does not match the requested numerical-identity target.</p>
<figure><img src="{shape}" alt="Current and ten-times-weighted penalties are both flat below the same positive KL threshold"><figcaption><strong>Figure 4. Increasing the weight preserves the same blind region.</strong> These are analytic penalty shapes, not new training results. The dotted linear-KL example retains pressure for every positive KL; its coefficient is illustrative, not tuned.</figcaption></figure>
<p><strong>I recommend treating preservation as a constraint and shrinkage as the reward within that constraint.</strong> For each fitting checkpoint i, require its fitting-token KL to stay below a numerical tolerance τ. Determine τ from same-path numerical checks, rather than treating 0.0001 as numerical error. Track the fraction of checkpoints passing and the worst checkpoint; an acceptable average alone is insufficient.</p>
<img src="{proposal_eq}" alt="Minimize negative shrinkage subject to each fitting checkpoint's KL remaining below a numerical tolerance">
<p><strong>A staged tightening schedule is a reasonable next experiment, but has not been tested here.</strong> Warm-start from the saved Chisel model, progressively lower the allowed KL, and increase the preservation multiplier when fitting-token violations persist. A linear-in-KL multiplier with a quadratic violation term avoids the existing positive-KL flat region once the tolerance reaches its final numerical level. Choose schedules from fitting data; first verify nonzero shrinkage survives the final tolerance, then test unseen tokens.</p>
<p><strong>Strict preservation may reduce the attainable shrinkage substantially.</strong> A successful loss reformulation must retain nonzero edits and norm reduction; returning to the zero-edit network is only a preservation sanity check. The current data do not tell us whether a large fixed coefficient or adaptive tightening will train better. No new objective has been silently substituted into the running experiment.</p>
<h2>The SOAP comparison used the wrong baseline</h2>
<p><strong>I incorrectly described the published optimizer as already ending in a matrix sign.</strong> The user’s msign(Ũ) response relied on that description. I then added that operation to both experimental versions, changing the intended baseline. The paired results therefore do not answer whether learning residuals improves the published optimizer.</p>
<p><strong>The added operation is the matrix sign, or polar factor, of the preconditioned update.</strong> In the pinned published implementation, SOAP first rotates and rescales momentum; call that result S. Shampoo then applies left and right inverse-fourth-root factors L and R. Before the common later normalization and geometric adjustments, the relevant change is:</p>
<img src="{soap_eq}" alt="Published update L S R becomes matrix sign of L S R in our fixed comparison">
<p><strong>This changes the update's shape, not just its learning rate.</strong> If a matrix has singular-value decomposition UΣVᵀ, its ideal polar factor is UVᵀ on the nonzero singular directions: the nonzero singular values become one. Our implementation approximates that operation with 12 BF16 Newton–Schulz iterations. Subsequent normalization, spectral shaping and row scaling still apply, so the final update need not have a flat spectrum.</p>
<p><strong>The learned run adds residuals before that same operation.</strong> Its pre-normalization update is:</p>
<img src="{learned_eq}" alt="Learned update is matrix sign of left factor plus learned residual, times SOAP momentum, times right factor plus learned residual">
<p><strong>The fixed comparison sets those learned residuals to zero while retaining the matrix sign.</strong> Both versions also form explicit Shampoo factors, which changes floating-point association relative to the published code. The pinned source contains a Muon matrix-sign helper, but the executed hidden-weight step calls shampoo_update instead. I should have traced that call before describing the baseline. The correction removes the added matrix sign and must verify that zero learned residuals reproduce the published update before further SOAP runs. Lag-1 hypergradients do not require the matrix sign.</p>
<p><a href="soap/index.html">SOAP curves and exact implementation</a> · <a href="chisel/index.html">Full-bank Chisel measurements and runtime</a> · <a href="components.zip">Download component measurements and analysis</a></p>
<details><summary>Measurement coverage and reproducibility</summary><p>The fixed fitting-token check covers 32 checkpoints × 128 sequences × 1024 tokens. This is not a frozen evaluation of all 800 training checkpoints, nor proof of preservation on every past example. Optimization uses 64 of the 128 cached fitting sequences per checkpoint visit, cycling the full panel every two epochs. The initial zero-output Chisel produces no edit and measured zero KL. That identity check alone does not determine the numerical tolerance for arbitrary edited evaluations.</p><p>Component identities are checked against each logged audit_state row. The norm loss is negative mean percentage L2 reduction, not squared-norm reduction. The trust penalty is 0.01 times squared realized edit norm divided by squared action radius. Data use the current transformer fit at source77eaccd. The SOAP comparison is traced to baseline/pr395/upstream.py at commit20dfc9150e0e99f83881c5a468fc2d8d0106e9b8 and rewriter/soap_residual.py from the executed source46dcd37.</p><p>Held-out checkpoint and unseen-sequence measurements remain saved in the downloadable records. They are not evidence that the seen-token preservation requirement has been met. The live run still optimizes the old tradeoff objective; its continued loss decrease should be read accordingly.</p></details></body></html>'''
    (out/'index.html').write_text(html)
    summary=dict(updated=updated,measurement_scope='32 fixed training checkpoints,128 fitting sequences each',seen_fit=derived,latest=latest,proposed_objective_status='recommendation, not implemented in active fit')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    with zipfile.ZipFile(out/'components.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        retained=[r for r in rows if r['event'] in ['audit_state','audit_summary','fit_schedule','start']]
        z.writestr('audits.jsonl',''.join(json.dumps(r)+'\n' for r in retained));z.write(out/'summary.json','summary.json');z.write(Path(__file__),'report_chisel_components.py')
        sidecar=a.metrics.with_suffix('.metadata.json')
        if sidecar.exists():z.write(sidecar,'snapshot_coverage.json')
    print(json.dumps(latest))

if __name__=='__main__':main()
