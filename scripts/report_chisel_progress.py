"""Render current, strictly parsed Chisel measurements as a mobile HTML report.

This consumes a closed, verified metrics snapshot. It never skips damaged rows.
The report is an offline-fit report, not evidence of an online GPT improvement.
"""
import argparse
import base64
import html
import io
import json
import subprocess
import zipfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})
BLUE, ORANGE, GREEN = "#176b9b", "#b45f06", "#39783d"


def records(path):
    raw = path.read_bytes()
    if b"\0" in raw or (raw and not raw.endswith(b"\n")):
        raise ValueError("Use a complete, verified metrics snapshot")
    return [json.loads(line) for line in raw.splitlines()]


def figure(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / (name + ".png"), dpi=170)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg")
    plt.close(fig)
    return "data:image/svg+xml;base64," + base64.b64encode(buffer.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("runs/chisel-transformer-convergence-v1/metrics.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/chisel"))
    args = parser.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    rows = records(args.metrics)
    audits = [r for r in rows if r["event"] == "audit_summary"]
    validation = [r for r in audits if r["role"] == "heldout_states"]
    if not validation: raise ValueError("No completed full validation")
    for row in validation:
        if row["state_ids"] != list(range(5, 1001, 5)):
            raise ValueError("Validation does not contain all 200 prescribed states")
    latest = validation[-1]; epoch = latest["completed_epochs"]
    states = [r for r in rows if r["event"] == "audit_state" and r["role"] == "heldout_states" and r["update"] == latest["update"]]
    assert len(states) == 200 and len({r["state_id"] for r in states}) == 200
    updates = [r for r in rows if r["event"] == "update"]
    schedule = [r for r in rows if r["event"] == "fit_schedule"][-1]
    final_state = next(r for r in states if r["state_id"] == 1000)
    fig, (ax, late_ax) = plt.subplots(2, 1, figsize=(4.5, 7.1), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    ax.plot([r["update"]/4 for r in updates], [r["objective"] for r in updates], color=BLUE, alpha=.3, lw=1, label="Training minibatches")
    for role, label, color in [("fixed_training", "32 fixed training states", BLUE), ("heldout_states", "All 200 validation states", ORANGE)]:
        selected = [r for r in audits if r["role"] == role]
        ax.plot([r["completed_epochs"] for r in selected], [r["objective"] for r in selected], "o-", color=color, label=label, markersize=4)
    ax.axhline(0, color="#555", ls="--", lw=1, label="No edit")
    ax.set_ylabel("Average Chisel objective")
    ax.legend(fontsize=9); ax.grid(alpha=.18)
    late_rows=[r for r in rows if r['event']=='audit_state' and r['role']=='heldout_states' and r['state_id']==1000]
    late_ax.plot([r['update']/4 for r in late_rows],[r['objective'] for r in late_rows],'o-',color=GREEN,markersize=4,label='GPT checkpoint 1000')
    late_ax.axhline(0,color='#555',ls='--',lw=1)
    late_ax.set(xlabel='Epochs through training checkpoints',ylabel='Checkpoint-1000 objective')
    late_ax.legend(fontsize=9);late_ax.grid(alpha=.18)
    loss = figure(fig, out, "loss")
    fig, ax = plt.subplots(2, 1, figsize=(4.5, 7.0), sharex=True)
    for row, color, label in [(validation[-2] if len(validation)>1 else validation[0], "#9aa4ab", "Previous check"), (latest, BLUE, "Latest check")]:
        selected = sorted([r for r in rows if r["event"]=="audit_state" and r["role"]=="heldout_states" and r["update"]==row["update"]], key=lambda r:r["state_id"])
        ax[0].plot([r["state_id"] for r in selected], [r["objective"] for r in selected], color=color, lw=1, label=f"{label}: epoch {row['completed_epochs']}")
        ax[1].plot([r["state_id"] for r in selected], [r["mean_kl"] for r in selected], color=color, lw=1)
    ax[0].axhline(0, color="#555", ls="--", lw=1); ax[0].set_ylabel("Chisel objective")
    ax[0].legend(fontsize=10); ax[1].axhline(1e-4, color=ORANGE, ls="--", lw=1, label="Soft penalty starts")
    ax[1].set(xlabel="GPT training iterations at checkpoint", ylabel="Prediction KL (nats/token)")
    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0)); ax[1].legend(fontsize=10)
    for a in ax:a.grid(alpha=.18)
    by_state = figure(fig, out, "by_state")
    above_budget=sum(r["mean_kl"] > 1e-4 for r in states)
    groups = latest["group_l2_percentage_reduction"]
    fig, ax = plt.subplots(2,1,figsize=(4.5,8.0))
    for a, suffix, title in zip(ax,["bias","weight"],["Bias groups","Matrix groups"]):
        selected = [(k,v) for k,v in groups.items() if k.endswith(suffix)]
        labels=[k.removesuffix("."+suffix).replace("attention.","Attention ").replace("mlp.","MLP ").replace("head","Output head").replace("embedding","Embedding") for k,v in selected]
        a.barh(labels,[v for k,v in selected],color=BLUE); a.axvline(0,color="#555",lw=.8)
        a.set_title(title,loc="left",fontsize=12); a.set_xlabel("L2 norm reduction (%)");a.grid(axis="x",alpha=.18)
        a.xaxis.set_major_locator(MaxNLocator(4))
        if suffix == "weight": a.ticklabel_format(axis="x",style="sci",scilimits=(0,0))
    group_plot = figure(fig,out,"groups")
    warm = [r for r in rows if r["event"]=="train_state" and r["epoch"]>=1]
    names = [("GPT + KL backward",["gpt_kl_backward_seconds"]), ("Edited GPT + KL forward",["student_kl_forward_total_seconds"]), ("Load GPT state",["state_load_seconds"]), ("Reconstruct cached teacher output",["teacher_target_total_seconds"]), ("Chisel forward + backward",["actor_forward_seconds","actor_backward_seconds"]), ("Build edit + norm gradient",["action_edit_forward_seconds","norm_forward_seconds","norm_edit_backward_seconds"])]
    timing = {label:1000*float(np.median([sum(r["component_seconds"].get(k,0) for k in keys) for r in warm[-800:]])) for label, keys in names}
    fig,ax=plt.subplots(figsize=(4.5,5.2))
    ys=np.arange(len(timing)); values=list(timing.values())
    ax.barh(ys,values,color=BLUE);ax.set_yticks(ys,[k.replace(" + "," +\n").replace("Reconstruct cached teacher output", "Reconstruct cached\nteacher output") for k in timing]);ax.invert_yaxis()
    ax.set_xlabel("Milliseconds per checkpoint\n(64 sequences)")
    ax.grid(axis="x",alpha=.18)
    for y,v in zip(ys,values):ax.text(v+5,y,f"{v:.0f}",va="center",fontsize=10)
    ax.set_xlim(0,max(values)*1.17)
    runtime=figure(fig,out,"runtime")
    update_seconds=float(np.median([r["wall_seconds"] for r in updates if r["epoch"]>=1]))
    matrix_reduction=np.mean([v for k,v in groups.items() if k.endswith("weight")])
    now=datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%B %d, %H:%M PT")
    readiness=("below" if final_state["objective"]<0 else "above")
    stopped=schedule.get("stop",False)
    status="The declared validation-plateau rule has fired." if stopped else "Training is continuing; convergence has not been established."
    body=f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Chisel: learning across 1000 GPT checkpoints</title>
<style>body{{font:18px/1.55 system-ui,sans-serif;color:#18222c;max-width:740px;margin:auto;padding:22px}}h1{{font-size:2rem;line-height:1.15}}h2{{font-size:1.3rem;margin-top:2rem}}figure{{margin:26px 0}}img{{width:100%;height:auto}}figcaption{{font-size:.94rem;color:#354452}}.muted{{color:#586673;font-size:.9rem}}code{{font-size:.85em;overflow-wrap:anywhere}}details{{border-top:1px solid #ddd;margin-top:28px;padding-top:15px}}a{{color:#176b9b}}strong{{font-weight:650}}</style><body>
<h1>Chisel is learning across the checkpoint bank</h1>
<p><strong>The new transformer reduces its average held-out objective, but that does not yet establish a useful online edit.</strong> After {epoch} epochs, validation loss is {latest['objective']:.2f}, relative to zero for no edit. The edit proposed at GPT checkpoint 1000 scores {readiness} the zero-edit reference.</p>
<p class="muted">Updated {now}. This page covers only the current transformer fit. <a href="../index.html">Research overview</a></p>
<p><strong>Every epoch trains on 800 GPT checkpoints and validation checks all 200 withheld checkpoints.</strong> Chisel uses the unchanged NanoGPT residual block, with four blocks and a width of 512. Each optimizer update averages 200 checkpoints, each evaluated on 64 previously seen training sequences. Validation runs every five epochs. {status}</p>
<figure><img src="{loss}" alt="Training and held-out Chisel loss over epochs, compared with zero for no edit"><figcaption><strong>Figure 1. Loss is falling on withheld checkpoints.</strong> Lower is better; negative means the percentage norm reward outweighs preservation and trust penalties. The thin curve changes checkpoint and sequence samples. The bold curves use fixed evaluation sets; each orange point includes all 200 withheld GPT states. The lower panel isolates checkpoint 1000, where we intend to begin applying Chisel. This is not GPT cross-entropy.</figcaption></figure>
<p><strong>The loss rewards equal percentage shrinkage across parameter types and penalizes changed predictions.</strong> For each of 15 groups, tensors with the same role are combined before taking their L2 norm. The mean percentage reduction enters with a minus sign. The preservation term is zero up to KL of 0.0001 nats/token, then grows quadratically with the excess relative to that scale. A small edit-size penalty is also included. At the latest check, {above_budget} of 200 checkpoints exceed that soft threshold.</p>
<figure><img src="{by_state}" alt="Chisel objective and prediction KL at each of 200 withheld GPT checkpoints"><figcaption><strong>Figure 2. The average can hide poor edits near the intended deployment checkpoint.</strong> Each point evaluates one frozen Chisel model at a withheld GPT checkpoint on 64 fixed past sequences. The previous and latest checks use the same data. Zero in the upper plot means no edit; the dashed line below marks where the KL penalty starts.</figcaption></figure>
<p><strong>Most measured shrinkage comes from biases.</strong> Matrix norm reductions remain much smaller. These measurements describe the change in a tensor's norm, not the magnitude of its edit: a large edit can change a norm very little. Normalization gains and standalone scalar parameters are excluded; embedding and output-head weights are included.</p>
<figure><img src="{group_plot}" alt="Percentage L2 norm reduction by functional group, with biases and matrices on separate scales"><figcaption><strong>Figure 3. Bias and matrix groups contribute very differently to the reward.</strong> Bars average the latest 200-checkpoint validation set. The panels have separate horizontal scales. Positive means a smaller group norm; negative means a larger norm. Every group has equal weight in the objective, regardless of its parameter count.</figcaption></figure>
<p><strong>The expensive part is differentiating preservation through GPT, not running Chisel.</strong> A warm optimizer update takes about {update_seconds:.0f} seconds across eight H100s, covering 12,800 checkpoint–sequence pairs. Each GPU processes 25 checkpoint states sequentially and accumulates gradients; Adam runs once after their global sum. That update processes 13.1 million tokens, so comparing its duration with one ordinary NanoGPT update would compare different workloads.</p>
<figure><img src="{runtime}" alt="Measured runtime components for one checkpoint and 64 sequences"><figcaption><strong>Figure 4. GPT preservation evaluation dominates the per-checkpoint cost.</strong> Bars are warm medians from the latest 800 training-state records. The teacher transformer is not rerun: its cached final activations reconstruct the original output through the original output head. Edited GPT forward and backward compute the finite-KL gradient. GPT still runs eagerly; these measurements do not establish a throughput-maximizing implementation or batch size.</figcaption></figure>
<p><strong>No continuing GPT run has received these Chisel edits.</strong> The current experiment tests whether the offline objective can be learned and generalize to withheld states and sequences. A good offline result would still need a fresh-batch preservation check and a matched online training comparison.</p>
<p><a href="reproduction.zip">Download report measurements, exact fitting source and configuration</a>.</p>
<details><summary>Configuration, stopping rule and saved state</summary>
<p>Source: <code>77eaccd</code>. Chisel has 15,504,960 parameters, rank-8 anchored matrix edits, patch size 64, four attention heads, and no persistent recurrent memory. Adam uses learning rate 0.00001, betas (0.9, 0.99), epsilon 0.00000001, no weight decay, and gradient clipping at global norm 1. Chisel is compiled; the recorded unique-graph count is one. Each state cycles through 128 cached training sequences; each validation state uses 64 fixed withheld sequences. Sequence holdout is global across checkpoints.</p>
<p>The KL penalty is the square of max(mean KL / 0.0001 − 1, 0), computed after averaging the sequence panel. The objective averages checkpoint losses. Exact-zero reference groups are frozen and contribute zero to the fixed 15-group mean. Matrix norms and realized edits are measured after casting to GPT's stored precision. Every fifth GPT checkpoint is withheld, so checkpoint validation tests interpolation within this run. It does not establish performance at later checkpoints. The fit compiles Chisel's forward/backward; GPT, chunked KL and Adam are not compiled together.</p>
<p>After three validation checks without an improvement of 0.001, the learning rate halves. Stopping requires at least 30 epochs, two reductions, five stale checks, and two validation intervals after the latest reduction. A 400-epoch safety limit is resumable and does not count as convergence. The best checkpoint tracks the actual minimum mean validation objective. These Chisel validation checks may select learning rates and stopping; official GPT validation never supplies fitting feedback.</p>
<p>Latest, best and five-epoch checkpoints contain the network, Adam state, schedule, RNG and data position. Durable directory: <code>/scratch/checkpoints/learnable-weight-decay/chisel-transformer-convergence-v1</code>. The original 1000-state bank and teacher cache remain preserved separately. This prototype uses additional GPT evaluations and therefore does not satisfy the original strict Track 3 training-computation budget.</p>
</details></body></html>"""
    (out/"index.html").write_text(body)
    summary=dict(epoch=epoch,validation_states_above_kl_threshold=above_budget,latest_validation=latest,state1000={k:final_state[k] for k in ['objective','mean_kl','mean_group_l2_percentage_reduction','realized_edit_norm']},schedule=schedule,median_update_seconds=update_seconds,timing_ms=timing,matrix_mean_norm_reduction=matrix_reduction)
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    temporary=out/"reproduction.zip.tmp"
    source=subprocess.check_output(['git','archive','--format=tar.gz','77eaccd',
        'baseline','rewriter','scripts','tests','configs','README.md'])
    latest_training={id(r) for r in warm[-800:]}
    report_rows=[r for r in rows if r['event']!='train_state' or id(r) in latest_training]
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr('source/chisel-fit-77eaccd.tar.gz',source)
        bundle.writestr('measurements/metrics.jsonl',''.join(json.dumps(r,allow_nan=False)+'\n' for r in report_rows))
        bundle.writestr('measurements/coverage.json',json.dumps(dict(
            retained='all validation, update and schedule records; latest 800 training-state records for timing',
            full_log=str(args.metrics),complete_log_durable='/scratch/checkpoints/learnable-weight-decay/chisel-transformer-convergence-v1/metrics.jsonl',
            original_record_count=len(rows),report_record_count=len(report_rows)),indent=2))
        bundle.write(out/'summary.json','measurements/summary.json')
        bundle.write('configs/chisel_transformer_convergence_v1.json','configuration.json')
        bundle.write(Path(__file__),'analysis/report_chisel_progress.py')
    temporary.replace(out/'reproduction.zip')
    print(json.dumps({"report":str(out/"index.html"),"epoch":epoch,"objective":latest["objective"]}))

if __name__=="__main__":main()
