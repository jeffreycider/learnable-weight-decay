"""CPU-only subtraction of pinned gamma=.1 and gamma=0 parameter diagnostics."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch


def statistics(norm,kl_total,all_total,names,clip_factor):
    n=torch.cat([norm[k].double().flatten() for k in names]);z=torch.cat([kl_total[k].double().flatten() for k in names]);t=torch.cat([all_total[k].double().flatten() for k in names]);k=z-n;r=t-z
    vectors={'norm':n,'kl_tail':k,'ranking':r,'total':t};lengths={key:float(v.norm()) for key,v in vectors.items()};cosines={}
    for a,b in [('norm','kl_tail'),('norm','ranking'),('kl_tail','ranking'),('ranking','total')]:
        den=lengths[a]*lengths[b];cosines[a+'_vs_'+b]=float(vectors[a]@vectors[b])/den if den else None
    return dict(norms=lengths,ranking_over_norm=lengths['ranking']/lengths['norm'] if lengths['norm'] else None,kl_tail_over_norm=lengths['kl_tail']/lengths['norm'] if lengths['norm'] else None,cosines=cosines,combined_clip_factor=clip_factor,after_combined_clip={k:v*clip_factor for k,v in lengths.items()} if clip_factor is not None else None,reconstruction_max_abs=float((n+k+r-t).abs().max()))


def summarize(norm,zero,full,owner,clip):
    names=list(norm)
    if owner=='shared':groups={'heads':[n for n in names if n.startswith(('matrix_output.','vector_output.'))],'body':[n for n in names if n.startswith('blocks.')],'other_representation':[n for n in names if not n.startswith(('matrix_output.','vector_output.','blocks.'))],'all':names};factors={k:clip.get('action_heads' if k=='heads' else 'representation') for k in groups};factors['all']=None
    else:groups={'encoder':[n for n in names if n.endswith('.encoder')],'decoder':[n for n in names if n.endswith('.decoder')],'trajectory':[n for n in names if n.startswith('trajectory_input.')],'all':names};factors={k:clip for k in groups}
    return {key:statistics(norm,zero,full,selected,factors[key]) for key,selected in groups.items() if selected}


def main():
    p=argparse.ArgumentParser();p.add_argument('--full',type=Path,required=True);p.add_argument('--zero',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    sf=json.loads((a.full/'summary.json').read_text());sz=json.loads((a.zero/'summary.json').read_text())
    if sf['checkpoint_sha256']!=sz['checkpoint_sha256'] or sz['ranking_weight']!=0:raise ValueError('unmatched policy or nonzero subtraction baseline')
    full=torch.load(a.full/'gradient_components.pt',map_location='cpu',weights_only=False);zero=torch.load(a.zero/'gradient_components.pt',map_location='cpu',weights_only=False)
    if full.keys()!=zero.keys():raise ValueError('state identities differ')
    rows=[];sn={};st={};sk={}
    for key in full:
        f=full[key];z=zero[key];fr=next(r for r in sf['rows'] if (r['domain'],r['state'])==key);zr=next(r for r in sz['rows'] if (r['domain'],r['state'])==key)
        if fr['context_ids']!=zr['context_ids'] or fr['mean_kl']!=zr['mean_kl']:raise ValueError('forward inputs/values changed between components')
        exact=all(torch.equal(f[family][n],z[family][n]) for family in ('shared_norm','private_norm') for n in f[family])
        if not exact:raise ValueError('norm-only gradient changed across component runs')
        rows.append(dict(domain=key[0],state=key[1],norm_gradient_bitexact=exact,private=summarize(f['private_norm'],z['private_total'],f['private_total'],'private',fr['private_unweighted']['all']['combined_clip_factor']),shared_state_weighted=summarize(f['shared_norm'],z['shared_total'],f['shared_total'],'shared',{})))
        for n in f['shared_total']:
            if n not in sn:sn[n]=f['shared_norm'][n].clone();st[n]=f['shared_total'][n].clone();sk[n]=z['shared_total'][n].clone()
            else:sn[n].add_(f['shared_norm'][n]);st[n].add_(f['shared_total'][n]);sk[n].add_(z['shared_total'][n])
    result=dict(checkpoint_sha256=sf['checkpoint_sha256'],checkpoint_update=sf['checkpoint_update'],ranking_weight=sf['ranking_weight'],scope='Same immutable policy/inputs; ranking = full gamma-gradient minus zero-gamma gradient, KL+tail = zero-gamma gradient minus norm-only; FP64 diagnostic subtraction of FP32 parameter gradients; no updates/query',shared_global_weighted=summarize(sn,sk,st,'shared',sf['shared_clip_factors']),rows=rows,counts={'gamma_run':sf['counts'],'zero_gamma_run':sz['counts']},total_additional_norm_actor_backwards=sf['additional_norm_only_actor_backwards']+sz['additional_norm_only_actor_backwards'])
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
