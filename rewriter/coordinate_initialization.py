"""Optional cached-weight QR initialization of fresh checkpoint-private maps.

This changes neither sensor nor write anchors. It uses only stored W B / W^T A
and full vector W. The existing seeded random frame supplies missing directions;
the private trajectory map is unchanged. No GPT call or optimizer step occurs.
"""
import torch


def weight_qr_frame(weight_features, fallback, *, rank_rtol=1e-10):
    """Ordered twice-reorthogonalized Gram--Schmidt, in CPU FP64.

    Directions inherit the input-column orientation, avoiding SVD sign choices.
    If n<L, repeat the available orthonormal directions to retain unit columns.
    rank_rtol is a numerical basis-rank test, not a ranking-loss tolerance.
    """
    if not 0<rank_rtol<1:raise ValueError('explicit numerical rank tolerance required')
    w=weight_features.detach().to(device='cpu',dtype=torch.float64)
    f=fallback.detach().to(device='cpu',dtype=torch.float64)
    if w.ndim!=2 or f.ndim!=2 or w.shape[0]!=f.shape[0] or min(f.shape)<1:
        raise ValueError('matching nonempty coordinate axes required')
    if not torch.isfinite(w).all() or not torch.isfinite(f).all():raise ValueError('nonfinite initialization input')
    n,latent=f.shape;target=min(n,latent);basis=[]
    def append(value,threshold):
        value=value.clone()
        for _ in range(2):
            for q in basis:value-=q*torch.dot(q,value)
        norm=value.norm()
        if float(norm)<=threshold:return False
        basis.append(value/norm);return True
    scale=float(w.norm(dim=0).max()) if w.shape[1] else 0.
    for column in w.T:
        if len(basis)==target:break
        append(column,scale*rank_rtol)
    weight_rank=len(basis)
    for column in f.T:
        if len(basis)==target:break
        append(column,max(float(column.norm()),1.)*rank_rtol)
    # Deterministic final completion covers degenerate caller-provided fallback.
    for i in range(n):
        if len(basis)==target:break
        column=torch.zeros(n,dtype=torch.float64);column[i]=1.;append(column,rank_rtol)
    if len(basis)!=target:raise RuntimeError('could not complete coordinate basis')
    value=torch.stack([basis[i%target] for i in range(latent)],dim=1).float()
    return value,dict(weight_rank=weight_rank,complement_columns=target-weight_rank,
        repeated_columns=latent-target,rank_rtol=rank_rtol,arithmetic='ordered CPU FP64 twice-reorthogonalized Gram-Schmidt')


@torch.no_grad()
def initialize_cached_weight_qr(private,snapshot,*,rank_rtol=1e-10):
    """Initialize a fresh ψ before its Adam is created; preserve trajectory RNG."""
    if getattr(private,'frame_mode','independent')!='independent':
        raise ValueError('cached-weight QR initializer supports independent frames only')
    private.validate_layout(snapshot)
    if any(p.grad is not None for p in private.parameters()):raise ValueError('fresh gradient-cleared private maps required')
    prepared=[];details={}
    for index,(name,obs) in enumerate(snapshot.matrices.items()):
        for side,frame,features in [('rows',private.matrix_rows[index],obs.weight_probes[0]),
                                    ('columns',private.matrix_columns[index],obs.weight_probes[1])]:
            value,stats=weight_qr_frame(features,frame.encoder,rank_rtol=rank_rtol)
            prepared.append((frame,value));details[name+':'+side]=stats
    for index,(name,obs) in enumerate(snapshot.vectors.items()):
        frame=private.vector_frames[index]
        value,stats=weight_qr_frame(obs.weight.reshape(-1,1),frame.encoder,rank_rtol=rank_rtol)
        prepared.append((frame,value));details[name+':vector']=stats
    # Validate everything before mutating any frame; Parameter identities stay.
    for frame,value in prepared:
        frame.encoder.copy_(value);frame.decoder.copy_(value)
    return dict(schema='cached_weight_qr_v1',frames=details,trajectory_initialization='unchanged original seeded private linear map',
        sensor_and_action_anchors='unchanged',gpt_forwards=0,backwards=0,optimizer_steps=0)


@torch.no_grad()
def initialize_warm_private(private,source,*,source_state_sha256,anchors_equal):
    """Explicit parameter-only initialization into a NEW checkpoint identity.

    Caller verifies original anchors and creates fresh Adam/dual afterward.
    Unlike resume, source moments/clocks are intentionally never accepted here.
    """
    if anchors_equal is not True:raise ValueError('this warm-start protocol requires exactly matching anchors')
    if not isinstance(source_state_sha256,str) or len(source_state_sha256)!=64:raise ValueError('source checkpoint-state SHA required')
    if any(p.grad is not None for p in private.parameters()):raise ValueError('fresh gradient-cleared private maps required')
    target=private.get_extra_state();old=source.get('_extra_state')
    if not isinstance(old,dict) or old.get('state_key')==target['state_key']:raise ValueError('explicit different source/target private identity required')
    if {k:v for k,v in old.items() if k!='state_key'}!={k:v for k,v in target.items() if k!='state_key'}:
        raise ValueError('warm-start architecture/layout differs')
    params=dict(private.named_parameters())
    if set(source)!=(set(params)|{'_extra_state'}):raise ValueError('warm-start parameter schema differs')
    for name,p in params.items():
        value=source[name]
        if not isinstance(value,torch.Tensor) or value.shape!=p.shape or value.dtype!=p.dtype or not torch.isfinite(value).all():
            raise ValueError('invalid warm-start parameter: '+name)
    for name,p in params.items():p.copy_(source[name])
    return dict(schema='parameter_only_warm_private_v1',source_state_key=old['state_key'],target_state_key=target['state_key'],
        source_state_sha256=source_state_sha256,anchors_exact=True,optimizer_initialization='fresh; no source moments or clocks copied',
        dual_initialization='fresh; caller owned',gpt_forwards=0,backwards=0,optimizer_steps=0)
