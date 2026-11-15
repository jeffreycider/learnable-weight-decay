"""Explicit transformer-Chisel architecture and checkpoint compatibility."""

ARCHITECTURE = 'nanogpt_blocks_patch_io_v1'


def add_chisel_arguments(parser):
    parser.add_argument('--chisel-model-dim',type=int,default=256)
    parser.add_argument('--chisel-layers',type=int,default=4)
    parser.add_argument('--chisel-patch-size',type=int,default=32)
    parser.add_argument('--chisel-trajectory-mask-features',action='store_true',help='new mixed-domain architecture: explicit valid-boundary mask features')


def chisel_architecture(args,*,rank,trajectory_dim):
    values=dict(rank=rank,trajectory_dim=trajectory_dim,model_dim=args.chisel_model_dim,
                num_layers=args.chisel_layers,patch_size=args.chisel_patch_size)
    if any(not isinstance(v,int) or v<=0 for v in values.values()):
        raise ValueError('positive integer transformer Chisel dimensions required')
    if getattr(args,'chisel_trajectory_mask_features',False):values['trajectory_mask_features']=True
    return dict(name=ARCHITECTURE,implementation='rewriter.transformer_chisel.TransformerChisel',
                residual_block='baseline.model.Block',normalization='baseline.model.RMSNorm',
                head_dim=128,attention_heads=args.chisel_model_dim//128,ff_width=4*args.chisel_model_dim,
                cuda_residual_dtype='bfloat16',parameter_dtype='float32',causal=True,**values)


def build_chisel(architecture):
    if architecture.get('name')!=ARCHITECTURE:raise ValueError('legacy Chisel architecture is retired')
    from .transformer_chisel import TransformerChisel
    if TransformerChisel.architecture!=ARCHITECTURE:raise ValueError('factory and core architecture markers differ')
    return TransformerChisel(**{key:architecture[key] for key in
        ('rank','trajectory_dim','model_dim','num_layers','patch_size')},
        **({'trajectory_mask_features':True} if architecture.get('trajectory_mask_features',False) else {}))


def validate_checkpoint_architecture(checkpoint,architecture):
    saved=checkpoint.get('metadata',{}).get('architecture')
    if saved!=architecture:
        raise ValueError('Chisel checkpoint architecture mismatch; legacy Perceiver/coordinate-MLP weights cannot initialize transformer Chisel')


def scale_output_heads(actor,scale):
    import torch
    with torch.no_grad():
        for head in actor.output_heads:
            head.weight.mul_(scale)
            if head.bias is not None:head.bias.mul_(scale)
