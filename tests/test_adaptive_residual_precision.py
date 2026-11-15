"""Explicit adaptive body arithmetic, legacy identity and unchanged private owners."""
import copy
import dataclasses
import pytest
import torch
from baseline.model import Block, Linear
from rewriter.adaptive_coordinate_chisel import AdaptiveCoordinateChisel
from rewriter.adaptive_sensing import query_weight_cores
from rewriter.private_probe_store import PrivateIdentity
from tests.test_controller import make_snapshot
from tests.test_distributed_probe_store import assert_tree_equal


def fixture(factor='fp64',residual=None,device='cpu'):
    torch.set_num_threads(1);torch.manual_seed(918)
    kwargs={} if residual is None else dict(residual_precision=residual)
    actor=AdaptiveCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,
        latent_tokens=2,fresh_weight_core=True,factor_precision=factor,**kwargs).to(device)
    def move(value):
        if isinstance(value,torch.Tensor):return value.to(device)
        if dataclasses.is_dataclass(value):return dataclasses.replace(value,**{f.name:move(getattr(value,f.name)) for f in dataclasses.fields(value)})
        if isinstance(value,dict):return {k:move(v) for k,v in value.items()}
        if isinstance(value,tuple):return tuple(move(v) for v in value)
        return value
    snapshot=move(make_snapshot());private=actor.make_private(snapshot,state_key=('test',0,6),seed=19)
    weights={n:torch.randn(o.rows.shape[0],o.columns.shape[0],device=device) for n,o in snapshot.matrices.items()}
    return actor,private,snapshot,weights


def forward(actor,private,snapshot,weights):
    # Sensing is caller-owned and must already honor its declared factor dtype.
    with torch.autocast(device_type=next(actor.parameters()).device.type,enabled=False):
        cores=query_weight_cores(weights,private,embedding_names=())
    return actor(snapshot,private,weight_cores=cores)


def test_default_omitted_and_explicit_bf16_are_identical():
    a,p,s,w=fixture();b,q,_,_=fixture(residual='bf16')
    assert 'residual_precision' not in a.configuration()
    assert a.configuration()['cuda_residual_dtype']=='bfloat16'
    assert_tree_equal(a.state_dict(),b.state_dict());assert_tree_equal(p.state_dict(),q.state_dict())
    assert_tree_equal(forward(a,p,s,w).dense(),forward(b,q,s,w).dense())


@pytest.mark.parametrize('factor',['fp32','fp64'])
def test_fp32_body_autocast_guard_nonzero_shared_private_gradients(factor):
    actor,private,snapshot,weights=fixture(factor,'fp32')
    assert type(actor.blocks[0]) is Block
    seen=[];hooks=[m.register_forward_hook(lambda m,i,o:seen.append((i[0].dtype,o.dtype)))
                  for block in actor.blocks for m in block.modules() if isinstance(m,Linear)]
    with torch.no_grad():
        for h in actor.output_heads:h.weight.normal_(std=.01)
        actor.blocks[0].attn.proj.weight.normal_(std=.01)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        action=forward(actor,private,snapshot,weights)
    sum(x.square().sum() for x in action.dense().values()).backward()
    for h in hooks:h.remove()
    assert seen and all(a==b==torch.float32 for a,b in seen)
    assert actor.weight_core_input.weight.grad.norm()>0
    assert actor.blocks[0].attn.q.weight.grad.norm()>0
    assert private.matrix_rows[0].raw.grad.norm()>0
    assert action.matrices['first'].C.dtype==(torch.float64 if factor=='fp64' else torch.float32)
    assert all(w.grad is None for w in weights.values())


def test_shared_mode_load_rejects_but_explicit_metadata_fork_preserves_all_numeric_owners():
    old,p,s,w=fixture();new,q,_,_=fixture(residual='fp32')
    assert PrivateIdentity.from_private(p,state_sha256='a'*64)==PrivateIdentity.from_private(q,state_sha256='a'*64)
    assert_tree_equal(p.state_dict(),q.state_dict())
    optimizer=torch.optim.Adam(old.parameters(),lr=1e-4);poptimizer=torch.optim.Adam(p.parameters(),lr=1e-4)
    with torch.no_grad():
        old.matrix_output.weight.normal_(std=.01)
    sum(x.square().sum() for x in forward(old,p,s,w).dense().values()).backward()
    optimizer.step();poptimizer.step();optimizer.zero_grad(set_to_none=True);poptimizer.zero_grad(set_to_none=True)
    parent=copy.deepcopy(dict(actor=old.state_dict(),private=p.state_dict(),adam=optimizer.state_dict(),private_adam=poptimizer.state_dict(),rng=torch.get_rng_state()))
    before=copy.deepcopy(new.state_dict())
    with pytest.raises(ValueError,match='architecture'):new.load_state_dict(parent['actor'])
    assert_tree_equal(new.state_dict(),before)
    # A caller must declare a new shared/run identity and rewrite ONLY shared
    # execution metadata. This test is not an automatic production migration.
    fork=copy.deepcopy(parent);fork['actor']['_extra_state']=new.configuration()
    new.load_state_dict(fork['actor']);q.load_state_dict(fork['private'])
    newopt=torch.optim.Adam(new.parameters(),lr=1e-4);newopt.load_state_dict(fork['adam'])
    qopt=torch.optim.Adam(q.parameters(),lr=1e-4);qopt.load_state_dict(fork['private_adam'])
    assert_tree_equal(newopt.state_dict(),parent['adam']);assert_tree_equal(qopt.state_dict(),parent['private_adam'])
    assert_tree_equal(q.state_dict(),parent['private'])
    for n,x in parent['actor'].items():
        if n!='_extra_state':assert_tree_equal(new.state_dict()[n],x)
    assert_tree_equal(old.state_dict(),parent['actor'])
    assert_tree_equal(torch.get_rng_state(),parent['rng'])


def test_invalid_mode_and_recipe_backend_guards():
    from tests.test_adaptive_training_driver import adaptive_recipe
    from scripts.train_coordinate_chisel import validate_recipe,build_actor
    for value in (None,True,'automatic','float32'):
        with pytest.raises(ValueError,match='residual_precision'):AdaptiveCoordinateChisel(residual_precision=value)
    r=adaptive_recipe();r['architecture']['residual_precision']='fp32';validate_recipe(r)
    assert build_actor(r).configuration()['residual_precision']=='fp32'
    for key in ('actor_backend','private_actor_backend'):
        bad=copy.deepcopy(r);bad[key]='adaptive_actor_graph';bad['architecture']['fresh_weight_core']=True
        with pytest.raises(ValueError,match='residual'):validate_recipe(bad)
    bad=copy.deepcopy(r);bad['action_family']='fixed_anchored_v1'
    with pytest.raises(ValueError,match='residual'):validate_recipe(bad)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA precision gate requires an explicitly assigned GPU')
def test_cuda_fp32_body_uses_exact_blocks_without_bf16_activations():
    from rewriter.gpt_precision import configure_precision
    configure_precision('fp32');actor,private,snapshot,weights=fixture('fp64','fp32','cuda')
    seen=[];hooks=[block.register_forward_hook(lambda m,i,o:seen.append((i[0].dtype,o.dtype))) for block in actor.blocks]
    with torch.autocast('cuda',dtype=torch.bfloat16):action=forward(actor,private,snapshot,weights)
    for h in hooks:h.remove()
    assert seen==[(torch.float32,torch.float32)]
    assert all(x.dtype==torch.float64 and x.count_nonzero()==0 for x in action.dense().values())
    sum(x.sum() for x in action.dense().values()).backward()
    assert actor.matrix_output.weight.grad.norm()>0


def test_fp32_body_actual_tiny_gpt_direct_and_detached_factor_credit():
    from tests.test_adaptive_preservation_actor import actual_actor_case
    from rewriter.adaptive_preservation import build_adaptive_proposal
    from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual,tokenwise_panel
    gradients=[];rows=[]
    for detached in (False,True):
        engine,_,_,snapshot,grouped,inputs=actual_actor_case()
        torch.manual_seed(418)
        actor=AdaptiveCoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,
            latent_tokens=2,fresh_weight_core=True,factor_precision='fp64',residual_precision='fp32')
        private=actor.make_private(snapshot,state_key=('test',0,6),seed=17)
        with torch.no_grad():actor.matrix_output.weight.normal_(std=.001)
        call=lambda s:actor(s,private,weight_cores=query_weight_cores(engine.parameters,private))
        teachers=[engine.cache_teacher(x[None],data_role='synthetic')[0] for x in inputs]
        config=TokenwiseConfig(mean_tolerance=1e-3,tail_tolerance=1e-3,scale=1e-3,rho=.01,ranking_weight=0.)
        dual=TokenwiseDual([6],config=config,dual_lr=.01)
        before={n:p.clone() for n,p in engine.parameters.items()}
        params,reg,_,_,credit=build_adaptive_proposal(call,snapshot,engine,grouped,radius_fraction=1.,detach_credit=detached)
        rows.append(tokenwise_panel(engine,params,reg,inputs,config=config,dual=dual,state_id=6,actor_credit=credit,teacher_provider=lambda a,b:teachers[a]))
        gradients.append({prefix+n:None if p.grad is None else p.grad.clone() for prefix,module in [('actor.',actor),('private.',private)] for n,p in module.named_parameters()})
        for n,p in engine.parameters.items():assert p.dtype==torch.float32 and p.grad is None;torch.testing.assert_close(p,before[n],atol=0,rtol=0)
    assert rows[0]==rows[1]
    for n,a in gradients[0].items():
        b=gradients[1][n];assert (a is None)==(b is None),n
        if a is not None:torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5,msg=n)
    assert gradients[1]['private.matrix_rows.0.raw'].norm()>0


def test_fp32_body_zero_heads_keep_identity_and_receive_first_gradient():
    actor,private,snapshot,weights=fixture('fp64','fp32')
    action=forward(actor,private,snapshot,weights)
    assert all(value.count_nonzero()==0 for value in action.dense().values())
    sum(value.sum() for value in action.dense().values()).backward()
    assert actor.matrix_output.weight.grad.norm()>0
    assert actor.vector_output.weight.grad.norm()>0
    # No claim that the zero core rotates frames on the first step.
    assert private.matrix_rows[0].raw.grad.count_nonzero()==0
