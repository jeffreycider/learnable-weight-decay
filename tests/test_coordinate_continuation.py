import copy,dataclasses
import pytest
import torch
from rewriter.coordinate_chisel import CoordinateChisel
from rewriter.meta_optimizer import build_meta_adam
from rewriter.tokenwise_reference import TokenwiseConfig,TokenwiseDual
from scripts.continue_coordinate_chisel import restore_numeric_state,validate_stage,stage_origin,configure_stage_optimizers,resolve_stage_panels,parent_readout_differences,read_parent_readout
from tests.test_controller import make_snapshot


def test_continuation_restores_every_numeric_owner_and_changes_only_declared_objective(monkeypatch):
    torch.set_num_threads(1);torch.manual_seed(67);snapshot=make_snapshot()
    actor=CoordinateChisel(rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2)
    private=actor.make_private(snapshot,state_key=('gpt',0,6),seed=8);key=('gpt',6)
    opt=build_meta_adam(actor,lr=1e-7,representation_lr_multiplier=10)
    local=torch.optim.Adam(private.parameters(),lr=1e-4,eps=1e-12)
    # Nontrivial moments for all owners, independent of zero output initialization.
    sum(p.square().sum() for p in actor.parameters()).backward();opt.step()
    sum(p.square().sum() for p in private.parameters()).backward();local.step()
    c=TokenwiseConfig(1e-10,1e-10,1e-8,.01,ranking_weight=2.,ranking_k=3,ranking_tie_policy='exact_teacher_ties')
    old=TokenwiseDual([6],config=c,dual_lr=.01);old.step([(6,1e-8)],expected_ids=[6])
    saved=copy.deepcopy(dict(actor=actor.state_dict(),shared_optimizer=opt.state_dict(),private={key:dict(parameters=private.state_dict(),optimizer=local.state_dict())},duals={'gpt':old.state_dict()},rng=torch.get_rng_state(),cuda_rng=torch.tensor([1],dtype=torch.uint8)))
    with torch.no_grad():
        for p in (*actor.parameters(),*private.parameters()):p.add_(5)
    new=TokenwiseDual([6],config=dataclasses.replace(c,ranking_loss_mode='relative_gap',ranking_weight=.1),dual_lr=.01)
    cuda=[];monkeypatch.setattr(torch.cuda,'set_rng_state',lambda value:cuda.append(value.clone()))
    restore_numeric_state(saved,actor,{key:private},opt,{key:local},{'gpt':new})
    for name,p in actor.state_dict().items():torch.testing.assert_close(p,saved['actor'][name],atol=0,rtol=0)
    for name,p in private.named_parameters():torch.testing.assert_close(p,saved['private'][key]['parameters'][name],atol=0,rtol=0)
    for before,after in [(saved['shared_optimizer'],opt.state_dict()),(saved['private'][key]['optimizer'],local.state_dict())]:
        assert before['param_groups']==after['param_groups']
        for pid,state in before['state'].items():
            for name,value in state.items():torch.testing.assert_close(after['state'][pid][name],value,atol=0,rtol=0)
    assert new.multipliers==old.multipliers and new.updates==old.updates
    assert len(cuda)==1 and torch.equal(torch.get_rng_state(),saved['rng'])
    bad=TokenwiseDual([6],config=dataclasses.replace(c,scale=1e-6),dual_lr=.01)
    with pytest.raises(ValueError,match='clock units'):restore_numeric_state(saved,actor,{key:private},opt,{key:local},{'gpt':bad})


def test_chained_stage_preserves_origin_and_accepts_closed_512_or_later():
    origin={'protocol':{'architecture':{'name':'fixture'}},'base_recipe_sha256':'base','mixed_index_sha256':'index'}
    metadata={'parent_metadata':{'parent_metadata':origin,'recipe':{'end_update':512}},'recipe':{'end_update':1024}}
    initial={'update':1024,'metadata':metadata,'private':{('gpt',6):{},('cifar',1):{}}}
    recipe=dict(schema='coordinate_seen_continuation_v2',start_update=1024,end_update=2048,checkpoint_every=32,readout_updates=[1024,1536,2048])
    before=copy.deepcopy(initial)
    start,resolved=validate_stage(recipe,initial)
    assert start==1024 and resolved is origin and initial==before
    validate_stage(recipe,initial,{**initial,'update':1536})
    for changes,match in [({'start_update':512},'parent update'),({'end_update':1024},'budget'),({'readout_updates':[2049]},'readout'),({'checkpoint_every':0},'cadence')]:
        with pytest.raises(ValueError,match=match):validate_stage({**recipe,**changes},initial)
    with pytest.raises(ValueError,match='budget'):validate_stage(recipe,initial,{**initial,'update':2049})
    with pytest.raises(ValueError,match='identities'):validate_stage(recipe,initial,{**initial,'private':{}})
    with pytest.raises(ValueError,match='budget'):validate_stage({k:v for k,v in recipe.items() if k!='start_update'},initial)
    with pytest.raises(ValueError,match='provenance'):stage_origin({'parent_metadata':{}})


def test_legacy_recipe_resolves_32_but_does_not_impose_future_stage_cap():
    origin=dict(protocol={},base_recipe_sha256='base',mixed_index_sha256='index')
    initial=dict(update=32,metadata=origin,private={})
    recipe=dict(schema='coordinate_seen_continuation_v1',end_update=512,checkpoint_every=32,readout_updates=[32,64,128,256,512])
    assert validate_stage(recipe,initial)==(32,origin)
    assert validate_stage({**recipe,'schema':'coordinate_seen_continuation_v2','start_update':512,'end_update':1024,'readout_updates':[512,1024]}, {**initial,'update':512})[0]==512


def test_low_lr_stage_resume_uses_parent_absolute_groups_without_changing_moments():
    head=torch.nn.Parameter(torch.ones(2));body=torch.nn.Parameter(torch.ones(3));probe=torch.nn.Parameter(torch.ones(4))
    shared=torch.optim.Adam([{'params':[head],'role':'action_heads','lr':1e-7,'eps':1e-8},{'params':[body],'role':'representation','lr':1e-4,'eps':1e-12}])
    local=torch.optim.Adam([probe],lr=1e-4,eps=1e-12);key=('gpt',6)
    (head.sum()+body.sum()+probe.sum()).backward();shared.step();local.step()
    initial=copy.deepcopy(dict(shared_optimizer=shared.state_dict(),private={key:dict(optimizer=local.state_dict())}))
    recipe=dict(primal_lr_factor=.1,shared_optimizer_transition='preserve',shared_optimizer_mode='separate_representation')
    before=[copy.deepcopy(o.state_dict()['state']) for o in (shared,local)]
    report=configure_stage_optimizers(initial,shared,{key:local},recipe)
    assert [g['lr'] for g in shared.param_groups]==[1e-8,1e-5] and local.param_groups[0]['lr']==1e-5
    for _ in range(2):assert configure_stage_optimizers(initial,shared,{key:local},recipe,resuming=True)==report
    for opt,states in zip((shared,local),before):
        for pid,state in states.items():
            for n,v in state.items():torch.testing.assert_close(opt.state_dict()['state'][pid][n],v,rtol=0,atol=0)
    local.param_groups[0]['lr']*=.1
    with pytest.raises(ValueError,match='resumed optimizer'):configure_stage_optimizers(initial,shared,{key:local},recipe,resuming=True)


def test_panel_expansion_keeps_current_parent_scope_and_requires_pinned_identity(tmp_path):
    import json,hashlib
    origin=dict(protocol={'joint_pilot':dict(gpt_sequences=16,cifar_images=1024)},base_recipe_sha256='base',mixed_index_sha256='index')
    row=dict(domain='gpt',local_id=6,context_ids=list(range(16)),mean_kl=1e-10,strict_topk_changes=0,topk_order_changes=2,topk_membership_changes=0,mean_group_l2_percentage_reduction=.1,group_l2_percentage_reduction={'head.bias':1.5},realized_edit_norm=.2)
    path=tmp_path/'metrics.jsonl';path.write_text(json.dumps(dict(event='fixed_fitting_readout',update=1024,rows=[row]))+'\n')
    pin=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    recipe=dict(fitting_panels={'gpt_sequences':64},parent_readout=pin)
    old,new,changed=resolve_stage_panels(recipe,origin)
    assert old==dict(gpt_sequences=16,cifar_images=1024) and new==dict(gpt_sequences=64,cifar_images=1024) and changed
    parent=dict(parent_metadata=origin,fitting_panel_config=new)
    assert resolve_stage_panels({},parent)==(new,new,False)
    assert resolve_stage_panels({**recipe,'fitting_panels':{'gpt_sequences':128}},parent)[0]==new
    for bad in [{'gpt_sequences':32},{'gpt_sequences':16},{'cifar_images':2048}]:
        with pytest.raises(ValueError):resolve_stage_panels({**recipe,'fitting_panels':bad},parent)
    with pytest.raises(ValueError,match='hash-pinned'):resolve_stage_panels({'fitting_panels':{'gpt_sequences':64}},origin)
    expected=read_parent_readout(pin,1024);assert not parent_readout_differences([row],expected)
    assert parent_readout_differences([{**row,'context_ids':list(range(1,17))}],expected)[0]['field']=='context_ids'
    with pytest.raises(ValueError,match='coverage'):parent_readout_differences([],expected)
    with pytest.raises(ValueError,match='unique'):read_parent_readout(pin,512)
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='SHA'):read_parent_readout(pin,1024)


def test_explicit_decoder_transition_preserves_dimensions_and_resume_identity():
    from scripts.continue_coordinate_chisel import resolve_actor_architecture
    arch=dict(name='checkpoint_coordinate_ed_nanogpt_v1',rank=2,trajectory_dim=12,model_dim=128,num_layers=1,latent_tokens=2,decoder_scale='sqrt_coordinates_over_latent_tokens')
    origin=dict(protocol={'architecture':arch},base_recipe_sha256='base',mixed_index_sha256='index')
    before,after=resolve_actor_architecture({'actor_architecture_transition':{'write_observation_frame':'decoder'}},origin)
    assert before==arch and after['write_observation_frame']=='decoder' and after['name'].endswith('writeframe_v2')
    chained=dict(parent_metadata=origin,actor_architecture=after)
    assert resolve_actor_architecture({},chained)==(after,after)
    with pytest.raises(ValueError,match='unrelated'):resolve_actor_architecture({'actor_architecture_transition':{'model_dim':256}},origin)
    with pytest.raises(ValueError,match='eager'):resolve_actor_architecture({'actor_architecture_transition':{'write_observation_frame':'decoder'},'actor_backend':'eager_cuda_graph_v1'},origin)


def test_explicit_tenfold_lr_transition_is_absolute_on_resume():
    p=torch.nn.Parameter(torch.ones(1));q=torch.nn.Parameter(torch.ones(1))
    shared=torch.optim.Adam([p],lr=1e-9);local=torch.optim.Adam([q],lr=1e-6,eps=1e-12)
    (p.sum()+q.sum()).backward();shared.step();local.step()
    initial=copy.deepcopy(dict(shared_optimizer=shared.state_dict(),private={('gpt',6):dict(optimizer=local.state_dict())}))
    recipe=dict(primal_lr_factor=10.,shared_optimizer_transition='preserve')
    r=configure_stage_optimizers(initial,shared,{('gpt',6):local},recipe)
    assert shared.param_groups[0]['lr']==pytest.approx(1e-8) and local.param_groups[0]['lr']==pytest.approx(1e-5)
    assert configure_stage_optimizers(initial,shared,{('gpt',6):local},recipe,resuming=True)==r
    for bad in (0,float('inf'),float('nan')):
        with pytest.raises(ValueError):configure_stage_optimizers(initial,shared,{('gpt',6):local},{**recipe,'primal_lr_factor':bad})
