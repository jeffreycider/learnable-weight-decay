from types import SimpleNamespace
import pytest
from scripts.adapt_coordinate_private import validate_recipe,measure_reused_context


def test_reused_gpt_compute_counts_each_forward_once_not_triangular():
    counter={'student_forwards':0,'student_backwards':0};total={}
    context={'domain':'gpt','engine':SimpleNamespace(compute=SimpleNamespace(as_dict=lambda:dict(counter)))}
    def measure(context,*,learning):
        counter['student_forwards']+=1;counter['student_backwards']+=int(learning)
        for k,v in counter.items():total[k]=total.get(k,0)+v
        total['cached_teacher_head_forwards']=total.get('cached_teacher_head_forwards',0)+1
        return {'value':1}
    contexts=SimpleNamespace(measure=measure,compute=total)
    for learning in (False,True,True,False):assert measure_reused_context(contexts,context,learning=learning)=={'value':1}
    assert total=={'student_forwards':4,'student_backwards':2,'cached_teacher_head_forwards':4}


def test_private_adaptation_rejects_protocol_drift():
    recipe=dict(schema='coordinate_private_adaptation_v1',shared_frozen=True,private_optimizer={'initialization':'fresh'},dual_initialization='fresh_zero',updates=128,readout_updates=[0,1,4,16,64,128],gpt_sequences=16,cifar_images=1024,cases=[dict(name='warm',domain='gpt',local_id=501,initializer='warm499')])
    validate_recipe(recipe)
    with pytest.raises(ValueError,match='fresh'):validate_recipe({**recipe,'shared_frozen':False})
    with pytest.raises(ValueError,match='budget'):validate_recipe({**recipe,'updates':64})
    with pytest.raises(ValueError,match='only499'):validate_recipe({**recipe,'cases':[{**recipe['cases'][0],'local_id':499}]})


def test_declared_cold_lr_transition_does_not_change_other_optimizer_settings():
    from scripts.adapt_coordinate_private import validate_fresh_optimizer
    parent=dict(lr=1e-5,eps=1e-12,betas=(.9,.99))
    validate_fresh_optimizer(parent,parent,{})
    for lr in (1e-4,1e-3):
        new={**parent,'lr':lr};recipe={'private_lr_transition':{'from':1e-5,'to':lr}}
        validate_fresh_optimizer(new,parent,recipe)
        with pytest.raises(ValueError):validate_fresh_optimizer(new,parent,{})
        with pytest.raises(ValueError):validate_fresh_optimizer({**new,'eps':1e-8},parent,recipe)
