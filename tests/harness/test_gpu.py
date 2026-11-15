import pytest
import torch
from chisel.models import CifarGPT
from chisel.observations import Observations
from chisel.muonh import MuonH

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA qualification')

def test_compiled_sensors_reuse_graphs_without_changing_gradients():
    from torch._dynamo.utils import counters
    model=CifarGPT(32,2,4,'mean',head_dim=8,causal=False,learned_qk_scale=True,expansion=2).cuda()
    sensor=Observations(model,r'blocks\.\d+\.(attn\.(q|k|v|proj)|mlp\.(fc|proj))\.weight')
    compiled=torch.compile(model,fullgraph=True,dynamic=False)
    counts=[]
    for step in range(5):
        sensor.active=True
        for squares,count in sensor.sums.values():squares.zero_();count.zero_()
        x=torch.randn(8,3,32,32,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=compiled(x).square().sum()
        loss.backward();sensor.finalize(1);model.zero_grad(set_to_none=True)
        counts.append(counters['stats']['unique_graphs'])
    assert counts[2]==counts[3]==counts[4],counts
    sensor.close()

def test_cuda_muonh_matches_literal_record():
    from test_muonh import reference
    torch.manual_seed(4)
    for shape in [(32,64),(64,32),(32,32)]:
        p=torch.nn.Parameter(torch.randn(*shape,device='cuda'));p.grad=torch.randn_like(p)
        update=compiled_reference_direction(p.grad.clone(),torch.zeros_like(p))
        radius=p.detach().norm();proposal=p.detach()-.018*update*radius/update.norm().clamp_min(1e-10)
        expected=proposal/proposal.norm().clamp_min(1e-10)*radius
        optimizer=MuonH([p],lr=.018,mu=.95,ns_steps=12,compile=True);optimizer.step()
        torch.testing.assert_close(p,expected,rtol=1e-5,atol=2e-6)


@torch.compile(fullgraph=True, dynamic=False)
def compiled_reference_direction(gradient, momentum):
    momentum.lerp_(gradient, 1 - .95)
    update=gradient.lerp_(momentum,.95).bfloat16()
    transposed=gradient.size(-2)>gradient.size(-1)
    if transposed:update=update.mT
    update=update/(update.norm(dim=(-2,-1),keepdim=True)+1e-7)
    for _ in range(12):
        a=update@update.mT;b=-1.5*a+.5*a@a;update=2*update+b@update
    if transposed:update=update.mT
    update*=max(1,gradient.size(-2)/gradient.size(-1))**.5
    return update


def test_cuda_capture_offline_online_pipeline(tmp_path):
    import json,yaml
    from pathlib import Path
    from chisel.harness import resolve,run
    if not Path('data/cifar10/train.pt').exists():pytest.skip('staged CIFAR required')
    source=resolve('recipes/cifar_capture.yaml')
    source.update(train_steps=4,output_dir=str(tmp_path/'source'),microbatch_size=8)
    source['model'].update(width=32,depth=2,head_dim=8)
    source['data'].update(batch_size=8,workers=0)
    source['post_optimizer']=[]
    for site in ['pre_forward','pre_optimizer']:
        for hook in source[site]:
            if 'every' in hook.get('hyperparams',{}):hook['hyperparams']['every']=1
    captured=run(source)
    assert captured.extras['training_forwards']==4
    receipts=sorted((tmp_path/'source/trajectory').rglob('*.ready.json'))[1:-1]
    entries=[json.loads(p.read_text()) for p in receipts]
    manifest=tmp_path/'manifest.yaml'
    manifest.write_text(yaml.safe_dump(dict(states=[dict(path=e['path'],sha256=e['sha256'],domain='cifar') for e in entries])))
    offline=resolve('recipes/offline.yaml')
    offline.update(train_steps=3,output_dir=str(tmp_path/'offline'),microbatch_size=8,validation_data=None)
    offline['model'].update(rank=2,width=32,depth=1,head_dim=8)
    offline['private_probes']['rank']=2
    offline['data'].update(manifest=str(manifest),batch_counts=dict(cifar=2))
    offline['post_optimizer'].append(dict(name=__name__+'.watch_compilation'))
    learned=run(offline)
    assert learned.extras['compile_counts'][1]==learned.extras['compile_counts'][2],learned.extras['compile_counts']
    assert learned.model.proj.weight.norm()>0
    assert learned.extras['training_forwards']==6
    online=resolve('recipes/cifar_online.yaml')
    for key in ['train_steps','model','data','microbatch_size']:
        online[key]=source[key]
    online['output_dir']=str(tmp_path/'online')
    online['chisel']['model']=offline['model'];online['chisel']['start_step']=1
    online['chisel']['pretrained']=str(tmp_path/'offline/latest_rank0.pt')
    online['post_optimizer']=[dict(name='chisel.online.after_optimizer')]
    deployed=run(online)
    assert deployed.extras['training_forwards']==4
    assert deployed.extras['consolidation'].pending_graph is None
    assert sum(x.norm() for x in deployed.extras['consolidation'].consolidated.values())>0


def watch_compilation():
    def hook(config,state):
        from torch._dynamo.utils import counters
        state.extras.setdefault('compile_counts',[]).append(counters['stats']['unique_graphs'])
    return hook
