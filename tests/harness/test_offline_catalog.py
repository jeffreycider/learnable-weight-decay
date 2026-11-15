import yaml
from chisel.offline_data import Catalog

def test_balanced_epochs_and_resume(tmp_path):
    path=tmp_path/'manifest.yaml'
    path.write_text(yaml.safe_dump(dict(states=[dict(domain=d,path=f'{d}{i}.pt',sha256=f'{d}{i}') for d in ['cifar','text'] for i in range(4)])))
    catalog=Catalog(path,dict(cifar=2,text=1),seed=7)
    first=next(catalog)
    assert [x['domain'] for x in first]==['cifar','cifar','text']
    saved=catalog.state_dict();resumed=Catalog(path,dict(cifar=2,text=1),seed=999);resumed.load_state_dict(saved)
    assert next(catalog)==next(resumed)
