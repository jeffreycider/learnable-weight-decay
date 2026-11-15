import hashlib
import torch
import numpy as np
from chisel.cifar import CifarStream, Mix

class Transform:
    def __call__(self,image):
        return torch.from_numpy(np.array(image).copy()).permute(2,0,1).float()+torch.rand(())

def stream(tmp_path, workers=0):
    p=tmp_path/'data.pt'
    if not p.exists():torch.save(dict(images=torch.randint(256,(40,8,8,3),dtype=torch.uint8),labels=torch.arange(40)%2),p)
    return CifarStream(str(p),hashlib.sha256(p.read_bytes()).hexdigest(),'train',3,2,8,3,11,12,Transform(),Mix(13,1,.5,1,2),workers,'cpu')

def test_batch_replay_resume_and_worker_independence(tmp_path):
    s=stream(tmp_path);multi=stream(tmp_path,2)
    for step in range(3):
        expected=next(s);parallel=next(multi);replayed=s.replay(step)
        for key in ('inputs','targets'):
            torch.testing.assert_close(expected[key],replayed[key],rtol=0,atol=0)
            torch.testing.assert_close(expected[key],parallel[key],rtol=0,atol=0)
    resumed=stream(tmp_path);resumed.load_state_dict(s.state_dict())
    expected=next(s);actual=next(resumed)
    for key in ('inputs','targets'):torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)

def test_augmentation_does_not_consume_model_rng(tmp_path):
    s=stream(tmp_path);before=torch.get_rng_state().clone()
    s.replay(0)
    assert torch.equal(before,torch.get_rng_state())
