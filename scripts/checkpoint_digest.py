"""Hash tensor contents, independently of torch.save container metadata."""
import argparse
import hashlib
import json
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

p=argparse.ArgumentParser()
p.add_argument('checkpoint')
p.add_argument('--output', required=True)
a=p.parse_args()
state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
weights=state['model']
hashes={}
for name, tensor in weights.items():
    tensor=tensor.detach().contiguous()
    hashes[name]=hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
result={'tensor_hashes':hashes,'aggregate':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()}
open(a.output,'w').write(json.dumps(result,indent=2)+'\n')
print(json.dumps({'aggregate':result['aggregate'],'tensors':len(hashes)}))
