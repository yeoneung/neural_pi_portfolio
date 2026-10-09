from pathlib import Path
import json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from npi.neural import ValueNetwork
from npi.cheap_bounds import FiniteSystem,scalar_bounds
from npi.model import Model
torch.set_num_threads(2)
checkpoint=ROOT/'results/confirmatory_20260925/main_pinn_seed101/checkpoint.pt'
data=torch.load(checkpoint,map_location='cpu',weights_only=True);meta=data['metadata'];m=Model.from_dict(meta['model'])
s=FiniteSystem.create(m)
net=ValueNetwork(m,64,3).double();net.load_state_dict(data['state_dict'])
spatial=np.stack(np.meshgrid(s.y,s.z,indexing='ij'),axis=-1).reshape(-1,2)[s.ids]
start=time.perf_counter();values=[]
with torch.no_grad():
 for tau in s.tau:
  x=np.column_stack([np.full(len(spatial),tau),spatial])
  values.append(net(torch.as_tensor(x,dtype=torch.float64))[:,0].numpy())
values=np.asarray(values);inference=time.perf_counter()-start
result=scalar_bounds(s,values)
center=np.argmin(np.sum(spatial**2,axis=1))
print(json.dumps({'setup_seconds':s.setup_seconds,'candidate_seconds':inference,'certificate_seconds':result['certificate_seconds'],
 'upper':result['upper_coefficients'][-1].tolist(),'lower':result['lower_coefficients'][-1].tolist(),
 'initial_policy_loss_bound':float(result['policy_loss_bound'][center]),'global_bound':result['global_one_sided_bound'],
 'minimum_slack':result['minimum_repair_slack']},indent=2))
