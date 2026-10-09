from pathlib import Path
import json,sys,time
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from npi.neural import ValueNetwork
from npi.cheap_bounds import FiniteSystem,scalar_bounds
from npi.boundary_neural import grid_values
from npi.model import Model
from npi.reference import Reference
torch.set_num_threads(2)
data=torch.load(ROOT/'results/confirmatory_20260925/main_pinn_seed101/checkpoint.pt',map_location='cpu',weights_only=True)
m=Model.from_dict(data['metadata']['model']);s=FiniteSystem.create(m)
net=ValueNetwork(m,64,3).double();net.load_state_dict(data['state_dict']);v=grid_values(net,s)
n=len(s.ids);ix=np.arange(n);eye=sparse.eye(n,format='csr');j=s.terminal.ravel()[s.ids].copy();jall=[j.copy()];global_bound=0.
start=time.perf_counter();maxgap=[]
for k,dt in enumerate(np.diff(s.tau),start=1):
 q=s.hamiltonians(v[k]).argmax(axis=0);old=j.copy()
 j=spsolve(eye-dt*s.ay[q*n+ix,:],j+dt*s.sources[q,ix]);jall.append(j.copy())
 gap=np.maximum(0.,s.hamiltonians(j).max(axis=0)-(j-old)/dt)
 maxgap.append(float(gap.max()));global_bound+=dt*maxgap[-1]
ref=Reference.load(ROOT/'results/crosscheck_20260925/bounded_80.npz')
print(json.dumps({'seconds':time.perf_counter()-start,'global_bound':global_bound,
 'max_final_gap':maxgap[-1],'max_policy_loss':float(np.max(ref.value[-1].ravel()[s.ids]-j))},indent=2))
result=scalar_bounds(s,np.asarray(jall),exponents=(0.,.5,1.))
print(json.dumps({'upper':result['upper_coefficients'][-1].tolist(),'certificate_seconds':result['certificate_seconds']},indent=2))
