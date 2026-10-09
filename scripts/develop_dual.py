from pathlib import Path
import json,sys,time
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.neural import ValueNetwork,predict
from npi.duality import simulate_dual,calibrate_multiplier,dual_statistics
torch.set_num_threads(2)
data=torch.load(ROOT/'results/confirmatory_20260925/main_pinn_seed101/checkpoint.pt',map_location='cpu',weights_only=True)
m=Model.from_dict(data['metadata']['model']);net=ValueNetwork(m,64,3).cuda();net.load_state_dict(data['state_dict'])
start=time.perf_counter();axes=[np.linspace(0,m.horizon,41),np.linspace(m.ymin,m.ymax,81),np.linspace(m.zmin,m.zmax,81)]
x=np.stack([a.ravel() for a in np.meshgrid(*axes,indexing='ij')],axis=1)
p=predict(net,x,chunk=8192)
nu=np.where(p['vy']>1e-8,-m.sigma_z*np.sqrt(1-m.rho*m.rho)*p['vyz']/np.maximum(p['vy'],1e-8),0.)
field=np.stack([p['q'],np.clip(nu,-2,2)],axis=1).reshape(*(len(a) for a in axes),2)
f=RegularGridInterpolator(axes,field);prep=time.perf_counter()-start
cal=simulate_dual(m,f,npaths=4096,nsteps=480,seed=81001);lam=calibrate_multiplier(m,cal)
validation=simulate_dual(m,f,npaths=8192,nsteps=480,seed=82001)
stats=dual_statistics(m,validation,lam)
print(json.dumps({'preparation_seconds':prep,'calibration_seconds':cal['simulation_seconds'],**{k:v for k,v in stats.items() if not isinstance(v,np.ndarray)}},indent=2))
out=ROOT/'results/dual_development_20260925';out.mkdir(exist_ok=False)
np.savez_compressed(out/'feedback.npz',tau=axes[0],y=axes[1],z=axes[2],fields=field)
(out/'statistics.json').write_text(json.dumps({k:v for k,v in stats.items() if not isinstance(v,np.ndarray)},indent=2))
