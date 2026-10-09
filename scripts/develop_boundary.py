from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.cheap_bounds import FiniteSystem
from npi.boundary_neural import train_boundary,grid_values
from npi.reference import solve
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=2000);p.add_argument('--ny',type=int,default=41);a=p.parse_args()
out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False)
m=Model.from_dict(json.loads((ROOT/'configs/confirmatory_study.json').read_text())['model'])
s=FiniteSystem.create(m,ny=a.ny,nz=3*(a.ny-1)//4+1,nt=3*(a.ny-1),na=a.ny)
settings={'width':64,'depth':3,'lr':1e-3,'steps':a.steps,'min_lr':1e-5,'device':'cuda','log_every':200}
ref=solve(m,ny=len(s.y),nz=len(s.z),nt=len(s.tau)-1,na=len(s.actions),generator_kind='bounded',tolerance=1e-11);ref.save(out/'reference.npz')
truth=ref.value.reshape(len(s.tau),-1)[:,s.ids]
y=np.repeat(s.y[:,None],len(s.z),axis=1).ravel()[s.ids]
rows=[]
for mode in ['hard','split']:
 net,meta=train_boundary(s,mode,201,settings,out/mode)
 values=grid_values(net,s);np.savez_compressed(out/(mode+'_values.npz'),values=values)
 row={'mode':mode,'seconds':meta['training_seconds'],'relative_value_error':float(np.linalg.norm(values-truth)/np.linalg.norm(truth)),
      'relative_lower_strip_error':float(np.linalg.norm((values-truth)[:,y< -1.5])/np.linalg.norm(truth[:,y< -1.5]))}
 rows.append(row);print('RESULT',json.dumps(row),flush=True)
 (out/'summary.json').write_text(json.dumps(rows,indent=2))
