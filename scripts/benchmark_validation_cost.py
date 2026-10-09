from pathlib import Path
import argparse,hashlib,json,sys,time,zipfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.reference import solve
from npi.cheap_bounds import FiniteSystem
from npi.neural import ValueNetwork,predict
from npi.deficit_only import evaluate

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--after-study');a=p.parse_args()
if a.after_study:
    while not (ROOT/a.after_study/'status.json').exists():time.sleep(5)
out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
path=ROOT/'results/confirmatory_20260925/main_pinn_seed106/checkpoint.pt'
data=torch.load(path,map_location='cpu',weights_only=True);m=Model.from_dict(data['metadata']['model'])
net=ValueNetwork(m,64,3).cuda();net.load_state_dict(data['state_dict'])
solve(m,ny=17,nz=13,nt=10,na=11,generator_kind='bounded')
predict(net,np.array([[1.,0.,0.]]));rows=[]
for repetition in range(3):
    methods=['direct_reference','deficit_correction'] if repetition%2==0 else ['deficit_correction','direct_reference']
    for method in methods:
        started=time.perf_counter()
        if method=='direct_reference':
            ref=solve(m,ny=81,nz=61,nt=240,na=81,generator_kind='bounded',tolerance=1e-11)
            row={'method':method,'repetition':repetition,'seconds':time.perf_counter()-started,
                 'linear_solves':sum(h['policy_iterations'] for h in ref.diagnostics['history']),
                 'max_bellman_residual':ref.diagnostics['max_discrete_bellman_residual']}
            ref.save(out/f'reference_rep{repetition}.npz')
        else:
            system=FiniteSystem.create(m)
            xyz=np.stack([x.ravel() for x in np.meshgrid(system.tau,system.y,system.z,indexing='ij')],axis=1)
            pred=predict(net,xyz,chunk=8192);nt=len(system.tau)
            values=pred['v'].reshape(nt,-1)[:,system.ids].astype(float)
            controls=pred['q'].reshape(nt,-1)[:,system.ids].astype(float)
            values[0]=system.terminal.ravel()[system.ids];preparation=time.perf_counter()-started
            result=evaluate(system,values,controls)
            center=np.argmin(np.sum(xyz[:len(system.y)*len(system.z)][system.ids,1:]**2,axis=1))
            row={'method':method,'repetition':repetition,'seconds':time.perf_counter()-started,
                 'preparation_seconds':preparation,'linear_solves':result['linear_solves'],
                 'initial_bound':float(result['policy_loss_bound'][center]),
                 'max_bellman_residual':max(h['residual'] for h in result['history'])}
            np.savez_compressed(out/f'deficit_rep{repetition}.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        rows.append(row);(out/'timings.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    loss=ref.value[-1].ravel()[system.ids]-result['policy_value']
    assert np.max(loss-result['policy_loss_bound'])<1e-6
hashes={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest()
        for sub in ['src','scripts'] for f in (ROOT/sub).rglob('*.py')}
with zipfile.ZipFile(out/'postrun_sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for name in hashes:archive.write(ROOT/name,name)
(out/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
(out/'metadata.json').write_text(json.dumps({'repetitions':3,'grid':[81,61,240,81],
 'checkpoint':str(path.relative_to(ROOT)),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
 'cost_scope':'full-grid reference versus deficit correction plus policy evaluation; dual screening timings are recorded separately',
 'source_capture':'after timing; no claim of pre-run capture','all_reported_discrete_inequalities_passed':True},indent=2))
(out/'status.json').write_text(json.dumps({'status':'success','timings':len(rows)}))
