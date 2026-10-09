from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.cheap_bounds import FiniteSystem
from npi.boundary_neural import BoundaryValueNetwork
from npi.reference import Reference,solve
from npi.simulation import simulate,paired_ce


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='results/boundary_revision_20260925');p.add_argument('--output',required=True)
    p.add_argument('--after-duality');a=p.parse_args()
    if a.after_duality:
        while not (ROOT/a.after_duality/'status.json').exists():time.sleep(5)
    run=ROOT/a.run;out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((run/'manifest.json').read_text());status=json.loads((run/'status.json').read_text())
    assert len(status)==16 and all(r['status']=='success' for r in status)
    m=Model.from_dict(json.loads((ROOT/manifest['boundary']['model_source']).read_text())['model']);torch.set_num_threads(2)
    rows=[];sim_rows=[];simulation_fields={};systems={};refs={}
    for grid in manifest['boundary']['grids']:
        key=grid['name'];s=FiniteSystem.create(m,**{k:grid[k] for k in ['ny','nz','nt','na']});systems[key]=s
        ref=(Reference.load(ROOT/'results/crosscheck_20260925/bounded_80.npz') if key=='fine'
             else solve(m,**{k:grid[k] for k in ['ny','nz','nt','na']},generator_kind='bounded',tolerance=1e-11))
        refs[key]=ref;ref.save(out/(key+'_reference.npz'))
    for item in status:
        s=systems[item['grid']];ref=refs[item['grid']];n=len(s.ids);indices=np.arange(n);eye=sparse.eye(n,format='csr')
        spatial=np.stack(np.meshgrid(s.y,s.z,indexing='ij'),axis=-1).reshape(-1,2)[s.ids]
        y,z=spatial[:,0],spatial[:,1];candidate=np.load(run/item['name']/'values.npz')['values']
        truth=ref.value.reshape(len(s.tau),-1)[:,s.ids];valid=s.tau>=.3-1e-10
        regions={'global':np.ones(n,dtype=bool),'lower_strip':(y<=-1.5+1e-10)&(abs(z)<=.3+1e-10),
                 'core':(abs(y)<=.7+1e-10)&(abs(z)<=.3+1e-10),'first_row':(y<s.y[1]+1e-10)&(abs(z)<=.3+1e-10)}
        row=dict(item)
        for name,mask in regions.items():
            row[name+'_value_error_pct']=100*float(np.linalg.norm((candidate-truth)[valid][:,mask])/np.linalg.norm(truth[valid][:,mask]))
        scale=m.utility_bundle(y)[1];rms=0.;qgrid=np.zeros_like(ref.value);j=s.terminal.ravel()[s.ids].copy()
        for k,dt in enumerate(np.diff(s.tau),start=1):
            h=s.hamiltonians(candidate[k]);q=h.argmax(axis=0)
            residual=(candidate[k]-candidate[k-1])/dt-h.max(axis=0);rms+=float(np.mean((residual/scale)**2))
            j=spsolve(eye-dt*s.ay[q*n+indices,:],j+dt*s.sources[q,indices])
            flat=qgrid[k].ravel();flat[s.ids]=s.actions[q]
            qgrid[k,0]=qgrid[k,1];qgrid[k,-1]=qgrid[k,-2];qgrid[k,:,0]=qgrid[k,:,1];qgrid[k,:,-1]=qgrid[k,:,-2]
        qgrid[0]=qgrid[1];row['scaled_discrete_residual_rms']=float(np.sqrt(rms/(len(s.tau)-1)))
        for label,y0 in [('center',0.),('near_boundary',-1.9)]:
            nearest=np.argmin((y-y0)**2+z*z)
            row[label+'_discrete_policy_loss']=float(truth[-1,nearest]-j[nearest])
        data=torch.load(run/item['name']/'checkpoint.pt',map_location='cpu',weights_only=True);meta=data['metadata']
        net=BoundaryValueNetwork(m,meta['settings']['width'],meta['settings']['depth'],meta['mode']).double();net.load_state_dict(data['state_dict'])
        with torch.no_grad():
            point=torch.tensor([[m.horizon,m.ymin,0.]],dtype=torch.float64)
            row['lower_interior_trace_jump']=float(net.interior(point)[0,0])-float(m.utility(m.ymin))
        np.savez_compressed(out/(item['name']+'_policy.npz'),tau=s.tau,y=s.y,z=s.z,q=qgrid)
        rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2));print('BOUNDARY',json.dumps(row),flush=True)
        if item['grid']=='fine' and item['seed']==301:simulation_fields[item['mode']]=(s,qgrid)
    # Independent continuous-state check for a pair selected by seed index.
    # Euler exit approximation is checked on two nested time grids.
    for y0 in [0.,-1.8,-1.95]:
        for nt in [480,960]:
            samples={}
            for mode,(s,q) in simulation_fields.items():
                feedback=RegularGridInterpolator((s.tau,s.y,s.z),q)
                paths=simulate(m,feedback,npaths=32768,nsteps=nt,fine_steps=960,seed=91001,initial_y=y0)
                samples[mode]=paths
                np.savez_compressed(out/f'paths_{mode}_y{y0}_nt{nt}.npz',**paths)
            contrast=paired_ce(m,samples['split']['payoff'],samples['hard']['payoff'])
            row={'initial_y':y0,'nt':nt,'paths':32768,
                 'split_CE_advantage_bp':-1e4*contrast['relative_CE_shortfall_vs_reference'],
                 'paired_mc_se_bp':1e4*contrast['paired_mc_se'],
                 'split_exit_fraction':samples['split']['exit_fraction'],'hard_exit_fraction':samples['hard']['exit_fraction']}
            sim_rows.append(row);(out/'simulation.json').write_text(json.dumps(sim_rows,indent=2));print('SIMULATION',json.dumps(row),flush=True)
    (out/'status.json').write_text(json.dumps({'status':'success','trained_models':len(rows),'simulation_comparisons':len(sim_rows)}))


if __name__=='__main__':main()
