"""Derivative-error estimates for saved fixed-policy evaluation trajectories.

References are independently solved fixed policies, NOT optimal values.
Finite differences and policy discretization are audited by mesh comparison;
the resulting estimates are not certified continuum derivative errors.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from npi.model import Model
from npi.neural import ValueNetwork, predict, derivatives, greedy, residual
from followup_io import DERIVATIVE_DEFAULT, project_path, verify_inputs, snapshot_sources


def load(path):
    d = torch.load(path, map_location='cpu', weights_only=True)
    n = ValueNetwork(Model.from_dict(d['model']), width=64, depth=3)
    n.load_state_dict(d['state_dict'])
    return n


def reference(path, model, points):
    a = np.load(path)
    axes = [a[k] for k in ('tau', 'y', 'z')]
    nt, ny, nz = map(len, axes)
    v = np.broadcast_to(model.utility(axes[1])[None, :, None], (nt, ny, nz)).copy()
    v.reshape(nt, -1)[:, a['interior']] = a['value']
    hy, hz = axes[1][1]-axes[1][0], axes[2][1]-axes[2][0]
    fields = dict(v=v[:, 1:-1, 1:-1],
        vy=(v[:, 2:, 1:-1]-v[:, :-2, 1:-1])/(2*hy),
        vyy=(v[:, 2:, 1:-1]-2*v[:, 1:-1, 1:-1]+v[:, :-2, 1:-1])/hy**2,
        vyz=(v[:, 2:, 2:]-v[:, 2:, :-2]-v[:, :-2, 2:]+v[:, :-2, :-2])/(4*hy*hz))
    return {k:RegularGridInterpolator([axes[0], axes[1][1:-1], axes[2][1:-1]], x)(points)
            for k,x in fields.items()}


def run(out, resume=False):
    torch.set_num_threads(2)
    if resume and (out/'completion.json').is_file():
        verify_inputs(out)
        print('Completed derivative analysis verified:',out)
        return
    out.mkdir(parents=True, exist_ok=False)
    files=[Path(__file__),ROOT/'scripts/followup_io.py']+list((ROOT/'src/npi').glob('*.py'))
    snapshot_sources(out,files)
    rows, hashes = [], {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    config = dict(seeds=[101,102,103], steps=list(range(0,1001,50)),
        meshes=['fine','finest'], time_points=[.3,.6,1.2,1.8,2.4,3.],
        core_y=list(np.linspace(-.6,.6,13)), lower_y=[-1.8,-1.7,-1.6],
        z=list(np.linspace(-.3,.3,7)),
        qualification='Finite-difference estimates of errors relative to two rounded-policy finite references. Mesh difference is a sensitivity diagnostic, not a rigorous error bound. All saved checkpoints and all three previously inspected seeds are included.')
    (out/'protocol.json').write_text(json.dumps(config, indent=2))
    for seed in config['seeds']:
        base = ROOT/f'results/npi_iteration_diagnostic_20260926/pi_seed{seed}/step_200.pt'
        fixed = load(base); m=fixed.market
        for region, ys in [('core',config['core_y']),('lower',config['lower_y'])]:
            points=np.stack([x.ravel() for x in np.meshgrid(config['time_points'],ys,config['z'],indexing='ij')],axis=1)
            scale=m.utility_bundle(points[:,1])[1]
            refs={}
            for mesh in config['meshes']:
                path=ROOT/f'results/npi_directional_bridge_20260926/{mesh}/seed{seed}/fixed_value.npz'
                hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
                refs[mesh]=reference(path,m,points)
            xp=torch.tensor(points,dtype=torch.float32)
            q=greedy(m,derivatives(fixed,xp)).detach()
            data=dict(points=points, scale=scale, fixed_q=q.numpy(), **{f'{mesh}_{k}':v for mesh,r in refs.items() for k,v in r.items()})
            for step in config['steps']:
                path=base if step==0 else ROOT/f'results/fixed_policy_accuracy_20260926/seed{seed}/step_{step}.pt'
                hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
                net=load(path); pred=predict(net,points)
                ev=residual(m,derivatives(net,xp),q).detach().numpy()/scale
                for mesh, ref in refs.items():
                    delta={k:pred[k]-ref[k] for k in ref}
                    b=m.phi(points[:,2])*delta['vy']+m.rho*m.sigma_z*delta['vyz']
                    c=delta['vy']-delta['vyy']
                    direction=(pred['q']-q.numpy())*(b-.5*(pred['q']+q.numpy())*c)
                    row=dict(seed=seed,region=region,step=step,mesh=mesh,n=len(points),
                        evaluation_rms=float(np.sqrt(np.mean(ev**2))),
                        directional_rms=float(np.sqrt(np.mean((direction/scale)**2))),
                        greedy_rms=float(np.sqrt(np.mean((pred['q']-m.greedy(ref['vy'],ref['vyy'],ref['vyz'],points[:,2]))**2))))
                    for k,e in delta.items():
                        row[k+'_rms']=float(np.sqrt(np.mean((e/scale)**2)))
                        row[k+'_mesh_difference']=float(np.sqrt(np.mean(((refs['fine'][k]-refs['finest'][k])/scale)**2)))
                    rows.append(row)
                data.update({f'step{step}_{k}':v for k,v in pred.items()})
            np.savez_compressed(out/f'seed{seed}_{region}.npz',**data)
        print('DERIVATIVE',seed,flush=True)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'input_sha256.json').write_text(json.dumps(hashes,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(10,5.5),sharex=True)
    for j,key in enumerate(['vy','vyy','vyz']):
        for i,region in enumerate(['core','lower']):
            ax=axs[i,j]
            for seed in config['seeds']:
                r=[r for r in rows if r['seed']==seed and r['region']==region and r['mesh']=='finest']
                line,=ax.plot([x['step'] for x in r],[x[key+'_rms'] for x in r],label=str(seed))
                ax.axhline(r[0][key+'_mesh_difference'],color=line.get_color(),ls=':',alpha=.65)
            ax.set_yscale('log');ax.set_title(region+': '+{'vy':r'$e_y$','vyy':r'$e_{yy}$','vyz':r'$e_{yz}$'}[key])
            if i==1:ax.set_xlabel('Policy-evaluation steps')
            if j==0:ax.set_ylabel('Scaled RMS estimate')
    axs[0,0].legend(title='Training seed',fontsize=8)
    fig.tight_layout();fig.savefig(out/'derivative_errors.pdf');fig.savefig(out/'derivative_errors.png',dpi=180)
    (out/'completion.json').write_text(json.dumps(dict(status='complete',rows=len(rows))))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=project_path,default=DERIVATIVE_DEFAULT)
    parser.add_argument('--resume',action='store_true',help='Verify and skip completed analysis; incomplete outputs are never overwritten.')
    args=parser.parse_args()
    run(args.output,args.resume)


if __name__=='__main__':main()
