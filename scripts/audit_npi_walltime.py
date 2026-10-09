"""Independent policy-bank deployment and paired path audit after timed training."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_npi_iteration import write_csv,write_json
from npi.model import Model
from npi.neural import ValueNetwork,predict
from npi.inexact import PolicyBank
from npi.simulation import simulate,paired_ce


def load_bank(data):
    m=Model.from_dict(data['model']);s=data['settings'];members=[]
    for item in data['policy_bank']:
        net=ValueNetwork(m,s['width'],s['depth']).cuda();net.load_state_dict(item['state_dict'])
        for p in net.parameters():p.requires_grad_(False)
        members.append((item['weight'],net))
    bank=object.__new__(PolicyBank);bank.members=members
    if abs(sum(w for w,_ in members)-1)>1e-10:raise ValueError('Invalid policy weights')
    return bank


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    study=Path(a.study)
    if not (study/'completion.json').exists():raise RuntimeError('Finish all timed runs before any audit')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((study/'protocol.json').read_text());m=Model.from_dict(cfg['model'])
    protocol=dict(training=str(study),paths=cfg['mc_paths'],mc_seed=cfg['mc_seed'],fine_steps=960,
        checkpoints_at_480=[10,20,30],checkpoints_at_960=[30],policy_grid=cfg['policy_grid'],
        scope='Post-training audit: 480-step curves and 960-step endpoint refinement, chosen before audit outcomes; common random paths, no online selection')
    write_json(out/'protocol.json',protocol)
    files=list(study.glob('*_seed*/budget_*.pt'))
    write_json(out/'input_sha256.json',{str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files+[Path(__file__)]})
    axes=[np.linspace(0,m.horizon,41),np.linspace(m.ymin,m.ymax,81),np.linspace(m.zmin,m.zmax,81)]
    points=np.stack([g.ravel() for g in np.meshgrid(*axes,indexing='ij')],axis=1)
    core=np.random.default_rng(9264003).uniform([0,-.7,-.3],[m.horizon,.7,.3],(4096,3))
    torch.set_num_threads(4);base={};rows=[]
    for steps in [480,960]:
        base[steps]=simulate(m,None,myopic=True,npaths=cfg['mc_paths'],nsteps=steps,fine_steps=960,seed=cfg['mc_seed'])['payoff']
        np.savez_compressed(out/f'myopic_{steps}.npz',payoff=base[steps])
    for path in sorted(files):
        data=torch.load(path,map_location='cpu',weights_only=True)
        bank=load_bank(data);started=time.perf_counter()
        q=bank.grid(points).reshape(41,81,81)
        policy=RegularGridInterpolator(axes,q)
        interpolation_rms=float(np.sqrt(np.mean((policy(core)-bank.grid(core))**2)))
        prep=time.perf_counter()-started
        label=path.parent.name+'_'+path.stem
        np.savez_compressed(out/f'{label}_grid.npz',tau=axes[0],y=axes[1],z=axes[2],q=q)
        for nsteps in ([480,960] if data['nominal_seconds']==30 else [480]):
            result=simulate(m,policy,npaths=cfg['mc_paths'],nsteps=nsteps,fine_steps=960,seed=cfg['mc_seed'])
            np.savez_compressed(out/f'{label}_paths_{nsteps}.npz',**result)
            ce=paired_ce(m,result['payoff'],base[nsteps])
            rows.append(dict(method=data['method'],seed=data['seed'],nominal_seconds=data['nominal_seconds'],
                elapsed_seconds=data['elapsed_seconds'],optimizer_steps=data['steps'],time_steps=nsteps,
                gain_bp=-1e4*ce['relative_CE_shortfall_vs_reference'],low_bp=-1e4*ce['paired_mc_95_high'],
                high_bp=-1e4*ce['paired_mc_95_low'],interpolation_rms_core=interpolation_rms,
                deployment_grid_seconds=prep,exit_fraction=result['exit_fraction']))
        write_csv(out/'policy_performance.csv',rows)
        print('AUDITED',label,flush=True)
    write_json(out/'completion.json',dict(status='complete',checkpoints=len(files),comparisons=len(rows)))


if __name__=='__main__':main()
