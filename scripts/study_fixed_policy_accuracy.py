"""Shared-prefix evaluation-length intervention on a strictly frozen policy."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import zipfile

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from diagnose_npi_iteration import load_network, write_csv, write_json
from npi.neural import sample, greedy, derivatives, predict
from npi.inexact import frozen_copy, evaluation_loss, coefficients, disagreement_metrics
from npi.simulation import simulate, paired_ce


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    config=dict(seeds=[101,102,103],inner_steps=[50,200,1000],saved_every=50,
        frozen_policy_source='results/npi_iteration_diagnostic_20260926/pi_seed{seed}/step_200.pt',
        optimizer='fresh Adam, identical state for all saved prefixes',lr=.001,
        lr_schedule='cosine in absolute step 200+j with T_max=10000, independent of selected inner budget',
        collocation_seed_offset=517000,probe_seed=9263001,mc_seed=9263002,
        mc_paths=32768,mc_steps=[480,960],mc_fine_steps=960,policy_grid=[41,81,81],
        empirical_disagreement_multiplier=2.,disagreement_threshold=.5,
        status='development intervention; three previously inspected training seeds; no speed claim')
    write_json(out/'protocol.json',config)
    sources=[ROOT/'src/npi/neural.py',ROOT/'src/npi/inexact.py',ROOT/'src/npi/simulation.py',Path(__file__)]
    inputs=[ROOT/config['frozen_policy_source'].format(seed=s) for s in config['seeds']]
    write_json(out/'input_sha256.json',{str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources+inputs})
    with zipfile.ZipFile(out/'sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sources:z.write(f,f.name)
        z.write(out/'protocol.json','protocol.json')
    torch.set_num_threads(4);history=[];mcrows=[];allpred=[]
    for seed,path in zip(config['seeds'],inputs):
        net=load_network(path);m=net.market;fixed=frozen_copy(net)
        setting=torch.load(path,map_location='cpu',weights_only=True)['settings']
        optimizer=torch.optim.Adam(net.parameters(),lr=.001,fused=True)
        stream=torch.Generator().manual_seed(seed+config['collocation_seed_offset'])
        rng=np.random.default_rng(config['probe_seed'])
        points=rng.uniform([0,m.ymin,m.zmin],[m.horizon,m.ymax,m.zmax],(8192,3))
        core=rng.uniform([0,-.7,-.3],[m.horizon,.7,.3],(4096,3))
        xp=torch.as_tensor(np.concatenate([points,core]),device='cuda',dtype=torch.float32)
        masks=dict(full=slice(0,8192),core=slice(8192,None))
        scale=torch.as_tensor(m.utility_bundle(xp[:,1].cpu().numpy())[1],device='cuda')
        fixed_q=greedy(m,derivatives(fixed,xp)).detach()
        prev=coefficients(net,xp);started=time.perf_counter()
        folder=out/f'seed{seed}';folder.mkdir()
        snapshots={0:frozen_copy(net)}
        for step in range(1,1001):
            absolute=200+step-1
            lr=1e-5+.5*(.001-1e-5)*(1+np.cos(np.pi*absolute/10000))
            optimizer.param_groups[0]['lr']=lr
            if (step-1)%50==0:
                x=sample(m,1024,stream,'cuda',torch.float32)
                q=greedy(m,derivatives(fixed,x)).detach()
            optimizer.zero_grad(set_to_none=True)
            loss=evaluation_loss(net,x,q)
            if not torch.isfinite(loss):raise FloatingPointError((seed,step))
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
            if step%50==0:
                now=coefficients(net,xp)
                for region,ix in masks.items():
                    metric=disagreement_metrics({k:v[ix] for k,v in now.items()},
                        {k:v[ix] for k,v in prev.items()},fixed_q[ix],scale[ix])
                    history.append(dict(seed=seed,inner_steps=step,region=region,seconds=time.perf_counter()-started,**metric))
                prev=now
                torch.save(dict(state_dict=net.state_dict(),model=m.to_dict(),settings=setting,seed=seed,step=step),folder/f'step_{step}.pt')
                if step in config['inner_steps']:snapshots[step]=frozen_copy(net)
        write_csv(out/'evaluation_history.csv',history)
        # A single frozen evaluation trajectory supplies every selected budget.
        axes=[np.linspace(0,m.horizon,41),np.linspace(m.ymin,m.ymax,81),np.linspace(m.zmin,m.zmax,81)]
        gp=np.stack([g.ravel() for g in np.meshgrid(*axes,indexing='ij')],axis=1)
        payoffs={}
        for step,model in snapshots.items():
            qgrid=predict(model,gp)['q'].reshape(41,81,81)
            policy=RegularGridInterpolator(axes,qgrid)
            pred=predict(model,core)
            np.savez_compressed(folder/f'core_{step}.npz',points=core,**pred)
            for nsteps in config['mc_steps']:
                result=simulate(m,policy,npaths=config['mc_paths'],nsteps=nsteps,fine_steps=960,seed=config['mc_seed'])
                np.savez_compressed(folder/f'paths_{step}_{nsteps}.npz',**result)
                payoffs[step,nsteps]=result['payoff']
                if step:
                    ce=paired_ce(m,result['payoff'],payoffs[0,nsteps])
                    mcrows.append(dict(seed=seed,inner_steps=step,time_steps=nsteps,
                        gain_bp=-1e4*ce['relative_CE_shortfall_vs_reference'],
                        low_bp=-1e4*ce['paired_mc_95_high'],high_bp=-1e4*ce['paired_mc_95_low']))
            write_csv(out/'policy_changes.csv',mcrows) if mcrows else None
            print('Frozen evaluation audit:',seed,step,flush=True)
    write_json(out/'completion.json',dict(status='complete',history_rows=len(history),mc_rows=len(mcrows)))


if __name__=='__main__':main()
