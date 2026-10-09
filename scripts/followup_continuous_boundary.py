"""Continuous-residual boundary comparisons, separate from the finite loss study."""
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import json
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.neural import ValueNetwork, derivatives, greedy, residual, sample, predict
from npi.inexact import evaluation_loss, frozen_copy
from npi.focused_evaluation import EvaluationCorrection, ScalarFeedback
from npi.simulation import simulate, paired_ce
from followup_io import BOUNDARY_DEFAULT, project_path, verify_inputs, snapshot_sources


class ContinuousEvaluator(torch.nn.Module):
    def __init__(self, model, mode, network, base=None):
        super().__init__()
        self.market=model
        self.base=base
        self.correction=EvaluationCorrection(model,mode=mode,**network)
        if base is not None:
            for p in base.parameters():p.requires_grad_(False)

    def forward(self,x):
        if self.base is not None: v=self.base(x)
        else:
            m=self.market;y=x[:,1:2]
            v=torch.exp((1-m.gamma1)*y)/(1-m.gamma1)+m.mixture*torch.exp((1-m.gamma2)*y)/(1-m.gamma2)
        return v+self.correction(x)


def write_json(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def initialize(out):
    config=json.loads((ROOT/'configs/continuous_boundary_followup.json').read_text())
    if not out.exists():
        out.mkdir(parents=True)
        write_json(out/'protocol.json',config)
        files=[Path(__file__),ROOT/'scripts/followup_io.py',ROOT/'configs/continuous_boundary_followup.json',
               ROOT/config['model_source']]+list((ROOT/'src/npi').glob('*.py'))
        files += [ROOT/config['base_source'].format(seed=s) for s in config['seeds']]
        write_json(out/'input_sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        snapshot_sources(out, files)
    elif json.loads((out/'protocol.json').read_text())!=config:raise ValueError('Protocol changed')
    verify_inputs(out)
    torch.set_num_threads(config['threads'])
    return config


def case(out,cfg,phase,seed,mode):
    dest=out/phase/f'seed{seed}'/mode
    if (dest/'completion.json').exists():return
    verify_inputs(out, allow_snapshots=False)
    dest.mkdir(parents=True,exist_ok=False)
    m=Model.from_dict(json.loads((ROOT/cfg['model_source']).read_text())['model'])
    base=None
    if phase=='frozen':
        saved=torch.load(ROOT/cfg['base_source'].format(seed=seed),map_location='cpu',weights_only=True)
        base=ValueNetwork(m,**cfg['network']);base.load_state_dict(saved['state_dict'])
    torch.manual_seed(seed+(cfg['correction_init_offset'] if phase=='frozen' else 0))
    net=ContinuousEvaluator(m,mode,cfg['network'],base)
    fixed=frozen_copy(base if base is not None else net)
    opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=.001)
    stream=torch.Generator().manual_seed(seed+cfg[phase+'_stream_offset'])
    rng=np.random.default_rng(cfg['probe_seed'])
    points=np.concatenate([rng.uniform([0,m.ymin,m.zmin],[m.horizon,m.ymax,m.zmax],(cfg['probe_full'],3)),
                           rng.uniform([0,-.7,-.3],[m.horizon,.7,.3],(cfg['probe_core'],3))])
    xp=torch.tensor(points,dtype=torch.float32); scale=m.utility_bundle(points[:,1])[1]
    axes=[np.linspace(0,m.horizon,cfg['policy_grid'][0]),np.linspace(m.ymin,m.ymax,cfg['policy_grid'][1]),np.linspace(m.zmin,m.zmax,cfg['policy_grid'][2])]
    grid=np.stack([a.ravel() for a in np.meshgrid(*axes,indexing='ij')],axis=1)
    history=[];started=time.perf_counter()
    def save(step):
        torch.save(dict(state_dict=net.state_dict(),model=m.to_dict(),mode=mode,phase=phase,seed=seed,step=step),dest/f'step_{step}.pt')
        d=derivatives(net,xp);q=greedy(m,derivatives(fixed,xp)).detach()
        ev=residual(m,d,q).detach().numpy()/scale
        curv=(d['vy']-d['vyy']).detach().numpy()/scale
        row=dict(phase=phase,seed=seed,mode=mode,step=step,eval_rms_full=float(np.sqrt(np.mean(ev[:cfg['probe_full']]**2))),
            eval_rms_core=float(np.sqrt(np.mean(ev[cfg['probe_full']:]**2))),curvature_min=float(curv.min()),
            curvature_nonpositive=int(np.sum(curv<=0)),seconds=time.perf_counter()-started)
        history.append(row);write_csv(dest/'history.csv',history)
        if step in ([50,1000] if phase=='frozen' else [200,400,2000]):
            qgrid=predict(net,grid,chunk=2048)['q'].reshape(cfg['policy_grid'])
            np.savez_compressed(dest/f'policy_{step}.npz',tau=axes[0],y=axes[1],z=axes[2],q=qgrid)
        print(json.dumps(row),flush=True)
    save(0)
    for step in range(1,cfg[phase+'_steps']+1):
        if phase=='iteration' and (step-1)%cfg['steps_per_policy']==0:fixed=frozen_copy(net)
        absolute=step-1+(200 if phase=='frozen' else 0)
        lr=cfg['learning_rate'];opt.param_groups[0]['lr']=lr['minimum']+.5*(lr['initial']-lr['minimum'])*(1+np.cos(np.pi*absolute/lr['cosine_horizon']))
        if (step-1)%cfg['resample_every']==0:
            x=sample(m,cfg['batch_size'],stream,'cpu',torch.float32)
            q=greedy(m,derivatives(fixed,x)).detach()
        opt.zero_grad(set_to_none=True)
        loss=evaluation_loss(net,x,q,cfg['shape_weight'])
        if not torch.isfinite(loss):raise FloatingPointError((phase,seed,mode,step))
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],1.);opt.step()
        if step in cfg[phase+'_checkpoints']:save(step)
    write_json(dest/'completion.json',dict(status='complete',seconds=time.perf_counter()-started))


def simulation(out,cfg,report_dir=None):
    from scipy.stats import norm
    report_dir=Path(report_dir) if report_dir is not None else out
    required=[out/phase/f'seed{seed}'/mode/'completion.json'
              for seed in cfg['seeds'] for phase in ['frozen','iteration'] for mode in cfg['modes']]
    missing=[str(path) for path in required if not path.exists()]
    if missing:raise FileNotFoundError('Training is incomplete: '+', '.join(missing))
    report_dir.mkdir(parents=True,exist_ok=True)
    m=Model.from_dict(json.loads((ROOT/cfg['model_source']).read_text())['model'])
    rows=[];crit=norm.ppf(1-.05/(2*24))
    for seed in cfg['seeds']:
        paths={}
        for phase,steps in [('frozen',[50,1000]),('iteration',[200,400,2000])]:
            for mode in cfg['modes']:
                dest=out/phase/f'seed{seed}'/mode
                for step in steps:
                    a=np.load(dest/f'policy_{step}.npz');policy=ScalarFeedback([a[k] for k in ['tau','y','z']],a['q'])
                    for nt in cfg['mc_steps']:
                        path=dest/f'paths_{step}_{nt}.npz'
                        if not path.exists():
                            verify_inputs(out,allow_snapshots=False)
                            result=simulate(m,policy,npaths=cfg['mc_paths'],nsteps=nt,fine_steps=max(cfg['mc_steps']),seed=cfg['mc_seed'])
                            np.savez_compressed(path,**result)
                        paths[phase,mode,step,nt]=np.load(path)['payoff']
        for nt in cfg['mc_steps']:
            comparisons=[('frozen_length_'+mode,('frozen',mode,1000,nt),('frozen',mode,50,nt),True) for mode in cfg['modes']]
            comparisons += [('frozen_representation',('frozen','split',1000,nt),('frozen','hard',1000,nt),True),
                            ('iteration_representation',('iteration','split',2000,nt),('iteration','hard',2000,nt),True)]
            comparisons += [('iteration_early_'+mode,('iteration',mode,400,nt),('iteration',mode,200,nt),False) for mode in cfg['modes']]
            for name,a,b,primary in comparisons:
                s=paired_ce(m,paths[a],paths[b]);gain=-1e4*s['relative_CE_shortfall_vs_reference'];se=1e4*s['paired_mc_se']
                rows.append(dict(seed=seed,time_steps=nt,comparison=name,primary=primary,gain_bp=gain,se_bp=se,
                    low_bp=gain-1.96*se,high_bp=gain+1.96*se,adjusted_low_bp=gain-crit*se,adjusted_high_bp=gain+crit*se))
        write_csv(report_dir/'simulation_summary.csv',rows);print('SIMULATION',seed,flush=True)
    write_json(report_dir/'simulation_completion.json',dict(status='complete',rows=len(rows),seeds=cfg['seeds']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['frozen','iteration','simulation'],required=True)
    p.add_argument('--output',type=project_path,default=BOUNDARY_DEFAULT,help='Result directory; relative paths use the project root.')
    args=p.parse_args()
    out=args.output;cfg=initialize(out)
    if args.phase=='simulation':simulation(out,cfg)
    else:
        for seed in cfg['seeds']:
            for mode in cfg['modes']:case(out,cfg,args.phase,seed,mode)


if __name__=='__main__':main()
