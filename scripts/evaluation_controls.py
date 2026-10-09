"""Matched CPU continuation, exact greedy decomposition, and empirical damping.

All comparisons reuse inspected seeds and are diagnostic. The damping coefficient
matches the update RMS on fixed core points; it is not a certified safe step.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np
import torch
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from npi.model import Model
from npi.neural import ValueNetwork, derivatives, greedy, residual, sample, predict
from npi.inexact import frozen_copy, evaluation_loss
from npi.focused_evaluation import ScalarFeedback
from npi.simulation import simulate, paired_ce
from followup_io import snapshot_sources, verify_inputs

OUT = ROOT / 'results/npi_evaluation_controls_20260929'
BOUNDARY = ROOT / 'results/continuous_boundary_followup_20260929'
DERIVATIVE = ROOT / 'results/derivative_curves_20260929'


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float)**2)))


def matching_step(short, long):
    denominator = rms(long)
    return min(1., rms(short) / denominator) if denominator else 0.


def quotient_terms(phi, rho_sigma, ey, eyz, dc, p_ref, cv):
    return np.asarray([phi*ey/cv, rho_sigma*eyz/cv, -p_ref*dc/cv])


def initialize():
    cfg = json.loads((ROOT/'configs/continuous_boundary_followup.json').read_text())
    design = dict(source_protocol=cfg, device='cpu', optimizer='Adam, unfused, fresh state',
        comparisons=['CPU single: 1000 minus 50', 'GPU single: 1000 minus 50 on common paths',
                     'CPU full correction: 1000 minus 50', 'CPU relaxed correction: 1000 minus 50'],
        inference='Diagnostic reuse of inspected seeds. Bonferroni 24 seedwise length contrasts; other contrasts descriptive.',
        damping_seeds=[101,102,103], damping_mc_seed=9297102,
        damping_rule='min(1,RMS_core(p50-q)/RMS_core(p1000-q)); fixed 546 points, no CE tuning',
        damping_inference='Three damped-minus-short contrasts, Bonferroni 3; conditional MC intervals only.',
        decomposition='Exact interior quotient identity; both finite references; steps 0, 50, 1000; cross terms retained.')
    if not OUT.exists():
        OUT.mkdir(parents=True)
        write_json(OUT/'protocol.json', design)
        files = [Path(__file__), ROOT/'scripts/followup_io.py', ROOT/'configs/continuous_boundary_followup.json',
                 ROOT/cfg['model_source']] + list((ROOT/'src/npi').glob('*.py'))
        files += [ROOT/cfg['base_source'].format(seed=s) for s in cfg['seeds']]
        files += list(DERIVATIVE.glob('seed*_core.npz'))
        files += [p for s in cfg['seeds'] for p in (ROOT/cfg['base_source'].format(seed=s)).parent.glob('policy_*.npz')]
        for s in cfg['seeds']:
            for mode in cfg['modes']:
                files += list((BOUNDARY/'frozen'/f'seed{s}'/mode).glob('paths_*.npz'))
        for s in design['damping_seeds']:
            files += [ROOT/f'results/npi_iteration_diagnostic_20260926/pi_seed{s}/step_200.pt']
            files += [ROOT/f'results/fixed_policy_accuracy_20260926/seed{s}/step_{k}.pt' for k in [50,1000]]
        write_json(OUT/'input_sha256.json', {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        snapshot_sources(OUT, files)
    else:
        assert json.loads((OUT/'protocol.json').read_text()) == design
    verify_inputs(OUT, allow_snapshots=False)
    torch.set_num_threads(cfg['threads'])
    m = Model.from_dict(json.loads((ROOT/cfg['model_source']).read_text())['model'])
    return cfg, design, m


def axes_points(m, cfg):
    axes = [np.linspace(0,m.horizon,cfg['policy_grid'][0]), np.linspace(m.ymin,m.ymax,cfg['policy_grid'][1]),
            np.linspace(m.zmin,m.zmax,cfg['policy_grid'][2])]
    points = np.stack([a.ravel() for a in np.meshgrid(*axes,indexing='ij')],axis=1)
    return axes, points


def load_net(path, m, cfg):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    net = ValueNetwork(m, **cfg['network']); net.load_state_dict(saved['state_dict'])
    return net


def train(cfg, m):
    axes, grid = axes_points(m,cfg)
    rng = np.random.default_rng(cfg['probe_seed'])
    points = np.concatenate([rng.uniform([0,m.ymin,m.zmin],[m.horizon,m.ymax,m.zmax],(cfg['probe_full'],3)),
                            rng.uniform([0,-.7,-.3],[m.horizon,.7,.3],(cfg['probe_core'],3))])
    xp = torch.tensor(points,dtype=torch.float32); scale = m.utility_bundle(points[:,1])[1]
    for seed in cfg['seeds']:
        dest = OUT/'cpu_single'/f'seed{seed}'
        if (dest/'completion.json').exists(): continue
        dest.mkdir(parents=True,exist_ok=True)
        net = load_net(ROOT/cfg['base_source'].format(seed=seed),m,cfg); fixed = frozen_copy(net)
        opt = torch.optim.Adam(net.parameters(),lr=.001, fused=False)
        stream = torch.Generator().manual_seed(seed+cfg['frozen_stream_offset'])
        rows=[]
        def save(step):
            torch.save(dict(state_dict=net.state_dict(), model=m.to_dict(), step=step, seed=seed),dest/f'step_{step}.pt')
            d=derivatives(net,xp); q=greedy(m,derivatives(fixed,xp)).detach()
            ev=residual(m,d,q).detach().numpy()/scale
            cv=(d['vy']-d['vyy']).detach().numpy()/scale
            rows.append(dict(seed=seed,step=step,eval_rms_full=rms(ev[:cfg['probe_full']]),
                             eval_rms_core=rms(ev[cfg['probe_full']:]),curvature_min=float(cv.min())))
            write_csv(dest/'history.csv',rows)
            if step in [50,1000]:
                qgrid=predict(net,grid,chunk=2048)['q'].reshape(cfg['policy_grid'])
                np.savez_compressed(dest/f'policy_{step}.npz',tau=axes[0],y=axes[1],z=axes[2],q=qgrid)
            print('CPU',seed,step,rows[-1]['eval_rms_full'],flush=True)
        save(0)
        for step in range(1,1001):
            lr=cfg['learning_rate']; absolute=200+step-1
            opt.param_groups[0]['lr']=lr['minimum']+.5*(lr['initial']-lr['minimum'])*(1+np.cos(np.pi*absolute/lr['cosine_horizon']))
            if (step-1)%cfg['resample_every']==0:
                x=sample(m,cfg['batch_size'],stream,'cpu',torch.float32)
                q=greedy(m,derivatives(fixed,x)).detach()
            opt.zero_grad(set_to_none=True); loss=evaluation_loss(net,x,q,cfg['shape_weight'])
            if not torch.isfinite(loss): raise FloatingPointError((seed,step))
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            if step in cfg['frozen_checkpoints']: save(step)
        write_json(dest/'completion.json',dict(status='complete'))


def payoff(dest, label, policy_file, m, cfg, nt, mc_seed):
    path=dest/f'paths_{label}_{nt}.npz'
    if not path.exists():
        a=np.load(policy_file); policy=ScalarFeedback([a[k] for k in ['tau','y','z']],a['q'])
        result=simulate(m,policy,npaths=cfg['mc_paths'],nsteps=nt,fine_steps=max(cfg['mc_steps']),seed=mc_seed)
        np.savez_compressed(path,**result)
    return np.load(path)['payoff']


def contrast(m, a, b, family):
    s=paired_ce(m,a,b); gain=-1e4*s['relative_CE_shortfall_vs_reference']; se=1e4*s['paired_mc_se']
    crit=norm.ppf(1-.05/(2*family))
    return dict(gain_bp=gain,se_bp=se,adjusted_low_bp=gain-crit*se,adjusted_high_bp=gain+crit*se)


def matched(cfg,m):
    rows=[]
    for seed in cfg['seeds']:
        dest=OUT/'cpu_single'/f'seed{seed}'
        assert (dest/'completion.json').exists()
        gpu=(ROOT/cfg['base_source'].format(seed=seed)).parent
        for nt in cfg['mc_steps']:
            p={}
            for step in [50,1000]:
                p['cpu',step]=payoff(dest,f'cpu_{step}',dest/f'policy_{step}.npz',m,cfg,nt,cfg['mc_seed'])
                p['gpu',step]=payoff(dest,f'gpu_{step}',gpu/f'policy_{step}.npz',m,cfg,nt,cfg['mc_seed'])
                for mode in cfg['modes']:
                    p[mode,step]=np.load(BOUNDARY/'frozen'/f'seed{seed}'/mode/f'paths_{step}_{nt}.npz')['payoff']
            for mode in ['cpu','gpu','hard','split']:
                rows.append(dict(seed=seed,time_steps=nt,comparison=mode+'_length',primary=True,**contrast(m,p[mode,1000],p[mode,50],24)))
            for mode in ['gpu','hard','split']:
                rows.append(dict(seed=seed,time_steps=nt,comparison=mode+'_minus_cpu_long',primary=False,**contrast(m,p[mode,1000],p['cpu',1000],1)))
        write_csv(OUT/'matched_comparisons.csv',rows)
        print('MATCHED MC',seed,flush=True)


def decomposition(cfg,m):
    rows=[]
    for seed in [101,102,103]:
        a=np.load(DERIVATIVE/f'seed{seed}_core.npz'); phi=m.phi(a['points'][:,2])
        for mesh in ['fine','finest']:
            cref=a[mesh+'_vy']-a[mesh+'_vyy']; pref=(phi*a[mesh+'_vy']+m.rho*m.sigma_z*a[mesh+'_vyz'])/cref
            for step in [0,50,1000]:
                vy=a[f'step{step}_vy']; vyy=a[f'step{step}_vyy']; vyz=a[f'step{step}_vyz']; cv=vy-vyy
                ey=vy-a[mesh+'_vy']; eyy=vyy-a[mesh+'_vyy']; eyz=vyz-a[mesh+'_vyz']; dc=ey-eyy
                gv=(phi*vy+m.rho*m.sigma_z*vyz)/cv
                mask=(cv>0)&(cref>0)&(gv>m.qmin)&(gv<m.qmax)&(pref>m.qmin)&(pref<m.qmax)
                terms=quotient_terms(phi,m.rho*m.sigma_z,ey,eyz,dc,pref,cv)[:,mask]
                error=(gv-pref)[mask]; identity=float(np.max(np.abs(error-terms.sum(axis=0))))
                assert identity<1e-6
                p=a[f'step{step}_q']; d=p-a['fixed_q']
                eta=(phi*ey+m.rho*m.sigma_z*eyz-.5*(p+a['fixed_q'])*dc)/a['scale']
                row=dict(seed=seed,mesh=mesh,step=step,points=len(mask),interior_points=int(mask.sum()),identity_max_error=identity,
                         ey_rms=rms(ey/a['scale']),eyy_rms=rms(eyy/a['scale']),eyz_rms=rms(eyz/a['scale']),
                         greedy_error_rms=rms(error),d_rms=rms(d),eta_scaled_rms=rms(eta),directional_scaled_rms=rms(d*eta))
                for j in range(3): row[f'a{j+1}_rms']=rms(terms[j])
                for i,j in [(0,1),(0,2),(1,2)]: row[f'cross_{i+1}{j+1}']=float(2*np.mean(terms[i]*terms[j]))
                row['sum_component_mse']=float(np.mean(terms**2,axis=1).sum())
                row['sum_cross_terms']=sum(row[f'cross_{i}{j}'] for i,j in [(1,2),(1,3),(2,3)])
                assert abs(row['sum_component_mse']+row['sum_cross_terms']-rms(error)**2)<1e-10
                rows.append(row)
    write_csv(OUT/'quotient_decomposition.csv',rows)


def damping(cfg,design,m):
    axes,grid=axes_points(m,cfg); rows=[]; coefficients=[]
    for seed in design['damping_seeds']:
        dest=OUT/'damping'/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
        a=np.load(DERIVATIVE/f'seed{seed}_core.npz')
        lam=matching_step(a['step50_q']-a['fixed_q'],a['step1000_q']-a['fixed_q'])
        coefficients.append(dict(seed=seed,lambda_value=lam,short_rms=rms(a['step50_q']-a['fixed_q']),
            long_rms=rms(a['step1000_q']-a['fixed_q']),matched_rms=rms(lam*(a['step1000_q']-a['fixed_q']))))
        qgrids={}
        for step in [0,50,1000]:
            path=dest/f'policy_{step}.npz'
            if not path.exists():
                source=(ROOT/f'results/npi_iteration_diagnostic_20260926/pi_seed{seed}/step_200.pt' if step==0 else
                        ROOT/f'results/fixed_policy_accuracy_20260926/seed{seed}/step_{step}.pt')
                net=load_net(source,m,cfg);qgrid=predict(net,grid,chunk=2048)['q'].reshape(cfg['policy_grid'])
                np.savez_compressed(path,tau=axes[0],y=axes[1],z=axes[2],q=qgrid)
            qgrids[step]=np.load(path)['q']
        np.savez_compressed(dest/'policy_damped.npz',tau=axes[0],y=axes[1],z=axes[2],q=qgrids[0]+lam*(qgrids[1000]-qgrids[0]))
        for nt in cfg['mc_steps']:
            p={k:payoff(dest,str(k),dest/f'policy_{k}.npz',m,cfg,nt,design['damping_mc_seed']) for k in [50,1000,'damped']}
            for label,k,b,primary in [('damped_minus_short','damped',50,True),('long_minus_short',1000,50,False),('damped_minus_long','damped',1000,False)]:
                rows.append(dict(seed=seed,time_steps=nt,comparison=label,primary=primary,**contrast(m,p[k],p[b],3 if primary else 1)))
        write_csv(OUT/'damping_comparisons.csv',rows); write_csv(OUT/'damping_coefficients.csv',coefficients)
        print('DAMPING',seed,lam,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','matched','decomposition','damping','all']);args=p.parse_args()
    cfg,design,m=initialize()
    for name,fn in [('train',lambda:train(cfg,m)),('matched',lambda:matched(cfg,m)),
                    ('decomposition',lambda:decomposition(cfg,m)),('damping',lambda:damping(cfg,design,m))]:
        if args.phase in [name,'all']:fn()


if __name__=='__main__': main()
