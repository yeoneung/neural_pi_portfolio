"""Run an explicitly labelled pilot; never writes over an existing run."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.reference import Reference,solve
from npi.neural import train,predict


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'configs/pilot.json'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--reference-only',action='store_true')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for folder in ['src','scripts'] for p in (ROOT/folder).rglob('*.py')}
    (out/'source_hashes.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    model=Model.from_dict(config['model'])
    print('Building independent reference...',flush=True)
    ref=solve(model,**config['reference']);ref.save(out/'reference.npz')
    print('Reference:',{k:v for k,v in ref.diagnostics.items() if k!='history'},flush=True)
    if args.reference_only:return
    ev=config['evaluation']
    axes=[np.linspace(0,model.horizon,ev['tau_points']),
          np.linspace(ev['ymin'],ev['ymax'],ev['y_points']),
          np.linspace(ev['zmin'],ev['zmax'],ev['z_points'])]
    grid=np.meshgrid(*axes,indexing='ij');points=np.stack([g.ravel() for g in grid],axis=1)
    vr=ref.interpolate(points);qr=ref.interpolate(points,'policy')
    summaries=[];profiles=[]
    for seed in config['seeds']:
        for method in config['methods']:
            net,meta=train(model,config['training'],method,seed,out/f'{method}_seed{seed}')
            pred=predict(net,points)
            valid=points[:,0]>0 # terminal is imposed exactly, do not dilute error.
            metrics={'method':method,'seed':seed,
                     'relative_l2_value':float(np.linalg.norm(pred['v'][valid]-vr[valid])/np.linalg.norm(vr[valid])),
                     'relative_l2_policy_discrete_reference':float(np.linalg.norm(pred['q'][valid]-qr[valid])/np.linalg.norm(qr[valid])),
                     'hjb_rms_raw_grid':float(np.sqrt(np.mean(pred['residual'][valid]**2))),
                     'hjb_max_raw_grid':float(np.max(np.abs(pred['residual'][valid]))),
                     'min_curvature_grid':float(np.min(pred['curvature'][valid])),
                     'curvature_violation_fraction_grid':float(np.mean(pred['curvature'][valid]<=0)),
                     'wall_seconds_including_diagnostics':meta['wall_seconds_including_diagnostics'],
                     'optimizer_steps':meta['total_optimizer_steps'],
                     'status':'pilot; continuum reference accuracy and cross-seed robustness not yet established'}
            summaries.append(metrics)
            np.savez_compressed(out/f'{method}_seed{seed}'/'evaluation.npz',points=points,reference_value=vr,reference_policy=qr,**pred)
            (out/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
            print(json.dumps(metrics,indent=2),flush=True)
            profile_points=np.column_stack([np.full(121,model.horizon),np.linspace(ev['ymin'],ev['ymax'],121),np.zeros(121)])
            profiles.append((f'{method}, seed {seed}',predict(net,profile_points)['q']))
    fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    xx=profile_points[:,1]
    ax[0].plot(np.exp(xx),ref.interpolate(profile_points,'policy'),'k--',label='Discrete reference (provisional)')
    for label,q in profiles:ax[0].plot(np.exp(xx),q,label=label)
    ax[0].set(xlabel='Wealth',ylabel='Volatility exposure q',title='Non-homothetic policy at z=0');ax[0].legend(fontsize=8)
    ax[1].plot(np.exp(xx),model.relative_risk_aversion(xx))
    ax[1].set(xlabel='Wealth',ylabel='Terminal relative risk aversion',title='Mixed-power utility')
    fig.suptitle('PILOT — not a publication accuracy claim',fontsize=10)
    fig.savefig(out/'pilot_policy.png',dpi=180);fig.savefig(out/'pilot_policy.pdf');plt.close(fig)


if __name__=='__main__':main()

