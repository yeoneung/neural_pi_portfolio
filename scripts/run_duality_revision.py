"""Reference-free portfolio dual diagnostics and independent checkpoint audit."""
from pathlib import Path
import argparse,hashlib,json,sys,time,zipfile
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.neural import ValueNetwork,predict
from npi.duality import simulate_factor_dual as simulate_dual,calibrate_multiplier,dual_statistics,UniformFeedback


def save_paths(folder,name,paths,statistics=None):
    arrays={k:v for k,v in paths.items() if isinstance(v,np.ndarray)}
    if statistics:arrays.update({k:v for k,v in statistics.items() if isinstance(v,np.ndarray)})
    np.savez_compressed(folder/(name+'.npz'),**arrays)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--after-boundary');p.add_argument('--after-analysis');a=p.parse_args()
    if a.after_boundary:
        while True:
            try:done=json.loads((ROOT/a.after_boundary/'status.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):done=[]
            if len(done)==16:break
            time.sleep(5)
    if a.after_analysis:
        while not (ROOT/a.after_analysis/'status.json').exists():time.sleep(5)
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False);(out/'grids').mkdir();(out/'paths').mkdir()
    manifest=json.loads((ROOT/'configs/cost_boundary_revision.json').read_text());cfg=manifest['duality']
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    hashes={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest()
            for sub in ['src','scripts','configs'] for f in (ROOT/sub).rglob('*') if f.suffix in ['.py','.json']}
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    with zipfile.ZipFile(out/'run_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in hashes:z.write(ROOT/name,name)
    torch.set_num_threads(2);rows=[];audits=[];selections=[];input_hashes={}
    for seed in cfg['training_seeds']:
        for method in cfg['methods']:
            candidates={}
            for step in cfg['checkpoints']:
                name=f'{method}_seed{seed}_step{step}'
                path=ROOT/f'results/confirmatory_20260925/main_{method}_seed{seed}'/('checkpoint.pt' if step==10000 else f'step_{step}.pt')
                input_hashes[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
                started=time.perf_counter();saved=torch.load(path,map_location='cpu',weights_only=True)
                meta=saved.get('metadata',saved);m=Model.from_dict(meta['model']);settings=meta['settings']
                net=ValueNetwork(m,settings['width'],settings['depth']).cuda();net.load_state_dict(saved['state_dict'])
                axes=[np.linspace(0,m.horizon,41),np.linspace(m.ymin,m.ymax,81),np.linspace(m.zmin,m.zmax,81)]
                x=np.stack([g.ravel() for g in np.meshgrid(*axes,indexing='ij')],axis=1)
                prediction=predict(net,x,chunk=8192)
                nu=np.where(prediction['vy']>1e-8,-m.sigma_z*np.sqrt(1-m.rho*m.rho)*prediction['vyz']/np.maximum(prediction['vy'],1e-8),0.)
                field=np.stack([prediction['q'],np.clip(nu,-cfg['shadow_exposure_clip'],cfg['shadow_exposure_clip'])],axis=1).reshape(*(len(ax) for ax in axes),2)
                feedback=UniformFeedback(axes,field);preparation=time.perf_counter()-started
                np.savez_compressed(out/'grids'/(name+'.npz'),tau=axes[0],y=axes[1],z=axes[2],fields=field)
                begin=time.perf_counter()
                cal=simulate_dual(m,feedback,npaths=cfg['calibration_paths'],nsteps=cfg['screening_steps'],fine_steps=cfg['fine_steps'],seed=cfg['calibration_seed'])
                lam=calibrate_multiplier(m,cal);calibration=time.perf_counter()-begin
                save_paths(out/'paths',name+'_calibration',cal)
                evaluation=simulate_dual(m,feedback,npaths=cfg['screening_paths'],nsteps=cfg['screening_steps'],fine_steps=cfg['fine_steps'],seed=cfg['screening_seed'])
                post=time.perf_counter();stats=dual_statistics(m,evaluation,lam);post=time.perf_counter()-post
                save_paths(out/'paths',name+'_screen',evaluation,stats)
                upper=stats['mean_gap']+norm.ppf(1-cfg['screening_alpha_per_look'])*stats['gap_mc_se']
                row={'name':name,'seed':seed,'method':method,'step':step,'preparation_seconds':preparation,'calibration_seconds':calibration,
                     'postprocessing_seconds':post,'total_screening_seconds':preparation+calibration+stats['simulation_seconds']+post,
                     'nonpositive_value_slope_fraction':float(np.mean(prediction['vy']<=1e-8)),
                     'shadow_clip_fraction':float(np.mean(abs(nu)>cfg['shadow_exposure_clip'])),
                     'screening_upper_with_three_look_correction':float(upper),
                     'screening_pass':bool(upper<=cfg['screening_tolerance_utility']),
                     **{k:v for k,v in stats.items() if not isinstance(v,np.ndarray)}}
                rows.append(row);candidates[step]=(feedback,lam)
                (out/'screening.json').write_text(json.dumps(rows,indent=2));print('SCREEN',json.dumps(row),flush=True)
            eligible=[r['step'] for r in rows if r['seed']==seed and r['method']==method and r['screening_pass']]
            selected=min(eligible) if eligible else None
            selection={'seed':seed,'method':method,'selected_step':selected,'interpretation':'retrospective; training was not actually stopped'}
            selections.append(selection);(out/'selection.json').write_text(json.dumps(selections,indent=2))
            for step in sorted({10000}|({selected} if selected is not None else set())):
                feedback,lam=candidates[step]
                for nt in cfg['audit_steps']:
                    name=f'{method}_seed{seed}_step{step}_nt{nt}'
                    sample=simulate_dual(m,feedback,npaths=cfg['independent_audit_paths'],nsteps=nt,fine_steps=cfg['fine_steps'],seed=cfg['audit_seed'])
                    stats=dual_statistics(m,sample,lam);save_paths(out/'paths',name+'_audit',sample,stats)
                    row={'name':name,'seed':seed,'method':method,'step':step,'nt':nt,
                         **{k:v for k,v in stats.items() if not isinstance(v,np.ndarray)}}
                    audits.append(row);(out/'audit.json').write_text(json.dumps(audits,indent=2));print('AUDIT',json.dumps(row),flush=True)
    (out/'input_sha256.json').write_text(json.dumps(input_hashes,indent=2))
    (out/'status.json').write_text(json.dumps({'status':'success','screens':len(rows),'audits':len(audits),'selections':len(selections)}))


if __name__=='__main__':main()
