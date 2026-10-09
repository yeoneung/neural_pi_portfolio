from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from npi.reference import Reference
from npi.simulation import simulate,paired_ce

def main():
    p=argparse.ArgumentParser();p.add_argument('--grids',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True);p.add_argument('--paths',type=int,default=65536)
    p.add_argument('--steps',type=int,default=240);p.add_argument('--fine-steps',type=int,default=480)
    p.add_argument('--pattern',default='main_*');p.add_argument('--resume',action='store_true');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=a.resume);ref=Reference.load(a.reference);m=ref.model
    settings={'paths':a.paths,'steps':a.steps,'fine_steps':a.fine_steps,'seed':70001,'reference':a.reference,'boundary':'first exit, linearly interpolated Euler crossing'}
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text())!=settings:raise ValueError('Resume configuration mismatch')
    else:(out/'config.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    qr=ref.derivative_policy();policy=RegularGridInterpolator((ref.tau,ref.y,ref.z),qr)
    kw={'npaths':a.paths,'nsteps':a.steps,'fine_steps':a.fine_steps}
    if a.resume and (out/'summary.json').exists():
        baseline=dict(np.load(out/'reference_paths.npz'));rows=json.loads((out/'summary.json').read_text())
    else:
        baseline=simulate(m,policy,**kw);np.savez_compressed(out/'reference_paths.npz',**baseline)
        myopic=simulate(m,None,myopic=True,**kw);np.savez_compressed(out/'myopic_paths.npz',**myopic)
        rows=[{'name':'myopic',**paired_ce(m,myopic['payoff'],baseline['payoff']),'exit_fraction':myopic['exit_fraction']}]
    (out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('Reference/myopic done',flush=True)
    for path in sorted(Path(a.grids).glob(a.pattern+'.npz')):
        if path.stem in {r['name'] for r in rows}:continue
        data=np.load(path);meta=json.loads(path.with_suffix('.json').read_text())
        if meta['model']!=m.to_dict():raise ValueError('Policy and reference models differ')
        interp=RegularGridInterpolator((data['tau'],data['y'],data['z']),np.stack([data['q'],data['residual']],axis=-1))
        result=simulate(m,interp,**kw);np.savez_compressed(out/(path.stem+'_paths.npz'),**result)
        row={'name':path.stem,'method':meta['method'],'seed':meta['seed'],**paired_ce(m,result['payoff'],baseline['payoff']),
             'exit_fraction':result['exit_fraction'],'initial_neural_value':meta['initial_value'],
             'mean_integrated_HJB_residual':float(np.mean(result['residual_integral'])),
             'ito_identity_discrepancy':float(np.mean(result['payoff']-result['residual_integral'])-meta['initial_value']),
             'identity_scope':'empirical only; Euler and residual/control interpolation errors remain'}
        rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)

if __name__=='__main__':main()
