"""Two shrinking-stencil families and independent mesh/time/action/domain audits."""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.reference import solve

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    m=Model.from_dict(json.loads((ROOT/'configs/stress_pilot.json').read_text())['model'])
    axes=[np.linspace(.3,3,6),np.linspace(-.6,.6,13),np.linspace(-.3,.3,9)]
    points=np.stack([g.ravel() for g in np.meshgrid(*axes,indexing='ij')],axis=1)
    cases=[]
    for kind in ['bounded','semi_lagrangian']:
        for n,nt,na in [(80,240,81),(120,360,121),(160,480,161)]:
            cases.append((f'{kind}_{n}',m,n+1,3*n//4+1,nt,na,kind))
    cases.extend([
        ('time',m,121,91,720,121,'bounded'),
        ('actions',m,121,91,360,241,'bounded'),
        ('domain',replace(m,ymin=-3,ymax=3,zmin=-2.25,zmax=2.25),181,136,360,121,'bounded')])
    rows=[];fields={}
    for name,market,ny,nz,nt,na,kind in cases:
        print('START '+name,flush=True)
        ref=solve(market,ny=ny,nz=nz,nt=nt,na=na,generator_kind=kind,tolerance=1e-11)
        ref.save(out/f'{name}.npz')
        d=RegularGridInterpolator((ref.tau,ref.y,ref.z),ref.derivative_policy())(points)
        fields[name]={'value':ref.interpolate(points),'policy':d}
        row={'name':name,**{k:v for k,v in ref.diagnostics.items() if k!='history'}}
        if 'bounded_120' in fields and name!='bounded_120':
            for key in ['value','policy']:
                row[key+'_change_vs_bounded_120']=float(np.linalg.norm(fields[name][key]-fields['bounded_120'][key])/np.linalg.norm(fields['bounded_120'][key]))
        rows.append(row)
        (out/'audit.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        np.savez_compressed(out/'comparison_fields.npz',points=points,**{name+'_'+key:arr for name,item in fields.items() for key,arr in item.items()})
        print(json.dumps(row),flush=True)

if __name__=='__main__':main()
