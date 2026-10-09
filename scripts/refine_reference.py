"""Additional reference refinements; compare policies reconstructed from values too."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.reference import solve


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    axes=[np.linspace(.1,1,6),np.linspace(-.6,.6,13),np.linspace(-.3,.3,9)]
    points=np.stack([g.ravel() for g in np.meshgrid(*axes,indexing='ij')],axis=1)
    previous=None;rows=[]
    for ny,nz,nt,na in [(81,41,120,81),(121,61,180,121),(161,81,240,161)]:
        ref=solve(Model(),ny=ny,nz=nz,nt=nt,na=na)
        name=f'{ny}x{nz}_t{nt}_a{na}'
        ref.save(out/f'{name}.npz')
        fields={'value':ref.interpolate(points),'discrete_policy':ref.interpolate(points,'policy'),
                'derivative_policy':RegularGridInterpolator((ref.tau,ref.y,ref.z),ref.derivative_policy())(points)}
        row={'name':name,'wall_seconds':ref.diagnostics['wall_seconds'],
             'discrete_bellman_residual':ref.diagnostics['max_discrete_bellman_residual']}
        if previous is not None:
            for key,v in fields.items():
                row[f'{key}_relative_change_vs_previous']=float(np.linalg.norm(v-previous[key])/np.linalg.norm(v))
        rows.append(row);previous=fields
        (out/'refinements.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
