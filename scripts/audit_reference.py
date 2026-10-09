"""Independent mesh, time, action and domain sensitivity on a fixed window."""
from pathlib import Path
import argparse,json,sys
from dataclasses import replace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.reference import solve


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    model=Model()
    specs=[('base',model,41,21,60,41),('mesh',model,81,41,60,41),
           ('time',model,41,21,120,41),('actions',model,41,21,60,81),
           ('domain',replace(model,ymin=-3,ymax=3,zmin=-1.5,zmax=1.5),61,31,60,41),
           ('joint',model,81,41,120,81)]
    grid=np.meshgrid(np.linspace(.1,1,6),np.linspace(-.6,.6,13),np.linspace(-.3,.3,9),indexing='ij')
    points=np.stack([x.ravel() for x in grid],axis=1)
    values={};policies={};rows=[]
    for name,m,ny,nz,nt,na in specs:
        print('REFERENCE AUDIT',name,flush=True)
        r=solve(m,ny=ny,nz=nz,nt=nt,na=na);r.save(out/f'{name}.npz')
        values[name]=r.interpolate(points);policies[name]=r.interpolate(points,'policy')
        row={'case':name,'ny':ny,'nz':nz,'nt':nt,'na':na,'wall_seconds':r.diagnostics['wall_seconds'],
             'discrete_residual':r.diagnostics['max_discrete_bellman_residual']}
        if name!='base':
            row['relative_l2_value_change_vs_base']=float(np.linalg.norm(values[name]-values['base'])/np.linalg.norm(values[name]))
            row['relative_l2_policy_change_vs_base']=float(np.linalg.norm(policies[name]-policies['base'])/np.linalg.norm(policies[name]))
        rows.append(row)
        (out/'audit.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(row,flush=True)
    # No automatic "certified" label: the experiment tolerances need to be
    # compared against the actual claimed effect size before publication.


if __name__=='__main__':main()
