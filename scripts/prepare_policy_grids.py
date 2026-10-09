"""Cache neural policies for independent simulation and audit interpolation."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.neural import ValueNetwork,predict

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--variant',default='main');p.add_argument('--seed',type=int,default=None)
    p.add_argument('--time-points',type=int,default=41);p.add_argument('--space-points',type=int,default=81);args=p.parse_args()
    run=Path(args.run);out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(2)
    rows=[]
    pattern=args.variant+'_*_seed'+('*' if args.seed is None else str(args.seed))+'/checkpoint.pt'
    for path in sorted(run.glob(pattern)):
        target=out/(path.parent.name+'.npz')
        if target.exists():
            rows.append(json.loads((out/(path.parent.name+'.json')).read_text()));continue
        data=torch.load(path,map_location='cpu',weights_only=True);meta=data['metadata'];m=Model.from_dict(meta['model'])
        net=ValueNetwork(m,meta['settings']['width'],meta['settings']['depth']);net.load_state_dict(data['state_dict'])
        axes=[np.linspace(0,m.horizon,args.time_points),np.linspace(m.ymin,m.ymax,args.space_points),np.linspace(m.zmin,m.zmax,args.space_points)]
        pts=np.stack([g.ravel() for g in np.meshgrid(*axes,indexing='ij')],axis=1)
        pred=predict(net,pts);shape=tuple(len(x) for x in axes);q=pred['q'].reshape(shape);res=pred['residual'].reshape(shape)
        interp=RegularGridInterpolator(axes,q)
        rng=np.random.default_rng(45678);test=rng.uniform([0,-.8,-.5],[m.horizon,.8,.5],size=(4096,3))
        true=predict(net,test)['q'];approx=interp(test)
        row={'name':path.parent.name,'model':meta['model'],'initial_value':float(predict(net,np.array([[m.horizon,0,0]]))['v'][0]),
             'relative_policy_interpolation_error_core':float(np.linalg.norm(approx-true)/np.linalg.norm(true)),
             'max_policy_interpolation_error_core':float(np.max(np.abs(approx-true))),
             'source_checkpoint':str(path),'method':meta['method'],'seed':meta['seed'],'grid_shape':list(shape)}
        np.savez_compressed(target,tau=axes[0],y=axes[1],z=axes[2],q=q,residual=res)
        (out/(path.parent.name+'.json')).write_text(json.dumps(row,indent=2),encoding='utf-8')
        rows.append(row);print(json.dumps(row),flush=True)
    (out/'interpolation_audit.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__':main()
