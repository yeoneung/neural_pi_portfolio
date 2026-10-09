"""Independent frozen-policy references for both matched CPU evaluation designs."""
from pathlib import Path
import argparse
import csv
import json
import sys
import time
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.neural import ValueNetwork, predict
from npi.focused_evaluation import ScalarFeedback
from npi.stencils import build
from npi.policy_error_audit import action_rows
from followup_continuous_boundary import ContinuousEvaluator
from followup_derivative_curves import reference
from followup_io import project_path, sha256, snapshot_sources, verify_inputs
DEFAULT=ROOT/'results/npi_matched_derivatives_20260929'


def write_json(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')


def write_csv(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))


def initialize(out):
    config=ROOT/'configs/matched_derivative_bridge.json';cfg=json.loads(config.read_text())
    if not out.exists():
        out.mkdir(parents=True)
        write_json(out/'protocol.json',cfg)
        files=[Path(__file__),config,ROOT/cfg['model_source']]
        files += [ROOT/'scripts'/s for s in ['followup_continuous_boundary.py','followup_derivative_curves.py','followup_io.py']]
        files += list((ROOT/'src/npi').glob('*.py'))
        files += [ROOT/cfg[mode+'_source'].format(seed=s,step=k) for s in cfg['seeds'] for mode in cfg['modes'] for k in cfg['steps']]
        write_json(out/'input_sha256.json',{p.relative_to(ROOT).as_posix():sha256(p) for p in files})
        snapshot_sources(out,files)
    assert json.loads((out/'protocol.json').read_text())==cfg
    verify_inputs(out,allow_snapshots=False)
    torch.set_num_threads(cfg['threads'])
    m=Model.from_dict(json.loads((ROOT/cfg['model_source']).read_text())['model'])
    return cfg,m


def load_net(cfg,m,mode,seed,step):
    a=torch.load(ROOT/cfg[mode+'_source'].format(seed=seed,step=step),map_location='cpu',weights_only=True)
    assert a['model']==m.to_dict()
    base=ValueNetwork(m,**cfg['network'])
    net=base if mode=='single' else ContinuousEvaluator(m,'hard',cfg['network'],base)
    net.load_state_dict(a['state_dict']);net.eval()
    return net


def reference_solve(ay,source,eye,terminal,points,actions,tau,policy):
    n=len(points);dt=tau[1]-tau[0];idx=np.arange(n)
    values=np.empty((len(tau),n));values[0]=terminal
    max_residual=0.
    for k in range(1,len(tau)):
        points[:,0]=tau[k];rows=action_rows(policy(points),actions,n)
        mat=eye-dt*ay[rows,:];rhs=values[k-1]+dt*source[rows//n,idx]
        values[k]=splu(mat.tocsc()).solve(rhs)
        max_residual=max(max_residual,float(np.max(np.abs(mat@values[k]-rhs))))
    return values,max_residual


def solve(out,cfg,m):
    axes=[np.linspace(0,m.horizon,cfg['policy_grid'][0]),np.linspace(m.ymin,m.ymax,cfg['policy_grid'][1]),np.linspace(m.zmin,m.zmax,cfg['policy_grid'][2])]
    gp=np.stack([a.ravel() for a in np.meshgrid(*axes,indexing='ij')],axis=1)
    for seed in cfg['seeds']:
        dest=out/f'seed{seed}';dest.mkdir(exist_ok=True)
        p=dest/'fixed_policy.npz'
        if not p.exists():
            q=predict(load_net(cfg,m,'single',seed,0),gp)['q'].reshape(cfg['policy_grid'])
            np.savez_compressed(p,tau=axes[0],y=axes[1],z=axes[2],q=q)
    for mesh,(ny,nz,nt,na) in cfg['meshes'].items():
        if all((out/f'seed{s}'/f'{mesh}_completion.json').exists() for s in cfg['seeds']):continue
        y=np.linspace(m.ymin,m.ymax,ny);z=np.linspace(m.zmin,m.zmax,nz)
        tau=np.linspace(0,m.horizon,nt+1);actions=np.linspace(m.qmin,m.qmax,na)
        stack,ids,_,diagnostics=build(m,y,z,actions,cfg['generator'].split()[0])
        boundary=np.setdiff1d(np.arange(ny*nz),ids);n=len(ids)
        yy,zz=np.meshgrid(y,z,indexing='ij');terminal=m.utility(yy).ravel()
        ay=stack[:,ids].tocsr();source=np.asarray(stack[:,boundary]@terminal[boundary]).reshape(na,n)
        del stack
        eye=sparse.eye(n,format='csc');points=np.column_stack([np.zeros(n),yy.ravel()[ids],zz.ravel()[ids]])
        write_json(out/f'{mesh}_stencil.json',diagnostics)
        for seed in cfg['seeds']:
            dest=out/f'seed{seed}';marker=dest/f'{mesh}_completion.json'
            if marker.exists():continue
            a=np.load(dest/'fixed_policy.npz');policy=ScalarFeedback([a[k] for k in ['tau','y','z']],a['q'])
            started=time.perf_counter();values,err=reference_solve(ay,source,eye,terminal[ids],points,actions,tau,policy)
            path=dest/f'{mesh}_value.npz'
            np.savez_compressed(path,tau=tau,y=y,z=z,interior=ids,value=values)
            write_json(marker,dict(status='complete',seconds=time.perf_counter()-started,maximum_linear_residual=err,sha256=sha256(path)))
            print('REFERENCE',mesh,seed,'seconds',round(time.perf_counter()-started,1),'residual',err,flush=True)
        del ay,source,eye,values


def diagnostics(a,cfg,m,seed):
    phi=m.phi(a['points'][:,2]);mask=np.ones(len(phi),dtype=bool)
    refs={}
    for mesh in cfg['meshes']:
        cr=a[mesh+'_vy']-a[mesh+'_vyy'];br=phi*a[mesh+'_vy']+m.rho*m.sigma_z*a[mesh+'_vyz']
        pr=np.divide(br,cr,out=np.full_like(br,np.nan),where=cr!=0)
        refs[mesh]=(cr,pr)
        mask &= (cr>0)&(pr>m.qmin)&(pr<m.qmax)
    for mode in cfg['modes']:
        for step in cfg['steps']:
            prefix=f'{mode}_{step}_';cv=a[prefix+'vy']-a[prefix+'vyy']
            b=phi*a[prefix+'vy']+m.rho*m.sigma_z*a[prefix+'vyz']
            p=np.divide(b,cv,out=np.full_like(b,np.nan),where=cv!=0)
            mask &= (cv>0)&(p>m.qmin)&(p<m.qmax)
    if not mask.any():raise ValueError('No common interior diagnostic points')
    rows=[]
    for mesh,(cr,pr) in refs.items():
        pref_clipped=m.greedy(a[mesh+'_vy'],a[mesh+'_vyy'],a[mesh+'_vyz'],a['points'][:,2])
        for mode in cfg['modes']:
            for step in cfg['steps']:
                pfx=f'{mode}_{step}_';p=a[pfx+'q'];cv=a[pfx+'vy']-a[pfx+'vyy']
                ey=a[pfx+'vy']-a[mesh+'_vy'];eyy=a[pfx+'vyy']-a[mesh+'_vyy'];eyz=a[pfx+'vyz']-a[mesh+'_vyz']
                dc=ey-eyy;db=phi*ey+m.rho*m.sigma_z*eyz
                terms=np.asarray([phi[mask]*ey[mask],m.rho*m.sigma_z*eyz[mask],-pr[mask]*dc[mask]])/cv[mask]
                pv=(phi[mask]*a[pfx+'vy'][mask]+m.rho*m.sigma_z*a[pfx+'vyz'][mask])/cv[mask]
                error=pv-pr[mask];err=float(np.max(np.abs(error-terms.sum(axis=0))))
                component_mse=float(np.mean(terms**2,axis=1).sum())
                cross=[float(2*np.mean(terms[i]*terms[j])) for i,j in [(0,1),(0,2),(1,2)]]
                assert err<1e-6 and abs(rms(error)**2-component_mse-sum(cross))<1e-10
                d=p-a['fixed_q'];eta=(db-.5*(p+a['fixed_q'])*dc)/a['scale']
                row=dict(seed=seed,mesh=mesh,mode=mode,step=step,points=len(mask),common_interior_points=int(mask.sum()),
                    greedy_error_rms=rms(error),all_point_clipped_error_rms=rms(p-pref_clipped),
                    a1_rms=rms(terms[0]),a2_rms=rms(terms[1]),a3_rms=rms(terms[2]),
                    cross12=cross[0],cross13=cross[1],cross23=cross[2],cross_sum=sum(cross),component_mse=component_mse,
                    combined_to_component_ratio=rms(error)/np.sqrt(component_mse) if component_mse else 0.,
                    ey_scaled_rms=rms(ey/a['scale']),eyy_scaled_rms=rms(eyy/a['scale']),eyz_scaled_rms=rms(eyz/a['scale']),
                    d_rms=rms(d),eta_scaled_rms=rms(eta),directional_scaled_rms=rms(d*eta),identity_error=err)
                rows.append(row)
    return rows


def analyze(out,cfg,m):
    points=np.stack([a.ravel() for a in np.meshgrid(cfg['time_points'],cfg['core_y'],cfg['core_z'],indexing='ij')],axis=1)
    rows=[]
    for seed in cfg['seeds']:
        dest=out/f'seed{seed}';data=dict(points=points,scale=m.utility_bundle(points[:,1])[1])
        for mesh in cfg['meshes']:
            marker=json.loads((dest/f'{mesh}_completion.json').read_text());path=dest/f'{mesh}_value.npz'
            assert sha256(path)==marker['sha256']
            data.update({mesh+'_'+k:v for k,v in reference(path,m,points).items()})
        for mode in cfg['modes']:
            for step in cfg['steps']:
                pred=predict(load_net(cfg,m,mode,seed,step),points)
                data.update({f'{mode}_{step}_'+k:v for k,v in pred.items()})
        for field in ['v','vy','vyy','vyz','q']:
            np.testing.assert_array_equal(data['single_0_'+field],data['full_0_'+field])
        data['fixed_q']=data['single_0_q']
        np.savez_compressed(dest/'diagnostic_arrays.npz',**data)
        rows+=diagnostics(data,cfg,m,seed)
        print('DIAGNOSTICS',seed,flush=True)
    write_csv(out/'summary.csv',rows)
    write_json(out/'completion.json',dict(status='complete',rows=len(rows),reference_solves=len(cfg['seeds'])*len(cfg['meshes'])))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=project_path,default=DEFAULT)
    p.add_argument('--phase',choices=['all','references','analysis'],default='all');args=p.parse_args()
    cfg,m=initialize(args.output)
    if args.phase in ['all','references']:solve(args.output,cfg,m)
    if args.phase in ['all','analysis']:analyze(args.output,cfg,m)


if __name__=='__main__':main()
