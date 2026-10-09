"""Retrospective finite-model attribution on saved frozen-policy evaluations.

No new neural training. Policies use the original experiment's 41x81x81
interpolation, then nearest-action discretization. All rows and all requested
seeds are retained. Generator differences avoid numerical differentiation of Uq.
"""
from pathlib import Path
import argparse, csv, hashlib, json, sys, time, zipfile
import numpy as np
import torch
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'))
from diagnose_npi_iteration import load_network
from npi.neural import predict
from npi.stencils import build
from npi.policy_error_audit import action_rows, compare_step
from npi.discrete_evaluation import certainty_equivalent
from scipy.sparse.linalg import splu


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def value_only(net, points, chunk=65536):
    ans=[]; device=next(net.parameters()).device
    with torch.no_grad():
        for start in range(0,len(points),chunk):
            x=torch.as_tensor(points[start:start+chunk],device=device,dtype=torch.float32)
            ans.append(net(x)[:,0].cpu().numpy().astype(float))
    return np.concatenate(ans)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--mesh',default='coarse',choices=['coarse','medium','fine','finest','time','action','alternate'])
    parser.add_argument('--seeds',type=int,nargs='+',default=[101,102,103])
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    meshes={'coarse':(41,31,120,81),'medium':(81,61,240,161),
        'fine':(121,91,360,241),'finest':(161,121,480,321),
        'time':(81,61,480,161),'action':(81,61,240,321),'alternate':(81,61,240,161)}
    ny,nz,nt,na=meshes[args.mesh];kind='semi_lagrangian' if args.mesh=='alternate' else 'bounded'
    protocol=dict(mesh=args.mesh,ny=ny,nz=nz,nt=nt,na=na,seeds=args.seeds,
        inner_steps=[50,1000],generator_kind=kind,policy_grid=[41,81,81],
        neural_dtype='float32',linear_algebra_dtype='float64',
        q_small_threshold=.05,core_y=.7,core_z=.3,
        interpretation='retrospective deterministic finite-model attribution, not a continuous error certificate',
        continuum_certified=False,floating_point_verified=False)
    write_json(out/'protocol.json',protocol)
    source_files=[Path(__file__),ROOT/'src/npi/policy_error_audit.py',ROOT/'src/npi/stencils.py',ROOT/'src/npi/neural.py']
    inputs=[]
    for seed in args.seeds:
        inputs.append(ROOT/f'results/npi_iteration_diagnostic_20260926/pi_seed{seed}/step_200.pt')
        inputs += [ROOT/f'results/fixed_policy_accuracy_20260926/seed{seed}/step_{j}.pt' for j in [50,1000]]
    write_json(out/'input_sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files+inputs})
    with zipfile.ZipFile(out/'sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in source_files:z.write(p,p.relative_to(ROOT))
    torch.set_num_threads(4);allrows=[];allstats=[];start=time.perf_counter()
    stack=None
    for seed in args.seeds:
        folder=out/f'seed{seed}';folder.mkdir()
        paths={0:ROOT/f'results/npi_iteration_diagnostic_20260926/pi_seed{seed}/step_200.pt'}
        paths.update({j:ROOT/f'results/fixed_policy_accuracy_20260926/seed{seed}/step_{j}.pt' for j in [50,1000]})
        networks={j:load_network(p) for j,p in paths.items()};m=networks[0].market
        axes=[np.linspace(0,m.horizon,41),np.linspace(m.ymin,m.ymax,81),np.linspace(m.zmin,m.zmax,81)]
        gp=np.stack([a.ravel() for a in np.meshgrid(*axes,indexing='ij')],axis=1)
        policies={}
        for j,net in networks.items():
            qgrid=predict(net,gp,chunk=8192)['q'].reshape(41,81,81)
            policies[j]=RegularGridInterpolator(axes,qgrid)
            np.savez_compressed(folder/f'policy_{j}.npz',tau=axes[0],y=axes[1],z=axes[2],q=qgrid)
        if stack is None:
            y=np.linspace(m.ymin,m.ymax,ny);z=np.linspace(m.zmin,m.zmax,nz)
            tau=np.linspace(0,m.horizon,nt+1);dt=tau[1];actions=np.linspace(m.qmin,m.qmax,na)
            stack,ids,_,diagnostics=build(m,y,z,actions,kind)
            n=len(ids);idx=np.arange(n);allids=np.arange(ny*nz);boundary=np.setdiff1d(allids,ids)
            ay=stack[:,ids].tocsr();ab=stack[:,boundary].tocsr();eye=sparse.eye(n,format='csc')
            yy,zz=np.meshgrid(y,z,indexing='ij');points=np.column_stack([np.zeros(n),yy.ravel()[ids],zz.ravel()[ids]])
            terminal=m.utility(yy).ravel();source=np.asarray(ab@terminal[boundary]).reshape(na,n)
            scale=m.utility_bundle(points[:,1])[1]
            fixed_masks={'full':np.ones(n,dtype=bool),'core':(np.abs(points[:,1])<=.7+1e-12)&(np.abs(points[:,2])<=.3+1e-12),
                'lower_wealth':points[:,1]<-1.5}
            center=np.where((np.abs(points[:,1])<1e-12)&(np.abs(points[:,2])<1e-12))[0].item()
            write_json(out/'stencil.json',diagnostics)
        uq=np.empty((nt+1,n));uq[0]=terminal[ids];qrows=[];qvalues=[];solve_error=0.
        for k in range(1,nt+1):
            points[:,0]=tau[k];q=policies[0](points);rows=action_rows(q,actions,n);qi=rows//n
            oq=ay[rows,:];sq=source[qi,idx];matrix=eye-dt*oq;rhs=uq[k-1]+dt*sq
            uq[k]=splu(matrix.tocsc()).solve(rhs)
            solve_error=max(solve_error,float(np.max(np.abs(matrix@uq[k]-rhs))))
            qrows.append(rows);qvalues.append(q)
        np.savez_compressed(folder/'fixed_value.npz',tau=tau,y=y,z=z,interior=ids,value=uq)
        print(f'{args.mesh} seed {seed}: frozen policy solved, residual {solve_error:.2e}',flush=True)
        ceq=float(certainty_equivalent(m,uq[-1,center]))
        for j in [50,1000]:
            up=terminal[ids].copy();adv=np.zeros(n);correction=np.zeros(n)
            # Attribute the same correction to disjoint low/high exposure and wealth regions.
            parts=np.zeros((n,3));max_identity=0.;max_global=0.;max_linear=solve_error
            aggregates={name:np.zeros(7) for name in ['full','core','lower_wealth','small_q','other_q']}
            layer_rows=[]
            for k in range(1,nt+1):
                points[:,0]=tau[k];p=policies[j](points);rows=action_rows(p,actions,n);pi=rows//n
                qr=qrows[k-1];qi=qr//n;oq=ay[qr,:];op=ay[rows,:];sq=source[qi,idx];sp=source[pi,idx]
                v=value_only(networks[j],points)
                up,adv,correction,fields=compare_step(oq,op,sq,sp,uq[k],v,up,adv,correction,dt)
                error=fields['directional_error'];av=fields['candidate_advantage'];au=fields['exact_advantage']
                small=np.abs(qvalues[k-1])<.05
                partition=np.column_stack([small,~small & (points[:,1]<-1.5),~small & (points[:,1]>=-1.5)])
                parts=splu((eye-dt*op).tocsc()).solve(parts+dt*error[:,None]*partition)
                max_identity=max(max_identity,fields['identity_error']);max_linear=max(max_linear,fields['linear_residual'])
                max_global=max(max_global,float(np.max(np.abs(up-uq[k]-(adv-correction)))))
                masks={**fixed_masks,'small_q':small,'other_q':~small}
                for name,mask in masks.items():
                    a=aggregates[name];a[0]+=mask.sum()
                    a[1]+=np.sum((error[mask]/scale[mask])**2)
                    a[2]+=np.sum((av[mask]/scale[mask])**2)
                    a[3]+=np.sum((au[mask]/scale[mask])**2)
                    a[4]+=np.count_nonzero(au[mask]<0)
                    a[5]+=np.count_nonzero((av[mask]>0)&(au[mask]<0))
                    a[6]+=np.sum(((v[mask]-uq[k,mask])/scale[mask])**2)
                if k==nt or k%max(1,nt//6)==0:
                    layer_rows.append(dict(k=k,tau=float(tau[k]),candidate_integral=float(adv[center]),
                        error_integral=float(correction[center]),value_difference=float(up[center]-uq[k,center])))
            cep=float(certainty_equivalent(m,up[center]));gain=1e4*(cep/ceq-1)
            row=dict(seed=seed,inner_steps=j,mesh=args.mesh,ny=ny,nz=nz,nt=nt,na=na,
                frozen_value=float(uq[-1,center]),proposal_value=float(up[center]),ce_gain_bp=gain,
                candidate_integral=float(adv[center]),error_integral=float(correction[center]),
                value_difference=float(up[center]-uq[-1,center]),small_q_error_integral=float(parts[center,0]),
                lower_other_q_error_integral=float(parts[center,1]),remaining_error_integral=float(parts[center,2]),
                maximum_local_identity_error=max_identity,maximum_propagated_identity_error=max_global,
                maximum_linear_solve_residual=max_linear,
                partition_identity_error=float(np.max(np.abs(parts.sum(axis=1)-correction))))
            allrows.append(row)
            for name,a in aggregates.items():
                allstats.append(dict(seed=seed,inner_steps=j,mesh=args.mesh,region=name,count=int(a[0]),
                    directional_error_rms_scaled=float(np.sqrt(a[1]/a[0])),candidate_advantage_rms_scaled=float(np.sqrt(a[2]/a[0])),
                    exact_advantage_rms_scaled=float(np.sqrt(a[3]/a[0])),negative_exact_advantage_fraction=float(a[4]/a[0]),
                    sign_reversal_fraction=float(a[5]/a[0]),value_error_rms_scaled=float(np.sqrt(a[6]/a[0]))))
            write_json(folder/f'layers_{j}.json',layer_rows)
            np.savez_compressed(folder/f'final_{j}.npz',value=up,candidate_integral=adv,error_integral=correction,
                small_q_error=parts[:,0],lower_other_q_error=parts[:,1],remaining_error=parts[:,2])
            print(f'{args.mesh} seed {seed} step {j}: CE={gain:.4f} bp, A={adv[center]:.6g}, E={correction[center]:.6g}, identity={max_global:.2e}',flush=True)
        for filename,rows in [('summary.csv',allrows),('regions.csv',allstats)]:
            with (out/filename).open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(out/'completion.json',dict(status='complete',seconds=time.perf_counter()-start,cases=len(allrows)))


if __name__=='__main__':main()
