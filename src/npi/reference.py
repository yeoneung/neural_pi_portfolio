"""Positive-coefficient wide-stencil reference solver.

Backward Euler in time-to-horizon and Howard iteration on a finite action
grid. Off-diagonal generator weights are nonnegative. Mixed diffusion is
represented by a positive diagonal stencil plus remaining axis diffusion.
Near boundaries, stencil endpoints are clamped to the prescribed boundary.
This boundary closure is explicit and must be tested by domain enlargement.
The solved discrete Bellman residual is recorded; that does not certify a
continuum error. Grid, action, time and domain audits remain necessary.
"""
from dataclasses import dataclass
import json
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator
from .model import Model


@dataclass
class Reference:
    model: Model
    tau: np.ndarray
    y: np.ndarray
    z: np.ndarray
    value: np.ndarray
    policy: np.ndarray
    diagnostics: dict

    def save(self, path):
        np.savez_compressed(path, tau=self.tau, y=self.y, z=self.z, value=self.value,
                            policy=self.policy, model=json.dumps(self.model.to_dict()),
                            diagnostics=json.dumps(self.diagnostics))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(Model.from_dict(json.loads(str(data['model']))), data['tau'], data['y'], data['z'],
                       data['value'], data['policy'], json.loads(str(data['diagnostics'])))

    def interpolate(self, points, field="value"):
        return RegularGridInterpolator((self.tau, self.y, self.z), getattr(self, field), bounds_error=True)(points)

    def derivative_policy(self):
        vy = np.gradient(self.value, self.y, axis=1, edge_order=2)
        vyy = np.gradient(vy, self.y, axis=1, edge_order=2)
        vyz = np.gradient(vy, self.z, axis=2, edge_order=2)
        return self.model.greedy(vy, vyy, vyz, self.z[None,None,:])


def build_generators(model, y, z, actions):
    """Rows map full-grid values to the generator at interior points."""
    ny, nz = len(y), len(z)
    hy, hz = y[1]-y[0], z[1]-z[0]
    ii, jj = np.meshgrid(np.arange(1,ny-1), np.arange(1,nz-1), indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    centers = ii*nz+jj
    n = len(centers)
    matrices, widest = [], 1
    for q in actions:
        rows, cols, vals = [], [], []
        diagonal = np.zeros(n)
        def add(di, dj, rate):
            rate = np.broadcast_to(np.asarray(rate,dtype=float),(n,))
            if np.min(rate) < -1e-12:
                raise ArithmeticError("Non-monotone stencil")
            rate = np.maximum(rate,0)
            target = np.clip(ii+di,0,ny-1)*nz+np.clip(jj+dj,0,nz-1)
            rows.extend(np.arange(n)); cols.extend(target); vals.extend(rate)
            diagonal[:] -= rate
        cross = model.rho*q*model.sigma_z
        rem_y, rem_z = q*q, model.sigma_z**2
        if abs(cross)>1e-15:
            # Pick an integer direction within the positivity cone. Search a
            # finite number of rational slopes rather than silently drop rho.
            lower, upper = abs(cross)/model.sigma_z**2, q*q/abs(cross)
            candidates=[]
            for iz in range(1,257):
                iy=max(1,int(np.ceil(lower*iz*hz/hy-1e-12)))
                if iy*hy/(iz*hz) <= upper*(1+1e-12):
                    candidates.append((max(iy,iz),iy+iz,iy,iz))
            if not candidates:
                raise ArithmeticError("No positive mixed-derivative stencil found")
            _,_,iy,iz=min(candidates)
            widest=max(widest,iy,iz)
            slope=iy*hy/(iz*hz)
            rem_y-=abs(cross)*slope
            rem_z-=abs(cross)/slope
            rate=abs(cross)/(2*iy*iz*hy*hz)
            sign=1 if cross>0 else -1
            add(iy,sign*iz,rate); add(-iy,-sign*iz,rate)
        add(1,0,rem_y/(2*hy*hy)); add(-1,0,rem_y/(2*hy*hy))
        add(0,1,rem_z/(2*hz*hz)); add(0,-1,rem_z/(2*hz*hz))
        by=model.rate+q*model.phi(z[jj])-0.5*q*q
        bz=model.kappa*(model.zbar-z[jj])
        add(1,0,np.maximum(by,0)/hy); add(-1,0,np.maximum(-by,0)/hy)
        add(0,1,np.maximum(bz,0)/hz); add(0,-1,np.maximum(-bz,0)/hz)
        rows.extend(np.arange(n)); cols.extend(centers); vals.extend(diagonal)
        mat=sparse.csr_matrix((vals,(rows,cols)),shape=(n,ny*nz))
        mat.eliminate_zeros()
        matrices.append(mat)
    return sparse.vstack(matrices,format='csr'),centers,widest


def solve(model, ny=61, nz=31, nt=100, na=61, tolerance=1e-9,
          max_policy_iterations=80, fixed_q=None, exact_fixed_boundary=False,
          generator_kind='legacy'):
    start=time.perf_counter()
    if min(ny,nz)<5 or nt<1 or na<2:
        raise ValueError("Invalid reference grid")
    if exact_fixed_boundary and fixed_q is None:
        raise ValueError("Exact fixed-policy boundaries require fixed_q")
    y=np.linspace(model.ymin,model.ymax,ny); z=np.linspace(model.zmin,model.zmax,nz)
    tau=np.linspace(0,model.horizon,nt+1); dt=tau[1]
    actions=np.linspace(model.qmin,model.qmax,na) if fixed_q is None else np.array([fixed_q])
    extra={'generator_kind':'legacy'}
    if generator_kind=='legacy':
        stack,interior,widest=build_generators(model,y,z,actions)
    else:
        from .stencils import build
        stack,interior,widest,extra=build(model,y,z,actions,generator_kind)
    n=len(interior); idx=np.arange(n)
    all_ids=np.arange(ny*nz); boundary=np.setdiff1d(all_ids,interior)
    ay=stack[:,interior].tocsr(); ab=stack[:,boundary].tocsr()
    eye=sparse.eye(n,format='csr')
    yy,zz=np.meshgrid(y,z,indexing='ij')
    initial=model.utility(yy)
    values=np.empty((nt+1,ny,nz)); policies=np.empty_like(values)
    values[0]=initial
    q_index=np.argmax((stack@initial.ravel()).reshape(len(actions),n),axis=0)
    policies[0]=model.greedy(*model.utility_bundle(yy)[1:],np.zeros_like(yy),zz)
    history=[]
    for k in range(1,nt+1):
        bfull=(model.constant_policy_value(tau[k],yy,fixed_q)
               if exact_fixed_boundary else initial).ravel()
        sources=np.asarray(ab@bfull[boundary]).reshape(len(actions),n)
        prev=values[k-1].ravel()[interior]
        u=prev.copy()
        for iteration in range(1,max_policy_iterations+1):
            rows=q_index*n+idx
            operator=ay[rows,:]
            u=spsolve(eye-dt*operator,prev+dt*sources[q_index,idx])
            hamiltonians=np.asarray(ay@u).reshape(len(actions),n)+sources
            best=np.argmax(hamiltonians,axis=0)
            bellman=(u-prev)/dt-np.max(hamiltonians,axis=0)
            residual=float(np.max(np.abs(bellman)))
            if residual <= tolerance*(1+float(np.max(np.abs(u)))):
                q_index=best
                break
            q_index=best
        else:
            raise RuntimeError(f"Howard iteration failed at step {k}: residual={residual}")
        full=bfull.copy(); full[interior]=u; values[k]=full.reshape(ny,nz)
        qfull=np.zeros(ny*nz); qfull[interior]=actions[q_index]
        qfull=qfull.reshape(ny,nz)
        qfull[0,:]=qfull[1,:];qfull[-1,:]=qfull[-2,:]
        qfull[:,0]=qfull[:,1];qfull[:,-1]=qfull[:,-2]
        policies[k]=qfull
        history.append({'step':k,'policy_iterations':iteration,'bellman_residual':residual})
    info={'scheme':'positive wide stencil / backward Euler / discrete-action Howard',
          'boundary':'exact fixed-policy verification' if exact_fixed_boundary else 'stopped liquidation utility',
          'ny':ny,'nz':nz,'nt':nt,'na':len(actions),'max_stencil_offset':widest,
          'wall_seconds':time.perf_counter()-start,
          'max_discrete_bellman_residual':max(h['bellman_residual'] for h in history),
          'mean_policy_iterations':float(np.mean([h['policy_iterations'] for h in history])),
          'fixed_q':fixed_q,'history':history,'continuum_certified':False,**extra}
    return Reference(model,tau,y,z,values,policies,info)
