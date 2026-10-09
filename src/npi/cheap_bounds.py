"""Scalar supersolution repairs for a finite positive-generator Bellman system.

No linear or Bellman systems are solved. Cost includes one full Hamiltonian
scan per candidate time layer. Exact-arithmetic inequalities are checked
numerically; no continuum or interval-arithmetic guarantee is asserted.
"""
from dataclasses import dataclass
import time
import numpy as np
from .stencils import build


@dataclass
class FiniteSystem:
    model: object
    y: np.ndarray
    z: np.ndarray
    tau: np.ndarray
    actions: np.ndarray
    ay: object
    sources: np.ndarray
    ids: np.ndarray
    boundary: np.ndarray
    terminal: np.ndarray
    setup_seconds: float

    @classmethod
    def create(cls,model,ny=81,nz=61,nt=240,na=81):
        started=time.perf_counter()
        y=np.linspace(model.ymin,model.ymax,ny);z=np.linspace(model.zmin,model.zmax,nz)
        tau=np.linspace(0,model.horizon,nt+1);actions=np.linspace(model.qmin,model.qmax,na)
        full,ids,_,_=build(model,y,z,actions,'bounded')
        boundary=np.setdiff1d(np.arange(ny*nz),ids)
        terminal=np.repeat(model.utility(y)[:,None],nz,axis=1)
        ay=full[:,ids].tocsr()
        sources=np.asarray(full[:,boundary]@terminal.ravel()[boundary]).reshape(na,len(ids))
        return cls(model,y,z,tau,actions,ay,sources,ids,boundary,terminal,time.perf_counter()-started)

    def hamiltonians(self,value):
        return np.asarray(self.ay@value).reshape(len(self.actions),len(self.ids))+self.sources


def scalar_bounds(system,candidate,controls=None,exponents=(0.,1.,2.,3.,4.)):
    """Two one-sided bounds with fixed positive weights exp(-p*y).

    candidate contains ONLY interior values with exact terminal data supplied
    here. The physical boundary payoff is in system.sources. If controls is
    None, deploy the discrete greedy selector of this candidate.
    """
    started=time.perf_counter();s=system;na=len(s.actions);n=len(s.ids);idx=np.arange(n)
    ys=np.repeat(s.y[:,None],len(s.z),axis=1).ravel()[s.ids]
    weights=np.exp(-ys[:,None]*np.asarray(exponents)[None,:])
    aw=np.asarray(s.ay@weights).reshape(na,n,len(exponents))
    upper=np.zeros(len(exponents));lower=np.zeros_like(upper)
    previous=s.terminal.ravel()[s.ids];global_upper=global_lower=0.;worst_slack=0.
    upper_history=[];lower_history=[];selected=[]
    for k,dt in enumerate(np.diff(s.tau),start=1):
        v=candidate[k];h=s.hamiltonians(v);maximum=h.max(axis=0)
        policy=h.argmax(axis=0) if controls is None else np.argmin(abs(controls[k,:,None]-s.actions),axis=1)
        r=(v-previous)/dt-maximum;d=maximum-h[policy,idx]
        deficit=maximum[None,:]-h
        denom=weights[None,:,:]-dt*aw
        if np.min(denom)<=0:raise ValueError('Weight denominator is nonpositive; refine time or change weights')
        up_forcing=np.maximum(-r,0.)
        up_rhs=weights[None,:,:]*upper[None,None,:]+dt*(up_forcing[None,:,None]-deficit[:,:,None])
        next_upper=np.maximum(0.,np.max(up_rhs/denom,axis=(0,1)))
        lo_forcing=np.maximum(r+d,0.)
        lo_denom=denom[policy,idx,:]
        lo_rhs=weights*lower[None,:]+dt*lo_forcing[:,None]
        next_lower=np.maximum(0.,np.max(lo_rhs/lo_denom,axis=0))
        worst_slack=min(worst_slack,float(np.min(denom*next_upper[None,None,:]-up_rhs)),
                        float(np.min(lo_denom*next_lower[None,:]-lo_rhs)))
        upper=next_upper;lower=next_lower;previous=v
        global_upper+=dt*float(np.max(up_forcing));global_lower+=dt*float(np.max(lo_forcing))
        upper_history.append(upper.copy());lower_history.append(lower.copy());selected.append(policy)
    # Every weight yields a valid bound; pointwise minimization preserves it.
    up=np.min(weights*upper[None,:],axis=1);lo=np.min(weights*lower[None,:],axis=1)
    return {'upper_correction':up,'lower_correction':lo,'policy_loss_bound':up+lo,
            'upper_coefficients':np.asarray(upper_history),'lower_coefficients':np.asarray(lower_history),
            'policy_indices':np.asarray(selected),'exponents':list(exponents),
            'global_one_sided_bound':global_upper+global_lower,'minimum_repair_slack':worst_slack,
            'certificate_seconds':time.perf_counter()-started,'linear_solves':0,'bellman_solves':0}
