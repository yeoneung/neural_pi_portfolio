"""Independent Euler simulation of the specified stopped diffusion.

Every policy receives the same seeded Brownian increments. Nested time grids
sum fine increments, making time-bias comparisons paired as well. A linear
crossing approximation pays utility at the first crossed boundary. This does
not resolve Brownian bridge exits, so time-step auditing is mandatory.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .discrete_evaluation import certainty_equivalent


@dataclass
class PolicyGrid:
    model: object
    tau: np.ndarray
    y: np.ndarray
    z: np.ndarray
    q: np.ndarray
    residual: np.ndarray|None=None

    def interpolator(self):
        fields=self.q if self.residual is None else np.stack([self.q,self.residual],axis=-1)
        interp=RegularGridInterpolator((self.tau,self.y,self.z),fields,bounds_error=True)
        return interp


def simulate(model,policy,npaths=32768,nsteps=240,seed=70001,fine_steps=480,
             initial_y=0.,initial_z=0.,myopic=False):
    if fine_steps%nsteps:raise ValueError('Use nested step counts dividing fine_steps')
    rng=np.random.default_rng(seed);ratio=fine_steps//nsteps;dt=model.horizon/nsteps
    y=np.full(npaths,float(initial_y));z=np.full(npaths,float(initial_z))
    alive=np.ones(npaths,dtype=bool);payoff=np.empty(npaths);integral=np.zeros(npaths);exit_time=np.full(npaths,model.horizon)
    for k in range(nsteps):
        # Drawing for all paths, including stopped ones, preserves coupling.
        dw=rng.standard_normal((ratio,npaths,2)).sum(axis=0)*np.sqrt(model.horizon/fine_steps)
        ix=np.flatnonzero(alive)
        if not len(ix):break
        tau=model.horizon-k*dt
        points=np.column_stack([np.full(len(ix),tau),y[ix],z[ix]])
        if myopic:
            q=np.clip(model.phi(z[ix])/model.relative_risk_aversion(y[ix]),model.qmin,model.qmax)
            resid=np.zeros(len(ix))
        else:
            data=policy(points)
            q=data if data.ndim==1 else data[:,0]
            resid=np.zeros(len(ix)) if data.ndim==1 else data[:,1]
            q=np.clip(q,model.qmin,model.qmax)
        yn=y[ix]+(model.rate+q*model.phi(z[ix])-.5*q*q)*dt+q*dw[ix,0]
        zn=z[ix]+model.kappa*(model.zbar-z[ix])*dt+model.sigma_z*(model.rho*dw[ix,0]+np.sqrt(1-model.rho**2)*dw[ix,1])
        fraction=np.ones(len(ix))
        for old,new,lo,hi in [(y[ix],yn,model.ymin,model.ymax),(z[ix],zn,model.zmin,model.zmax)]:
            for hit,bound in [(new<lo,lo),(new>hi,hi)]:
                local=np.ones(len(ix));local[hit]=(bound-old[hit])/(new[hit]-old[hit])
                fraction=np.minimum(fraction,local)
        crossed=fraction<1
        yhit=np.clip(y[ix]+fraction*(yn-y[ix]),model.ymin,model.ymax)
        integral[ix]+=dt*fraction*resid
        if np.any(crossed):
            ids=ix[crossed];payoff[ids]=model.utility(yhit[crossed]);alive[ids]=False
            exit_time[ids]=(k+fraction[crossed])*dt
        keep=ix[~crossed];y[keep]=yn[~crossed];z[keep]=zn[~crossed]
    payoff[alive]=model.utility(y[alive])
    return {'payoff':payoff,'residual_integral':integral,'exit_time':exit_time,
            'exit_fraction':float(np.mean(exit_time<model.horizon))}


def paired_ce(model,payoff,reference):
    j=float(np.mean(payoff));jr=float(np.mean(reference))
    ce=float(certainty_equivalent(model,j));cer=float(certainty_equivalent(model,jr))
    # Delta-method paired uncertainty for 1 - g^{-1}(J)/g^{-1}(J_ref).
    gp=lambda w:w**(-model.gamma1)+model.mixture*w**(-model.gamma2)
    influence=-(payoff-j)/(gp(ce)*cer)+ce*(reference-jr)/(cer**2*gp(cer))
    se=float(np.std(influence,ddof=1)/np.sqrt(len(payoff)))
    loss=1-ce/cer
    return {'mean_utility':j,'CE':ce,'reference_CE':cer,'relative_CE_shortfall_vs_reference':loss,
            'paired_mc_se':se,'paired_mc_95_low':loss-1.96*se,'paired_mc_95_high':loss+1.96*se,
            'interpretation':'benchmark-relative shortfall, not a certified optimality gap'}
