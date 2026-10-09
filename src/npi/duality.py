"""Portfolio dual upper bounds using bounded state-price processes.

The auxiliary market extends the original coefficients beyond the stopping
domain and allows a stopped strategy to continue in cash. This gives an upper
bound for the original stopped problem when r>=0. Monte Carlo discretization
and sampling errors remain separate from the continuous-time theorem.
"""
import time
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates


class UniformFeedback:
    """The same trilinear interpolation, evaluated by SciPy's compiled kernel."""
    def __init__(self,axes,fields):
        self.lower=np.array([axis[0] for axis in axes])
        self.step=np.array([axis[1]-axis[0] for axis in axes])
        self.upper=np.array([axis[-1] for axis in axes])
        for axis in axes:
            if not np.allclose(np.diff(axis),axis[1]-axis[0]):raise ValueError('Uniform grid required')
        self.fields=[np.ascontiguousarray(fields[...,k],dtype=float) for k in range(fields.shape[-1])]

    def __call__(self,points):
        if np.any(points<self.lower-1e-10) or np.any(points>self.upper+1e-10):raise ValueError('Outside feedback grid')
        coordinates=((points-self.lower)/self.step).T
        return np.column_stack([map_coordinates(field,coordinates,order=1,mode='nearest',prefilter=False) for field in self.fields])


def inverse_marginal(model,x):
    x=np.asarray(x,dtype=float)
    if np.any(x<=0):raise ValueError('Marginal utility argument must be positive')
    logx=np.log(x);logw=-logx/model.gamma1
    for _ in range(30):
        a=-model.gamma1*logw
        b=np.log(model.mixture)-model.gamma2*logw if model.mixture>0 else np.full_like(a,-np.inf)
        denom=np.logaddexp(a,b)
        elasticity=model.gamma1*np.exp(a-denom)+model.gamma2*np.exp(b-denom)
        step=(denom-logx)/elasticity
        logw+=np.clip(step,-2.,2.)
        if np.max(np.abs(step))<2e-13:break
    else:raise RuntimeError('Inverse marginal did not converge')
    return np.exp(logw)


def conjugate(model,x):
    wealth=inverse_marginal(model,x)
    return model.utility(np.log(wealth))-x*wealth


def simulate_dual(model,feedback,npaths=8192,nsteps=480,fine_steps=960,seed=81001,initial_y=0.,initial_z=0.):
    if fine_steps%nsteps:raise ValueError('Time grids must be nested')
    if model.rate<0:raise ValueError('Cash-extension domination requires nonnegative rate')
    started=time.perf_counter();dt=model.horizon/nsteps;ratio=fine_steps//nsteps
    rng=np.random.default_rng(seed);y=np.full(npaths,initial_y);z=np.full(npaths,initial_z);logm=np.zeros(npaths)
    payoff=np.empty(npaths);alive=np.ones(npaths,dtype=bool)
    for k in range(nsteps):
        dw=rng.standard_normal((ratio,npaths,2)).sum(axis=0)*np.sqrt(model.horizon/fine_steps)
        points=np.column_stack([np.full(npaths,model.horizon-k*dt),np.clip(y,model.ymin,model.ymax),np.clip(z,model.zmin,model.zmax)])
        data=feedback(points);q=np.clip(data[:,0],model.qmin,model.qmax);nu=np.clip(data[:,1],-2.,2.)
        phi=model.phi(np.clip(z,model.zmin,model.zmax))
        yn=y+(model.rate+q*phi-.5*q*q)*dt+q*dw[:,0]
        zn=z+model.kappa*(model.zbar-z)*dt+model.sigma_z*(model.rho*dw[:,0]+np.sqrt(1-model.rho**2)*dw[:,1])
        logm+=(-model.rate-.5*(phi*phi+nu*nu))*dt-phi*dw[:,0]-nu*dw[:,1]
        fraction=np.ones(npaths)
        for old,new,lo,hi in [(y,yn,model.ymin,model.ymax),(z,zn,model.zmin,model.zmax)]:
            for hit,bound in [(new<lo,lo),(new>hi,hi)]:
                eligible=alive&hit
                local=np.ones(npaths);local[eligible]=(bound-old[eligible])/(new[eligible]-old[eligible])
                fraction=np.minimum(fraction,local)
        hit=alive&(fraction<1)
        payoff[hit]=model.utility(np.clip(y[hit]+fraction[hit]*(yn[hit]-y[hit]),model.ymin,model.ymax))
        alive[hit]=False;y,z=yn,zn
    payoff[alive]=model.utility(y[alive])
    wealth=np.exp(y);density=np.exp(logm)
    return {'payoff':payoff,'terminal_wealth_auxiliary':wealth,'density':density,
            'density_wealth':density*wealth,'exit_fraction':float(np.mean(~alive)),
            'simulation_seconds':time.perf_counter()-started,'initial_wealth':float(np.exp(initial_y))}


def calibrate_multiplier(model,paths):
    density=paths['density'];initial=paths['initial_wealth']
    equation=lambda loglam:float(np.mean(density*inverse_marginal(model,np.exp(loglam)*density))-initial)
    return float(np.exp(brentq(equation,-20,20,xtol=1e-12)))


def simulate_factor_dual(model,feedback,npaths=8192,nsteps=480,fine_steps=960,seed=81001,initial_y=0.,initial_z=0.):
    """Stop the auxiliary deflator at the factor exit shared by all controls.

    Monitoring uses grid endpoints, including wealth overshoot payoffs. This
    is an explicit time-discrete stopped market with an exact conditional
    product-martingale identity, not a Brownian-bridge continuous-exit solver.
    """
    if fine_steps%nsteps:raise ValueError('Time grids must be nested')
    if model.rate<0:raise ValueError('Cash extension requires r>=0')
    started=time.perf_counter();dt=model.horizon/nsteps;ratio=fine_steps//nsteps
    rng=np.random.default_rng(seed);y=np.full(npaths,float(initial_y));z=np.full(npaths,float(initial_z));logm=np.zeros(npaths)
    aux_alive=np.ones(npaths,dtype=bool);original_alive=np.ones(npaths,dtype=bool)
    payoff=np.empty(npaths);wealth_exited=np.zeros(npaths,dtype=bool)
    for k in range(nsteps):
        dw=rng.standard_normal((ratio,npaths,2)).sum(axis=0)*np.sqrt(model.horizon/fine_steps)
        ix=np.flatnonzero(aux_alive)
        if not len(ix):break
        points=np.column_stack([np.full(len(ix),model.horizon-k*dt),np.clip(y[ix],model.ymin,model.ymax),np.clip(z[ix],model.zmin,model.zmax)])
        data=feedback(points);q=np.clip(data[:,0],model.qmin,model.qmax);nu=np.clip(data[:,1],-2.,2.)
        phi=model.phi(z[ix])
        yn=y[ix]+(model.rate+q*phi-.5*q*q)*dt+q*dw[ix,0]
        zn=z[ix]+model.kappa*(model.zbar-z[ix])*dt+model.sigma_z*(model.rho*dw[ix,0]+np.sqrt(1-model.rho**2)*dw[ix,1])
        logm[ix]+=(-model.rate-.5*(phi*phi+nu*nu))*dt-phi*dw[ix,0]-nu*dw[ix,1]
        factor_hit=(zn<=model.zmin)|(zn>=model.zmax)
        wealth_hit=(yn<=model.ymin)|(yn>=model.ymax)
        hit=original_alive[ix]&(factor_hit|wealth_hit)
        payoff[ix[hit]]=model.utility(yn[hit]);wealth_exited[ix[original_alive[ix]&wealth_hit]]=True
        original_alive[ix[hit]]=False;aux_alive[ix[factor_hit]]=False
        y[ix]=yn;z[ix]=zn
    payoff[original_alive]=model.utility(y[original_alive])
    wealth=np.exp(y);density=np.exp(logm)
    return {'payoff':payoff,'terminal_wealth_auxiliary':wealth,'density':density,
            'density_wealth':density*wealth,'exit_fraction':float(np.mean(~original_alive)),
            'factor_exit_fraction':float(np.mean(~aux_alive)),'wealth_exit_fraction':float(np.mean(wealth_exited)),
            'simulation_seconds':time.perf_counter()-started,'initial_wealth':float(np.exp(initial_y)),
            'monitoring':'grid endpoints; overshoot wealth is paid; auxiliary stops at factor exit'}


def dual_statistics(model,paths,multiplier):
    x=multiplier*paths['density'];payoff=paths['payoff']
    # E[M_T W_T]=W_0 for bounded adapted exposures in the auxiliary market.
    # This control variate avoids the large variance of lambda*w0+g*(lambda*M).
    upper_sample=conjugate(model,x)+multiplier*paths['density_wealth']
    gap=upper_sample-payoff
    se=float(np.std(gap,ddof=1)/np.sqrt(len(gap)))
    mart=paths['density_wealth'];mse=float(np.std(mart,ddof=1)/np.sqrt(len(mart)))
    return {'multiplier':multiplier,'mean_policy_utility':float(np.mean(payoff)),
            'mean_dual_upper':float(np.mean(upper_sample)),'mean_gap':float(np.mean(gap)),
            'gap_mc_se':se,'gap_one_sided_95_upper':float(np.mean(gap)+1.6448536269514722*se),
            'gap_two_sided_95_low':float(np.mean(gap)-1.959963984540054*se),
            'gap_two_sided_95_high':float(np.mean(gap)+1.959963984540054*se),
            'martingale_mean':float(np.mean(mart)),'martingale_mc_se':mse,
            'exit_fraction':paths['exit_fraction'],'simulation_seconds':paths['simulation_seconds'],
            'factor_exit_fraction':paths.get('factor_exit_fraction'),
            'wealth_exit_fraction':paths.get('wealth_exit_fraction'),
            'monitoring':paths.get('monitoring','linear Euler crossing for original payoff; auxiliary terminal horizon'),
            'interval_scope':'pointwise asymptotic Monte Carlo interval; time/interpolation bias excluded',
            'gap_samples':gap,'upper_samples':upper_sample}
