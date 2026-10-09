"""Time only the published deficit correction and independent policy solve.

The older diagnostic also computes two absolute envelopes; including those
unneeded solves would unfairly inflate the cost of the deficit correction.
"""
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def evaluate(system,candidate,controls,tolerance=1e-10):
    s=system;started=time.perf_counter();n=len(s.ids);ix=np.arange(n)
    eye=sparse.eye(n,format='csr');j=s.terminal.ravel()[s.ids].copy();correction=np.zeros(n)
    history=[];linear_solves=0
    for k,dt in enumerate(np.diff(s.tau),start=1):
        v=candidate[k];previous=candidate[k-1] if k>1 else s.terminal.ravel()[s.ids]
        h=s.hamiltonians(v);maxh=h.max(axis=0);r=(v-previous)/dt-maxh
        pi=np.argmin(abs(controls[k,:,None]-s.actions[None,:]),axis=1)
        j=spsolve(eye-dt*s.ay[pi*n+ix,:],j+dt*s.sources[pi,ix]);linear_solves+=1
        old=correction.copy();deficits=maxh[None,:]-h;forcing=np.maximum(-r,0);best=h.argmax(axis=0)
        for iteration in range(1,81):
            correction=spsolve(eye-dt*s.ay[best*n+ix,:],old+dt*(forcing-deficits[best,ix]));linear_solves+=1
            hs=np.asarray(s.ay@correction).reshape(len(s.actions),n)-deficits
            residual=(correction-old)/dt-hs.max(axis=0)-forcing
            best=hs.argmax(axis=0)
            if np.max(abs(residual))<=tolerance*(1+np.max(abs(correction))):break
        else:raise RuntimeError('Deficit correction failed to converge')
        history.append({'step':k,'iterations':iteration,'residual':float(np.max(abs(residual)))})
    return {'correction':correction,'policy_value':j,'policy_loss_bound':candidate[-1]+correction-j,
            'linear_solves':linear_solves,'history':history,'solve_seconds':time.perf_counter()-started}
