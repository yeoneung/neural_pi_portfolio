"""Componentwise finite-grid residual bounds by monotone propagation.

These auxiliary solves cost comparable work to a low-dimensional reference;
they are verification diagnostics, not an accelerated high-dimensional solver.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .reference import build_generators
from .stencils import build


def propagate(reference,controls,candidate,tolerance=1e-10):
    m=reference.model;actions=np.linspace(m.qmin,m.qmax,reference.diagnostics['na'])
    kind=reference.diagnostics.get('generator_kind','legacy')
    if kind=='legacy':stack,ids,_=build_generators(m,reference.y,reference.z,actions)
    else:stack,ids,_,_=build(m,reference.y,reference.z,actions,kind)
    n=len(ids);idx=np.arange(n);boundary=np.setdiff1d(np.arange(len(reference.y)*len(reference.z)),ids)
    ay=stack[:,ids].tocsr();ab=stack[:,boundary].tocsr();eye=sparse.eye(n,format='csr')
    b=m.utility(reference.y)[:,None]*np.ones((1,len(reference.z)))
    source=np.asarray(ab@b.ravel()[boundary]).reshape(len(actions),n)
    e=np.zeros(n);f=np.zeros(n);upper=np.zeros(n);j=b.ravel()[ids].copy();best=np.zeros(n,dtype=int)
    global_value=0.;global_policy=0.;history=[]
    for k,dt in enumerate(np.diff(reference.tau),start=1):
        u=candidate[k].ravel()[ids]
        prev=candidate[k-1].ravel()[ids] if k>1 else b.ravel()[ids]
        h=np.asarray(ay@u).reshape(len(actions),n)+source
        r=(u-prev)/dt-np.max(h,axis=0)
        qi=np.argmin(np.abs(controls[k].ravel()[ids,None]-actions[None,:]),axis=1)
        deficit=np.maximum(0,np.max(h,axis=0)-h[qi,idx]);rows=qi*n+idx
        selected_matrix=eye-dt*ay[rows,:]
        f_rhs=f+dt*(np.abs(r)+deficit)
        j_rhs=j+dt*source[qi,idx]
        f=spsolve(selected_matrix,f_rhs)
        j=spsolve(selected_matrix,j_rhs)
        linear_residual=max(float(np.max(np.abs(selected_matrix@f-f_rhs))),
                            float(np.max(np.abs(selected_matrix@j-j_rhs))))
        previous_e=e.copy()
        for iteration in range(1,81):
            e=spsolve(eye-dt*ay[best*n+idx,:],previous_e+dt*np.abs(r))
            eh=np.asarray(ay@e).reshape(len(actions),n);best=np.argmax(eh,axis=0)
            residual=(e-previous_e)/dt-np.max(eh,axis=0)-np.abs(r)
            if np.max(np.abs(residual))<=tolerance*(1+np.max(np.abs(e))):break
        else:raise RuntimeError('Envelope Howard iteration did not converge')
        # Preserve the candidate's action-wise Hamiltonian deficits. This
        # penalizes adversarial actions that are poor for the candidate value.
        deficits=np.max(h,axis=0)[None,:]-h
        previous_upper=upper.copy();upper_best=np.argmax(h,axis=0)
        upper_source=np.maximum(-r,0.)
        for upper_iteration in range(1,81):
            selected=upper_best*n+idx
            upper=spsolve(eye-dt*ay[selected,:],previous_upper+dt*(upper_source-deficits[upper_best,idx]))
            upper_h=np.asarray(ay@upper).reshape(len(actions),n)-deficits
            upper_best=np.argmax(upper_h,axis=0)
            upper_residual=(upper-previous_upper)/dt-np.max(upper_h,axis=0)-upper_source
            if np.max(np.abs(upper_residual))<=tolerance*(1+np.max(np.abs(upper))):break
        else:raise RuntimeError('Deficit envelope Howard iteration did not converge')
        global_value+=dt*float(np.max(np.abs(r)))
        global_policy+=dt*(2*float(np.max(np.abs(r)))+float(np.max(deficit)))
        history.append({'step':k,'envelope_iterations':iteration,'envelope_residual':float(np.max(np.abs(residual))),
                        'deficit_envelope_iterations':upper_iteration,'deficit_envelope_residual':float(np.max(np.abs(upper_residual))),
                        'fixed_policy_linear_system_residual':linear_residual})
    def full(x):
        result=np.zeros(b.size);result[ids]=x;return result.reshape(b.shape)
    policy_value=b.copy().ravel();policy_value[ids]=j
    return {'value_envelope':full(e),'policy_evaluation_envelope':full(f),
            'policy_loss_bound':full(e+f),'policy_value':policy_value.reshape(b.shape),
            'deficit_upper_envelope':full(upper),'deficit_policy_loss_bound':full(u+upper-j),
            'global_value_bound':global_value,'global_policy_bound':global_policy,
            'history':history,'continuum_certified':False,'floating_point_validated':False}
