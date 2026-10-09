"""Independent fixed-policy values and finite-grid a posteriori diagnostics.

These quantify the stopped, time/action/space-discrete control problem only.
They are not Monte Carlo welfare estimates or continuum certificates.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .reference import Reference,build_generators


def evaluate_policy(reference, controls, candidate_values=None):
    m=reference.model
    if reference.diagnostics['fixed_q'] is not None:
        raise ValueError('Require an optimal discrete reference')
    if controls.shape!=reference.value.shape:
        raise ValueError('Controls must be specified on the reference grid')
    actions=np.linspace(m.qmin,m.qmax,reference.diagnostics['na'])
    kind=reference.diagnostics.get('generator_kind','legacy')
    if kind=='legacy':stack,ids,_=build_generators(m,reference.y,reference.z,actions)
    else:
        from .stencils import build
        stack,ids,_,_=build(m,reference.y,reference.z,actions,kind)
    n=len(ids);idx=np.arange(n);boundary=np.setdiff1d(np.arange(len(reference.y)*len(reference.z)),ids)
    ay=stack[:,ids].tocsr();ab=stack[:,boundary].tocsr();eye=sparse.eye(n,format='csr')
    boundary_full=m.utility(reference.y)[:,None]*np.ones((1,len(reference.z)))
    source=np.asarray(ab@boundary_full.ravel()[boundary]).reshape(len(actions),n)
    values=np.empty_like(reference.value);values[0]=boundary_full
    history=[]
    for k,dt in enumerate(np.diff(reference.tau),start=1):
        q=controls[k].ravel()[ids]
        qi=np.argmin(np.abs(q[:,None]-actions[None,:]),axis=1)
        rows=qi*n+idx
        u=spsolve(eye-dt*ay[rows,:],values[k-1].ravel()[ids]+dt*source[qi,idx])
        full=boundary_full.copy().ravel();full[ids]=u;values[k]=full.reshape(boundary_full.shape)
        solve_residual=(u-values[k-1].ravel()[ids])/dt-(ay[rows,:]@u+source[qi,idx])
        record={'step':k,'dt':float(dt),'linear_solve_residual_max':float(np.max(np.abs(solve_residual)))}
        if candidate_values is not None:
            # Projection enforces prescribed terminal/lateral data exactly.
            uc=candidate_values[k].ravel()[ids]
            up=candidate_values[k-1].ravel()[ids] if k>1 else boundary_full.ravel()[ids]
            h=np.asarray(ay@uc).reshape(len(actions),n)+source
            best=np.max(h,axis=0)
            residual=(uc-up)/dt-best
            gap=best-h[qi,idx]
            record.update(residual_max=float(np.max(np.abs(residual))),
                          greedy_deficit_max=float(np.max(gap)))
        history.append(record)
    info={'boundary':'stopped liquidation utility',
          'policy_representation':'continuous neural greedy rounded to nearest action; fixed-policy discrete solve',
          'continuum_certified':False,'floating_point_verified':False,'history':history}
    if candidate_values is not None:
        info['finite_grid_value_error_bound']=sum(r['dt']*r['residual_max'] for r in history)
        info['finite_grid_policy_loss_bound']=sum(r['dt']*(2*r['residual_max']+r['greedy_deficit_max']) for r in history)
        info['actual_max_policy_loss_vs_reference']=float(np.max(reference.value-values))
        info['actual_max_projected_value_error']=float(np.max(np.abs(candidate_values[1:,1:-1,1:-1]-reference.value[1:,1:-1,1:-1])))
    return Reference(m,reference.tau,reference.y,reference.z,values,controls,info)


def certainty_equivalent(model,value):
    """Inverse terminal utility, in wealth units, on the stopped wealth range."""
    value=np.asarray(value,dtype=float)
    if np.any(value<model.utility(model.ymin)-1e-8) or np.any(value>model.utility(model.ymax)+1e-8):
        raise ValueError('Utility outside stopped-domain reward range')
    lo=np.full_like(value,model.ymin);hi=np.full_like(value,model.ymax)
    for _ in range(70):
        mid=(lo+hi)/2
        below=model.utility(mid)<value
        lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
    return np.exp((lo+hi)/2)
