import numpy as np
import pytest
from npi.model import Model
from npi.reference import solve
from npi.stencils import build


@pytest.mark.parametrize('kind',['bounded','semi_lagrangian'])
def test_positive_generators_and_mixed_moment(kind):
    m=Model(rho=-.7,qmin=-.8);y=np.linspace(-2,2,61);z=np.linspace(-1,1,31)
    actions=np.array([-.3,-1e-6,0,1e-6,.3]);a,ids,_,meta=build(m,y,z,actions,kind)
    np.testing.assert_allclose(a@np.ones(len(y)*len(z)),0,atol=1e-11)
    n=len(ids);yy,zz=np.meshgrid(y,z,indexing='ij');yi=yy.ravel()[ids];zi=zz.ravel()[ids]
    middle=(abs(yi)<.3)&(abs(zi)<.2)
    for k,q in enumerate(actions):
        mat=a[k*n:(k+1)*n].tocoo();assert np.all(mat.data[mat.col!=ids[mat.row]]>=0)
        expected=(m.rate+q*m.phi(zi)-q*q/2)*zi+m.kappa*(m.zbar-zi)*yi+m.rho*q*m.sigma_z
        np.testing.assert_allclose((a@(yy*zz).ravel()).reshape(len(actions),n)[k,middle],expected[middle],atol=1e-10)


@pytest.mark.parametrize('kind',['bounded','semi_lagrangian'])
def test_correlated_frozen_policy_matches_independent_exact_formula(kind):
    m=Model(horizon=.2,phi1=0,rho=-.7)
    errors=[]
    for ny,nz,nt in [(21,11,20),(41,21,40),(81,41,80)]:
        ref=solve(m,ny=ny,nz=nz,nt=nt,na=3,fixed_q=.15,exact_fixed_boundary=True,generator_kind=kind)
        p=np.array([[.2,0,0],[.2,.4,0],[.1,-.4,0]])
        exact=m.constant_policy_value(p[:,0],p[:,1],.15)
        errors.append(np.max(np.abs(ref.interpolate(p)-exact)/np.abs(exact)))
    assert errors[-1]<errors[0]
    assert errors[-1]<.003


def test_bounded_small_action_perturbation_shrinks_with_mesh():
    defects=[]
    for n in [20,80,320]:
        # Assembly on a small local lattice with the desired physical spacing.
        h=4/n;y=np.arange(-3,4)*h;z=y.copy()
        _,_,_,meta=build(Model(rho=-.7),y,z,np.linspace(-.02,.02,21),'bounded')
        defects.append(meta['max_added_diagonal_variance_sum'])
    assert defects[-1]<defects[0]
