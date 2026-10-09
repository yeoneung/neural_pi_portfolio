from dataclasses import replace
import numpy as np
import torch
from npi.model import Model
from npi.reference import build_generators,solve
from npi.neural import ValueNetwork,derivatives,greedy,residual


def test_mixed_utility_is_increasing_concave_and_not_homothetic():
    m=Model();y=np.linspace(-1,1,20)
    _,gy,gyy=m.utility_bundle(y)
    assert np.all(gy>0) and np.all(gyy-gy<0)
    assert np.ptp(m.relative_risk_aversion(y))>0.5
    pure=replace(m,mixture=0)
    np.testing.assert_allclose(pure.relative_risk_aversion(y),m.gamma1)


def test_quadratic_greedy_matches_brute_force_with_bad_curvature():
    rng=np.random.default_rng(1);m=Model()
    vy,vyy,vyz,z=[rng.normal(size=30) for _ in range(4)]
    q=m.greedy(vy,vyy,vyz,z)
    grid=np.linspace(m.qmin,m.qmax,10001)
    a=m.phi(z)*vy+m.rho*m.sigma_z*vyz;b=vyy-vy
    scores=a[:,None]*grid+0.5*b[:,None]*grid**2
    assert np.max(np.max(scores,axis=1)-(a*q+0.5*b*q*q))<1e-10


def test_generator_is_monotone_and_consistent_on_quadratics():
    m=Model();y=np.linspace(-2,2,41);z=np.linspace(-1,1,31)
    a,ids,_=build_generators(m,y,z,np.array([0,0.1,0.4]))
    np.testing.assert_allclose(np.asarray(a.sum(axis=1)),0,atol=1e-12)
    n=len(ids)
    for k in range(3):
        mat=a[k*n:(k+1)*n].tocoo()
        off=mat.col!=ids[mat.row]
        assert np.all(mat.data[off]>=0)
    yy,zz=np.meshgrid(y,z,indexing='ij')
    computed=(a@(yy*zz).ravel()).reshape(3,n)
    yi=yy.ravel()[ids];zi=zz.ravel()[ids]
    middle=(abs(yi)<0.7)&(abs(zi)<0.3)
    for k,q in enumerate([0,0.1,0.4]):
        expected=(m.rate+q*m.phi(zi)-q*q/2)*zi+m.kappa*(m.zbar-zi)*yi+m.rho*q*m.sigma_z
        np.testing.assert_allclose(computed[k,middle],expected[middle],atol=1e-10)


def test_fixed_policy_reference_converges_to_independent_exact_solution():
    m=Model(horizon=0.2,phi1=0,rho=0)
    errors=[]
    for ny,nz,nt in [(21,11,20),(41,21,40)]:
        ref=solve(m,ny=ny,nz=nz,nt=nt,na=3,fixed_q=0.15,exact_fixed_boundary=True)
        points=np.array([[0.2,0,0],[0.2,0.4,0],[0.1,-0.4,0]])
        exact=m.constant_policy_value(points[:,0],points[:,1],0.15)
        errors.append(np.max(np.abs(ref.interpolate(points)-exact)/np.abs(exact)))
    assert errors[1]<errors[0]
    assert errors[1]<0.004


def test_hard_boundary_and_terminal_and_initial_greedy():
    m=Model();net=ValueNetwork(m,width=8,depth=2).double()
    with torch.no_grad():net.net[-1].weight.normal_()
    x=torch.tensor([[0,0.2,0.1],[0.4,m.ymin,0.2],[0.3,0.1,m.zmax]],dtype=torch.float64)
    np.testing.assert_allclose(net(x).detach().numpy()[:,0],m.utility(x[:,1].numpy()),atol=1e-12)
    fresh=ValueNetwork(m,width=8,depth=2).double()
    d=derivatives(fresh,x)
    q=greedy(m,d).detach().numpy()
    expected=np.clip(m.phi(x[:,2].numpy())/m.relative_risk_aversion(x[:,1].numpy()),m.qmin,m.qmax)
    np.testing.assert_allclose(q,expected)


def test_residual_policy_gap_identity():
    m=Model();net=ValueNetwork(m,width=8,depth=2).double()
    x=torch.tensor([[0.2,0.1,0.2],[0.8,-0.5,-0.1]],dtype=torch.float64)
    d=derivatives(net,x);q=greedy(m,d);old=torch.zeros_like(q)
    diff=residual(m,d,q)-residual(m,d,old)
    formula=0.5*(d['vy']-d['vyy'])*(q-old)**2
    torch.testing.assert_close(diff,formula)
