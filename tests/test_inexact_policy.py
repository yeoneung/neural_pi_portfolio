import numpy as np
import torch
from npi.inexact import directional_terms, PolicyBank
from npi.model import Model
from npi.neural import ValueNetwork, derivatives, greedy, residual


def test_directional_error_identity_at_clipped_and_interior_actions():
    rng=np.random.default_rng(9271)
    b=rng.normal(size=1000);c=rng.uniform(.2,4,1000)
    db=rng.normal(scale=.3,size=1000);dc=rng.uniform(-.1,.1,1000)
    q=rng.uniform(-.8,.8,1000);qp=np.clip(b/c,-.8,.8)
    gain,error=directional_terms(b,c,q,qp,db,dc)
    true=lambda a:(b-db)*a-.5*(c-dc)*a*a
    np.testing.assert_allclose(gain-error,true(qp)-true(q),atol=1e-14)
    assert np.sum(np.abs(qp)==.8)>100


def test_certified_damping_bound_against_independent_true_hamiltonians():
    rng=np.random.default_rng(9272)
    b=rng.normal(size=4000);c=rng.uniform(.5,4,4000)
    q=rng.uniform(-.8,.8,4000);d=np.clip(b/c,-.8,.8)-q
    eps1=.03;eps2=.05
    margin=(b-c*q)*d-eps1*np.abs(d)
    good=(margin>0)&(np.abs(d)>1e-10)
    theta=np.zeros_like(d);theta[good]=np.minimum(1,margin[good]/((c[good]+eps2)*d[good]**2))
    # Independent bounded perturbations satisfy |delta_b-q delta_c|<=eps1.
    dc=rng.uniform(-eps2,eps2,len(d));db=q*dc+rng.uniform(-eps1,eps1,len(d))
    qp=q+theta*d
    true=lambda a:(b-db)*a-.5*(c-dc)*a*a
    assert np.min(true(qp)-true(q))>=-1e-14
    assert np.sum(good)>3000


def test_zero_exposure_residual_does_not_control_wealth_curvature():
    class Perturbation(torch.nn.Module):
        def __init__(self,n):super().__init__();self.n=n
        def forward(self,x):
            t,y,z=x[:,0],x[:,1],x[:,2]
            chi=t*(1-t)*(1-y*y)**2*(1-z*z)**2
            phase=self.n*(y+.03*(t-.5))
            return (chi*torch.cos(phase)/self.n**2)[:,None]
    m=Model();x=torch.tensor([[.5,0.,0.],[.25,.2,.3],[.7,-.4,-.2]],dtype=torch.float64)
    previous=None
    for n in [16,32,64,128]:
        d=derivatives(Perturbation(n),x)
        rr=residual(m,d,torch.zeros(len(x),dtype=x.dtype))
        assert float(rr.detach().abs().max())*n*n<2.
        assert abs(float(d['vyy'][0].detach()))>.24
        if previous is not None:assert float(d['v'].detach().abs().max())<previous
        previous=float(d['v'].detach().abs().max())


def test_damped_policy_is_convex_mixture_of_controls():
    torch.manual_seed(9274);m=Model(qmin=-.8)
    net=ValueNetwork(m,8,1);bank=PolicyBank(net)
    x=torch.rand(25,3)
    old=bank.controls(x)
    with torch.no_grad():net.net[-1].weight.fill_(.05)
    new=greedy(m,derivatives(net,x)).detach()
    bank.update(net,.5)
    torch.testing.assert_close(bank.controls(x),.5*(old+new))
    assert abs(sum(w for w,_ in bank.members)-1)<1e-14
