import numpy as np
import torch
from npi.model import Model
from npi.boundary_neural import BoundaryValueNetwork
from npi.duality import conjugate,inverse_marginal,simulate_dual,simulate_factor_dual,dual_statistics,UniformFeedback
from npi.cheap_bounds import FiniteSystem,scalar_bounds
from npi.reference import solve


def test_split_network_has_exact_payoff_but_can_represent_positive_interior_jump():
    m=Model();hard=BoundaryValueNetwork(m,8,2,'hard').double();split=BoundaryValueNetwork(m,8,2,'split').double()
    split.load_state_dict(hard.state_dict())
    with torch.no_grad():hard.net[-1].bias.fill_(.2);split.net[-1].bias.fill_(.2)
    boundary=torch.tensor([[.5,m.ymin,0.],[.5,m.ymax,0.],[.5,0.,m.zmin],[0.,0.,0.]],dtype=torch.float64)
    for net in [hard,split]:
        np.testing.assert_allclose(net(boundary).detach().numpy()[:,0],m.utility(boundary[:,1]),atol=1e-12)
    inside=torch.tensor([[.5,m.ymin+1e-8,0.]],dtype=torch.float64)
    payoff=m.utility(m.ymin)
    assert float(split(inside).detach()[0,0])-payoff>1.
    assert abs(float(hard(inside).detach()[0,0])-payoff)<1e-4


def test_mixed_fenchel_inequality_and_inverse_marginal():
    m=Model();x=np.logspace(-5,5,101);w=inverse_marginal(m,x)
    np.testing.assert_allclose(w**(-m.gamma1)+m.mixture*w**(-m.gamma2),x,rtol=1e-11)
    conjugated=conjugate(m,x)
    np.testing.assert_allclose(conjugated+x*w,m.utility(np.log(w)),rtol=1e-12)
    trial=np.logspace(-2,2,101)
    assert np.min(conjugated+x*trial-m.utility(np.log(trial)))>=-1e-10


def test_dual_gap_vanishes_pathwise_for_exact_unconstrained_crra_policy():
    m=Model(horizon=.2,phi1=0.,rho=0.,mixture=0.,ymin=-10,ymax=10,zmin=-10,zmax=10)
    q=m.phi0/m.gamma1
    feedback=lambda x:np.column_stack([np.full(len(x),q),np.zeros(len(x))])
    paths=simulate_dual(m,feedback,npaths=1024,nsteps=8,fine_steps=16,seed=5)
    lam=np.exp((1-m.gamma1)*(m.rate+m.phi0**2/(2*m.gamma1))*m.horizon)
    stats=dual_statistics(m,paths,lam)
    np.testing.assert_allclose(stats['gap_samples'],0.,atol=1e-12)
    assert paths['exit_fraction']==0


def test_scalar_repair_bounds_nontrivial_finite_value_error_without_solves():
    m=Model(horizon=.1)
    s=FiniteSystem.create(m,ny=17,nz=13,nt=10,na=11)
    ref=solve(m,ny=17,nz=13,nt=10,na=11,generator_kind='bounded',tolerance=1e-12)
    truth=ref.value.reshape(len(s.tau),-1)[:,s.ids]
    candidate=truth+.02*s.tau[:,None]*np.sin(np.arange(len(s.ids)))[None,:]
    result=scalar_bounds(s,candidate,exponents=(0.,1.))
    assert np.max(truth[-1]-candidate[-1]-result['upper_correction'])<1e-9
    assert result['linear_solves']==result['bellman_solves']==0
    assert result['minimum_repair_slack']>-1e-10


def test_uniform_interpolator_agrees_with_regular_grid_interpolator():
    from scipy.interpolate import RegularGridInterpolator
    rng=np.random.default_rng(3);axes=[np.linspace(0,3,5),np.linspace(-2,2,7),np.linspace(-1.5,1.5,9)]
    fields=rng.normal(size=(5,7,9,2));points=rng.uniform([0,-2,-1.5],[3,2,1.5],size=(1000,3))
    np.testing.assert_allclose(UniformFeedback(axes,fields)(points),RegularGridInterpolator(axes,fields)(points),atol=5e-14)


def test_common_factor_horizon_stops_density_and_wealth_at_the_same_time():
    m=Model(horizon=.3,phi0=0,phi1=0,rho=0,zmin=-.02,zmax=.02,rate=0)
    feedback=lambda x:np.zeros((len(x),2))
    paths=simulate_factor_dual(m,feedback,npaths=1024,nsteps=30,fine_steps=60,seed=9)
    np.testing.assert_array_equal(paths['density_wealth'],np.ones(1024))
    np.testing.assert_array_equal(paths['payoff'],np.full(1024,m.utility(0)))
    assert paths['factor_exit_fraction']>.9 and paths['wealth_exit_fraction']==0


def test_deficit_only_agrees_with_full_envelope_diagnostics():
    from npi.transport_bounds import propagate
    from npi.deficit_only import evaluate
    m=Model(horizon=.1);s=FiniteSystem.create(m,ny=17,nz=13,nt=10,na=11)
    ref=solve(m,ny=17,nz=13,nt=10,na=11,generator_kind='bounded',tolerance=1e-12)
    controls=np.zeros_like(ref.value)
    full=propagate(ref,controls,ref.value)
    fast=evaluate(s,ref.value.reshape(len(s.tau),-1)[:,s.ids],controls.reshape(len(s.tau),-1)[:,s.ids])
    np.testing.assert_allclose(fast['policy_loss_bound'],full['deficit_policy_loss_bound'].ravel()[s.ids],atol=1e-10)
