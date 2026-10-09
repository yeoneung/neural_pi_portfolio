"""Analytic regression cases; these numerical checks are not theorem proofs."""
import numpy as np
from scipy.integrate import quad
from npi.model import Model
from npi.duality import conjugate,dual_statistics


def test_exact_cash_submodel_has_zero_policy_loss_but_positive_dual_gap():
    # Only the utility helper uses Model. The analytic submodel has phi=nu=0,
    # no factor exit, and admissible set {0}, so the only policy is optimal.
    m=Model(horizon=3.,gamma1=2.,gamma2=5.,mixture=.3)
    y0=1.97;barrier=2.;theta=(barrier-y0)/m.rate
    initial=np.exp(y0);terminal=np.exp(y0+m.rate*m.horizon);density=np.exp(-m.rate*m.horizon)
    marginal=terminal**(-m.gamma1)+m.mixture*terminal**(-m.gamma2)
    multiplier=marginal/density
    paths={'payoff':np.full(4,m.utility(barrier)),
           'density':np.full(4,density),'density_wealth':np.full(4,density*terminal),
           'exit_fraction':1.,'simulation_seconds':0.}
    stats=dual_statistics(m,paths,multiplier)
    cash_floor=m.utility(barrier+m.rate*(m.horizon-theta))-m.utility(barrier)
    np.testing.assert_allclose(density*terminal,initial,rtol=1e-14)
    np.testing.assert_allclose(stats['gap_samples'],cash_floor,rtol=1e-12)
    assert cash_floor>0
    # Fenchel equality is exact here, despite the positive gap versus the
    # optimal value of the original stopped problem.
    np.testing.assert_allclose(conjugate(m,multiplier*density)+multiplier*density*terminal,
                               m.utility(np.log(terminal)),atol=1e-13)


def test_trace_term_is_necessary_in_stopped_ito_identity():
    m=Model(horizon=3.,gamma1=2.,gamma2=5.,mixture=.3)
    y0=1.97;barrier=2.;theta=(barrier-y0)/m.rate;offset=.4
    # Smooth interior branch v(t,y)=psi(y)+offset*(T-t), cash dynamics only.
    # At upper exit the assigned payoff differs from this branch's trace.
    actual_minus_initial=m.utility(barrier)-(m.utility(y0)+offset*m.horizon)
    residual_integral=quad(lambda s:m.rate*m.utility_bundle(y0+m.rate*s)[1]-offset,0,theta,
                           epsabs=1e-12,epsrel=1e-12)[0]
    trace_discrepancy=-offset*(m.horizon-theta)
    np.testing.assert_allclose(actual_minus_initial,trace_discrepancy+residual_integral,atol=1e-12)
    assert abs(actual_minus_initial-residual_integral)>.5


def test_fenchel_curvature_bounds_and_integral_identity():
    m=Model(gamma1=2.,gamma2=5.,mixture=.3)
    for inv,wealth in [(.4,.7),(2.,.9),(.2,1.3),(3.,4.)]:
        x=inv**(-m.gamma1)+m.mixture*inv**(-m.gamma2)
        curvature=lambda w:m.gamma1*w**(-m.gamma1-1)+m.mixture*m.gamma2*w**(-m.gamma2-1)
        gap=float(conjugate(m,x)+x*wealth-m.utility(np.log(wealth)))
        integral=(wealth-inv)**2*quad(lambda s:(1-s)*curvature(inv+s*(wealth-inv)),0,1,
                                     epsabs=1e-11,epsrel=1e-11)[0]
        lo=.5*curvature(max(inv,wealth))*(wealth-inv)**2
        hi=.5*curvature(min(inv,wealth))*(wealth-inv)**2
        np.testing.assert_allclose(gap,integral,rtol=1e-11)
        assert lo<=gap+1e-11 and gap<=hi+1e-11
