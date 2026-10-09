import numpy as np
from npi.model import Model
from npi.simulation import simulate,paired_ce


def test_zero_control_deterministic_wealth_and_paired_zero_shortfall():
    m=Model(horizon=.5,zmin=-100,zmax=100)
    zero=lambda p:np.zeros(len(p))
    result=simulate(m,zero,npaths=32,nsteps=10,fine_steps=20)
    np.testing.assert_allclose(result['payoff'],m.utility(m.rate*m.horizon),atol=1e-12)
    pair=paired_ce(m,result['payoff'],result['payoff'])
    assert abs(pair['relative_CE_shortfall_vs_reference'])<1e-14
    assert pair['paired_mc_se']<1e-14


def test_deterministic_crossing_pays_at_boundary_not_overshoot():
    m=Model(horizon=1.,rate=.1,zmin=-100,zmax=100)
    result=simulate(m,lambda p:np.zeros(len(p)),npaths=16,nsteps=3,fine_steps=6,initial_y=m.ymax-.05)
    np.testing.assert_allclose(result['payoff'],m.utility(m.ymax),atol=1e-12)
    np.testing.assert_allclose(result['exit_time'],.5,atol=1e-12)
