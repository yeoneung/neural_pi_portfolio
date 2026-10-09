import numpy as np
from npi.model import Model
from npi.reference import solve
from npi.discrete_evaluation import evaluate_policy,certainty_equivalent


def test_optimal_discrete_policy_recovers_reference_and_inferior_policy_bound():
    m=Model(horizon=.2)
    ref=solve(m,ny=21,nz=11,nt=10,na=21,tolerance=1e-12)
    exact=evaluate_policy(ref,ref.policy,ref.value)
    np.testing.assert_allclose(exact.value,ref.value,atol=1e-10)
    wrong=evaluate_policy(ref,np.zeros_like(ref.policy),ref.value)
    assert np.min(ref.value-wrong.value)>-1e-9
    observed=np.max(ref.value-wrong.value)
    assert observed>1e-5
    assert observed<=wrong.diagnostics['finite_grid_policy_loss_bound']+1e-9


def test_mixed_utility_ce_inversion():
    m=Model();ys=np.linspace(m.ymin,m.ymax,17)
    np.testing.assert_allclose(certainty_equivalent(m,m.utility(ys)),np.exp(ys),rtol=1e-12)
