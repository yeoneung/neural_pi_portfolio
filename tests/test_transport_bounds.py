import numpy as np
from npi.model import Model
from npi.reference import solve
from npi.transport_bounds import propagate


def test_exact_value_transported_deficit_equals_fixed_policy_loss():
    ref=solve(Model(horizon=.2),ny=21,nz=11,nt=10,na=21,generator_kind='bounded',tolerance=1e-12)
    out=propagate(ref,np.zeros_like(ref.policy),ref.value)
    loss=ref.value[-1]-out['policy_value']
    assert np.min(loss)>-1e-9
    assert np.max(loss)>1e-5
    np.testing.assert_allclose(out['value_envelope'],0,atol=1e-9)
    np.testing.assert_allclose(out['policy_loss_bound'],loss,atol=1e-9)
    np.testing.assert_allclose(out['deficit_policy_loss_bound'],loss,atol=1e-9)
    assert np.max(out['policy_loss_bound'])<=out['global_policy_bound']+1e-8
