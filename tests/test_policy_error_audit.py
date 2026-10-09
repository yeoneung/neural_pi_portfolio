import numpy as np
from scipy import sparse
from npi.policy_error_audit import fixed_evaluation, compare_step


def test_propagated_difference_with_changing_policies_and_boundary_sources():
    # Two transient states, absorbing boundary sources, time-varying policies.
    dt = .17
    terminal = np.array([-.8, -.3])
    oq = [sparse.csr_matrix([[-1.-.1*k, .4], [.2, -.9]]) for k in range(5)]
    op = [sparse.csr_matrix([[-.8, .3], [.1+.02*k, -.7]]) for k in range(5)]
    sq = [np.array([-.1, -.2-.01*k]) for k in range(5)]
    sp = [np.array([-.12-.01*k, -.13]) for k in range(5)]
    uq, residual = fixed_evaluation(oq, sq, terminal, dt)
    up_independent, _ = fixed_evaluation(op, sp, terminal, dt)
    up = terminal.copy(); adv = np.zeros(2); correction = np.zeros(2)
    for k in range(5):
        v = uq[k+1] + np.array([.07*(k+1), -.03])
        up, adv, correction, fields = compare_step(
            oq[k], op[k], sq[k], sp[k], uq[k+1], v, up, adv, correction, dt)
        np.testing.assert_allclose(up, up_independent[k+1], atol=1e-14)
        np.testing.assert_allclose(up-uq[k+1], adv-correction, atol=1e-14)
        assert fields['identity_error'] < 1e-14
    assert residual < 1e-14


def test_exact_evaluation_has_zero_correction():
    q = sparse.csr_matrix([[-1., .2], [.3, -.8]])
    p = sparse.csr_matrix([[-.7, .1], [.4, -.9]])
    source = np.array([-.1, -.2]); terminal = np.array([-1., -.5]); dt = .2
    uq, _ = fixed_evaluation([q], [source], terminal, dt)
    _, _, correction, fields = compare_step(q, p, source, source, uq[1], uq[1],
        terminal, np.zeros(2), np.zeros(2), dt)
    np.testing.assert_array_equal(correction, 0.)
    np.testing.assert_array_equal(fields['directional_error'], 0.)
