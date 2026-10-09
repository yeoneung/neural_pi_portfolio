"""Finite-model policy-difference attribution, without differentiating a grid value.

The generator rows include boundary columns. The propagated candidate advantage
minus the propagated evaluation-error correction equals the policy-value change
in the specified backward-Euler model. No continuum or interval certificate.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def action_rows(controls, actions, n):
    """Nearest action, choosing the lower index at a tie."""
    indices = np.argmin(np.abs(np.asarray(controls)[:, None] - actions[None, :]), axis=1)
    return indices * n + np.arange(n)


def fixed_evaluation(operators, sources, terminal, dt):
    """Evaluate a time-dependent frozen generator on the interior nodes."""
    values = [np.asarray(terminal, dtype=float).copy()]
    solve_residual = 0.
    eye = sparse.eye(len(terminal), format='csc')
    for operator, source in zip(operators, sources):
        matrix = eye - dt * operator
        rhs = values[-1] + dt * source
        u = splu(matrix.tocsc()).solve(rhs)
        solve_residual = max(solve_residual, float(np.max(np.abs(matrix @ u - rhs))))
        values.append(u)
    return np.asarray(values), solve_residual


def compare_step(operator_q, operator_p, source_q, source_p, uq, v,
                 previous_p, previous_advantage, previous_correction, dt):
    """One time layer of the exact finite-model decomposition."""
    difference = operator_p - operator_q
    boundary_difference = source_p - source_q
    candidate_advantage = difference @ v + boundary_difference
    exact_advantage = difference @ uq + boundary_difference
    directional_error = difference @ (v - uq)
    matrix = sparse.eye(len(uq), format='csc') - dt * operator_p
    rhs = np.column_stack([
        previous_p + dt * source_p,
        previous_advantage + dt * candidate_advantage,
        previous_correction + dt * directional_error,
    ])
    result = splu(matrix.tocsc()).solve(rhs)
    return result[:, 0], result[:, 1], result[:, 2], {
        'candidate_advantage': candidate_advantage,
        'exact_advantage': exact_advantage,
        'directional_error': directional_error,
        'identity_error': float(np.max(np.abs(candidate_advantage - directional_error - exact_advantage))),
        'linear_residual': float(np.max(np.abs(matrix @ result - rhs))),
    }
