import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve
from npi.model import Model
from npi.cheap_bounds import FiniteSystem
from npi.focused_evaluation import EvaluationCorrection, ScalarFeedback, full_evaluation_rms


def test_scalar_feedback_preserves_original_trilinear_policy():
    from scipy.interpolate import RegularGridInterpolator
    axes = [np.linspace(0, 3, 5), np.linspace(-2, 2, 7), np.linspace(-1.5, 1.5, 9)]
    rng = np.random.default_rng(31)
    fields = rng.uniform(-.8, .8, (5, 7, 9))
    points = rng.uniform([0, -2, -1.5], [3, 2, 1.5], (100, 3))
    np.testing.assert_allclose(ScalarFeedback(axes, fields)(points),
                               RegularGridInterpolator(axes, fields)(points), atol=1e-14)


def test_matched_initial_functions_and_only_lower_face_released():
    m = Model()
    hard = EvaluationCorrection(m, 8, 2, 'hard').double()
    split = EvaluationCorrection(m, 8, 2, 'split').double()
    split.load_state_dict(hard.state_dict())
    points = torch.tensor([[.5, 0., 0.], [.5, m.ymin, 0.],
                           [.5, m.ymax, 0.], [.5, 0., m.zmin],
                           [.5, 0., m.zmax], [0., .3, .1]], dtype=torch.float64)
    np.testing.assert_array_equal(hard(points).detach(), split(points).detach())
    with torch.no_grad():
        hard.net[-1].bias.fill_(.2)
        split.net[-1].bias.fill_(.2)
    a = hard(points).detach().numpy()[:, 0]
    b = split(points).detach().numpy()[:, 0]
    assert a[0] == b[0] > 0
    assert a[1] == 0 and b[1] > 0
    np.testing.assert_array_equal(a[2:], 0.)
    np.testing.assert_array_equal(b[2:], 0.)


def test_full_residual_uses_boundary_sources_and_previous_layer():
    s = FiniteSystem.create(Model(horizon=.2), ny=13, nz=11, nt=6, na=9)
    n = len(s.ids)
    policy = np.full((len(s.tau), n), 5, dtype=int)
    operator = s.ay[5*n+np.arange(n)]
    matrix = sparse.eye(n)-s.tau[1]*operator
    values = [s.terminal.ravel()[s.ids]]
    for _ in s.tau[1:]:
        values.append(spsolve(matrix.tocsc(), values[-1]+s.tau[1]*s.sources[5]))
    values = np.asarray(values)
    assert full_evaluation_rms(s, values, policy)['eval_rms_full'] < 1e-11
    values[2, 5] += .1
    assert full_evaluation_rms(s, values, policy)['eval_rms_full'] > 1e-5
