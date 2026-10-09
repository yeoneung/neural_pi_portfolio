"""Matched correction spaces for frozen-policy boundary interventions.

The base evaluator is fixed. Zero initial output makes the hard and split
corrections identical as functions at step zero. Only the lower-wealth factor
differs; absorbing boundary rewards enter the finite operator independently.
"""
import numpy as np
import torch
from .neural import ValueNetwork
from .duality import UniformFeedback


class ScalarFeedback:
    """Scalar adapter for the existing vector-field uniform interpolator."""
    def __init__(self, axes, values):
        self.interpolator = UniformFeedback(axes, np.asarray(values)[..., None])

    def __call__(self, points):
        return self.interpolator(points)[:, 0]


class EvaluationCorrection(ValueNetwork):
    def __init__(self, model, width=64, depth=3, mode='hard'):
        super().__init__(model, width, depth)
        if mode not in ('hard', 'split'):
            raise ValueError(mode)
        self.mode = mode

    def forward(self, x):
        m = self.market
        tau, y, z = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        yn = 2*(y-m.ymin)/(m.ymax-m.ymin)-1
        zn = 2*(z-m.zmin)/(m.zmax-m.zmin)-1
        inp = torch.cat([2*tau/m.horizon-1, yn, zn], dim=1)
        scale = torch.exp((1-m.gamma1)*y)+m.mixture*torch.exp((1-m.gamma2)*y)
        bubble = (1-yn**2 if self.mode == 'hard' else 1-yn)*(1-zn**2)
        return tau*bubble*scale*self.net(inp)


def network_values(network, points, chunk=32768):
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    values = []
    with torch.no_grad():
        for start in range(0, len(points), chunk):
            x = torch.as_tensor(points[start:start+chunk], device=device, dtype=dtype)
            values.append(network(x)[:, 0].cpu().numpy().astype(float))
    return np.concatenate(values)


def grid_points(system):
    yy, zz = np.meshgrid(system.y, system.z, indexing='ij')
    space = np.column_stack([yy.ravel()[system.ids], zz.ravel()[system.ids]])
    points = np.concatenate([np.column_stack([np.full(len(space), t), space])
                             for t in system.tau])
    return space, points


def full_evaluation_rms(system, values, policy_indices):
    """Independent float64 residual on every node and every time layer."""
    n = len(system.ids)
    ys = np.repeat(system.y[:, None], len(system.z), axis=1).ravel()[system.ids]
    zs = np.repeat(system.z[None, :], len(system.y), axis=0).ravel()[system.ids]
    scale = system.model.utility_bundle(ys)[1]
    core = (np.abs(ys) <= .7+1e-12) & (np.abs(zs) <= .3+1e-12)
    sums = np.zeros(2)
    maximum = 0.
    for k, dt in enumerate(np.diff(system.tau), 1):
        rows = policy_indices[k]*n+np.arange(n)
        h = system.ay[rows]@values[k]+system.sources[policy_indices[k], np.arange(n)]
        residual = ((values[k]-values[k-1])/dt-h)/scale
        sums += [np.mean(residual**2), np.mean(residual[core]**2)]
        maximum = max(maximum, float(np.max(np.abs(residual))))
    return dict(eval_rms_full=float(np.sqrt(sums[0]/(len(system.tau)-1))),
                eval_rms_core=float(np.sqrt(sums[1]/(len(system.tau)-1))),
                eval_max_scaled=maximum)
