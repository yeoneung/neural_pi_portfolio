"""Directional diagnostics for inexact neural policy evaluation.

The disagreement between successive critics is an empirical proxy, never a
certified upper bound on error relative to the true policy value.
"""
import copy
import numpy as np
import torch
from .neural import derivatives, greedy, residual


def frozen_copy(net):
    out = copy.deepcopy(net).eval()
    for parameter in out.parameters():
        parameter.requires_grad_(False)
    return out


def evaluation_loss(net, x, q, shape_weight=1.):
    m = net.market
    d = derivatives(net, x)
    scale = torch.exp((1-m.gamma1)*x[:, 1])+m.mixture*torch.exp((1-m.gamma2)*x[:, 1])
    raw = residual(m, d, q)
    shape = torch.relu(-d['vy']/scale)**2+torch.relu((d['vyy']-d['vy'])/scale)**2
    return torch.mean((raw/scale)**2)+shape_weight*torch.mean(shape)


def coefficients(net, x):
    d = derivatives(net, x)
    m = net.market
    result = dict(b=m.phi(x[:, 2])*d['vy']+m.rho*m.sigma_z*d['vyz'],
                  c=d['vy']-d['vyy'], q=greedy(m, d), v=d['v'])
    result['hjb'] = residual(m, d, result['q'])
    return {k: v.detach() for k, v in result.items()}


def directional_terms(b, c, old_q, new_q, delta_b, delta_c):
    """Exact candidate advantage and signed error contribution, array generic."""
    dq = new_q-old_q
    advantage = dq*(b-.5*c*(new_q+old_q))
    error = dq*(delta_b-.5*(new_q+old_q)*delta_c)
    return advantage, error


def disagreement_metrics(current, previous, old_q, scale, multiplier=2.):
    advantage, error_proxy = directional_terms(current['b'], current['c'], old_q,
        current['q'], current['b']-previous['b'], current['c']-previous['c'])
    rms = lambda x: float(torch.mean(x*x).sqrt())
    gain = rms(advantage/scale)
    uncertainty = multiplier*rms(error_proxy/scale)
    return dict(gain_rms=gain, disagreement_rms=uncertainty,
                disagreement_ratio=uncertainty/max(gain, 1e-15),
                q_change_rms=rms(current['q']-old_q),
                eval_rms=rms((current['hjb']-advantage)/scale),
                curvature_violation_fraction=float((current['c']<=0).float().mean()))


class PolicyBank:
    """Exact convex mixture of greedy feedbacks, without fitting another actor."""
    def __init__(self, net):
        self.members = [(1., frozen_copy(net))]

    def controls(self, x):
        q = torch.zeros(len(x), device=x.device, dtype=x.dtype)
        for weight, net in self.members:
            q = q+weight*greedy(net.market, derivatives(net, x)).detach()
        return q

    def update(self, net, damping=1.):
        if not 0 < damping <= 1:
            raise ValueError('damping must belong to (0,1]')
        self.members = ([] if damping == 1 else [(w*(1-damping), n) for w, n in self.members])
        self.members.append((damping, frozen_copy(net)))

    def state(self):
        return [dict(weight=w, state_dict={k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
                for w, net in self.members]

    def grid(self, points, chunk=4096):
        device = next(self.members[0][1].parameters()).device
        values = []
        for start in range(0, len(points), chunk):
            x = torch.as_tensor(points[start:start+chunk], device=device, dtype=torch.float32)
            values.append(self.controls(x).cpu().numpy())
        return np.concatenate(values)
