"""Terminal-wealth control with a mixed-power, non-homothetic utility.

State: log wealth y and OU predictor z. Control q is wealth-normalized
volatility exposure; dW/W = (r + q*phi(z)) dt + q dB.
The bounded-domain experiment liquidates at the first domain exit, paying
g(W_exit). This is a specified stopped problem, not an unidentified
whole-space PDE truncated without boundary information.
"""
from dataclasses import asdict, dataclass, fields
import numpy as np


@dataclass(frozen=True)
class Model:
    horizon: float = 1.0
    rate: float = 0.03
    phi0: float = 0.30
    phi1: float = 0.20
    kappa: float = 1.0
    zbar: float = 0.0
    sigma_z: float = 0.20
    rho: float = -0.40
    gamma1: float = 2.0
    gamma2: float = 4.0
    mixture: float = 0.10
    qmin: float = 0.0
    qmax: float = 0.8
    ymin: float = -2.0
    ymax: float = 2.0
    zmin: float = -1.0
    zmax: float = 1.0

    def __post_init__(self):
        if self.horizon <= 0 or self.sigma_z <= 0 or abs(self.rho) >= 1:
            raise ValueError("Need T>0, sigma_z>0 and |rho|<1")
        if not (self.ymin < self.ymax and self.zmin < self.zmax and self.qmin < self.qmax):
            raise ValueError("Invalid state or action interval")
        if min(self.gamma1, self.gamma2) <= 0 or 1 in (self.gamma1, self.gamma2) or self.mixture < 0:
            raise ValueError("Positive risk aversions different from one and mixture>=0 required")

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown model keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self):
        return asdict(self)

    def phi(self, z):
        return self.phi0 + self.phi1 * z

    def utility(self, y):
        return np.exp((1-self.gamma1)*np.asarray(y))/(1-self.gamma1) + self.mixture*np.exp((1-self.gamma2)*np.asarray(y))/(1-self.gamma2)

    def utility_bundle(self, y):
        a = np.exp((1-self.gamma1)*np.asarray(y))
        b = self.mixture*np.exp((1-self.gamma2)*np.asarray(y))
        return self.utility(y), a+b, (1-self.gamma1)*a+(1-self.gamma2)*b

    def relative_risk_aversion(self, y):
        _, gy, gyy = self.utility_bundle(y)
        return (gy-gyy)/gy

    def constant_policy_value(self, tau, y, q):
        """Whole-space exact fixed-q value, valid when phi1=0.

        Used ONLY as a verification case with its own exact lateral data.
        It is not the mixed-utility optimal value.
        """
        if self.phi1 != 0:
            raise ValueError("This formula requires constant risk premium")
        ans = np.zeros(np.broadcast_shapes(np.shape(tau), np.shape(y)))
        for weight, gamma in [(1.0, self.gamma1), (self.mixture, self.gamma2)]:
            p = 1-gamma
            exponent = p*(self.rate+q*self.phi0)-0.5*gamma*p*q*q
            ans = ans + weight*np.exp(p*np.asarray(y)+exponent*np.asarray(tau))/p
        return ans

    def greedy(self, vy, vyy, vyz, z):
        """Exact maximizer of the quadratic Hamiltonian on [qmin,qmax].

        Handles loss of concavity by comparing endpoints. No curvature
        clipping changes the optimization problem.
        """
        linear = self.phi(z)*vy + self.rho*self.sigma_z*vyz
        quadratic = vyy-vy
        root = np.divide(-linear, quadratic, out=np.zeros_like(np.asarray(linear, dtype=float)), where=quadratic < -1e-14)
        root = np.clip(root, self.qmin, self.qmax)
        lo = linear*self.qmin+0.5*quadratic*self.qmin**2
        hi = linear*self.qmax+0.5*quadratic*self.qmax**2
        endpoint = np.where(hi > lo, self.qmax, self.qmin)
        return np.where(quadratic < -1e-14, root, endpoint)

