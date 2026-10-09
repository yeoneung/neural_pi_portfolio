"""Scientific invariants for the continuous boundary intervention and derivative audit."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from followup_continuous_boundary import ContinuousEvaluator
from followup_derivative_curves import reference
from npi.model import Model
from npi.neural import ValueNetwork, derivatives


class FollowupTests(unittest.TestCase):
    def test_common_initial_function_and_derivatives(self):
        m=Model();base=ValueNetwork(m)
        with torch.no_grad():base.net[-1].weight.fill_(.01)
        x=torch.tensor([[.2,-.5,.1],[.7,.5,-.1]])
        for mode in ['hard','split']:
            net=ContinuousEvaluator(m,mode,dict(width=64,depth=3),base)
            for k in ['v','vy','vyy','vyz']:
                torch.testing.assert_close(derivatives(net,x)[k],derivatives(base,x)[k],rtol=0,atol=0)

    def test_retained_data_and_lower_boundary_freedom(self):
        m=Model()
        for mode in ['hard','split']:
            net=ContinuousEvaluator(m,mode,dict(width=64,depth=3))
            with torch.no_grad():net.correction.net[-1].bias.fill_(1.)
            x=torch.tensor([[0.,0.,0.],[.5,m.ymax,0.],[.5,0.,m.zmin],[.5,0.,m.zmax]])
            np.testing.assert_allclose(net(x).detach().numpy()[:,0],m.utility(x[:,1].numpy()),atol=1e-6)
            lower=torch.tensor([[.5,m.ymin,0.]])
            difference=float((net(lower)[0,0]-m.utility(m.ymin)).detach())
            self.assertEqual(difference==0,mode=='hard')

    def test_derivative_extraction_on_quadratic(self):
        tau=np.linspace(0,1,5);y=np.linspace(-2,2,41);z=np.linspace(-1,1,31)
        ids=np.arange(len(y)*len(z)).reshape(len(y),len(z))[1:-1,1:-1].ravel()
        tt,yy,zz=np.meshgrid(tau,y,z,indexing='ij')
        v=tt+yy+2*yy**2+3*yy*zz+zz**2
        pts=np.array([[.25,-.2,0.],[.75,.4,.2]])
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'reference.npz'
            np.savez(path,tau=tau,y=y,z=z,interior=ids,value=v.reshape(len(tau),-1)[:,ids])
            ref=reference(path,Model(),pts)
        np.testing.assert_allclose(ref['vy'],1+4*pts[:,1]+3*pts[:,2],atol=1e-12)
        np.testing.assert_allclose(ref['vyy'],4,atol=1e-12)
        np.testing.assert_allclose(ref['vyz'],3,atol=1e-12)


if __name__=='__main__':unittest.main()
