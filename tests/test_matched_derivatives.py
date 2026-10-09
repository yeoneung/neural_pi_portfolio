"""Reference time stepping with liquidation sources and shared-mask diagnostics."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from matched_derivative_bridge import reference_solve,diagnostics
from npi.model import Model


class MatchedDerivativeTests(unittest.TestCase):
    def test_time_varying_policy_and_boundary_source(self):
        ay=sparse.csr_matrix([[-2.],[-3.]])
        source=np.array([[4.],[9.]])
        policy=lambda points:np.where(points[:,0]<=.5,0.,1.)
        values,error=reference_solve(ay,source,sparse.eye(1,format='csc'),np.array([5.]),np.zeros((1,3)),
            np.array([0.,1.]),np.array([0.,.5,1.]),policy)
        np.testing.assert_allclose(values[:,0],[5.,3.5,3.2],atol=1e-14)
        self.assertLess(error,1e-14)

    def test_common_mask_excludes_clipped_point_for_every_design(self):
        m=Model(qmin=-.8,phi1=0)
        cfg=dict(meshes={'fine':None,'finest':None},modes=['single','full'],steps=[0,50])
        a=dict(points=np.zeros((2,3)),scale=np.ones(2),fixed_q=np.full(2,.3))
        for mesh in cfg['meshes']:
            for field,value in [('vy',1.),('vyy',0.),('vyz',0.)]:a[mesh+'_'+field]=np.full(2,value)
        for mode in cfg['modes']:
            for step in cfg['steps']:
                for field,value in [('vy',1.),('vyy',0.),('vyz',0.),('q',.3)]:a[f'{mode}_{step}_'+field]=np.full(2,value)
        # The last full evaluator is clipped at one point; all other rows must
        # exclude that same point from their interior quotient decomposition.
        a['full_50_vyy'][1]=.9;a['full_50_q'][1]=.8
        rows=diagnostics(a,cfg,m,701)
        self.assertTrue(all(r['common_interior_points']==1 for r in rows))
        self.assertTrue(all(r['greedy_error_rms']==0 for r in rows))
        self.assertGreater(next(r['all_point_clipped_error_rms'] for r in rows if r['mode']=='full' and r['step']==50),0)


if __name__=='__main__':unittest.main()
