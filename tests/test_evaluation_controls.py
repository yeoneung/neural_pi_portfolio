"""Checks for the algebra and the empirical, non-certified step matching."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluation_controls import quotient_terms, matching_step, rms


class ControlsTests(unittest.TestCase):
    def test_exact_quotient_and_cross_terms(self):
        rng=np.random.default_rng(48)
        cref=rng.uniform(.5,3,100);cv=rng.uniform(.5,3,100)
        pref=rng.uniform(-.3,.3,100);phi=rng.uniform(.1,.5,100)
        ey=rng.normal(0,.02,100);eyz=rng.normal(0,.03,100)
        bv=pref*cref+phi*ey-.245*eyz
        parts=quotient_terms(phi,-.245,ey,eyz,cv-cref,pref,cv)
        np.testing.assert_allclose(parts.sum(axis=0),bv/cv-pref,atol=1e-15)
        mse=np.mean(parts**2,axis=1).sum()+sum(2*np.mean(parts[i]*parts[j]) for i,j in [(0,1),(0,2),(1,2)])
        self.assertAlmostEqual(mse,rms(bv/cv-pref)**2,places=14)

    def test_matching_is_global_norm_not_direction_or_certificate(self):
        short=np.array([1.,0.]);long=np.array([0.,4.])
        lam=matching_step(short,long)
        self.assertEqual(lam,.25)
        self.assertAlmostEqual(rms(short),rms(lam*long))
        self.assertFalse(np.allclose(short,lam*long))
        self.assertEqual(matching_step(short,np.zeros(2)),0)
        self.assertEqual(matching_step(long,short),1)


if __name__=='__main__':unittest.main()
