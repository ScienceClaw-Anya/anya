import unittest
import numpy as np
from .tools import table,choose,FEATURES,VIEWS
from .audit_features import audit_table


class QueryValueTests(unittest.TestCase):
    def test_true_pair_features_only_use_revealed_signal(self):
        distance=np.array([[20.,30.,40.],[30.,20.,50.],[25.,40.,30.]])
        observed=np.zeros((3,2,1,641)); observed[...,0]=distance[:,0,None,None]; observed[...,-512:]=2.
        x=table(distance,observed,np.array([.5,.8,.9]),[0],[1,2])
        self.assertEqual(x.shape,(2,44)); self.assertTrue(np.isfinite(x).all())
        with self.assertRaises(ValueError): table(distance,observed,np.array([.5,.8,.9]),[0],[0])

    def test_expert_views_are_distinct_and_action_ties_stable(self):
        self.assertEqual(len(FEATURES),44)
        self.assertNotEqual(VIEWS['geometry'],VIEWS['neural'])
        self.assertNotEqual(VIEWS['neural'],VIEWS['waveform'])
        self.assertEqual(choose(['p002','p001'],[1.,1.]),'p001')
        with self.assertRaises(ValueError): choose(['p001'],[np.nan])

    def test_independent_cached_audit_matches_missing_trial_features(self):
        rng=np.random.default_rng(14)
        distance=rng.uniform(15,100,(5,8)); distance[0,1]=0
        observed=rng.normal(size=(5,3,4,641))
        observed[...,0]=distance[:,:4,None].transpose(0,2,1)
        observed[0,:,1,:]=0; observed[2,0,2,:]=0
        scores=np.array([.1,.3,.3,.9,.99]); selected=[0,1,2,3]
        np.testing.assert_allclose(audit_table(distance,observed,scores,selected,[4,5,6,7]),table(distance,observed,scores,selected,[4,5,6,7]),rtol=0,atol=1e-12)


if __name__=='__main__': unittest.main()
