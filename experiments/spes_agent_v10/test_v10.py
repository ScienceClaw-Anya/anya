import unittest
import numpy as np
from .features import from_revealed,expected_rank_gain,FEATURES
from experiments.spes_agent_v8.audit import independent_auc

class PosteriorFeatureTests(unittest.TestCase):
    def test_hidden_waveforms_do_not_change_contact_features(self):
        rng=np.random.default_rng(71);raw=rng.normal(size=(9,4,15,641)).astype(np.float32);raw[...,0]=30
        selected=[1,3,5,7];score=np.linspace(.1,.9,9);a=from_revealed(raw,selected,score)
        raw[:,:,[i for i in range(15) if i not in selected],1:]+=10000
        np.testing.assert_array_equal(a,from_revealed(raw,selected,score));self.assertEqual(a.shape,(9,len(FEATURES)))
    def test_unavailable_traces_are_ignored(self):
        rng=np.random.default_rng(5);raw=rng.normal(size=(9,4,15,641));raw[...,0]=30;raw[0,:,1,0]=0
        score=np.linspace(.1,.9,9);a=from_revealed(raw,[1,3,5,7],score);raw[0,:,1,1:]+=1e5
        np.testing.assert_array_equal(a,from_revealed(raw,[1,3,5,7],score))
    def test_hard_posterior_matches_pairwise_auc_gain(self):
        labels=np.array([0,0,1,1]);before=np.array([.5,.8,.2,.5]);after=np.array([.1,.2,.7,.8])
        self.assertAlmostEqual(expected_rank_gain(labels,before,after),independent_auc(labels,after)-independent_auc(labels,before))
        self.assertAlmostEqual(expected_rank_gain(labels,before,after),-expected_rank_gain(labels,after,before))
    def test_equal_posterior_has_no_order_preference_without_ties(self):
        before=np.array([.1,.2,.4,.8]);after=before[::-1]
        self.assertAlmostEqual(expected_rank_gain(np.full(4,.3),before,after),0)

if __name__=='__main__':unittest.main()
