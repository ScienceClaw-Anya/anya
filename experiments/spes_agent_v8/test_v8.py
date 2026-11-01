import unittest
import numpy as np
from unittest.mock import patch
from .features import from_revealed,FEATURES

class FeatureIsolationTests(unittest.TestCase):
    def test_cached_feature_matrix_matches_original(self):
        from .numeric import feature_matrix
        rng=np.random.default_rng(619)
        raw=rng.normal(size=(9,5,15,641)).astype(np.float32)
        raw[...,0]=rng.uniform(12,95,size=(9,5,15))
        raw[0,:,2,:]=0
        scores=np.linspace(.03,.92,9)
        for selected in [[1,3,5,7],[1,3,5,7,10,11,12]]:
            candidates=[q for q in range(15) if q not in selected]
            expected=np.vstack([from_revealed(raw,selected,scores,q) for q in candidates])
            actual=feature_matrix(raw,selected,scores,candidates)
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
            changed=raw.copy()
            changed[:,:,candidates,1:]+=10000
            np.testing.assert_array_equal(actual,feature_matrix(changed,selected,scores,candidates))
    def test_hidden_signal_and_labels_not_used(self):
        rng=np.random.default_rng(2);raw=rng.normal(size=(6,4,12,641));raw[...,0]=rng.uniform(14,90,size=(6,4,12))
        selected=[1,3,5,7];score=np.linspace(.1,.9,6);a=from_revealed(raw,selected,score,2)
        changed=raw.copy();unseen=[i for i in range(12) if i not in selected];changed[:,:,unseen,1:]+=1e5
        b=from_revealed(changed,selected,score,2);np.testing.assert_array_equal(a,b);self.assertEqual(len(a),len(FEATURES))
        # Altering an actual revealed waveform can change descriptive features.
        changed=raw.copy();changed[:,:,selected,-512:]*=10
        self.assertFalse(np.array_equal(a,from_revealed(changed,selected,score,2)))
    def test_unavailable_or_repeated_candidates_rejected(self):
        raw=np.ones((6,4,12,641));raw[...,0]=20;raw[:,:,2,0]=0;score=np.linspace(.1,.9,6)
        with self.assertRaises(ValueError):from_revealed(raw,[1,3,5,7],score,2)
        with self.assertRaises(ValueError):from_revealed(raw,[1,3,5,7],score,3)
    def test_target_labels_do_not_change_features_or_rollout_choices(self):
        from scipy.special import expit
        from .targets import rows_for_patient
        rng=np.random.default_rng(11);raw=rng.normal(size=(6,4,16,641));raw[...,0]=rng.uniform(14,90,size=(6,4,16))
        labels=np.array([0,0,0,1,1,1])
        def score(model,array,stats,selected):return expit(array[:,:,selected,-1].mean(axis=(1,2)))
        def policy(learner,x):return {'bag_mean':x[:,13]}
        for phase in [0,1]:
            with patch('experiments.spes_agent_v8.targets.score_subset',side_effect=score),patch('experiments.spes_agent_v8.targets.predict',side_effect=policy):
                a=rows_for_patient(raw,labels,None,[],0,7,phase,{});b=rows_for_patient(raw,1-labels,None,[],0,7,phase,{})
            np.testing.assert_array_equal(a['features'],b['features']);np.testing.assert_array_equal(a['prefix'],b['prefix']);np.testing.assert_array_equal(a['candidate'],b['candidate'])
            np.testing.assert_allclose(a['gain'],-b['gain'],atol=1e-12)

if __name__=='__main__':unittest.main()
