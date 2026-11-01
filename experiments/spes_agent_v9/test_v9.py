import unittest
from unittest.mock import patch
import numpy as np
from .features import hypothetical_input,make_features,normalise_hypothesis,FEATURES,METHODS

class HypothesisIsolationTests(unittest.TestCase):
    def test_hypothesis_ignores_hidden_waveforms(self):
        rng=np.random.default_rng(43);raw=rng.normal(size=(8,4,15,641)).astype(np.float32);raw[...,0]=rng.uniform(10,90,size=raw[...,0].shape)
        selected=[1,3,5,7];candidate=2;changed=raw.copy();hidden=[i for i in range(15) if i not in selected];changed[:,:,hidden,1:]+=1e6
        for method in METHODS:
            a,aw=hypothetical_input(raw,selected,candidate,method);b,bw=hypothetical_input(changed,selected,candidate,method)
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(aw,bw)
            np.testing.assert_array_equal(a[:,:,:4],raw[:,:,selected]);self.assertEqual(a.shape[2],5)
    def test_features_respond_to_observed_data_only(self):
        rng=np.random.default_rng(9);raw=rng.normal(size=(8,4,15,641)).astype(np.float32);raw[...,0]=30
        selected=[1,3,5,7];scores=np.linspace(.1,.9,8);candidate=[2,4]
        def score(model,arrays,stats):return [1/(1+np.exp(-a[...,-1].mean(axis=(1,2)))) for a in arrays]
        with patch('experiments.spes_agent_v9.features.score_hypotheses',side_effect=score):
            a,_=make_features(raw,selected,scores,candidate,None,[])
            hidden=[i for i in range(15) if i not in selected];raw[:,:,hidden,1:]+=10000
            b,_=make_features(raw,selected,scores,candidate,None,[])
        np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(2,len(FEATURES)))
    def test_cropped_normalization_matches_original(self):
        rng=np.random.default_rng(63);raw=rng.normal(size=(8,4,5,641)).astype(np.float32);raw[0,:,1]=0
        stats=[{'mean':2.,'std':3.},{'mean':4.,'std':5.},{'mean':6.,'std':7.}]
        expected=raw.copy()
        for segment,stat in zip([slice(0,1),slice(1,129),slice(129,None)],stats):
            view=expected[...,segment];mask=np.abs(view)>1e-12;view[mask]=(view[mask]-stat['mean'])/stat['std']
        np.testing.assert_array_equal(normalise_hypothesis(raw,stats),np.concatenate([expected[...,:1],expected[...,-512:]],axis=-1))
    def test_repeated_candidate_rejected(self):
        raw=np.ones((8,4,15,641),dtype=np.float32)
        with self.assertRaises(ValueError):hypothetical_input(raw,[1,3,5,7],3,'profile_average')

if __name__=='__main__':unittest.main()
