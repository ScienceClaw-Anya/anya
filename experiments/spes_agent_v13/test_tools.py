import unittest
import numpy as np
from scipy.special import expit
from .tools import features, boundary_offset, uncertainty_weights, acquisition_table, choose


class ToolTests(unittest.TestCase):
    def test_prior_odds_double_class_weight(self):
        true = np.array([.05,.2,.6]); r = 5.
        weighted = (r*r*true)/(1-true+r*r*true)
        np.testing.assert_allclose(expit(features(weighted,'prior_platt',r)),true,atol=1e-12)

    def test_patient_shift_invariance_and_learned_boundary(self):
        score = expit(np.array([-1.,0.,1.,2.,3.,4.])); labels = np.array([0,0,0,1,1,1])
        shifted = expit(np.array([-1.,0.,1.,2.,3.,4.])+2)
        np.testing.assert_allclose(features(score,'centered_platt',5), features(shifted,'centered_platt',9),atol=1e-12)
        self.assertAlmostEqual(boundary_offset(score,labels),boundary_offset(shifted,labels))
        n = boundary_offset(score,labels)
        w = uncertainty_weights(np.column_stack([score]*3),'learned_boundary',5,n)
        self.assertEqual(int(np.argmax(w)),3)

    def test_observation_only_selection_and_unique_pair_budget(self):
        kernel = np.array([[1.,.1,.2],[.1,.5,.9]])
        matrix = np.array([[.2,.4,.5],[.8,.6,.9]])
        rows = acquisition_table(kernel,[0],[0,1,2],matrix,'zero_boundary',5,0)
        self.assertEqual({r['id'] for r in rows},{'p001','p002'})
        self.assertEqual(choose(rows),'p002')
        self.assertEqual(choose([{'id':'p002','gain':1},{'id':'p001','gain':1}]),'p001')

    def test_unknown_recipes_and_nonfinite_scores_fail(self):
        with self.assertRaises(ValueError): features([np.nan],'raw_platt',5)
        with self.assertRaises(ValueError): features([.2],'invented',5)
        with self.assertRaises(ValueError): uncertainty_weights(np.ones((3,2)),'prior_uncertainty',5,0)


if __name__ == '__main__': unittest.main()
