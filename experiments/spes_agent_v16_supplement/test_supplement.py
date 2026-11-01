import unittest
import numpy as np
from .analysis import paired_summary,slate,pool_metrics


class DiagnosticTests(unittest.TestCase):
    def test_candidate_oracle_is_only_conditional_and_matches_random_pool_size(self):
        entry=slate(list(range(7)),list(range(7,20)),[[7,8,9],[8,9,10]],7,0,36,0)
        self.assertEqual(entry['proposal_pool'],[7,8,9,10]); self.assertTrue(all(len(pool)==4 and len(set(pool))==4 for pool in entry['matched_random_pools']))
        self.assertEqual(entry,slate(list(range(7)),list(range(7,20)),[[7,8,9],[8,9,10]],7,0,36,0))
        with self.assertRaises(ValueError): slate(list(range(6)),list(range(7,20)),[[7,8,9]],7,0,36,0)

    def test_random_oracle_averages_maxima_not_max_over_replicates(self):
        entry={'actual_final_query':7,'proposal_pool':[7,8],'matched_random_pools':[[7,9],[8,10]]}
        result=pool_metrics({7:.6,8:.8,9:.7,10:.9},entry,8)
        self.assertAlmostEqual(result['oracle_minus_actual'],.2)
        self.assertAlmostEqual(result['matched_random_oracle_mean_AUROC8'],.8)
        self.assertAlmostEqual(result['oracle_minus_matched_random_oracle'],0.)

    def test_resampling_unit_is_patient_difference(self):
        result=paired_summary([.1,.1,.1,.1,.1]); self.assertEqual(result['patients'],5)
        np.testing.assert_allclose(result['paired_95CI'],[.1,.1]); self.assertAlmostEqual(result['mean'],.1)
        with self.assertRaises(ValueError): paired_summary([[.1,.2],[.3,.4]])


if __name__=='__main__': unittest.main()
