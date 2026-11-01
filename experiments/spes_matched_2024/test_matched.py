import unittest
from experiments.spes_matched_2024.experiment import disjoint, aggregate_cases


class MatchedTests(unittest.TestCase):
    def test_overlap_and_missing_patient_rejected(self):
        valid={'train':list(range(21)),'val':list(range(21,28)),'test':list(range(28,35))}
        disjoint(valid)
        with self.assertRaises(ValueError): disjoint(dict(valid,test=[0]+list(range(29,35))))

    def test_repeats_are_averaged_before_patient_bootstrap(self):
        rows=[{'patient':p,'seed':s,'baseline_AUROC':.5,'candidate_AUROC':.5+(.1 if p==1 else -.1)*s/3} for p in [1,2] for s in range(1,6)]
        r=aggregate_cases(rows)
        self.assertEqual(r['patients'],2); self.assertEqual(r['patient_repeat_cases'],10)
        self.assertAlmostEqual(r['paired_gain'],0)
        self.assertAlmostEqual(r['paired_nominal_95CI'][0],-.1)
        self.assertAlmostEqual(r['paired_nominal_95CI'][1],.1)
        with self.assertRaises(ValueError): aggregate_cases(rows[:-1])


if __name__=='__main__': unittest.main()
