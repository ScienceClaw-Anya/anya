import unittest
import numpy as np
from .tools import expand,pair_data,fit_models,predict,VIEWS


class RelativeRankingTests(unittest.TestCase):
    def test_state_offsets_cancel_without_inflating_loss_weight(self):
        x=np.arange(36).reshape(12,3)/20; gain=np.arange(12)/20; groups=[(0,)]*6+[(1,)]*6
        a,b,w=pair_data(x,gain,groups); aa,bb,ww=pair_data(x,gain+np.r_[np.ones(6)*.7,np.ones(6)*-.2],groups)
        np.testing.assert_allclose(a,aa); np.testing.assert_allclose(b,bb,atol=1e-15); np.testing.assert_array_equal(w,ww)
        self.assertEqual(len(b),30); self.assertAlmostEqual(float(w.sum()),12.)

    def test_expansion_has_separate_view_dimensions(self):
        z=np.arange(88).reshape(2,44)/40
        expected={'geometry':23,'neural':71,'waveform':223,'monolithic':384}
        for name,columns in VIEWS.items():
            result=expand(z,columns); self.assertEqual(result.shape,(2,expected[name])); np.testing.assert_array_equal(result[:,:len(columns)],z[:,columns])
        self.assertAlmostEqual(expand(z,list(range(44)))[0,44],z[0,0]*z[0,10])

    def test_model_pair_predictions_invariant_to_within_state_row_order(self):
        rng=np.random.default_rng(17); x=rng.normal(size=(12,44)); gain=rng.normal(size=12); groups=[(0,)]*6+[(1,)]*6
        a=fit_models(x,gain,groups); perm=np.r_[np.arange(6)[::-1],np.arange(6,12)[::-1]]
        b=fit_models(x[perm],gain[perm],[groups[i] for i in perm])
        for name in VIEWS: np.testing.assert_allclose(predict(a,x)[name],predict(b,x)[name],rtol=0,atol=1e-10)

    def test_incomplete_states_and_nonfinite_values_fail(self):
        with self.assertRaises(ValueError): pair_data(np.ones((5,3)),np.ones(5),[(0,)]*5)
        with self.assertRaises(ValueError): pair_data(np.ones((6,3)),np.full(6,np.nan),[(0,)]*6)


if __name__=='__main__': unittest.main()
