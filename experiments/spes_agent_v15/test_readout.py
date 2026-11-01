import unittest
import numpy as np
from .experiment import design


class ReadoutTests(unittest.TestCase):
    def test_sparse_context_encodes_missingness_and_query_interaction(self):
        embedding=np.arange(48).reshape(3,16); fraction=np.array([0.,.5,1.])
        x=design(embedding,4,fraction,True)
        self.assertEqual(x.shape,(3,34)); np.testing.assert_array_equal(x[:,:16],embedding)
        np.testing.assert_array_equal(x[:,16:32],embedding*.5)
        np.testing.assert_array_equal(x[:,-2],.5*np.ones(3)); np.testing.assert_array_equal(x[:,-1],fraction)
        np.testing.assert_array_equal(design(embedding,8,fraction,False),embedding)

    def test_unaligned_or_nonfinite_feature_shapes_fail(self):
        with self.assertRaises(ValueError): design(np.zeros((3,15)),4,np.ones(3),True)
        with self.assertRaises(ValueError): design(np.full((3,16),np.nan),4,np.ones(3),True)
        with self.assertRaises(ValueError): design(np.zeros((3,16)),9,np.ones(3),True)
        with self.assertRaises(ValueError): design(np.zeros((3,16)),4,np.array([1.,1.,2.]),True)


if __name__=='__main__': unittest.main()
