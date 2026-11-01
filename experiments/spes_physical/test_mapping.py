import unittest
import numpy as np
from scipy.spatial.distance import cdist
from .mapping import match_pairs,physical_input

class PhysicalMappingTests(unittest.TestCase):
    def test_different_compact_columns_recover_one_physical_pair(self):
        coords=np.array([[0,0,0],[7,2,1],[1,11,3]],dtype=float);centers=np.array([[22,4,2],[5,29,9],[32,28,7]],dtype=float)
        orders=np.array([[0,1,2],[1,2,0],[2,0,1]]);distance=cdist(coords,centers);compact=np.take_along_axis(distance,orders,axis=1)
        index,audit=match_pairs(compact,coords,centers);self.assertTrue(audit['columns'][0]['conflicting_verified_pairs'])
        raw=np.zeros((3,2,3,641),dtype=np.float32)
        for i in range(3):
            for col,pair in enumerate(orders[i]):raw[i,:,col,0]=compact[i,col];raw[i,:,col,1:]=100*i+pair
        observed=physical_input(raw,index,[1]);np.testing.assert_array_equal(observed[:,0,0,-1],[1,101,201])
        for i in range(3):
            for col,pair in enumerate(orders[i]):
                if pair!=1:raw[i,:,col,1:]+=10000
        np.testing.assert_array_equal(observed,physical_input(raw,index,[1]))
    def test_ambiguous_geometry_is_rejected(self):
        index,audit=match_pairs(np.array([[20.]]),np.array([[0.,0.,0.]]),np.array([[20.,0.,0.],[-20.,0.,0.]]))
        self.assertEqual(audit['ambiguous_cells'],1);self.assertTrue((index==-1).all())
    def test_duplicate_pair_conditions_are_not_silently_pooled(self):
        index,audit=match_pairs(np.array([[20.,20.]]),np.array([[0.,0.,0.]]),np.array([[20.,0.,0.]]))
        self.assertEqual(index[0,0],-2);self.assertEqual(audit['duplicate_recording_pair_assignments'],1)
        self.assertFalse(physical_input(np.ones((1,2,2,641)),index,[0]).any())
    def test_trial_geometry_changes_are_rejected(self):
        raw=np.ones((1,2,2,641));raw[0,1,0,0]=20
        with self.assertRaises(ValueError):physical_input(raw,np.array([[0]]),[0])

if __name__=='__main__':unittest.main()
