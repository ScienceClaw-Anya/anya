import unittest
import numpy as np
from .neural import fixed_queries

class SparseProtocolTests(unittest.TestCase):
    def test_fixed_queries_match_sequential_random_policy(self):
        for patient in [7,21,53]:
            remaining=[0,2,3,4,5,6,7,8,9,12,14];expected=[]
            for step in range(8):
                seed=6190+2*100+patient+step*100003
                expected.append(remaining.pop(int(np.random.default_rng(seed).integers(len(remaining)))))
            self.assertEqual(fixed_queries([0,2,3,4,5,6,7,8,9,12,14],2,patient),expected)
    def test_training_and_validation_use_actual_whole_columns(self):
        try:import torch
        except ImportError:self.skipTest('Torch is required for tensor-mapping test')
        from .neural import sparse_batch
        x=torch.arange(3*4*12*5,dtype=torch.float32).reshape(3,4,12,5);patient=torch.tensor([7,21,7]);available={7:list(range(12)),21:list(range(8))}
        rng=np.random.default_rng(54);out=sparse_batch(x,patient,available,rng=rng)
        self.assertTrue(1<=out.shape[2]<=8)
        # Equal patient IDs in a training batch receive the same actual columns.
        offsets0=out[0,0,:,0]/5;offsets2=(out[2,0,:,0]-x[2,0,0,0])/5
        torch.testing.assert_close(offsets0,offsets2)
        fixed={7:list(range(8)),21:list(range(8))};out=sparse_batch(x,patient,available,fixed=fixed)
        torch.testing.assert_close(out,x[:,:,:8])

if __name__=='__main__':unittest.main()
