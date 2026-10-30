import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from training_consensus import canonical,PERMS
from mixed_label_targets import canonical_transforms,soft_targets

class MixedTargetTests(unittest.TestCase):
    def test_canonical_keys_and_transform_direction(self):
        rng=np.random.default_rng(45145);board=rng.integers(-1,2,(256,64),dtype=np.int8);turn=rng.choice([-1,1],256)
        key,sym=canonical_transforms(board,turn);np.testing.assert_array_equal(key,canonical(board,turn)[0])
        aligned=np.take_along_axis(board*turn[:,None],PERMS[sym],axis=1)
        actual,_=canonical_transforms(aligned,np.ones(256,np.int8));np.testing.assert_array_equal(actual,key)
        self.assertEqual(tuple(sym.shape),(256,))
    def test_reorder_and_inverse_policy_alignment(self):
        data=dict(occurrence_count=np.array([4,2]),wdl_count=np.array([[2,1,1],[0,2,0]]),policy_offsets=np.array([0,2,3]),policy_action=np.array([5,7,63]),policy_mass=np.array([1.,3.,2.]))
        sym=np.array([3,5]);t=soft_targets(data,np.array([1,0]),sym)
        self.assertEqual(t['policy'][0,PERMS[3,63]],1);self.assertEqual(t['policy'][1,PERMS[5,5]],.25);self.assertEqual(t['policy'][1,PERMS[5,7]],.75)
        np.testing.assert_array_equal(t['policy'].sum(1),[1,1]);np.testing.assert_array_equal(t['value'],[-1,.25]);np.testing.assert_array_equal(t['wdl'],[[0,1,0],[.5,.25,.25]])

if __name__=='__main__':unittest.main()
