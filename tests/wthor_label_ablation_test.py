import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from training_consensus import KEY
from wthor_label_ablation import first_labels,policy_targets

class LabelAblationTests(unittest.TestCase):
    def test_first_means_first_in_frozen_occurrence_order_not_majority(self):
        keys=np.array([(9,2),(1,2),(1,2),(9,2),(1,2),(1,2)],dtype=KEY)
        occ=dict(keys=keys,source=np.array([0,1,0,0,0,0]),outcome=np.array([-1,1,0,1,-1,-1]),action_bits=np.array([1<<63,1,6,8,4,4],np.uint64),game_index=np.arange(6))
        expected=np.array([(1,2),(9,2)],dtype=KEY);hard=first_labels(occ,expected)
        np.testing.assert_array_equal(hard['first_occurrence'],[2,0]);np.testing.assert_array_equal(hard['value'],[0,-1]);np.testing.assert_array_equal(hard['wdl'],[[0,0,1],[0,1,0]])
        np.testing.assert_array_equal(hard['action'],[1,63])
    def test_target_switch_keeps_rows_and_probability_mass(self):
        data=dict(keys=np.array([(1,2),(9,2)],dtype=KEY),policy_offsets=np.array([0,2,3]),policy_action=np.array([1,2,63]),policy_probability=np.array([.25,.75,1],np.float32))
        hard=dict(action=np.array([1,63]));a=policy_targets(data,hard,'hard');b=policy_targets(data,hard,'soft')
        self.assertEqual(a.shape,b.shape);np.testing.assert_array_equal(a.sum(1),[1,1]);np.testing.assert_array_equal(b.sum(1),[1,1]);self.assertEqual(a[0,1],1);self.assertEqual(b[0,1],.25)
    def test_position_set_mismatch_rejected(self):
        occ=dict(keys=np.array([(1,2)],dtype=KEY),source=np.array([0]),outcome=np.array([1]),action_bits=np.array([1],np.uint64),game_index=np.array([0]))
        with self.assertRaises(ValueError):first_labels(occ,np.array([(2,3)],dtype=KEY))

if __name__=='__main__':unittest.main()
