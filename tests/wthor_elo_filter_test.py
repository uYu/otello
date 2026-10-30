import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wthor_elo_filter import both_below,choose_candidate

class WthorFilterTests(unittest.TestCase):
    def test_only_both_known_and_strictly_below_are_removed(self):
        a=[1299,1300,1299,1500,np.nan,1200];b=[1299,1200,1300,1200,1200,np.nan]
        np.testing.assert_array_equal(both_below(a,b,1300),[True,False,False,False,False,False])
    def test_selection_enforces_both_retention_floors(self):
        def row(t,g,u,rate):return dict(threshold=t,gameRetentionVsWthor=g,uniqueRetentionVsWthor=u,theoryComparableGames=100,recordedVs24EmptyWdlDisagreementRate=rate,uniquePositions=int(u*1000))
        rows=[row(None,1,1,.1),row(1500,.9,.9,.23),row(1600,.64,.63,.22),row(1650,.55,.53,.21),row(1700,.7,.59,.20)]
        self.assertEqual(choose_candidate(rows)['threshold'],1600)
        with self.assertRaises(ValueError):choose_candidate([row(1800,.4,.4,.1)])
    def test_equal_occurrences_do_not_inherit_elo_weights(self):
        from training_consensus import KEY,aggregate
        key=np.array([(1,2),(1,2),(1,2)],dtype=KEY)
        data=aggregate(key,np.array([1,2,2],np.uint64),np.array([1,0,-1]),np.ones(3),np.zeros(3,np.uint8))
        np.testing.assert_array_equal(data['policy_mass'],[1,2]);np.testing.assert_array_equal(data['policy_quality_mass'],data['policy_mass'])
        np.testing.assert_array_equal(data['wdl_count'],[[1,1,1]])

if __name__=='__main__':unittest.main()
