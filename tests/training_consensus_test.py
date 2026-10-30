import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from training_consensus import EloPool,expected_score,actor_weight,rate_records,canonical,aggregate,boards_from_keys,PERMS,INVERSE

class ConsensusTests(unittest.TestCase):
    def test_elo_expectation_and_zero_sum(self):
        p=EloPool(4);self.assertEqual(p.update(1,2,1),(0.5,16.));self.assertEqual(p.rating[1],1516);self.assertEqual(p.rating[2],1484)
        self.assertAlmostEqual(p.rating.sum(),6000);before=p.rating.copy();self.assertIsNone(p.update(0,1,1));self.assertIsNone(p.update(1,1,1));np.testing.assert_array_equal(before,p.rating)
        p=EloPool(4);self.assertEqual(p.update(1,2,.5),(0.5,0.))
        self.assertGreater(expected_score(1700,1500),.5);self.assertAlmostEqual(expected_score(1700,1500)+expected_score(1500,1700),1)
        self.assertGreater(32*(1-expected_score(1500,1700)),16)
        self.assertEqual(expected_score(0,1e9),0);self.assertEqual(expected_score(1e9,0),1)
    def test_holdout_and_own_outcome_cannot_change_training_weights(self):
        def game(i,year,split,result):return dict(gameId=str(i),year=year,rawRecordIndex=i,blackPlayerIndex=1,whitePlayerIndex=2,complete=True,split=split,blackResult=result)
        rows=[game(0,2000,'train',10),game(1,2000,'test',-10),game(2,2001,'train',-10),game(3,2001,'train',10)]
        a,_,p,_=rate_records(rows,4);changed=[dict(r,blackResult=-r['blackResult']) if r['split']=='test' else r for r in rows];b,_,q,_=rate_records(changed,4)
        np.testing.assert_array_equal(p.rating,q.rating)
        for x,y in zip(a,b):self.assertEqual(x['blackTrainYearStartElo'],y['blackTrainYearStartElo'])
        changed=[dict(r,blackResult=-r['blackResult']) if r['gameId']=='2' else r for r in rows];c,_,_,_=rate_records(changed,4)
        self.assertEqual(a[2]['blackTrainYearStartElo'],c[2]['blackTrainYearStartElo']);self.assertEqual(a[3]['blackTrainYearStartElo'],c[3]['blackTrainYearStartElo'])
        self.assertEqual(a[0]['blackTrainYearStartElo'],1500)
    def test_quality_is_neutral_without_evidence_and_monotonic(self):
        self.assertEqual(actor_weight(None,100),.5);self.assertEqual(actor_weight(2000,0),.5)
        self.assertLess(actor_weight(1200,30),actor_weight(1500,30));self.assertLess(actor_weight(1500,30),actor_weight(1800,30))
    def test_d4_and_color_invariance_with_action_stabilizers(self):
        rng=np.random.default_rng(40140);b=rng.integers(-1,2,(128,64),dtype=np.int8);t=rng.choice([-1,1],128).astype(np.int8);m=rng.integers(64,size=128)
        key,bits=canonical(b,t,m)
        for p,inv in zip(PERMS,INVERSE):
            k,v=canonical(b[:,p],t,inv[m]);np.testing.assert_array_equal(k,key);np.testing.assert_array_equal(v,bits)
        k,v=canonical(-b,-t,m);np.testing.assert_array_equal(k,key);np.testing.assert_array_equal(v,bits)
        board=np.zeros((1,64),np.int8);board[0,[28,35]]=1;board[0,[27,36]]=-1
        k,v=canonical(board,np.ones(1,np.int8),np.array([37]));self.assertEqual(int(v[0]).bit_count(),4)
        roundtrip,_=canonical(boards_from_keys(key),np.ones(len(key),np.int8));np.testing.assert_array_equal(roundtrip,key)
    def test_aggregate_keeps_conflicting_results_and_multiple_actions(self):
        board=np.zeros((3,64),np.int8);board[:,[28,35]]=1;board[:,[27,36]]=-1
        key,bits=canonical(board,np.ones(3,np.int8),np.array([37,26,19]))
        r=aggregate(key,bits,np.array([1,0,-1]),np.array([.25,.5,.75]),np.array([0,1,0]))
        np.testing.assert_array_equal(r['wdl_count'],[[1,1,1]]);np.testing.assert_array_equal(r['source_count'],[[2,1]])
        self.assertEqual(r['occurrence_count'][0],3);self.assertEqual(len(r['policy_action']),4)
        np.testing.assert_array_equal(r['policy_mass'],[.75]*4);np.testing.assert_allclose(r['policy_quality_mass'],[.375]*4)

    def test_targets_keep_soft_value_and_downweight_singleton(self):
        from consensus_training_targets import targets
        data=dict(occurrence_count=np.array([10,2]),wdl_count=np.array([[6,4,0],[0,1,1]]),policy_offsets=np.array([0,1,3]),policy_action=np.array([7,2,3]),policy_mass=np.array([10.,1.,1.]),policy_quality_mass=np.array([.5,.25,.75]),policy_quality_total=np.array([.5,1.]),matched_train_group=np.array([0,1]))
        t=targets(data)
        np.testing.assert_allclose(t['value_mean'],[.2,-.5]);np.testing.assert_allclose(t['policy_probability'],[1,.5,.5])
        np.testing.assert_allclose(t['policy_elo_probability'],[1,.25,.75]);np.testing.assert_allclose(t['policy_loss_weight_neutral_one'],[.1,1.])

if __name__=='__main__':unittest.main()
