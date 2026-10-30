import sys
from pathlib import Path
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from pattern_model import encode
from pattern_structure import StructureNet
from pattern_global_policy import GlobalPolicyNet,permutations


class GlobalPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        data=np.load(ROOT/'.build/training/v1/positions.npz')
        self.board=data['validation_board'][:512];self.turn=data['validation_turn'][:512]
        self.codes=torch.from_numpy(encode(self.board,self.turn).astype(np.int64))
        self.phase=torch.from_numpy((64-np.count_nonzero(self.board,axis=1)).astype(np.float32)/60)
        self.state=torch.load(ROOT/'models/pattern32-symmetry-v1.37.pt',weights_only=True)

    def test_identity_frozen_value_and_parameter_budget(self):
        base=StructureNet('symmetry');base.load_state_dict(self.state)
        expected=base(self.codes,self.phase)
        for channels,count in [(1,237),(2,464)]:
            net=GlobalPolicyNet(channels,self.state)
            self.assertEqual(sum(p.numel() for p in net.parameters() if p.requires_grad),count)
            actual=net(self.codes,self.phase)
            for a,b in zip(expected,actual):torch.testing.assert_close(a,b,atol=0,rtol=0)
            actual[2].square().mean().backward()
            self.assertGreater(float(net.global_up.coefficients.grad.abs().sum()),0)
            self.assertTrue(all(p.grad is None for name,p in net.named_parameters() if not name.startswith('global_')))

    def test_nonzero_head_preserves_d4_and_value(self):
        for channels in [1,2]:
            net=GlobalPolicyNet(channels,self.state)
            with torch.no_grad():
                for name,p in net.named_parameters():
                    if name.startswith('global_'):p.uniform_(-.2,.2)
                expected=net(self.codes,self.phase)
                for perm in permutations(8):
                    actual=net(torch.from_numpy(encode(self.board[:,perm],self.turn).astype(np.int64)),self.phase)
                    torch.testing.assert_close(actual[0],expected[0],atol=0,rtol=0)
                    torch.testing.assert_close(actual[1],expected[1],atol=0,rtol=0)
                    torch.testing.assert_close(actual[2][:,np.argsort(perm)],expected[2],atol=2e-6,rtol=0)
                base=StructureNet('symmetry');base.load_state_dict(self.state);original=base(self.codes,self.phase)
                torch.testing.assert_close(expected[0],original[0],atol=0,rtol=0)
                torch.testing.assert_close(expected[1],original[1],atol=0,rtol=0)

    def test_cached_features_match_full_policy(self):
        net=GlobalPolicyNet(1,self.state)
        with torch.no_grad():
            net.global_up.coefficients.uniform_(-.2,.2)
            value,wdl,projection,base=net.frozen_features(self.codes,self.phase)
            actual=net(self.codes,self.phase)
            torch.testing.assert_close(actual[0],value,atol=0,rtol=0)
            torch.testing.assert_close(actual[1],wdl,atol=0,rtol=0)
            torch.testing.assert_close(actual[2],base+net.policy_correction(projection),atol=0,rtol=0)

    def test_full_training_updates_shared_encoder(self):
        net=GlobalPolicyNet(1,freeze_baseline=False)
        self.assertEqual(sum(p.numel() for p in net.parameters() if p.requires_grad),916987)
        value,wdl,policy=net(self.codes,self.phase)
        loss=value.square().mean()+wdl.square().mean()+policy.square().mean();loss.backward()
        for parameter in [net.embedding.weight,net.trunk.weight,net.head.weight,net.global_up.coefficients]:
            self.assertGreater(float(parameter.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
