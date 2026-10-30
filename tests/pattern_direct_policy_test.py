"""Geometry and trainability tests for the untrained direct policy prototype."""
from pathlib import Path
import sys
import os
import struct
import tempfile
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from pattern_model import encode, parity, native_library
from pattern_policy_mlp import DirectPolicyNet, export
from pattern_structure import StructureNet


class DirectPolicyTests(unittest.TestCase):
    def test_initial_value_path_matches_symmetry_control(self):
        torch.manual_seed(20261029);control=StructureNet('symmetry')
        torch.manual_seed(20261029);candidate=DirectPolicyNet()
        for a,b in [(control.embedding.weight,candidate.embedding.weight),(control.trunk.weight,candidate.trunk.weight),
                    (control.trunk.bias,candidate.trunk.bias),(control.head.weight[:4],candidate.head.weight),
                    (control.head.bias[:4],candidate.head.bias)]:
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_native_export_and_invalid_formats(self):
        torch.set_num_threads(2);torch.manual_seed(3819)
        rng=np.random.default_rng(3819)
        board=rng.integers(-1,2,(256,64),dtype=np.int8);turn=np.array([1,-1]*128,np.int8)
        phase=torch.from_numpy((64-np.count_nonzero(board,axis=1)).astype(np.float32)/60)
        net=DirectPolicyNet().eval()
        with torch.no_grad():
            net.embedding.weight.mul_(8)
            expected=tuple(x.numpy() for x in net(torch.from_numpy(encode(board,turn).astype(np.int64)),phase))
        suffix='dylib' if sys.platform=='darwin' else 'so'
        library=Path(os.environ.get('OTELLO_DIRECT_POLICY_LIBRARY',ROOT/f'.build/native-pattern-direct-policy-v1.38/libothello.{suffix}'))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'direct.otpat';binary=export(net,path)
            self.assertEqual(struct.unpack_from('<8s8I',binary),(b'OTPAT5\0\0',28503,32,32,97,4,256,74,8))
            self.assertEqual(len(binary),1832076)
            self.assertLess(parity(library,path,board,turn,expected)['maxPolicyLogitError'],2e-5)
            lib=native_library(library)
            malformed=[binary[:-1],binary+b'extra']
            for offset in range(8,40,4):
                bad=bytearray(binary);struct.pack_into('<I',bad,offset,0);malformed.append(bytes(bad))
            for bad in malformed:
                path.write_bytes(bad);self.assertFalse(lib.ot_nnue_load(str(path).encode()))

    def test_widths_export_native_and_symmetry(self):
        torch.set_num_threads(2)
        rng=np.random.default_rng(39139)
        board=rng.integers(-1,2,(128,64),dtype=np.int8);turn=np.array([1,-1]*64,np.int8)
        phase=torch.from_numpy((64-np.count_nonzero(board,axis=1)).astype(np.float32)/60)
        suffix='dylib' if sys.platform=='darwin' else 'so'
        library=Path(os.environ.get('OTELLO_DIRECT_POLICY_LIBRARY',ROOT/f'.build/native-pattern-policy-width-v1.39/libothello.{suffix}'))
        for width in [8,16,32]:
            torch.manual_seed(20261029);control=StructureNet('symmetry')
            torch.manual_seed(20261029);net=DirectPolicyNet(width).eval()
            for a,b in [(control.embedding.weight,net.embedding.weight),(control.trunk.weight,net.trunk.weight),
                        (control.trunk.bias,net.trunk.bias),(control.head.weight[:4],net.head.weight),
                        (control.head.bias[:4],net.head.bias)]:torch.testing.assert_close(a,b,rtol=0,atol=0)
            with torch.no_grad():
                net.embedding.weight.mul_(8)
                expected=net(torch.from_numpy(encode(board,turn).astype(np.int64)),phase)
                for mirror in [False,True]:
                    for rotation in range(4):
                        grid=np.arange(64).reshape(8,8)
                        perm=np.rot90(np.fliplr(grid) if mirror else grid,rotation).reshape(64)
                        actual=net(torch.from_numpy(encode(board[:,perm],turn).astype(np.int64)),phase)
                        for a,b in zip(actual[:2],expected[:2]):torch.testing.assert_close(a,b,rtol=0,atol=0)
                        torch.testing.assert_close(actual[2][:,np.argsort(perm)],expected[2],rtol=0,atol=0)
                with tempfile.TemporaryDirectory() as directory:
                    path=Path(directory)/'candidate.otpat';binary=export(net,path)
                    self.assertEqual(struct.unpack_from('<I',binary,36)[0],width)
                    self.assertEqual(len(binary),1832076+(width-8)*154)
                    self.assertLess(parity(library,path,board,turn,tuple(x.numpy() for x in expected))['maxPolicyLogitError'],2e-5)
        with self.assertRaises(ValueError):DirectPolicyNet(9)

    def test_existing_mlp8_export_unchanged(self):
        torch.set_num_threads(2)
        net=DirectPolicyNet()
        net.load_state_dict(torch.load(ROOT/'models/pattern32-direct-mlp8-v1.38.pt',weights_only=True))
        with tempfile.TemporaryDirectory() as directory:
            binary=export(net,Path(directory)/'regression.otpat')
            self.assertEqual(binary,(ROOT/'models/pattern32-direct-mlp8-v1.38.otpat').read_bytes())

    def test_shared_head_is_trainable_and_d4_equivariant(self):
        torch.set_num_threads(2);torch.manual_seed(3818)
        rng=np.random.default_rng(3818)
        board=rng.integers(-1,2,(64,64),dtype=np.int8);turn=np.array([1,-1]*32,np.int8)
        phase=torch.from_numpy((64-np.count_nonzero(board,axis=1)).astype(np.float32)/60)
        net=DirectPolicyNet()
        self.assertEqual(sum(p.numel() for p in net.policy_mlp.parameters()),609)
        code=torch.from_numpy(encode(board,turn).astype(np.int64))
        value,wdl,logits=net(code,phase)
        self.assertEqual(tuple(logits.shape),(64,64))
        loss=(value.square().mean()+wdl.square().mean()+logits.square().mean())
        loss.backward()
        self.assertGreater(net.embedding.weight.grad.abs().sum().item(),0)
        self.assertGreater(net.policy_mlp[0].weight.grad.abs().sum().item(),0)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001);optimizer.step()
        with torch.no_grad():
            expected=net(code,phase)
            renamed=net(torch.from_numpy(encode(-board,-turn).astype(np.int64)),phase)
            for a,b in zip(expected,renamed):torch.testing.assert_close(a,b,rtol=0,atol=0)
            grid=np.arange(64).reshape(8,8)
            for mirror in [False,True]:
                for rotation in range(4):
                    perm=np.rot90(np.fliplr(grid) if mirror else grid,rotation).reshape(64)
                    actual=net(torch.from_numpy(encode(board[:,perm],turn).astype(np.int64)),phase)
                    torch.testing.assert_close(actual[0],expected[0],rtol=0,atol=0)
                    torch.testing.assert_close(actual[1],expected[1],rtol=0,atol=0)
                    torch.testing.assert_close(actual[2][:,np.argsort(perm)],expected[2],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
