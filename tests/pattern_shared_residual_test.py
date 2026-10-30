import ctypes
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from pattern_model import encode,parity
from pattern_structure import StructureNet
from pattern_shared_residual import SharedResidualNet,export
import shared_residual_experiment as e


class SharedResidualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.library=Path(os.environ.get('OTELLO_PATTERN_LIBRARY',str(e.LIB)))
        rng=np.random.default_rng(48148)
        self.boards=rng.integers(-1,2,(512,64),dtype=np.int8)
        self.turns=np.array([1,-1]*256,np.float32)
        self.codes=torch.from_numpy(encode(self.boards,self.turns).astype(np.int64))
        self.phase=torch.from_numpy((64-np.count_nonzero(self.boards,axis=1)).astype(np.float32)/60)

    def test_identity_initialization_and_trainable_residual(self):
        torch.manual_seed(20261029);base=StructureNet('symmetry')
        torch.manual_seed(20261029);net=SharedResidualNet(16)
        for key,value in base.state_dict().items():self.assertTrue(torch.equal(value,net.state_dict()[key]),key)
        expected=base(self.codes,self.phase);actual=net(self.codes,self.phase)
        for a,b in zip(expected,actual):torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertEqual(sum(p.numel() for p in net.parameters()),917822)
        (actual[0].square().sum()+actual[2].square().sum()).backward()
        self.assertGreater(float(net.residual_up.weight.grad.abs().sum()),0)
        with torch.no_grad():net.residual_up.weight.uniform_(-.2,.2)
        net.zero_grad();out=net(self.codes,self.phase);(out[0].square().sum()+out[2].square().sum()).backward()
        self.assertGreater(float(net.residual_down.weight.grad.abs().sum()),0)

    def test_native_saturated_residual_symmetry_and_bad_files(self):
        for width in [8,16]:
            torch.manual_seed(48148);net=SharedResidualNet(width).eval()
            with torch.no_grad():
                net.embedding.weight.mul_(8)
                net.residual_down.weight.uniform_(-1,1);net.residual_down.bias.uniform_(-1,1)
                net.residual_up.weight.uniform_(-1,1);net.residual_up.bias.uniform_(-1,1)
                prediction=tuple(x.numpy() for x in net(self.codes,self.phase))
                grid=np.arange(64).reshape(8,8)
                for mirror in [False,True]:
                    for rotation in range(4):
                        perm=np.rot90(np.fliplr(grid) if mirror else grid,rotation).reshape(64)
                        transformed=tuple(x.numpy() for x in net(torch.from_numpy(encode(self.boards[:,perm],self.turns).astype(np.int64)),self.phase))
                        np.testing.assert_array_equal(transformed[0],prediction[0]);np.testing.assert_array_equal(transformed[1],prediction[1])
                        np.testing.assert_allclose(transformed[2][:,np.argsort(perm)],prediction[2],atol=2e-6,rtol=0)
                swapped=net(torch.from_numpy(encode(-self.boards,-self.turns).astype(np.int64)),self.phase)
                for a,b in zip(swapped,prediction):np.testing.assert_array_equal(a.numpy(),b)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'model.otpat';binary=export(net,path)
                self.assertEqual(len(binary),1833692+2*64*width+4*(width+32)+4)
                self.assertLess(parity(self.library,path,self.boards,self.turns,prediction)['maxPolicyLogitError'],2e-5)
                lib=ctypes.CDLL(str(self.library));lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p
                damaged=bytearray(binary);struct.pack_into('<I',damaged,36,15)
                for i,data in enumerate([binary[:-1],binary+b'X',damaged]):
                    bad=Path(folder)/f'bad-{i}.otpat';bad.write_bytes(data)
                    self.assertFalse(lib.ot_nnue_load(str(bad).encode()))

    def test_residual_parameters_invalidate_search_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            torch.manual_seed(48148);net=SharedResidualNet(16)
            a=Path(folder)/'a.otpat';export(net,a)
            with torch.no_grad():net.residual_up.weight.uniform_(-1,1)
            b=Path(folder)/'b.otpat';export(net,b)
            pairs=[e.load(self.library,p) for p in [a,b]]
            try:
                board=np.zeros(64,np.int8);board[[28,35]]=1;board[[27,36]]=-1
                e.search(pairs[0],board,1,simulations=128)
                reused,_=e.search(pairs[0],board,1,simulations=128,clear=False)
                changed,_=e.search(pairs[1],board,1,simulations=128,clear=False)
                self.assertGreater(reused[3][3],0);self.assertEqual(changed[3][3],0)
            finally:
                for lib,handle in pairs:lib.ot_nnue_destroy(handle)


if __name__=='__main__':unittest.main()
