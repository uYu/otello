import ctypes
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from pattern_model import encode, parity
from pattern_structure import StructureNet, canonical_rows, export


class StructureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        suffix = 'dylib' if sys.platform == 'darwin' else 'so'
        self.library = Path(os.environ.get('OTELLO_PATTERN_LIBRARY', ROOT/f'.build/native-pattern-structure-v1.37/libothello.{suffix}'))

    def test_full_ternary_table_orbits(self):
        mapping, representatives = canonical_rows()
        self.assertEqual(len(representatives), 28503)
        np.testing.assert_array_equal(mapping[representatives], np.arange(28503))
        for first, length, number in [(0, 8, 4), (26244, 4, 1), (26325, 5, 1), (26568, 6, 1), (27297, 7, 1), (29484, 8, 1), (36045, 9, 1)]:
            size = 3**length
            ids = np.arange(size)
            digits = ids[:, None]//(3**np.arange(length)) % 3
            transformed = digits.reshape(-1,3,3).transpose(0,2,1).reshape(-1,9) if length == 9 else digits[:, ::-1]
            changed = transformed @ (3**np.arange(length))
            for table in range(number):
                offset = first+table*size
                np.testing.assert_array_equal(mapping[offset+ids], mapping[offset+changed])

    def test_independent_export_and_native_predictions(self):
        rng = np.random.default_rng(37137)
        board = rng.integers(-1, 2, (256, 64), dtype=np.int8)
        turn = np.array([1, -1]*128, np.float32)
        phase = torch.from_numpy((64-np.count_nonzero(board, axis=1)).astype(np.float32)/60)
        codes = torch.from_numpy(encode(board,turn).astype(np.int64))
        for variant, rows, inputs, size in [('symmetry',28503,97,1833692), ('pool10',55728,321,3590428)]:
            torch.manual_seed(37137)
            net = StructureNet(variant).eval()
            with torch.no_grad():
                net.embedding.weight.mul_(8)
                net.head.bias.uniform_(-1,1)
                wanted = tuple(x.numpy() for x in net(codes, phase))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'test.otpat'
                binary = export(net,path)
                self.assertEqual(struct.unpack_from('<8s7I',binary), (b'OTPAT4\0\0', rows,32,32,inputs,46,256,1 if variant=='symmetry' else 2))
                self.assertEqual(len(binary), size)
                self.assertLess(parity(self.library,path,board,turn,wanted)['maxPolicyLogitError'], 2e-5)
            if variant == 'symmetry':
                grid = np.arange(64).reshape(8,8)
                with torch.no_grad():
                    for mirror in [False,True]:
                        for rotation in range(4):
                            perm = np.rot90(np.fliplr(grid) if mirror else grid,rotation).reshape(64)
                            prediction = tuple(x.numpy() for x in net(torch.from_numpy(encode(board[:,perm],turn).astype(np.int64)),phase))
                            np.testing.assert_array_equal(prediction[0],wanted[0])
                            np.testing.assert_array_equal(prediction[1],wanted[1])
                            np.testing.assert_allclose(prediction[2][:,np.argsort(perm)],wanted[2],atol=2e-6,rtol=0)
            else:
                self.assertEqual(tuple(net.pool.shape),(38,10))
                np.testing.assert_array_equal(net.counts.numpy(),[4,4,4,4,4,4,4,4,2,4])


if __name__ == '__main__':
    unittest.main()
