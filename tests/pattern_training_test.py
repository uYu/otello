import ctypes
import importlib.util
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
import pattern_model as pattern
spec = importlib.util.spec_from_file_location('pattern_rules', ROOT/'scripts/compare-resnet.py')
rules = importlib.util.module_from_spec(spec); spec.loader.exec_module(rules)


class PatternTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(1832)
        suffix = 'dylib' if sys.platform == 'darwin' else 'so'
        self.library = Path(os.environ.get('OTELLO_PATTERN_LIBRARY', ROOT/f'.build/native/libothello.{suffix}'))

    def test_wide_outcome_integer_export(self):
        rng=np.random.default_rng(1864)
        board=rng.integers(-1,2,(128,64),dtype=np.int8);turn=np.array([1,-1]*64,np.float32)
        net=pattern.PatternNet(hidden=64).eval()
        with torch.no_grad():
            expected=tuple(x.numpy() for x in net(torch.from_numpy(pattern.encode(board,turn).astype(np.int64)),torch.from_numpy((64-np.count_nonzero(board,axis=1)).astype(np.float32)/60)))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'h64.otpat';binary=pattern.export(net,path,outcome=True)
            self.assertEqual(struct.unpack_from('<I',binary,16)[0],64)
            self.assertEqual(len(binary),3585368)
            self.assertLess(pattern.parity(self.library,path,board,turn,expected)['maxPolicyLogitError'],2e-5)

    def test_outcome_metrics_use_result_sign_instead_of_disc_margin(self):
        spec = importlib.util.spec_from_file_location('pattern_outcome_training', ROOT/'scripts/train-pattern-outcome.py')
        trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
        values = np.array([1., -1., 0.],np.float32)
        wdl = np.array([[8.,0.,0.],[0.,8.,0.],[0.,0.,8.]],np.float32)
        logits = np.zeros((3,64),np.float32)
        mask = np.zeros((3,64),bool); mask[:,0]=True
        scores = trainer.metric((values,wdl,logits),np.array([.25,-.75,0.]),np.zeros(3,int),mask)
        self.assertEqual(scores['mse'],0)
        self.assertEqual(scores['nonDrawSignAccuracy'],1)
        self.assertEqual(scores['wdlAccuracy'],1)
        self.assertEqual(scores['policyTop1'],1)

    def test_masks_and_augmented_next_moves_against_rule_oracle(self):
        rng = np.random.default_rng(1832)
        board, turn = rules.initial(), 1
        grid = np.arange(64).reshape(8, 8)
        while rules.legal(board, turn) or rules.legal(board, -turn):
            legal = rules.legal(board, turn)
            mask = pattern.legal_mask(np.array([board], np.int8), np.array([turn]))[0]
            self.assertEqual(list(np.flatnonzero(mask)), legal)
            move = int(rng.choice(legal)) if legal else -1
            if move >= 0:
                for flip in (False, True):
                    for rotation in range(4):
                        perm = np.rot90(np.fliplr(grid) if flip else grid, rotation).reshape(64)
                        transformed = np.array(board)[perm].tolist()
                        transformed_move = int(np.argsort(perm)[move])
                        self.assertIn(transformed_move, rules.legal(transformed, turn))
            board, turn = rules.play(board, turn, move)

    def test_all_heads_integer_export_and_invalid_files(self):
        rng = np.random.default_rng(1832)
        board = rng.integers(-1, 2, (256, 64), dtype=np.int8)
        turn = np.array([1, -1]*128, np.float32)
        net = pattern.PatternNet().eval()
        # Exercise positive and negative quantized activations beyond initialization.
        with torch.no_grad():
            net.embedding.weight.mul_(8)
            net.head.bias.uniform_(-1, 1)
            codes = torch.from_numpy(pattern.encode(board, turn).astype(np.int64))
            phase = torch.from_numpy((64-np.count_nonzero(board, axis=1)).astype(np.float32)/60)
            expected = tuple(x.numpy() for x in net(codes, phase))
            swapped = tuple(x.numpy() for x in net(torch.from_numpy(pattern.encode(-board, -turn).astype(np.int64)), phase))
        for a, b in zip(expected, swapped):
            np.testing.assert_array_equal(a, b)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.otpat'
            binary = pattern.export(net, path)
            outcome_path = Path(directory)/'outcome.otpat'
            outcome_binary = pattern.export(net, outcome_path, outcome=True)
            self.assertEqual(outcome_binary[:8], b'OTPAT2\0\0')
            self.assertEqual(outcome_binary[8:], binary[8:])
            pattern.parity(self.library, outcome_path, board, turn, expected)
            lib = pattern.native_library(self.library)
            ptr = ctypes.POINTER(ctypes.c_int)
            lib.ot_nnue_mcts_play.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]+[ctypes.c_int]*3+[ptr, ptr, ptr, ctypes.POINTER(ctypes.c_double)]
            for model, expected_score in [(path, -106400), (outcome_path, -100000)]:
                handle = lib.ot_nnue_load(str(model).encode())
                self.assertTrue(handle)
                try:
                    cells = (ctypes.c_int*64)(*[1]*64)
                    out, selection, visits, stats = (ctypes.c_int*136)(), (ctypes.c_int*6)(), (ctypes.c_int*64)(), (ctypes.c_double*72)()
                    self.assertEqual(lib.ot_nnue_mcts_play(handle,cells,-1,0,16,1832,0,0,0,out,selection,visits,stats),1)
                    self.assertEqual(out[1],expected_score)
                    self.assertEqual(out[5],1)
                finally:
                    lib.ot_nnue_destroy(handle)
            result = pattern.parity(self.library, path, board, turn, expected)
            self.assertLess(result['maxPolicyLogitError'], 2e-5)
            lib = pattern.native_library(self.library)
            malformed = [binary[:-1], binary+b'extra', binary[:31]]
            for offset in range(8, 32, 4):
                bad = bytearray(binary); bad[offset:offset+4] = struct.pack('<I', 0); malformed.append(bytes(bad))
            bad = bytearray(binary)
            bias_offset = 32+2*(pattern.ROWS*pattern.DIM+pattern.HIDDEN*pattern.INPUTS)
            bad[bias_offset:bias_offset+4] = struct.pack('<i', 10000001); malformed.append(bytes(bad))
            for bad in malformed:
                path.write_bytes(bad)
                self.assertFalse(lib.ot_nnue_load(str(path).encode()))


if __name__ == '__main__':
    unittest.main()
