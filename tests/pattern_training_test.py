import ctypes
import importlib.util
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
        self.library = ROOT/f'.build/native-v1.8/libothello.{suffix}'

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
