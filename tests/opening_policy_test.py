import ctypes
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import opening_policy


class OpeningPolicyTests(unittest.TestCase):
    def test_symmetry_dedup_preserves_start_and_folds_equivalent_first_moves(self):
        self.assertEqual(len(opening_policy.SYMMETRIES), 4)
        self.assertEqual(len({opening_policy.canonical_moves([m]) for m in (19, 26, 37, 44)}), 1)
        self.assertEqual(opening_policy.transform(-1, 2), -1)

    def test_native_play_matches_independent_seeded_sampler(self):
        suffix = 'dylib' if sys.platform == 'darwin' else 'so'
        lib = ctypes.CDLL(str(ROOT / f'.build/native-v1.3/libothello.{suffix}'))
        ptr = ctypes.POINTER(ctypes.c_int)
        lib.ot_nnue_load.argtypes = [ctypes.c_char_p]
        lib.ot_nnue_load.restype = ctypes.c_void_p
        lib.ot_nnue_destroy.argtypes = [ctypes.c_void_p]
        lib.ot_nnue_play.argtypes = [ctypes.c_void_p, ptr] + [ctypes.c_int] * 4 + [ctypes.c_uint] + [ctypes.c_int] * 3 + [ptr, ptr]
        lib.ot_nnue_search.argtypes = [ctypes.c_void_p, ptr] + [ctypes.c_int] * 4 + [ctypes.c_uint, ptr]
        model = lib.ot_nnue_load(str(ROOT / 'models/nnue256-v1.2.otnnue').encode())
        self.assertTrue(model)
        board = [0] * 64
        board[28] = board[35] = 1
        board[27] = board[36] = -1
        cells = (ctypes.c_int * 64)(*board)
        out, meta = (ctypes.c_int * 136)(), (ctypes.c_int * 6)()
        choices = set()
        try:
            for seed in range(32):
                self.assertEqual(lib.ot_nnue_play(model, cells, 1, 0, 4, 2, seed, 12, 32, 64, out, meta), 1)
                candidates = [dict(move=out[7 + 2*i], score=out[8 + 2*i]) for i in range(out[6])]
                move, score, audited = opening_policy.choose(board, 1, candidates, seed, exact=bool(out[5]))
                self.assertEqual((out[0], out[1]), (move, score))
                self.assertEqual(list(meta), [audited[k] for k in ('eligible', 'temperatureMilli', 'lossLimit', 'bestScore', 'scoreLoss', 'sampled')])
                choices.add(move)
            self.assertGreater(len(choices), 1)
            self.assertEqual(lib.ot_nnue_play(model, cells, 1, 0, 4, 2, 7, 0, 32, 64, out, meta), 1)
            deterministic = (out[0], out[1])
            self.assertEqual(lib.ot_nnue_search(model, cells, 1, 0, 4, 2, 7, out), 1)
            self.assertEqual(deterministic, (out[0], out[1]))
            self.assertEqual(lib.ot_nnue_play(model, cells, 1, 0, 4, 2, 7, -1, 32, 64, out, meta), -1)
        finally:
            lib.ot_nnue_destroy(model)


if __name__ == '__main__':
    unittest.main()
