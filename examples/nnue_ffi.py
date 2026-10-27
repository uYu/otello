"""Offline C ABI: load the delivered model, evaluate and search; no torch needed."""
import ctypes
from pathlib import Path
import sys
import secrets

root = Path(__file__).resolve().parents[1]
suffix = 'dylib' if sys.platform == 'darwin' else 'so'
lib = ctypes.CDLL(str(root / f'.build/native/libothello.{suffix}'))
ptr = ctypes.POINTER(ctypes.c_int)
lib.ot_nnue_load.argtypes = [ctypes.c_char_p]
lib.ot_nnue_load.restype = ctypes.c_void_p
lib.ot_nnue_evaluate.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ptr]
lib.ot_nnue_search.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_uint, ptr]
lib.ot_nnue_mcts_play.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint] + [ctypes.c_int] * 3 + [ptr, ptr, ptr, ctypes.POINTER(ctypes.c_double)]
lib.ot_nnue_destroy.argtypes = [ctypes.c_void_p]
lib.ot_nnue_error.restype = ctypes.c_char_p
model = lib.ot_nnue_load(str(root / 'models/nnue256-v1.2.otnnue').encode())
if not model:
    raise RuntimeError(lib.ot_nnue_error().decode())
try:
    cells = (ctypes.c_int * 64)()
    cells[28] = cells[35] = 1
    cells[27] = cells[36] = -1
    score = ctypes.c_int()
    if lib.ot_nnue_evaluate(model, cells, 1, ctypes.byref(score)) != 1:
        raise RuntimeError(lib.ot_nnue_error().decode())
    output = (ctypes.c_int * 136)()
    policy = (ctypes.c_int * 6)()
    seed = secrets.randbits(31)
    visits, stats = (ctypes.c_int * 64)(), (ctypes.c_double * 72)()
    if lib.ot_nnue_mcts_play(model, cells, 1, 100, 0, seed, 12, 32, 64, output, policy, visits, stats) != 1:
        raise RuntimeError(lib.ot_nnue_error().decode())
    print(dict(seed=seed, openingSampled=bool(policy[5]), evaluation=score.value, move=output[0], score=output[1], search='mcts', maximumVisitedDepth=output[2], simulations=output[4]))
finally:
    lib.ot_nnue_destroy(model)
