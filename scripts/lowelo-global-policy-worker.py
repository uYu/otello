#!/usr/bin/env python3
"""JSON-lines adapter for explicit candidate Pattern logits (no default search)."""
import ctypes
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
lib = ctypes.CDLL(str(ROOT / '.build/native/libothello.dylib'))
ptr = ctypes.POINTER(ctypes.c_int)
lib.ot_nnue_load.argtypes = [ctypes.c_char_p]
lib.ot_nnue_load.restype = ctypes.c_void_p
lib.ot_nnue_destroy.argtypes = [ctypes.c_void_p]
lib.ot_nnue_error.restype = ctypes.c_char_p
lib.ot_pattern_predict.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
handle = lib.ot_nnue_load(str(ROOT / sys.argv[1]).encode())
assert handle, lib.ot_nnue_error().decode()
try:
    for line in sys.stdin:
        request = json.loads(line)
        board = request['board']
        assert len(board) == 64 and all(c in [-1, 0, 1] for c in board)
        assert request['turn'] in [-1, 1]
        out = (ctypes.c_double * 68)()
        assert lib.ot_pattern_predict(handle, (ctypes.c_int * 64)(*board), request['turn'], out) == 1
        print(json.dumps(dict(value=out[0], logits=list(out)[4:])), flush=True)
finally:
    lib.ot_nnue_destroy(handle)
