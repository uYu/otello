"""Use the native engine directly, without a browser, WebAssembly or network."""
import ctypes
import json
import sys
from pathlib import Path

library = ctypes.CDLL(str(Path(sys.argv[1]).resolve()))
pointer = ctypes.POINTER(ctypes.c_int)
library.ot_legal.argtypes = [pointer, ctypes.c_int, pointer]
library.ot_legal.restype = ctypes.c_int
library.ot_search.argtypes = [pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, pointer]
library.ot_search.restype = ctypes.c_int
cells = (ctypes.c_int * 64)()
cells[27] = cells[36] = -1
cells[28] = cells[35] = 1
moves, result = (ctypes.c_int * 64)(), (ctypes.c_int * 136)()
count = library.ot_legal(cells, 1, moves)
if library.ot_search(cells, 1, 500, 64, 2, 1, result) != 1:
    raise RuntimeError("Native search failed")
print(json.dumps({"legalMoves": list(moves[:count]), "move": result[0], "score": result[1], "depth": result[2], "nodes": result[4], "ttHits": result[135]}))
