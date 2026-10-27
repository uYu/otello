"""Native Egaroucid with its opening book; no browser or network required."""
import ctypes
import json
import sys
from pathlib import Path

library = ctypes.CDLL(str(Path(sys.argv[1]).resolve()))
pointer = ctypes.POINTER(ctypes.c_int)
library.ot_egaroucid_create.argtypes = [ctypes.c_char_p]
library.ot_egaroucid_create.restype = ctypes.c_void_p
library.ot_egaroucid_search.argtypes = [ctypes.c_void_p, pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int, pointer]
library.ot_egaroucid_search.restype = ctypes.c_int
library.ot_egaroucid_destroy.argtypes = [ctypes.c_void_p]
library.ot_egaroucid_destroy.restype = None
library.ot_egaroucid_last_error.restype = ctypes.c_char_p
resources = sys.argv[2].encode() if len(sys.argv) > 2 else None
engine = library.ot_egaroucid_create(resources)
if not engine:
    raise RuntimeError(library.ot_egaroucid_last_error().decode())
try:
    cells, result = (ctypes.c_int * 64)(), (ctypes.c_int * 8)()
    cells[27] = cells[36] = -1
    cells[28] = cells[35] = 1
    if library.ot_egaroucid_search(engine, cells, 1, 5, 1, result) != 1:
        raise RuntimeError(library.ot_egaroucid_last_error().decode())
    print(json.dumps({"move": result[0], "value": result[1], "depth": result[2], "nodes": result[4], "bookHit": bool(result[5]), "useBook": bool(result[7])}))
finally:
    library.ot_egaroucid_destroy(engine)
