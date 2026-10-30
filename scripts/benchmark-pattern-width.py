#!/usr/bin/env python3
"""Measure trained width variants after training has completed."""
import ctypes,hashlib,json,statistics,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'models/v1.29-pattern-width-benchmark.json'
assert not OUTPUT.exists(),'Refusing to overwrite benchmark'
training=json.loads((ROOT/'models/v1.29-pattern-width.json').read_text())
libpath=ROOT/'.build/native-pattern-h64-v1.29/libothello.dylib'
assert hashlib.sha256(libpath.read_bytes()).hexdigest()==training['librarySha256']
lib=ctypes.CDLL(str(libpath));ptr=ctypes.POINTER(ctypes.c_int)
lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p;lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p];lib.ot_pattern_predict.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
paths=[ROOT/'models/pattern32-outcome-v1.25.otpat']+[ROOT/v['model'] for v in training['variants']]
handles=[];data=np.load(ROOT/'.build/training/v1/positions.npz');boards=[(ctypes.c_int*64)(*map(int,b)) for b in data['validation_board'][:64]];turns=data['validation_turn'][:64]
try:
 for p in paths:
  handle=lib.ot_nnue_load(str(p).encode());assert handle;handles.append(handle)
 samples=[[] for _ in paths];n=8192
 for round_index in range(9):
  for i in [(round_index+j)%len(paths) for j in range(len(paths))]:
   out=(ctypes.c_double*68)();tick=time.perf_counter_ns();checksum=0.
   for j in range(n):
    k=j%64;assert lib.ot_pattern_predict(handles[i],boards[k],int(turns[k]),out)==1;checksum+=out[0]
   samples[i].append(dict(microseconds=(time.perf_counter_ns()-tick)/n/1000,checksum=checksum))
 rows=[dict(model=str(p.relative_to(ROOT)),modelSha256=hashlib.sha256(p.read_bytes()).hexdigest(),medianMicroseconds=statistics.median(s['microseconds'] for s in samples[i]),samples=samples[i]) for i,p in enumerate(paths)]
 report=dict(library=str(libpath.relative_to(ROOT)),librarySha256=training['librarySha256'],scriptSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),positions=64,rounds=9,callsPerRound=n,results=rows,limits='Interleaved full-joint C ABI inference including Python overhead; same fused native implementation for all three models; no playing-strength claim')
 OUTPUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([(r['model'],r['medianMicroseconds']) for r in rows]))
finally:
 for h in handles:lib.ot_nnue_destroy(h)
