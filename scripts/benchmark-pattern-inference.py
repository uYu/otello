#!/usr/bin/env python3
"""Interleaved Pattern joint-inference timing and deterministic MCTS parity."""
import ctypes,hashlib,json,statistics,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/pattern32-outcome-v1.25.otpat'
OUTPUT=ROOT/'models/v1.29-pattern-inference-benchmark.json'
paths=[ROOT/'.build/native-v1.25-reference/libothello.dylib',ROOT/'.build/native-pattern-h64-v1.29/libothello-pre-fusion.dylib',ROOT/'.build/native-pattern-h64-v1.29/libothello.dylib']
names=['approved v1.25','H64-capable before fusion','fused inference']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if OUTPUT.exists():raise ValueError('Refusing to overwrite benchmark')
data=np.load(ROOT/'.build/training/v1/positions.npz');boards=data['validation_board'][:64];turns=data['validation_turn'][:64]
ptr=ctypes.POINTER(ctypes.c_int);cells=[(ctypes.c_int*64)(*map(int,row)) for row in boards]
libs=[];handles=[];predictions=[]
try:
 for path in paths:
  lib=ctypes.CDLL(str(path));lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p;lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p];lib.ot_pattern_predict.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_double)];lib.ot_nnue_mcts_play.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]+[ctypes.c_int]*3+[ptr,ptr,ptr,ctypes.POINTER(ctypes.c_double)]
  handle=lib.ot_nnue_load(str(MODEL).encode());assert handle;libs.append(lib);handles.append(handle)
  out=(ctypes.c_double*68)();rows=[]
  for b,t in zip(cells,turns):assert lib.ot_pattern_predict(handle,b,int(t),out)==1;rows.append(list(out))
  predictions.append(rows)
 assert all(np.array_equal(predictions[0],row) for row in predictions[1:]),'Joint inference changed'
 samples=[[] for _ in libs];n=8192
 for round_index in range(9):
  for index in [(round_index+j)%len(libs) for j in range(len(libs))]:
   lib,handle=libs[index],handles[index];out=(ctypes.c_double*68)();tick=time.perf_counter_ns();checksum=0.
   for j in range(n):
    k=j%len(cells);assert lib.ot_pattern_predict(handle,cells[k],int(turns[k]),out)==1;checksum+=out[0]
   samples[index].append(dict(microseconds=(time.perf_counter_ns()-tick)/n/1000,checksum=checksum))
 # Fixed simulations remove clock-dependent branch selection from equivalence checks.
 search_results=[];search_samples=[]
 for lib,handle in zip(libs,handles):
  rows=[];tick=time.perf_counter_ns();nodes=0
  for b,t in zip(cells,turns):
   lib.ot_mcts_clear_cache();out,selection,visits,stats=(ctypes.c_int*136)(),(ctypes.c_int*6)(),(ctypes.c_int*64)(),(ctypes.c_double*72)()
   assert lib.ot_nnue_mcts_play(handle,b,int(t),0,1024,1832,12,32,64,out,selection,visits,stats)==1
   nodes+=stats[0];out[3]=0;stats[5]=0;rows.append([list(out),list(selection),list(visits),list(stats)])
  search_samples.append(dict(milliseconds=(time.perf_counter_ns()-tick)/1e6,simulations=int(nodes)));search_results.append(rows)
 assert all(row==search_results[0] for row in search_results[1:]),'Deterministic MCTS changed'
 results=[dict(name=name,library=str(path.relative_to(ROOT)),librarySha256=sha(path),medianMicroseconds=statistics.median(s['microseconds'] for s in rows),samples=rows,fixedSimulationSearch=search) for name,path,rows,search in zip(names,paths,samples,search_samples)]
 report=dict(model=str(MODEL.relative_to(ROOT)),modelSha256=sha(MODEL),scriptSha256=sha(Path(__file__)),positions=64,rounds=9,callsPerRound=n,jointOutputsExactlyEqual=True,fixedSimulationMctsExactlyEqual=True,simulationsPerPosition=1024,results=results,speedupVsApproved=results[0]['medianMicroseconds']/results[-1]['medianMicroseconds'],speedupVsPreFusion=results[1]['medianMicroseconds']/results[-1]['medianMicroseconds'],limits='Python/C ABI full-joint prediction overhead included; interleaved warm-cache timing; fixed-simulation timings are single passes, not strength matches')
 OUTPUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
finally:
 for lib,handle in zip(libs,handles):lib.ot_nnue_destroy(handle)
