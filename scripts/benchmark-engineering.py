#!/usr/bin/env python3
"""Fixed-work search timing before/after engineering; models and choices unchanged."""
import importlib.util,json,time,statistics,ctypes
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('engineering_check',ROOT/'scripts/check-engineering.py');check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)
with np.load(ROOT/'.build/training/v1/positions.npz') as data:
 boards=data['test_board'][::500][:12];turns=data['test_turn'][::500][:12]
results=[]
for label,reference,model in [('v1.9','.build/native-v1.9-reference/libothello.dylib','models/nnue256-outcome-v1.9.otnnue'),('deep','.build/native-deep/libothello.dylib','models/nnue256-deep-v1.10.otnnue')]:
 libs=[check.load(reference),check.load('.build/native-engineering/libothello.dylib')]
 handles=[lib.ot_nnue_load(str(ROOT/model).encode()) for lib in libs];assert all(handles)
 times=[[],[]]
 try:
  for round in range(4):
   for index in ([0,1] if round%2==0 else [1,0]):
    lib,handle=libs[index],handles[index];elapsed=0
    for board,turn in zip(boards,turns):
     cells=(ctypes.c_int*64)(*map(int,board));lib.ot_mcts_clear_cache()
     start=time.perf_counter();check.search(lib,handle,cells,int(turn));elapsed+=time.perf_counter()-start
    if round:times[index].append(elapsed)
  old,new=map(statistics.median,times)
  results.append(dict(model=label,modelSha256=check.sha(model),referenceLibrarySha256=check.sha(reference),optimizedLibrarySha256=check.sha('.build/native-engineering/libothello.dylib'),referenceSeconds=times[0],optimizedSeconds=times[1],referenceMedianSeconds=old,optimizedMedianSeconds=new,speedup=old/new))
 finally:
  for lib,handle in zip(libs,handles):lib.ot_nnue_destroy(handle)
report=dict(scope='Same model, same positions, same 512-simulation budget; 12 cold searches per round; one warmup then three measured rounds with alternating engine order. Includes one ctypes request per search on both sides. Not match strength or a guaranteed time-budget speedup.',positions=12,simulations=512,results=results)
(ROOT/'.build/engineering/search-benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
