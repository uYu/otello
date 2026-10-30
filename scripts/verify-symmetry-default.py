#!/usr/bin/env python3
"""Check embedded native weights and frozen-candidate deterministic parity."""
import ctypes,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/v1.37-default-check';WORK.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('symmetry_release_benchmark',ROOT/'scripts/benchmark-pattern-structure.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
active=json.loads((ROOT/'models/default-model.json').read_text());model=ROOT/active['defaultModel']
assert active['version']=='v1.37' and sha(model)==active['modelSha256']
suffix='dylib' if sys.platform=='darwin' else 'so'
new,handle=b.load(ROOT/f'.build/native/libothello.{suffix}',model)
old,previous=b.load(ROOT/f'.build/native-pattern-structure-v1.37/libothello.{suffix}',model)
ptr=ctypes.POINTER(ctypes.c_int);new.ot_default_model_sha256.restype=ctypes.c_char_p
new.ot_default_evaluate.argtypes=[ptr,ctypes.c_int,ptr]
assert new.ot_default_model_sha256().decode()==active['modelSha256']
data=np.load(ROOT/'.build/training/v1/positions.npz');fixture=[]
try:
 for split in ['validation','test']:
  boards,turns=data[split+'_board'],data[split+'_turn']
  expected=b.h.native(old,previous,boards,turns)
  actual=b.h.native(new,handle,boards,turns)
  assert np.array_equal(expected,actual),split
  for board,turn in zip(boards,turns):
   cells=(ctypes.c_int*64)(*map(int,board));out=(ctypes.c_int*136)();score=ctypes.c_int()
   assert new.ot_default_evaluate(cells,int(turn),out)==1
   new.ot_nnue_evaluate.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ptr]
   assert new.ot_nnue_evaluate(handle,cells,int(turn),ctypes.byref(score))==1
   assert out[0]==score.value
   if split=='validation':fixture.append(dict(board=board.tolist(),turn=int(turn),score=score.value))
 for board,turn in zip(data['validation_board'][:32],data['validation_turn'][:32]):
  assert b.search(old,previous,board,turn)[0]==b.search(new,handle,board,turn)[0]
finally:new.ot_nnue_destroy(handle);old.ot_nnue_destroy(previous)
(WORK/'fixture.json').write_text(json.dumps(dict(modelSha256=active['modelSha256'],positions=fixture)))
result=dict(nativeEmbeddedModelSha256=active['modelSha256'],nativeExplicitEmbeddedPositions=12000,all68OutputsExactlyMatchCandidate=True,fixedSimulationRoots=32,simulationsPerRoot=1024)
(WORK/'native.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
