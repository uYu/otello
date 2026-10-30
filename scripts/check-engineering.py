#!/usr/bin/env python3
"""Check optimized inference and fixed-work MCTS against frozen native libraries."""
import ctypes,hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PTR=ctypes.POINTER(ctypes.c_int)
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def load(path):
 lib=ctypes.CDLL(str(ROOT/path));lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p
 lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p];lib.ot_nnue_evaluate.argtypes=[ctypes.c_void_p,PTR,ctypes.c_int,PTR]
 lib.ot_nnue_policy.argtypes=[ctypes.c_void_p,PTR,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
 lib.ot_nnue_mcts_play.argtypes=[ctypes.c_void_p,PTR,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]+[ctypes.c_int]*3+[PTR,PTR,PTR,ctypes.POINTER(ctypes.c_double)]
 return lib

def search(lib,handle,board,turn):
 out=(ctypes.c_int*136)();p=(ctypes.c_int*6)();visits=(ctypes.c_int*64)();stats=(ctypes.c_double*72)()
 assert lib.ot_nnue_mcts_play(handle,board,turn,0,512,20261028,12,32,64,out,p,visits,stats)==1
 return (list(out)[:3],list(out)[5:135],list(p),list(visits),list(stats)[:5],list(stats)[6:])

def main():
 optimized='.build/native-engineering/libothello.dylib';new=load(optimized)
 new.ot_nnue_infer.argtypes=[ctypes.c_void_p,PTR,ctypes.c_int,PTR,ctypes.POINTER(ctypes.c_double)]
 new.ot_legal.argtypes=[PTR,ctypes.c_int,PTR];new.ot_play.argtypes=[PTR,ctypes.c_int,ctypes.c_int,PTR]
 with np.load(ROOT/'.build/training/v1/positions.npz') as data:
  boards=data['test_board'][:1024];turns=data['test_turn'][:1024]
 result=[]
 for label,reference,model in [('v1.9','.build/native-v1.9-reference/libothello.dylib','models/nnue256-outcome-v1.9.otnnue'),('deep','.build/native-deep/libothello.dylib','models/nnue256-deep-v1.10.otnnue')]:
  old=load(reference);a=old.ot_nnue_load(str(ROOT/model).encode());b=new.ot_nnue_load(str(ROOT/model).encode());assert a and b
  value_error=policy_error=0
  try:
   specials=[(np.ones(64,dtype=np.int8),-1),(np.array([1]*32+[-1]*32,dtype=np.int8),1)]
   for board,turn in list(zip(boards,turns))+specials:
    cells=(ctypes.c_int*64)(*map(int,board));turn=int(turn);v=ctypes.c_int();fused=ctypes.c_int();p,q=(ctypes.c_double*65)(),(ctypes.c_double*65)()
    assert old.ot_nnue_evaluate(a,cells,turn,ctypes.byref(v))==new.ot_nnue_infer(b,cells,turn,ctypes.byref(fused),q)==1
    assert old.ot_nnue_policy(a,cells,turn,p)==1
    value_error=max(value_error,abs(v.value-fused.value));policy_error=max(policy_error,float(abs(np.array(p)-np.array(q)).max()))
   assert value_error==policy_error==0,(label,value_error,policy_error)
   old.ot_mcts_clear_cache();new.ot_mcts_clear_cache()
   for i in range(24):
    cells=(ctypes.c_int*64)(*map(int,boards[i*40]));turn=int(turns[i*40]);assert search(old,a,cells,turn)==search(new,b,cells,turn),(label,i)
   # A reachable sequence exercises retained subtrees as well as cold roots.
   old.ot_mcts_clear_cache();new.ot_mcts_clear_cache();cells=(ctypes.c_int*64)();cells[28]=cells[35]=1;cells[27]=cells[36]=-1;turn=1
   for i in range(20):
    assert search(old,a,cells,turn)==search(new,b,cells,turn),(label,'warm',i)
    moves=(ctypes.c_int*64)();n=new.ot_legal(cells,turn,moves);out=(ctypes.c_int*136)()
    assert new.ot_play(cells,turn,moves[i%n] if n else -1,out)==1
    cells=(ctypes.c_int*64)(*list(out)[:64]);turn=out[64]
   result.append(dict(model=label,modelSha256=sha(model),referenceLibrarySha256=sha(reference),optimizedLibrarySha256=sha(optimized),inferencePositions=1026,valueError=value_error,policyError=policy_error,coldSearchPositions=24,warmSearchPositions=20,simulations=512,identical=True))
  finally:old.ot_nnue_destroy(a);new.ot_nnue_destroy(b)
 report=dict(results=result,scope='Exact value/probability parity and fixed-simulation moves, scores, candidates, root visits, selection, depth, reuse and tree statistics; elapsed excluded')
 (ROOT/'.build/engineering/parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
