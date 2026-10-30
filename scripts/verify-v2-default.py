#!/usr/bin/env python3
"""Compare released native inference and search with the exact measured candidate."""
import ctypes,json
from pathlib import Path
import numpy as np
import global_policy_native_experiment as e
ROOT=e.ROOT;WORK=ROOT/'.build/v2-default-check';WORK.mkdir(parents=True,exist_ok=True)
a=e.read(ROOT/'models/default-model.json');model=ROOT/a['defaultModel']
assert a['version']=='v2' and e.sha(model)==a['modelSha256']==e.sha(e.MODEL)
new=e.load(ROOT/f'.build/native/libothello.{e.SUFFIX}',model);old=e.load(e.LIB,e.MODEL)
lib,handle=new;ptr=e.ptr
lib.ot_default_model_sha256.restype=ctypes.c_char_p
lib.ot_default_evaluate.argtypes=[ptr,ctypes.c_int,ptr]
lib.ot_default_infer.argtypes=[ptr,ctypes.c_int,ptr,ctypes.POINTER(ctypes.c_double)]
lib.ot_nnue_infer.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ptr,ctypes.POINTER(ctypes.c_double)]
assert lib.ot_default_model_sha256().decode()==a['modelSha256']
data=np.load(ROOT/'.build/training/v1/positions.npz');fixture=[];count=0
try:
    for split in ['validation','test']:
        boards,turns=data[split+'_board'],data[split+'_turn']
        assert np.array_equal(e.native(new,boards,turns),e.native(old,boards,turns)),split
        for board,turn in zip(boards,turns):
            cells=(ctypes.c_int*64)(*map(int,board));score=ctypes.c_int();other=ctypes.c_int();value=ctypes.c_int()
            probs=(ctypes.c_double*65)();explicit=(ctypes.c_double*65)()
            assert lib.ot_default_infer(cells,int(turn),ctypes.byref(score),probs)==1
            assert lib.ot_default_evaluate(cells,int(turn),ctypes.byref(value))==1
            assert lib.ot_nnue_infer(handle,cells,int(turn),ctypes.byref(other),explicit)==1
            assert score.value==other.value==value.value and list(probs)==list(explicit)
            if split=='validation':fixture.append(dict(board=board.tolist(),turn=int(turn),score=score.value,probabilities=list(probs)))
            count+=1
    for board,turn in zip(data['validation_board'][:32],data['validation_turn'][:32]):
        assert e.search(new,board,turn,simulations=2048)[0]==e.search(old,board,turn,simulations=2048)[0]
finally:
    lib.ot_nnue_destroy(handle);old[0].ot_nnue_destroy(old[1])
e.write(WORK/'fixture.json',dict(modelSha256=a['modelSha256'],positions=fixture))
result=dict(nativeEmbeddedModelSha256=a['modelSha256'],nativeExplicitEmbeddedPositions=count,all68OutputsExactlyMatchMeasuredCandidate=True,all65EmbeddedPolicyProbabilitiesExactlyMatchExplicit=True,fixedSimulationRoots=32,simulationsPerRoot=2048)
e.write(WORK/'native.json',result);print(json.dumps(result))
