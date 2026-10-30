#!/usr/bin/env python3
"""Check frozen large-CNN CPU predictions against the batched MPS callback ABI."""
import ctypes,importlib.util,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('big_eval',ROOT/'scripts/big-cnn-evaluator.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    checkpoint=ROOT/'.build/training/teacher-cache/big-cnn-v1.24-joint-snapshot1/teacher.pt'
    evaluator=m.Evaluator(checkpoint)
    with np.load(ROOT/'.build/training/v1/positions.npz') as d:b=d['validation_board'][:16];t=d['validation_turn'][:16]
    with np.load(ROOT/'.build/training/policy/labels.npz') as d:legal=d['validation_legal'][:16]
    b=np.concatenate([b,-b]).astype(np.int32);t=np.concatenate([t,-t]).astype(np.int32);legal=np.concatenate([legal,legal])
    cpu=m.big.BigCnn().eval();cpu.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'])
    with torch.inference_mode():
        v,p=cpu(torch.from_numpy(b),torch.from_numpy(t));p=p.masked_fill(~torch.from_numpy(legal),-torch.inf).softmax(1).numpy();v=v.numpy()
    masks=np.array([sum(1<<int(i) for i in np.flatnonzero(row[:64])) for row in legal],dtype=np.uint64)
    checks=[]
    for n in (1,8,16,32):
        scores=np.zeros(n,dtype=np.int32);probs=np.zeros((n,65),dtype=np.float64)
        assert evaluator.callback(n,b[:n].ctypes.data_as(m.PTR),t[:n].ctypes.data_as(m.PTR),masks[:n].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            scores.ctypes.data_as(m.PTR),probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))==1,evaluator.errors
        score_error=float(abs(scores-v[:n]*1000).max());policy_error=float(abs(probs-p[:n]).max())
        assert score_error<=.51 and policy_error<1e-4,(n,score_error,policy_error)
        assert (probs[~legal[:n]]==0).all() and np.allclose(probs.sum(1),1)
        checks.append(dict(batch=n,maxValueIntegerRoundingError=score_error,maxPolicyError=policy_error))
        if n==32:
            assert np.array_equal(scores[:16],scores[16:]) and np.allclose(probs[:16],probs[16:],atol=1e-6)
    forced=np.ones((1,64),dtype=np.int32);forced[0,0]=0;forced[0,1]=-1;turn=np.array([-1],np.int32);mask=np.zeros(1,dtype=np.uint64)
    scores=np.zeros(1,np.int32);probs=np.zeros((1,65),np.float64)
    assert evaluator.callback(1,forced.ctypes.data_as(m.PTR),turn.ctypes.data_as(m.PTR),mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),scores.ctypes.data_as(m.PTR),probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))==1
    assert probs[0,64]==1 and probs[:,:64].sum()==0
    result=dict(complete=True,modelSha256=m.big.h.sha(checkpoint),checks=checks,colorInversion=True,forcedPass=True,
        limits='CPU FP32 vs MPS callback parity; existing native outcome MCTS batch/proof contracts were previously validated',
        sourceSha256={p:m.big.h.sha(ROOT/p) for p in ['scripts/big-cnn-evaluator.py','scripts/verify-big-cnn-evaluator.py']})
    m.big.h.write_json(ROOT/'models/v1.26-big-cnn-callback-verification.json',result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
