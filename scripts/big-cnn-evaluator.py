#!/usr/bin/env python3
"""Batched MPS CNN evaluator for the existing C++ outcome MCTS callback ABI."""
import ctypes,importlib.util,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('big_trainer',ROOT/'scripts/train-big-cnn.py')
big=importlib.util.module_from_spec(spec);spec.loader.exec_module(big)
PTR=ctypes.POINTER(ctypes.c_int)
CALLBACK=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_int,PTR,PTR,ctypes.POINTER(ctypes.c_uint64),PTR,ctypes.POINTER(ctypes.c_double))

class Evaluator:
    def __init__(self,checkpoint):
        assert torch.backends.mps.is_available();torch.set_num_threads(2)
        bundle=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.net=big.BigCnn(bundle['channels'],bundle['blocks'],bundle['headWidth']).eval()
        self.net.load_state_dict(bundle['state_dict']);self.net.to('mps');self.errors=[];self.calls=0;self.positions=0;self.seconds=0.
        self.callback=CALLBACK(self.evaluate)
    def predict(self,board,turn,legal):
        with torch.inference_mode():
            value,logits=self.net(torch.from_numpy(board.astype(np.int8,copy=False)).to('mps'),torch.from_numpy(turn.astype(np.int8,copy=False)).to('mps'))
            policy=logits.masked_fill(~torch.from_numpy(legal).to('mps'),-torch.inf).softmax(1)
            return value.cpu().numpy(),policy.cpu().numpy()
    def evaluate(self,n,board,turn,masks,score,policy):
        try:
            tick=time.perf_counter();b=np.ctypeslib.as_array(board,shape=(n*64,)).reshape(n,64);t=np.ctypeslib.as_array(turn,shape=(n,));mask=np.ctypeslib.as_array(masks,shape=(n,))
            legal=np.zeros((n,65),bool);legal[:,:64]=((mask[:,None]>>np.arange(64,dtype=np.uint64))&1)!=0;legal[:,64]=mask==0
            v,p=self.predict(b,t,legal)
            scaled=v.astype(np.float64)*1000
            np.ctypeslib.as_array(score,shape=(n,))[:]=np.where(scaled>=0,np.floor(scaled+.5),np.ceil(scaled-.5)).astype(np.int32)
            np.ctypeslib.as_array(policy,shape=(n*65,))[:]=p.astype(np.float64).reshape(-1)
            self.calls+=1;self.positions+=n;self.seconds+=time.perf_counter()-tick
            return 1
        except BaseException as error:self.errors.append(repr(error));return -1
    def warmup(self):
        with np.load(ROOT/'.build/training/v1/positions.npz') as d:
            b=d['validation_board'][:32];t=d['validation_turn'][:32]
        with np.load(ROOT/'.build/training/policy/labels.npz') as d:legal=d['validation_legal'][:32]
        for n in (1,8,16,32):
            for _ in range(2):self.predict(b[:n],t[:n],legal[:n])
        torch.mps.synchronize()
