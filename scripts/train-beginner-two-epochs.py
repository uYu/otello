#!/usr/bin/env python3
"""Separate weak-policy candidates: low-Elo imitation plus explicit error-target training."""
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/beginner-low-elo-mistakes-v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    meta=json.loads((ROOT/'models/beginner-low-elo-mistakes-v2-data.json').read_text())
    assert sha(WORK/'positions.npz')==meta['cacheSha256']
    data=np.load(WORK/'positions.npz');assert len(data['train_board'])==10000
    default=json.loads((ROOT/'models/v2-default.json').read_text())
    lib=ctypes.CDLL(str(ROOT/f'.build/native/libothello.{"dylib" if sys.platform=="darwin" else "so"}'))
    ptr=ctypes.POINTER(ctypes.c_int);lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p
    lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p];lib.ot_pattern_predict.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
    handle=lib.ot_nnue_load(str(ROOT/default['defaultModel']).encode());assert handle
    error=[]
    for board,legal in zip(data['train_board'],data['train_legal']):
        out=(ctypes.c_double*68)();assert lib.ot_pattern_predict(handle,(ctypes.c_int*64)(*map(int,board)),1,out)==1
        logits=-np.array(list(out)[4:]);logits=np.where(legal,logits,-np.inf);logits-=logits.max()
        probability=np.exp(logits);error.append(probability/probability.sum())
    lib.ot_nnue_destroy(handle);error=np.array(error,np.float32)
    hard=np.zeros((10000,64),np.float32);hard[np.arange(10000),data['train_move']]=1
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    grid=np.arange(64).reshape(8,8)
    perms=np.stack([np.rot90(np.fliplr(grid) if mirror else grid,r).reshape(64) for mirror in (False,True) for r in range(4)])
    invs=np.argsort(perms,axis=1)
    def features(b):return np.stack((b==1,b==-1,b==0),2).astype(np.float32).reshape(-1,192)
    for error_fraction in (.8,):
        name=f'beginner-error-policy-{int(error_fraction*100)}-e2-v1';output=ROOT/f'models/{name}.json';assert not output.exists()
        torch.manual_seed(20261101);rng=np.random.default_rng(20261101);net=nn.Linear(192,64)
        initial=hashlib.sha256(torch.cat([p.detach().flatten() for p in net.parameters()]).numpy().tobytes()).hexdigest()
        optimizer=torch.optim.Adam(net.parameters(),lr=.003);target=(1-error_fraction)*hard+error_fraction*error
        for epoch in range(2):
            for batch in rng.permutation(10000).reshape(-1,250):
                sym=rng.integers(8);p=perms[sym]
                x=torch.from_numpy(features(data['train_board'][batch][:,p]));mask=torch.from_numpy(data['train_legal'][batch][:,p])
                logits=net(x).masked_fill(~mask,-1e9)
                loss=-(torch.from_numpy(target[batch][:,p])*nn.functional.log_softmax(logits,1)).sum(1).mean()
                optimizer.zero_grad();loss.backward();optimizer.step()
        model=dict(format='otello-beginner-policy-v1',version=name,parameters=12352,policyWeights=net.weight.detach().tolist(),policyBias=net.bias.detach().tolist(),
                   architecture='8x8x3 -> 64 logits',symmetry='D4 mean; learned probability sampling')
        output.write_text(json.dumps(model,separators=(',',':'))+'\n')
        fixtures=[]
        with torch.no_grad():
            boards=data['test_board'][:12]
            prediction=np.mean([net(torch.from_numpy(features(boards[:,p]))).numpy()[:,inv] for p,inv in zip(perms,invs)],axis=0)
            for board,logits in zip(boards,prediction):fixtures.append(dict(board=board.astype(int).tolist(),turn=1,logits=logits.tolist()))
        report=dict(version=name,parameters=12352,modelSha256=sha(output),initialWeightsSha256=initial,
                    controlledParent='models/beginner-error-policy-80-v1-training.json',optimizerUpdates=80,status='two-epoch candidate; no default promotion',
                    initialization='random; all parameters jointly trained; no v2 initialization or frozen parameters',
                    trainPositions=10000,uniqueCanonicalTrainPositions=10000,epochs=2,batchSize=250,optimizer='Adam',learningRate=.003,seed=20261101,
                    dataReport='models/beginner-low-elo-mistakes-v2-data.json',dataReportSha256=sha(ROOT/'models/beginner-low-elo-mistakes-v2-data.json'),
                    cacheSha256=sha(WORK/'positions.npz'),errorTargetFraction=error_fraction,
                    target='(1-errorFraction)*actual low-Elo recorded move + errorFraction*softmax(-frozen v2 legal Pattern logits)',
                    targetQualification='Deliberate error-target training, not pure imitation. Inverse learned prior is a training-only proxy for poor moves, not proof of a search blunder.',
                    v2ModelSha256=default['modelSha256'],sourceSha256=sha(Path(__file__)),fixtures=fixtures,
                    inference='Only the new model probability distribution; no v2, heuristic, uniform mixture, book or MCTS at play time',
                    dataNotices='data/README.md',limits='Must measure match strength and inspect behavior; synthetic weak targets are distinct from original low-Elo imitation candidates.')
        parent=json.loads((ROOT/'models/beginner-error-policy-80-v1-training.json').read_text())
        assert initial==parent['initialWeightsSha256']
        assert report['cacheSha256']==parent['cacheSha256']
        output.with_name(name+'-training.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(model=name,errorTargetFraction=error_fraction)),flush=True)


if __name__=='__main__':main()
