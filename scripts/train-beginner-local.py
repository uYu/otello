#!/usr/bin/env python3
"""Learn novice local move preferences from the fixed 10k low-Elo dataset."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.build/training/beginner-low-elo1300-v1/positions.npz'
DATA_REPORT=ROOT/'models/beginner-low-elo1300-v1-data.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    meta=json.loads(DATA_REPORT.read_text());assert sha(CACHE)==meta['cacheSha256']
    data=np.load(CACHE);assert len(data['train_board'])==10000
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    grid=np.arange(64).reshape(8,8)
    perms=np.stack([np.rot90(np.fliplr(grid) if mirror else grid,r).reshape(64) for mirror in (False,True) for r in range(4)])
    inverses=np.argsort(perms,axis=1)
    for hidden in (1,4):
        name=f'beginner-low-elo1300-local{hidden}-v1';output=ROOT/f'models/{name}.json'
        assert not output.exists(),'Preserve trained candidates'
        torch.manual_seed(20261101);rng=np.random.default_rng(20261101)
        net=nn.Conv2d(1,1,3,padding=1) if hidden==1 else nn.Sequential(nn.Conv2d(1,4,3,padding=1),nn.Tanh(),nn.Conv2d(4,1,1))
        initial_sha=hashlib.sha256(torch.cat([p.detach().flatten() for p in net.parameters()]).numpy().tobytes()).hexdigest()
        optimizer=torch.optim.Adam(net.parameters(),lr=.003)

        def predict(board):
            with torch.no_grad():
                return net(torch.from_numpy(board.astype(np.float32).reshape(-1,1,8,8))).numpy().reshape(-1,64)

        def metric(split):
            logits=np.where(data[split+'_legal'],predict(data[split+'_board']),-np.inf)
            moves=data[split+'_move'];logits-=logits.max(1,keepdims=True)
            ce=np.log(np.exp(logits).sum(1))-logits[np.arange(len(moves)),moves]
            return dict(positions=len(moves),crossEntropy=float(ce.mean()),top1=float((logits.argmax(1)==moves).mean()),
                        uniformCrossEntropy=float(np.log(data[split+'_legal'].sum(1)).mean()))

        history=[dict(epoch=0,validation=metric('validation'))];started=time.perf_counter()
        for epoch in range(1,31):
            order=rng.permutation(10000)
            for offset in range(0,10000,250):
                batch=order[offset:offset+250];symmetry=rng.integers(8);p,inv=perms[symmetry],inverses[symmetry]
                x=torch.from_numpy(data['train_board'][batch][:,p].astype(np.float32).reshape(-1,1,8,8).copy())
                mask=torch.from_numpy(data['train_legal'][batch][:,p]);move=torch.from_numpy(inv[data['train_move'][batch]].astype(np.int64))
                logits=net(x).reshape(-1,64).masked_fill(~mask,-torch.inf)
                loss=nn.functional.cross_entropy(logits,move)
                optimizer.zero_grad();loss.backward();optimizer.step()
            if epoch in (1,5,15,30):
                history.append(dict(epoch=epoch,validation=metric('validation')))
        state=net.state_dict()
        model=dict(format='otello-beginner-local-policy-v1',version=name,hidden=hidden,
                   parameters=sum(p.numel() for p in net.parameters()),input='8x8x1 signed squares relative to mover',
                   architecture='shared 3x3 convolution -> one logit per square' if hidden==1 else 'shared 3x3 convolution to4 -> tanh -> shared 1x1 convolution to1 logit',
                   padding='zero; outside board and empty are both0',inference='single view; legal masked argmax',
                   kernelWeights=(state['weight'] if hidden==1 else state['0.weight']).reshape(hidden,9).tolist(),
                   hiddenBias=(state['bias'] if hidden==1 else state['0.bias']).tolist(),
                   outputWeights=None if hidden==1 else state['2.weight'].ravel().tolist(),
                   outputBias=0 if hidden==1 else float(state['2.bias'][0]))
        output.write_text(json.dumps(model,separators=(',',':'))+'\n')
        fixtures=[dict(board=b.astype(int).tolist(),turn=1,logits=v.tolist()) for b,v in zip(data['test_board'][:12],predict(data['test_board'][:12]))]
        report=dict(version=name,modelSha256=sha(output),parameters=model['parameters'],initialization='random; all parameters trainable; no pretrained weights or freezing',
                    initialWeightsSha256=initial_sha,seed=20261101,trainPositions=10000,epochs=30,batchSize=250,learningRate=.003,
                    optimizer='Adam',loss='legal-masked CE on the low-Elo actor actual recorded move',
                    dataReport=str(DATA_REPORT.relative_to(ROOT)),dataReportSha256=sha(DATA_REPORT),cacheSha256=sha(CACHE),
                    checkpoint='fixed final epoch30; no validation/test selection',history=history,test=metric('test'),fixtures=fixtures,
                    sourceSha256=sha(Path(__file__)),seconds=time.perf_counter()-started,
                    environment=dict(torch=torch.__version__,numpy=np.__version__,threads=2,device='cpu'),dataNotices='data/README.md',
                    inference='Only learned local logits; deterministic legal argmax. No global features, hand-coded corner preference, book, MCTS, value lookahead or stochastic sampling',
                    limits='Restricted 3x3 receptive field cannot see distant threats or infer full legal-flip rays. Weakness/coherence requires match checks; not a human win-rate claim.')
        output.with_name(name+'-training.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(model=name,test=report['test'])),flush=True)


if __name__=='__main__':
    main()
