#!/usr/bin/env python3
"""Two-epoch low-LR control: real low-Elo moves only, deterministic argmax."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/beginner-low-elo1300-v1'
NAME='beginner-pure-lowelo-e2-lr0003-argmax-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=ROOT/f'models/{NAME}.json';assert not output.exists()
    meta_path=ROOT/'models/beginner-low-elo1300-v1-data.json';meta=json.loads(meta_path.read_text())
    assert sha(WORK/'positions.npz')==meta['cacheSha256']
    data=np.load(WORK/'positions.npz');assert len(data['train_board'])==10000
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(20261101)
    rng=np.random.default_rng(20261101);net=nn.Linear(192,64)
    initial=hashlib.sha256(torch.cat([p.detach().flatten() for p in net.parameters()]).numpy().tobytes()).hexdigest()
    parent=json.loads((ROOT/'models/beginner-error-policy-80-v1-training.json').read_text())
    assert initial==parent['initialWeightsSha256']
    optimizer=torch.optim.Adam(net.parameters(),lr=.0003)
    grid=np.arange(64).reshape(8,8)
    perms=np.stack([np.rot90(np.fliplr(grid) if mirror else grid,r).reshape(64) for mirror in (False,True) for r in range(4)])
    inverses=np.argsort(perms,axis=1)
    def features(b):return np.stack((b==1,b==-1,b==0),2).astype(np.float32).reshape(-1,192)
    for epoch in range(2):
        for batch in rng.permutation(10000).reshape(-1,250):
            symmetry=rng.integers(8);p,inv=perms[symmetry],inverses[symmetry]
            x=torch.from_numpy(features(data['train_board'][batch][:,p]))
            mask=torch.from_numpy(data['train_legal'][batch][:,p]);moves=torch.from_numpy(inv[data['train_move'][batch]].astype(np.int64))
            loss=nn.functional.cross_entropy(net(x).masked_fill(~mask,-1e9),moves)
            optimizer.zero_grad();loss.backward();optimizer.step()
    model=dict(format='otello-beginner-policy-v1',version=NAME,parameters=12352,architecture='8x8x3 ->64 linear logits',
               policyWeights=net.weight.detach().tolist(),policyBias=net.bias.detach().tolist(),symmetry='D4 mean; deterministic legal argmax')
    output.write_text(json.dumps(model,separators=(',',':'))+'\n')
    with torch.no_grad():
        boards=data['test_board'][:12]
        predictions=np.mean([net(torch.from_numpy(features(boards[:,p]))).numpy()[:,inv] for p,inv in zip(perms,inverses)],axis=0)
    fixtures=[dict(board=b.astype(int).tolist(),turn=1,logits=p.tolist()) for b,p in zip(boards,predictions)]
    report=dict(version=NAME,modelSha256=sha(output),parameters=12352,initialWeightsSha256=initial,
                initialization='random; all parameters trainable; no pretrained initialization or freezing',
                trainPositions=10000,uniqueCanonicalTrainPositions=9718,epochs=2,optimizerUpdates=80,batchSize=250,
                optimizer='Adam',learningRate=.0003,seed=20261101,errorTargetFraction=0,
                target='Only actual low-Elo actor recorded legal move; no synthetic errors or v2 targets',
                dataReport=str(meta_path.relative_to(ROOT)),dataReportSha256=sha(meta_path),cacheSha256=sha(WORK/'positions.npz'),
                sourceSha256=sha(Path(__file__)),fixtures=fixtures,dataNotices='data/README.md',
                inference='D4 mean logits, deterministic legal argmax, coordinate tie-break; no sampling/noise/book/MCTS/heuristic',
                status='candidate; no default promotion',
                limits='Data and target differ from the existing error-target model. This control does not isolate only inference randomness; matched 50-game results remain screening evidence.')
    output.with_name(NAME+'-training.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(model=NAME,epochs=2,learningRate=.0003)))


if __name__=='__main__':main()
