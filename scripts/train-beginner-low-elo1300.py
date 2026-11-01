#!/usr/bin/env python3
"""Train policy/value variants on exactly 10,000 established low-Elo actor positions."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/beginner-low-elo1300-v1'
DATA_REPORT=ROOT/'models/beginner-low-elo1300-v1-data.json'
SEED=20261101
EPOCHS=30


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    metadata=json.loads(DATA_REPORT.read_text())
    assert sha(WORK/'positions.npz')==metadata['cacheSha256']
    kinds=('policy','value8','value32')
    for kind in kinds:
        assert not (ROOT/f'models/beginner-low-elo1300-{kind}-v1.json').exists(),'Preserve trained candidates'
    data=np.load(WORK/'positions.npz')
    assert len(data['train_board'])==10000
    torch.set_num_threads(2);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(SEED)
    grid=np.arange(64).reshape(8,8)
    perms=np.stack([np.rot90(np.fliplr(grid) if mirror else grid,r).reshape(64) for mirror in (False,True) for r in range(4)])
    inverses=np.argsort(perms,axis=1)
    nets=dict(policy=nn.Linear(192,64),value8=nn.Sequential(nn.Linear(64,8),nn.Tanh(),nn.Linear(8,1),nn.Tanh()),
              value32=nn.Sequential(nn.Linear(192,32),nn.Tanh(),nn.Linear(32,1),nn.Tanh()))
    initial={kind:hashlib.sha256(torch.cat([p.detach().flatten() for p in net.parameters()]).numpy().tobytes()).hexdigest() for kind,net in nets.items()}
    optimizers={kind:torch.optim.Adam(net.parameters(),lr=.003) for kind,net in nets.items()}

    def features(b):
        return np.stack((b==1,b==-1,b==0),axis=2).astype(np.float32).reshape(-1,192)

    def predict(kind,b):
        with torch.no_grad():
            outputs=[]
            for p,inv in zip(perms,inverses):
                x=torch.from_numpy(b[:,p].astype(np.float32).copy() if kind=='value8' else features(b[:,p]))
                value=nets[kind](x).numpy()
                outputs.append(value[:,inv] if kind=='policy' else value.ravel())
            return np.mean(outputs,axis=0)

    def metric(kind,split):
        prediction=predict(kind,data[split+'_board'])
        if kind=='policy':
            mask=data[split+'_legal'];moves=data[split+'_move']
            logits=np.where(mask,prediction,-np.inf);logits-=logits.max(1,keepdims=True)
            ce=np.log(np.exp(logits).sum(1))-logits[np.arange(len(moves)),moves]
            return dict(positions=len(moves),crossEntropy=float(ce.mean()),top1=float((logits.argmax(1)==moves).mean()),
                        uniformCrossEntropy=float(np.log(mask.sum(1)).mean()),uniformTop1=float((1/mask.sum(1)).mean()))
        target=data[split+'_target']
        return dict(positions=len(target),outcomeMse=float(np.mean((prediction-target)**2)),
                    zeroPredictorMse=float(np.mean(target**2)),nonDrawSignAccuracy=float((np.sign(prediction[target!=0])==target[target!=0]).mean()),
                    alwaysPredictLossAccuracy=float((target[target!=0]<0).mean()))

    history={kind:[dict(epoch=0,validation=metric(kind,'validation'))] for kind in kinds}
    started=time.perf_counter()
    for epoch in range(1,EPOCHS+1):
        order=rng.permutation(10000)
        for offset in range(0,10000,250):
            batch=order[offset:offset+250];symmetry=rng.integers(8);p,inv=perms[symmetry],inverses[symmetry]
            board=data['train_board'][batch][:,p]
            x=torch.from_numpy(features(board));signed=torch.from_numpy(board.astype(np.float32).copy())
            mask=torch.from_numpy(data['train_legal'][batch][:,p]);move=torch.from_numpy(inv[data['train_move'][batch]].astype(np.int64))
            target=torch.from_numpy(data['train_target'][batch])
            losses=dict(policy=nn.functional.cross_entropy(nets['policy'](x).masked_fill(~mask,-torch.inf),move),
                        value8=torch.mean((nets['value8'](signed).ravel()-target)**2),
                        value32=torch.mean((nets['value32'](x).ravel()-target)**2))
            for kind,loss in losses.items():
                optimizers[kind].zero_grad();loss.backward();optimizers[kind].step()
        if epoch in (1,5,15,30):
            for kind in kinds:
                row=dict(epoch=epoch,validation=metric(kind,'validation'));history[kind].append(row)
            print(json.dumps(dict(epoch=epoch,metrics={k:history[k][-1]['validation'] for k in kinds})),flush=True)
    for kind,net in nets.items():
        name=f'beginner-low-elo1300-{kind}-v1';state=net.state_dict()
        if kind=='policy':
            model=dict(format='otello-beginner-policy-v1',version=name,parameters=12352,
                       architecture='8x8x3 interleaved own/opponent/empty -> 8x8 linear logits',
                       policyWeights=state['weight'].tolist(),policyBias=state['bias'].tolist(),symmetry='D4 mean; legal masked argmax')
        else:
            width=8 if kind=='value8' else 32
            model=dict(format='otello-beginner-v1' if kind=='value8' else 'otello-beginner-board-value-v1',version=name,
                       parameters=sum(p.numel() for p in net.parameters()),hidden=width,
                       input='signed64' if kind=='value8' else 'interleaved own/opponent/empty192',
                       hiddenWeights=state['0.weight'].tolist(),hiddenBias=state['0.bias'].tolist(),
                       outputWeights=state['2.weight'][0].tolist(),outputBias=float(state['2.bias'][0]),symmetry='D4 mean')
        output=ROOT/f'models/{name}.json';output.write_text(json.dumps(model,separators=(',',':'))+'\n')
        boards=data['test_board'][:12];predictions=predict(kind,boards)
        fixtures=[dict(board=b.astype(int).tolist(),turn=1,**({'logits':v.tolist()} if kind=='policy' else {'value':float(v)})) for b,v in zip(boards,predictions)]
        report=dict(version=name,kind=kind,modelSha256=sha(output),parameters=model['parameters'],initialWeightsSha256=initial[kind],
                    initialization='random; all parameters trainable; no pretrained model, freezing or distillation',
                    seed=SEED,epochs=EPOCHS,trainPositions=10000,batchSize=250,learningRate=.003,optimizer='Adam',
                    dataReport=str(DATA_REPORT.relative_to(ROOT)),dataReportSha256=sha(DATA_REPORT),cacheSha256=sha(WORK/'positions.npz'),
                    sourceSha256=sha(Path(__file__)),history=history[kind],test=metric(kind,'test'),fixtures=fixtures,
                    target='Recorded low-Elo actor next legal move' if kind=='policy' else 'Recorded final WDL from low-Elo actor perspective',
                    inference='D4 mean. Policy masked deterministic argmax; values deterministic argmax of immediate legal successors, with pass-aware perspective. No book, search, heuristics, noise or terminal override.',
                    checkpoint='fixed final epoch30; no validation/test checkpoint selection',
                    environment=dict(torch=torch.__version__,numpy=np.__version__,device='cpu',threads=2),
                    seconds=time.perf_counter()-started,dataNotices='data/README.md',
                    limits='Low Elo is an internal estimate; value outcome labels are biased by the low-rated actor population. Strength and suitability must be measured.')
        output.with_name(name+'-training.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(model=name,test=report['test'])),flush=True)


if __name__=='__main__':
    main()
