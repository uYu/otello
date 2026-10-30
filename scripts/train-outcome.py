#!/usr/bin/env python3
"""Fit an AlphaGo-style scalar outcome value on frozen v1.8 features/policy."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import time
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('policy_trainer',ROOT/'scripts/train-policy.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
VALUE_OFFSET=24+128*256*2+256*4
VALUE_END=67612


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OutcomeNet(nn.Module):
    def __init__(self,binary):
        super().__init__()
        if binary[:8]!=b'OTNNUE3\0' or len(binary)!=104936:
            raise ValueError('Expected the v1.8 256-wide policy/value parent')
        self.transform=policy.PolicyNet(b'OTNNUE2\0'+binary[8:VALUE_END]).transform
        self.transform.requires_grad_(False)
        self.head=nn.Linear(512,1)
        w=np.frombuffer(binary,dtype='<i2',count=512,offset=VALUE_OFFSET)
        b=np.frombuffer(binary,dtype='<i4',count=1,offset=VALUE_END-4)
        with torch.no_grad():
            self.head.weight.copy_(torch.from_numpy(w.copy()).float().reshape(1,512)/256)
            self.head.bias.copy_(torch.from_numpy(b.copy()).float()/65536)

    def forward(self,boards,turns):
        with torch.no_grad():
            x=policy.value.features(boards)
            black=self.transform(x).clamp(0,1)
            white=self.transform(torch.cat((x[:,64:],x[:,:64]),dim=1)).clamp(0,1)
            actor=(turns>0).unsqueeze(1)
            joined=torch.cat((torch.where(actor,black,white),torch.where(actor,white,black)),dim=1)
        return torch.tanh(self.head(joined).squeeze(1))


def export(net,parent):
    w=np.rint(net.head.weight.detach().numpy().reshape(-1)*256)
    b=np.rint(net.head.bias.detach().numpy()*65536)
    for a,dtype in ((w,'<i2'),(b,'<i4')):
        bounds=np.iinfo(dtype)
        if not np.isfinite(a).all() or a.min()<bounds.min or a.max()>bounds.max:
            raise ValueError('Value weights exceed format limits')
    return b'OTNNUE4\0'+parent[8:VALUE_OFFSET]+w.astype('<i2').tobytes()+b.astype('<i4').tobytes()+parent[VALUE_END:]


def quantized_predict(binary,boards,turns):
    arrays=(np.frombuffer(binary,dtype='<i2',count=128*256,offset=24).reshape(128,256),
            np.frombuffer(binary,dtype='<i4',count=256,offset=24+128*256*2),
            np.frombuffer(binary,dtype='<i2',count=512,offset=VALUE_OFFSET),
            np.frombuffer(binary,dtype='<i4',count=1,offset=VALUE_END-4))
    return policy.value.quantized_predict(arrays,boards,turns)


def metric(prediction,target):
    error=prediction-target;non_draw=target!=0
    return dict(outcomeMse=float(np.mean(error**2)),expectedScoreMse=float(np.mean(error**2)/4),
        nonDrawSignAccuracy=float(np.mean(np.sign(prediction[non_draw])==target[non_draw])) if non_draw.any() else None,
        draws=int((~non_draw).sum()),zeroOutcomeMse=float(np.mean(target**2)))


def train(args):
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise ValueError('Choose a fresh candidate output')
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed)
    meta=json.loads(args.cache.with_suffix('.json').read_text())
    if sha(args.cache)!=meta['cacheSha256'] or meta['seed']!=args.seed:
        raise ValueError('Position cache checksum/seed mismatch')
    baseline=json.loads(args.baseline_manifest.read_text())
    if sha(args.parent)!=baseline['modelSha256']:
        raise ValueError('Parent differs from the approved baseline')
    data=dict(np.load(args.cache));parent=args.parent.read_bytes();net=OutcomeNet(parent)
    opt=torch.optim.AdamW(net.head.parameters(),lr=0.001,weight_decay=0.001)
    boards=torch.from_numpy(data['train_board']);turns=torch.from_numpy(data['train_turn'])
    targets=torch.from_numpy(np.sign(data['train_target']).astype(np.float32))
    permutations=torch.from_numpy(policy.value.symmetries())

    def predict(split):
        result=[]
        with torch.no_grad():
            for start in range(0,len(data[split+'_board']),1024):
                result.append(net(torch.from_numpy(data[split+'_board'][start:start+1024]),
                                  torch.from_numpy(data[split+'_turn'][start:start+1024])).numpy())
        return np.concatenate(result)

    validation=np.sign(data['validation_target'])
    initial=metric(predict('validation'),validation)
    best=initial['outcomeMse'];best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
    history=[dict(epoch=0,**initial)];started=time.perf_counter()
    for epoch in range(args.epochs):
        for indices in torch.randperm(len(boards)).split(args.batch_size):
            selection=permutations[torch.randint(8,(len(indices),))]
            batch=torch.gather(boards[indices],1,selection)
            prediction=net(batch,turns[indices]);loss=((prediction-targets[indices])**2).mean()
            opt.zero_grad();loss.backward();opt.step()
        score=metric(predict('validation'),validation);history.append(dict(epoch=epoch+1,**score))
        if score['outcomeMse']<best:
            best=score['outcomeMse'];best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
        print(json.dumps(history[-1]),flush=True)
    net.load_state_dict(best_state);binary=export(net,parent)
    assert binary[8:VALUE_OFFSET]==parent[8:VALUE_OFFSET] and binary[VALUE_END:]==parent[VALUE_END:]
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(binary)
    torch.save(net.state_dict(),args.output.with_suffix('.pt'))
    report=dict(version='v1.9-outcome-candidate',status='candidate',modelFormat='OTNNUE4',
        valueTarget='side-to-move terminal outcome: win +1, draw 0, loss -1',
        architecture='frozen v1.8 NNUE transform and policy; train only 512->1 tanh outcome value',
        parameters=52098,trainableParameters=513,bytes=len(binary),sha256=sha(args.output),
        parent=str(args.parent),parentSha256=sha(args.parent),baselineVersion=baseline['version'],
        featuresUnchanged=True,policyBytesUnchanged=True,dataset=meta,epochs=args.epochs,
        seed=args.seed,batchSize=args.batch_size,learningRate=0.001,weightDecay=0.001,
        bestEpoch=min(history,key=lambda x:x['outcomeMse'])['epoch'],history=history,
        seconds=time.perf_counter()-started,torchVersion=torch.__version__,numpyVersion=np.__version__,
        trainingScriptSha256=sha(Path(__file__)),helperSha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/train-policy.py',ROOT/'scripts/train-nnue.py']},metrics={})
    for split in ('validation','test'):
        fp=predict(split);qp=quantized_predict(binary,data[split+'_board'],data[split+'_turn']);truth=np.sign(data[split+'_target'])
        report['metrics'][split]=dict(float=metric(fp,truth),quantized=metric(qp,truth),
            maxQuantizationError=float(abs(fp-qp).max()))
    lib=ctypes.CDLL(str(args.library.resolve()));ptr=ctypes.POINTER(ctypes.c_int)
    lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p
    lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p]
    lib.ot_nnue_evaluate.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ptr]
    lib.ot_nnue_policy.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
    handle=lib.ot_nnue_load(str(args.output.resolve()).encode());old=lib.ot_nnue_load(str(args.parent.resolve()).encode())
    if not handle or not old:raise ValueError('Native model load failed')
    count=min(1024,len(data['test_board']));maximum=0;policy_error=0
    expected=quantized_predict(binary,data['test_board'][:count],data['test_turn'][:count])*1000
    try:
        for i in range(count):
            cells=(ctypes.c_int*64)(*map(int,data['test_board'][i]));turn=int(data['test_turn'][i]);score=ctypes.c_int()
            if lib.ot_nnue_evaluate(handle,cells,turn,ctypes.byref(score))!=1:raise ValueError('Native outcome value failed')
            maximum=max(maximum,abs(score.value-expected[i]))
            a,b=(ctypes.c_double*65)(),(ctypes.c_double*65)()
            assert lib.ot_nnue_policy(handle,cells,turn,a)==lib.ot_nnue_policy(old,cells,turn,b)==1
            policy_error=max(policy_error,float(abs(np.array(a)-np.array(b)).max()))
    finally:
        lib.ot_nnue_destroy(handle);lib.ot_nnue_destroy(old)
    if maximum>0.501 or policy_error!=0:raise ValueError('Native value parity or frozen policy parity failed')
    report.update(cppParityPositions=count,cppMaxRoundingError=maximum,cppPolicyMaxError=policy_error)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('history','dataset')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    active=json.loads((ROOT/'models/default-model.json').read_text())
    p.add_argument('--cache',type=Path,default=ROOT/'.build/training/v1/positions.npz')
    p.add_argument('--parent',type=Path,default=ROOT/active['defaultModel'])
    p.add_argument('--baseline-manifest',type=Path,default=ROOT/active['baselineManifest'])
    p.add_argument('--output',type=Path,default=ROOT/'.build/training/outcome/nnue-outcome-v1.9.otnnue')
    p.add_argument('--library',type=Path,default=ROOT/('.build/native-outcome/libothello.'+('dylib' if sys.platform=='darwin' else 'so')))
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=20261027)
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.threads)<1:p.error('Positive counts required')
    train(args)
