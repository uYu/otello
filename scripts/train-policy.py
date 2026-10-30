#!/usr/bin/env python3
"""Add a supervised policy bottleneck to frozen v1.7 NNUE features/value bytes."""
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
spec=importlib.util.spec_from_file_location('value_trainer',ROOT/'scripts/train-nnue.py')
value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PolicyNet(nn.Module):
    def __init__(self, binary):
        super().__init__()
        if struct.unpack('<8sIIII',binary[:24]) != (b'OTNNUE2\0',128,256,256,256) or len(binary)!=67612:
            raise ValueError('Expected frozen 256-wide turn NNUE')
        self.transform=nn.Linear(128,256)
        w=np.frombuffer(binary,dtype='<i2',count=128*256,offset=24).reshape(128,256)
        b=np.frombuffer(binary,dtype='<i4',count=256,offset=24+128*256*2)
        with torch.no_grad():
            self.transform.weight.copy_(torch.from_numpy(w.T.copy()).float()/256)
            self.transform.bias.copy_(torch.from_numpy(b.copy()).float()/256)
        self.transform.requires_grad_(False)
        self.hidden=nn.Linear(512,32)
        self.policy=nn.Linear(32,65)

    def forward(self, boards, turns):
        with torch.no_grad():
            x=value.features(boards)
            black=self.transform(x).clamp(0,1)
            white=self.transform(torch.cat((x[:,64:],x[:,:64]),dim=1)).clamp(0,1)
            actor=(turns>0).unsqueeze(1)
            joined=torch.cat((torch.where(actor,black,white),torch.where(actor,white,black)),dim=1)
        return self.policy(self.hidden(joined).clamp(0,1))


def quantize(net):
    result=[]
    for tensor,scale,dtype in [(net.hidden.weight,256,'<i2'),(net.hidden.bias,65536,'<i4'),
                               (net.policy.weight,256,'<i2'),(net.policy.bias,65536,'<i4')]:
        a=np.rint(tensor.detach().numpy()*scale);limits=np.iinfo(dtype)
        if not np.isfinite(a).all() or a.min()<limits.min or a.max()>limits.max:
            raise ValueError('Policy weights outside export range')
        result.append(a.astype(dtype))
    return result


def quantized_probabilities(binary,weights,boards,turns,legal):
    w=np.frombuffer(binary,dtype='<i2',count=128*256,offset=24).reshape(128,256).astype(np.int64)
    b=np.frombuffer(binary,dtype='<i4',count=256,offset=24+128*256*2).astype(np.int64)
    x=np.concatenate((boards==1,boards==-1),axis=1).astype(np.int64)
    black=np.clip(x@w+b,0,256);white=np.clip(np.concatenate((x[:,64:],x[:,:64]),axis=1)@w+b,0,256)
    own=np.where((turns>0)[:,None],black,white);opp=np.where((turns>0)[:,None],white,black)
    h=np.clip((np.concatenate((own,opp),axis=1)@weights[0].astype(np.int64).T+weights[1])/65536,0,1)
    logits=h@weights[2].astype(np.float64).T/256+weights[3]/65536
    logits=np.where(legal,logits,-np.inf);logits-=logits.max(axis=1,keepdims=True)
    probs=np.exp(logits);return probs/probs.sum(axis=1,keepdims=True)


def metrics(probs,move,legal):
    chosen=np.maximum(probs[np.arange(len(move)),move],1e-300)
    return dict(crossEntropy=float(-np.log(chosen).mean()),
        top1=float((probs.argmax(axis=1)==move).mean()),
        uniformCrossEntropy=float(np.log(legal.sum(axis=1)).mean()),
        uniformTop1=float((1/legal.sum(axis=1)).mean()))


def train(args):
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise ValueError('Choose a fresh candidate output')
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed)
    labels=dict(np.load(args.labels));data=dict(np.load(args.cache))
    meta=json.loads(args.labels.with_suffix('.json').read_text())
    if sha(args.labels)!=meta['cacheSha256'] or sha(args.cache)!=meta['positionsSha256']:
        raise ValueError('Policy/position cache mismatch')
    baseline=json.loads((ROOT/'models/v1.7-default.json').read_text())
    if sha(args.parent)!=baseline['modelSha256']:
        raise ValueError('Parent differs from approved v1.7 baseline')
    binary=args.parent.read_bytes();net=PolicyNet(binary)
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=0.001,weight_decay=0.001)
    boards=torch.from_numpy(data['train_board']);turns=torch.from_numpy(data['train_turn'])
    moves=torch.from_numpy(labels['train_move'].astype(np.int64));legal=torch.from_numpy(labels['train_legal'])
    perms=torch.from_numpy(value.symmetries());inverse=torch.argsort(perms,dim=1)
    best,best_state,history=float('inf'),None,[];started=time.perf_counter()

    def predict(split):
        parts=[]
        with torch.no_grad():
            for start in range(0,len(data[split+'_board']),1024):
                board=torch.from_numpy(data[split+'_board'][start:start+1024])
                turn=torch.from_numpy(data[split+'_turn'][start:start+1024])
                mask=torch.from_numpy(labels[split+'_legal'][start:start+1024])
                parts.append(net(board,turn).masked_fill(~mask,-torch.inf).softmax(1).numpy())
        return np.concatenate(parts)

    for epoch in range(args.epochs):
        net.train()
        for indices in torch.randperm(len(boards)).split(args.batch_size):
            symmetry=torch.randint(8,(len(indices),));p=perms[symmetry]
            board=torch.gather(boards[indices],1,p)
            move=inverse[symmetry,moves[indices]]
            mask=torch.cat((torch.gather(legal[indices,:64],1,p),legal[indices,64:]),dim=1)
            logits=net(board,turns[indices]).masked_fill(~mask,-torch.inf)
            loss=nn.functional.cross_entropy(logits,move)
            opt.zero_grad();loss.backward();opt.step()
        net.eval();score=metrics(predict('validation'),labels['validation_move'],labels['validation_legal'])
        history.append(dict(epoch=epoch+1,**score));print(json.dumps(history[-1]),flush=True)
        if score['crossEntropy']<best:
            best=score['crossEntropy'];best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
    net.load_state_dict(best_state);net.eval();weights=quantize(net)
    output=b'OTNNUE3\0'+binary[8:]+struct.pack('<II',32,65)+b''.join(w.tobytes() for w in weights)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(output)
    torch.save(net.state_dict(),args.output.with_suffix('.pt'))
    report=dict(version='v1.8-policy-candidate',status='candidate',architecture='frozen 128->256 dual-view NNUE; value 512->1 unchanged; policy 512->32 clipped ReLU->65',
        parameters=33537+512*32+32+32*65+65,trainableParameters=512*32+32+32*65+65,
        bytes=len(output),sha256=sha(args.output),parent=str(args.parent),parentSha256=sha(args.parent),
        valueBytesUnchanged=output[8:len(binary)]==binary[8:],epochs=args.epochs,batchSize=args.batch_size,
        seed=args.seed,bestEpoch=min(history,key=lambda x:x['crossEntropy'])['epoch'],history=history,
        seconds=time.perf_counter()-started,labels=meta,trainingScriptSha256=sha(Path(__file__)),
        torchVersion=torch.__version__,numpyVersion=np.__version__,metrics={})
    for split in ('validation','test'):
        board,turn=data[split+'_board'],data[split+'_turn'];mask=labels[split+'_legal']
        fp=predict(split);qp=quantized_probabilities(binary,weights,board,turn,mask)
        report['metrics'][split]=dict(float=metrics(fp,labels[split+'_move'],mask),
            quantized=metrics(qp,labels[split+'_move'],mask),maxQuantizationError=float(abs(fp-qp).max()))
    lib=ctypes.CDLL(str(args.library.resolve()));ptr=ctypes.POINTER(ctypes.c_int)
    lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p
    lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p]
    lib.ot_nnue_policy.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
    lib.ot_nnue_evaluate.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ptr]
    handle=lib.ot_nnue_load(str(args.output.resolve()).encode());parent=lib.ot_nnue_load(str(args.parent.resolve()).encode())
    if not handle or not parent:
        raise ValueError('Native policy/parent load failed')
    maximum=0;count=min(1024,len(data['test_board']))
    expected=quantized_probabilities(binary,weights,data['test_board'][:count],data['test_turn'][:count],labels['test_legal'][:count])
    try:
        for i in range(count):
            cells=(ctypes.c_int*64)(*map(int,data['test_board'][i]));turn=int(data['test_turn'][i]);probs=(ctypes.c_double*65)()
            if lib.ot_nnue_policy(handle,cells,turn,probs)!=1:raise ValueError('Native policy failed')
            maximum=max(maximum,float(abs(np.array(probs)-expected[i]).max()))
            a,b=ctypes.c_int(),ctypes.c_int()
            assert lib.ot_nnue_evaluate(handle,cells,turn,ctypes.byref(a))==1
            assert lib.ot_nnue_evaluate(parent,cells,turn,ctypes.byref(b))==1
            if a.value!=b.value:raise ValueError('Frozen value output changed')
    finally:
        lib.ot_nnue_destroy(handle);lib.ot_nnue_destroy(parent)
    if maximum>1e-10:raise ValueError('Native policy differs from quantized Python')
    report.update(cppParityPositions=count,cppMaxProbabilityError=maximum,valueParity=True)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('history','labels')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=ROOT/'.build/training/v1/positions.npz')
    p.add_argument('--labels',type=Path,default=ROOT/'.build/training/policy/labels.npz')
    p.add_argument('--parent',type=Path,default=ROOT/'models/nnue256-v1.2.otnnue')
    p.add_argument('--output',type=Path,default=ROOT/'.build/training/policy/nnue-policy-v1.8.otnnue')
    p.add_argument('--library',type=Path,default=ROOT/('.build/native/libothello.'+('dylib' if sys.platform=='darwin' else 'so')))
    p.add_argument('--epochs',type=int,default=15);p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=20261027)
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.threads)<1:p.error('Positive counts required')
    train(args)
