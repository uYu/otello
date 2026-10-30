#!/usr/bin/env python3
"""Train a shared residual NNUE trunk on the approved v1.9 outcome baseline."""
import argparse,ctypes,hashlib,importlib.util,json,struct,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('outcome_trainer',ROOT/'scripts/train-outcome.py')
outcome=importlib.util.module_from_spec(spec);spec.loader.exec_module(outcome)
policy=outcome.policy
BASE_SIZE=104936

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class DeepNet(nn.Module):
    def __init__(self,parent):
        super().__init__()
        if parent[:8]!=b'OTNNUE4\0' or len(parent)!=BASE_SIZE:raise ValueError('Expected v1.9 scalar-outcome parent')
        base=outcome.OutcomeNet(b'OTNNUE3\0'+parent[8:])
        self.transform=base.transform;self.head=base.head
        self.down=nn.Linear(256,32);self.up=nn.Linear(32,256)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
        self.policy_hidden=nn.Linear(512,32);self.policy_out=nn.Linear(32,65)
        offset=67620
        with torch.no_grad():
            for tensor,count,dtype,scale in [(self.policy_hidden.weight,32*512,'<i2',256),(self.policy_hidden.bias,32,'<i4',65536),(self.policy_out.weight,65*32,'<i2',256),(self.policy_out.bias,65,'<i4',65536)]:
                a=np.frombuffer(parent,dtype=dtype,count=count,offset=offset).copy();offset+=a.nbytes
                tensor.copy_(torch.from_numpy(a).reshape(tensor.shape)/scale)
        self.policy_hidden.requires_grad_(False);self.policy_out.requires_grad_(False)

    def forward(self,boards,turns):
        with torch.no_grad():
            x=policy.value.features(boards).to(self.transform.weight.dtype)
            a=self.transform(x).clamp(0,1);b=self.transform(torch.cat((x[:,64:],x[:,:64]),1)).clamp(0,1)
            both=torch.cat((a,b),0)
        both=(both+self.up(self.down(both).clamp(0,1))).clamp(0,1)
        a,b=both.chunk(2,0);actor=(turns>0).unsqueeze(1)
        joined=torch.cat((torch.where(actor,a,b),torch.where(actor,b,a)),1)
        return torch.tanh(self.head(joined).squeeze(1)),self.policy_out(self.policy_hidden(joined).clamp(0,1))

def packed(tensor,scale,dtype):
    a=np.rint(tensor.detach().cpu().numpy()*scale);bounds=np.iinfo(dtype)
    if not np.isfinite(a).all() or a.min()<bounds.min or a.max()>bounds.max:raise ValueError('Deep weights outside format range')
    return a.astype(dtype).tobytes()

def export(net,parent):
    value=packed(net.head.weight,256,'<i2')+packed(net.head.bias,65536,'<i4')
    block=b''.join(packed(tensor,scale,dtype) for tensor,scale,dtype in [(net.down.weight,256,'<i2'),(net.down.bias,65536,'<i4'),(net.up.weight,256,'<i2'),(net.up.bias,65536,'<i4')])
    return b'OTNNUE5\0'+parent[8:outcome.VALUE_OFFSET]+value+parent[outcome.VALUE_END:]+struct.pack('<II',256,32)+block

def quantized_net(binary):
    if binary[:8]!=b'OTNNUE5\0' or len(binary)!=138864 or struct.unpack_from('<II',binary,BASE_SIZE)!=(256,32):raise ValueError('Invalid deep model')
    net=DeepNet(b'OTNNUE4\0'+binary[8:BASE_SIZE]).double();offset=BASE_SIZE+8
    with torch.no_grad():
        for tensor,count,dtype,scale in [(net.down.weight,32*256,'<i2',256),(net.down.bias,32,'<i4',65536),(net.up.weight,256*32,'<i2',256),(net.up.bias,256,'<i4',65536)]:
            a=np.frombuffer(binary,dtype=dtype,count=count,offset=offset).copy();offset+=a.nbytes
            tensor.copy_(torch.from_numpy(a).reshape(tensor.shape)/scale)
    return net

def predictions(net,data,labels,split):
    values=[];probabilities=[]
    with torch.no_grad():
        for start in range(0,len(data[split+'_board']),1024):
            v,p=net(torch.from_numpy(data[split+'_board'][start:start+1024]),torch.from_numpy(data[split+'_turn'][start:start+1024]))
            mask=torch.from_numpy(labels[split+'_legal'][start:start+1024])
            values.append(v.numpy());probabilities.append(p.masked_fill(~mask,-torch.inf).softmax(1).numpy())
    return np.concatenate(values),np.concatenate(probabilities)

def metrics(values,probs,data,labels,split):
    v=outcome.metric(values,np.sign(data[split+'_target']));p=policy.metrics(probs,labels[split+'_move'],labels[split+'_legal'])
    return dict(value=v,policy=p,loss=v['outcomeMse']+p['crossEntropy'])

def train(args):
    if args.output.exists() or args.output.with_suffix('.json').exists():raise ValueError('Choose a fresh experiment path')
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed)
    baseline=json.loads(args.baseline_manifest.read_text());meta=json.loads(args.cache.with_suffix('.json').read_text());label_meta=json.loads(args.labels.with_suffix('.json').read_text())
    assert sha(args.parent)==baseline['modelSha256'] and baseline['version']=='v1.9'
    assert sha(args.cache)==meta['cacheSha256']==label_meta['positionsSha256'] and sha(args.labels)==label_meta['cacheSha256']
    assert meta['seed']==args.seed
    parent=args.parent.read_bytes();data=dict(np.load(args.cache));labels=dict(np.load(args.labels));net=DeepNet(parent)
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.001,weight_decay=.001)
    boards=torch.from_numpy(data['train_board']);turns=torch.from_numpy(data['train_turn']);target=torch.from_numpy(np.sign(data['train_target']).astype(np.float32))
    moves=torch.from_numpy(labels['train_move'].astype(np.int64));legal=torch.from_numpy(labels['train_legal']);perms=torch.from_numpy(policy.value.symmetries());inverse=torch.argsort(perms,dim=1)
    initial=metrics(*predictions(net,data,labels,'validation'),data,labels,'validation')
    best=initial['loss'];state={k:v.detach().clone() for k,v in net.state_dict().items()};history=[dict(epoch=0,**initial)];started=time.perf_counter()
    for epoch in range(args.epochs):
        for indices in torch.randperm(len(boards)).split(args.batch_size):
            symmetry=torch.randint(8,(len(indices),));p=perms[symmetry]
            board=torch.gather(boards[indices],1,p);move=inverse[symmetry,moves[indices]]
            mask=torch.cat((torch.gather(legal[indices,:64],1,p),legal[indices,64:]),1)
            v,logits=net(board,turns[indices]);loss=((v-target[indices])**2).mean()+nn.functional.cross_entropy(logits.masked_fill(~mask,-torch.inf),move)
            opt.zero_grad();loss.backward();opt.step()
        score=metrics(*predictions(net,data,labels,'validation'),data,labels,'validation');history.append(dict(epoch=epoch+1,**score));print(json.dumps(history[-1]),flush=True)
        if score['loss']<best:best=score['loss'];state={k:v.detach().clone() for k,v in net.state_dict().items()}
    net.load_state_dict(state);binary=export(net,parent);qnet=quantized_net(binary)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(binary);torch.save(net.state_dict(),args.output.with_suffix('.pt'))
    report=dict(version='v1.10-deep-candidate',status='candidate',modelFormat='OTNNUE5',valueTarget='side-to-move win +1, draw 0, loss -1',architecture='frozen 128->256; shared 256->32 clipped ReLU->256 residual clipped ReLU; ordered 512 features to value and frozen policy',parameters=sum(p.numel() for p in net.parameters()),trainableParameters=sum(p.numel() for p in net.parameters() if p.requires_grad),bytes=len(binary),sha256=sha(args.output),parent=str(args.parent),parentSha256=sha(args.parent),baselineVersion='v1.9',featuresUnchanged=binary[8:outcome.VALUE_OFFSET]==parent[8:outcome.VALUE_OFFSET],policyBytesUnchanged=binary[outcome.VALUE_END:BASE_SIZE]==parent[outcome.VALUE_END:],loss='outcome MSE + legal-mask recorded-move cross entropy; coefficient 1 each',epochs=args.epochs,seed=args.seed,batchSize=args.batch_size,learningRate=.001,weightDecay=.001,initialResidual='zero up projection: identity at epoch 0',bestEpoch=min(history,key=lambda x:x['loss'])['epoch'],history=history,seconds=time.perf_counter()-started,dataset=meta,labels=label_meta,trainingScriptSha256=sha(__file__),helperSha256={p:sha(ROOT/p) for p in ['scripts/train-outcome.py','scripts/train-policy.py','scripts/train-nnue.py']},torchVersion=torch.__version__,numpyVersion=np.__version__,metrics={})
    for split in ('validation','test'):
        fp=predictions(net,data,labels,split);qp=predictions(qnet,data,labels,split)
        report['metrics'][split]=dict(float=metrics(*fp,data,labels,split),quantized=metrics(*qp,data,labels,split),maxValueQuantizationError=float(abs(fp[0]-qp[0]).max()),maxPolicyQuantizationError=float(abs(fp[1]-qp[1]).max()))
    lib=ctypes.CDLL(str(args.library.resolve()));ptr=ctypes.POINTER(ctypes.c_int)
    lib.ot_nnue_load.argtypes=[ctypes.c_char_p];lib.ot_nnue_load.restype=ctypes.c_void_p;lib.ot_nnue_destroy.argtypes=[ctypes.c_void_p]
    lib.ot_nnue_evaluate.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ptr];lib.ot_nnue_policy.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
    handle=lib.ot_nnue_load(str(args.output.resolve()).encode());assert handle
    values,probs=predictions(qnet,data,labels,'test');count=min(1024,len(values));value_error=policy_error=0
    try:
        for i in range(count):
            cells=(ctypes.c_int*64)(*map(int,data['test_board'][i]));turn=int(data['test_turn'][i]);score=ctypes.c_int();prob=(ctypes.c_double*65)()
            assert lib.ot_nnue_evaluate(handle,cells,turn,ctypes.byref(score))==1 and lib.ot_nnue_policy(handle,cells,turn,prob)==1
            value_error=max(value_error,abs(score.value-values[i]*1000));policy_error=max(policy_error,float(abs(np.array(prob)-probs[i]).max()))
    finally:lib.ot_nnue_destroy(handle)
    assert value_error<=.501 and policy_error<1e-10,(value_error,policy_error)
    report.update(cppParityPositions=count,cppMaxRoundingError=value_error,cppMaxPolicyError=policy_error)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('history','dataset','labels')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);active=json.loads((ROOT/'models/default-model.json').read_text())
    p.add_argument('--parent',type=Path,default=ROOT/active['defaultModel']);p.add_argument('--baseline-manifest',type=Path,default=ROOT/active['baselineManifest'])
    p.add_argument('--cache',type=Path,default=ROOT/'.build/training/v1/positions.npz');p.add_argument('--labels',type=Path,default=ROOT/'.build/training/policy/labels.npz')
    p.add_argument('--output',type=Path,default=ROOT/'.build/training/deep/nnue-deep-v1.10.otnnue');p.add_argument('--library',type=Path,default=ROOT/'.build/native-deep/libothello.dylib')
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=1024);p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=20261027)
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.threads)<1:p.error('Positive counts required')
    train(args)
