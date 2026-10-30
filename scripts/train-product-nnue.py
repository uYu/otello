#!/usr/bin/env python3
"""Controlled QAT training: ordinary NNUE256 and parameter-efficient paired-product NNUE."""
import argparse, importlib.util, json, shutil, struct, tarfile, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cnn_helpers',ROOT/'scripts/train-cnn-policy-suite.py')
suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)
h=suite.helpers
CONFIGS=[dict(name='linear-control',product=False),dict(name='paired-product',product=True)]

def quant(x,scale):
    y=torch.round(x*scale)/scale
    return x+(y-x).detach()

def affine(layer,x,bias_scale):
    return nn.functional.linear(x,quant(layer.weight,256),quant(layer.bias,bias_scale))

class ProductNet(nn.Module):
    def __init__(self,product):
        super().__init__();self.product=product
        self.transform=nn.Linear(128,256)
        width=256 if product else 512
        self.value=nn.Linear(width,1)
        self.policy_hidden=nn.Linear(width,32)
        self.policy_out=nn.Linear(32,65)
    def forward(self,boards,turns):
        own=boards*turns[:,None]
        x=torch.cat((own==1,own==-1),1).float().to(self.transform.weight.dtype)
        a=affine(self.transform,x,256).clamp(0,1)
        b=affine(self.transform,torch.cat((x[:,64:],x[:,:64]),1),256).clamp(0,1)
        if self.product:
            def pair(z):
                raw=z[:,:128]*z[:,128:]
                rounded=torch.floor(raw*256+.5)/256
                return raw+(rounded-raw).detach()
            a,b=pair(a),pair(b)
        joined=torch.cat((a,b),1)
        return affine(self.value,joined,65536).squeeze(1).tanh(),affine(self.policy_out,affine(self.policy_hidden,joined,65536).clamp(0,1),65536)

def packed(tensor,scale,dtype):
    a=np.rint(tensor.detach().cpu().numpy()*scale);bounds=np.iinfo(dtype)
    assert np.isfinite(a).all() and a.min()>=bounds.min and a.max()<=bounds.max
    return a.astype(dtype).tobytes()

def export(net):
    out=struct.pack('<8sIIII',b'OTNNUE6\0' if net.product else b'OTNNUE4\0',128,256,256,256)
    out+=packed(net.transform.weight.T,256,'<i2')+packed(net.transform.bias,256,'<i4')
    if net.product:out+=struct.pack('<III',256,32,65)
    out+=packed(net.value.weight,256,'<i2')+packed(net.value.bias,65536,'<i4')
    if not net.product:out+=struct.pack('<II',32,65)
    for layer in (net.policy_hidden,net.policy_out):
        out+=packed(layer.weight,256,'<i2')+packed(layer.bias,65536,'<i4')
    return out

def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'protocol.json').exists() and not (ROOT/'models/v1.22-product-nnue.json').exists()
    started=time.monotonic()
    progress=dict(experiment='cnn-offline-training',complete=False,stage='queued',currentName='等待 C16/H32 对战结束',
                  currentChannels=256,epoch=0,maxEpochs=30,completedModels=0,totalModels=2)
    def update(**kwargs):
        progress.update(kwargs,seconds=round(time.monotonic()-started,2));h.write_json(args.output/'progress.json',progress)
    update()
    monitor=ROOT/'models/current-experiment.json';config=json.loads(monitor.read_text())
    config['experiments'].append(dict(name='乘积交互 NNUE · 普通结构同训练对照',focus='product NNUE',games=0,
        runtimeReport=str((args.output/'progress.json').relative_to(ROOT)),report='models/v1.22-product-nnue.json'))
    h.write_json(monitor,config)
    while True:
        match=json.loads(args.wait_match.read_text()) if args.wait_match.exists() else {}
        if match.get('complete') or match.get('stopped'):break
        time.sleep(5)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    assert args.device!='mps' or torch.backends.mps.is_available()
    device=torch.device(args.device)
    cache=ROOT/'.build/training/v1/positions.npz';labels_path=ROOT/'.build/training/policy/labels.npz'
    meta=json.loads(cache.with_suffix('.json').read_text());label_meta=json.loads(labels_path.with_suffix('.json').read_text())
    assert h.sha(cache)==meta['cacheSha256']==label_meta['positionsSha256'] and h.sha(labels_path)==label_meta['cacheSha256']
    baseline=json.loads((ROOT/'models/v1.9-default.json').read_text())
    assert h.sha(ROOT/baseline['defaultModel'])==baseline['modelSha256']
    data=dict(np.load(cache));labels=dict(np.load(labels_path))
    for split in ('train','validation','test'):
        for key in ('legal','move'):data[split+'_'+key]=labels[split+'_'+key]
    sources=['scripts/train-product-nnue.py','scripts/verify-product-nnue.py','scripts/benchmark-product-nnue.py',
             'scripts/check-engineering.py','scripts/train-cnn-policy-suite.py','scripts/train-mixed-5m.py',
             'scripts/train-deep.py','scripts/train-outcome.py','scripts/train-policy.py','scripts/train-nnue.py']
    sources += [str(p.relative_to(ROOT)) for p in (ROOT/'experiments/product-nnue-v1.22').rglob('*') if p.is_file()]
    protocol=dict(version='v1.22-product-nnue',status='candidate',defaultPromoted=False,
        architecture='shared 128->256 clipped linear transform per view; product splits 256 into halves and computes 128 rounded products per view; value 256->1 and policy 256->32->65; control uses original 512-dimensional dual-view heads',
        confounds='Product changes interaction and compresses head input 512 to256; not a pure multiplication-only ablation. Same-protocol fully trained linear control is included.',
        configs=CONFIGS,epochs=30,batchSize=1024,learningRate=.001,cosineEtaMin=.00001,weightDecay=.001,valueWeight=4.,policyWeight=1.,
        training='Both from scratch, all layers jointly trained, identical data/order/D4 random sequence; QAT of weights and biases throughout, STE for product rounding; no distillation or custom game features',
        quantization='transform weights/bias scale256, heads weight256 bias65536; product integer (a*b+128)//256, a,b in0..256',
        checkpointSelection='validation MSE+CE minimum and value MSE minimum; test after both models freeze',
        seed=20261027,dataSeed=20261028,device=args.device,dataset=meta,labels=label_meta,
        baselineManifest='models/v1.9-default.json',baselineModelSha256=baseline['modelSha256'],
        librarySha256=h.sha(args.library),nativeVerification=json.loads((ROOT/'.build/training/product-nnue-v1.22-checks/report.json').read_text()),
        torchVersion=torch.__version__,sourceSha256={p:h.sha(ROOT/p) for p in sources})
    h.write_json(args.output/'protocol.json',protocol)
    archive=ROOT/'models/archive/v1.22-product-nnue-sources.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'w:gz') as tar:
        for p in sources:tar.add(ROOT/p,arcname=p)
    boards=torch.from_numpy(data['train_board']);turns=torch.from_numpy(data['train_turn'])
    target=torch.from_numpy(np.sign(data['train_target']).astype(np.float32));move=torch.from_numpy(data['train_move'].astype(np.int64));legal=torch.from_numpy(data['train_legal'])
    perms=torch.from_numpy(h.policy.value.symmetries());inverse=torch.argsort(perms,1)
    reports=[]
    for conf in CONFIGS:
        torch.manual_seed(20261027);rng=torch.Generator().manual_seed(20261028)
        net=ProductNet(conf['product']).to(device);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30,eta_min=.00001)
        history=[];best=dict(joint=float('inf'),value=float('inf'));best_epoch={}
        update(stage='training',currentName=conf['name'],epoch=0,validation=None)
        for epoch in range(1,31):
            net.train();order=torch.randperm(len(boards),generator=rng)
            for offset in range(0,len(boards),1024):
                ids=order[offset:offset+1024];sym=torch.randint(8,(len(ids),),generator=rng);p=perms[sym]
                b=torch.gather(boards[ids],1,p).to(device);t=turns[ids].to(device)
                mask=torch.cat((torch.gather(legal[ids,:64],1,p),legal[ids,64:]),1).to(device)
                m=torch.where(move[ids]==64,64,inverse[sym,move[ids].clamp(max=63)]).to(device)
                v,logits=net(b,t);loss=4*(v-target[ids].to(device)).square().mean()+nn.functional.cross_entropy(logits.masked_fill(~mask,-1e9),m)
                opt.zero_grad();loss.backward();opt.step()
                if offset%65536==0:update(epoch=epoch,processedPositions=(epoch-1)*len(boards)+offset+len(ids))
            score=h.metrics(*suite.predict(net,data,'validation',device),data,'validation');assert np.isfinite(score['selectionLoss'])
            history.append(dict(epoch=epoch,learningRate=opt.param_groups[0]['lr'],**score))
            for criterion,metric in [('joint',score['selectionLoss']),('value',score['value']['outcomeMse'])]:
                if metric<best[criterion]:
                    best[criterion]=metric;best_epoch[criterion]=epoch
                    torch.save(dict(product=conf['product'],config=conf,criterion=criterion,epoch=epoch,
                        state_dict={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}),args.output/f"{conf['name']}-{criterion}.pt")
            scheduler.step();print(json.dumps(dict(name=conf['name'],**history[-1])),flush=True)
            update(epoch=epoch,validation=score,bestEpoch=best_epoch)
        reports.append(dict(**conf,history=history,bestEpoch=best_epoch,bestValidation=best))
        h.write_json(args.output/'training-checkpoints.json',reports);update(completedModels=len(reports))
    update(stage='frozen-evaluation');evaluated=[]
    for conf in CONFIGS:
        for criterion in ('joint','value'):
            name=conf['name']+'-'+criterion;path=args.output/(name+'.pt');bundle=torch.load(path,map_location='cpu',weights_only=True)
            net=ProductNet(conf['product']).double().eval();net.load_state_dict(bundle['state_dict'])
            native_path=args.output/(name+'.otnnue');native_path.write_bytes(export(net))
            row=dict(name=name,product=conf['product'],epoch=bundle['epoch'],parameters=sum(p.numel() for p in net.parameters()),
                     bytes=native_path.stat().st_size,sha256=h.sha(native_path),metrics={},nativeParity={})
            for split in ('validation','test'):
                py=suite.predict(net,data,split,torch.device('cpu'));native=h.native_predictions(native_path,data,split,args.library)
                ve,pe=float(abs(py[0]-native[0]).max()),float(abs(py[1]-native[1]).max());assert ve<=.000501 and pe<1e-10,(name,split,ve,pe)
                row['metrics'][split]=h.metrics(*native,data,split)
                row['nativeParity'][split]=dict(positions=len(py[0]),maxValueRoundingError=ve,maxPolicyError=pe)
                if split=='test':np.savez_compressed(args.output/(name+'-predictions.npz'),value=native[0],policy=native[1])
            for suffix in ('.pt','.otnnue'):
                dest=ROOT/'models'/('nnue256-'+name+'-v1.22'+suffix);assert not dest.exists();shutil.copy2(args.output/(name+suffix),dest)
            row['model']='models/nnue256-'+name+'-v1.22.otnnue';evaluated.append(row)
    baseline_metrics={split:h.metrics(*h.native_predictions(ROOT/baseline['defaultModel'],data,split,args.library),data,split) for split in ('validation','test')}
    h.write_json(args.output/'evaluated.json',evaluated);update(stage='speed-benchmark')
    import subprocess,sys
    subprocess.run([sys.executable,str(ROOT/'scripts/benchmark-product-nnue.py'),'--output',str(args.output/'speed.json')],check=True)
    report=dict(**protocol,complete=True,experiment='cnn-offline-training',candidates=evaluated,trainingReports=reports,
                baselineMetrics=baseline_metrics,nativeSpeed=json.loads((args.output/'speed.json').read_text()),seconds=round(time.monotonic()-started,2))
    assert all(h.sha(ROOT/p)==digest for p,digest in protocol['sourceSha256'].items())
    h.write_json(args.output/'report.json',report);h.write_json(ROOT/'models/v1.22-product-nnue.json',report)
    lines=['# 配对乘积 NNUE 离线实验','',protocol['architecture'],'',protocol['confounds'],'',
           '原百万局面、冻结验证/测试集；普通NNUE与乘积NNUE均从头QAT联合训练30轮，相同数据增强顺序与超参数。默认保持v1.9，尚未对战。','',
           '| 模型 | 胜负MSE | 非和局准确率 | 走法命中率 |','|---|---:|---:|---:|']
    for name,score in [('v1.9',baseline_metrics['test'])]+[(r['name'],r['metrics']['test']) for r in evaluated]:
        lines.append(f"| {name} | {score['value']['outcomeMse']:.6f} | {score['value']['nonDrawSignAccuracy']:.2%} | {score['policy']['top1']:.2%} |")
    doc=ROOT/'models/v1.22-product-nnue-comparison.md';doc.write_text('\n'.join(lines)+'\n\n同一原生batch32速度和全量量化导出一致性见JSON报告。\n')
    artifacts=[ROOT/'models/v1.22-product-nnue.json',doc]+list((ROOT/'models').glob('nnue256-*-v1.22.*'))
    h.write_json(ROOT/'models/archive/v1.22-product-nnue-sources.json',dict(archive=str(archive.relative_to(ROOT)),sha256=h.sha(archive),files=protocol['sourceSha256'],artifactSha256={str(p.relative_to(ROOT)):h.sha(p) for p in artifacts}))
    update(stage='complete',complete=True,candidates=[dict(r,channels=256) for r in evaluated],baselineMetrics=baseline_metrics)
    print(json.dumps(dict(complete=True,seconds=report['seconds'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'.build/training/product-nnue-v1.22-run1')
    p.add_argument('--wait-match',type=Path,default=ROOT/'.build/training/light-cnn-match-v1.21/c16h32-vs-v1.9-b32-2s-20-run1.json')
    p.add_argument('--library',type=Path,default=ROOT/'.build/native-product-nnue-v1.22/libothello.dylib')
    p.add_argument('--device',choices=['cpu','mps'],default='mps');args=p.parse_args()
    try:main(args)
    except Exception as exc:
        path=args.output/'progress.json'
        if path.exists():
            d=json.loads(path.read_text());d.update(stage='failed',error=str(exc));h.write_json(path,d)
        raise
