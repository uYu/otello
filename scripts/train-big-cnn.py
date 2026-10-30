#!/usr/bin/env python3
"""Large residual policy/outcome CNN, trained offline on the original frozen million positions."""
import argparse,importlib.util,json,shutil,tarfile,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cnn_suite',ROOT/'scripts/train-cnn-policy-suite.py')
suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)
h=suite.helpers

class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv1=nn.Conv2d(c,c,3,padding=1,bias=False);self.bn1=nn.BatchNorm2d(c)
        self.conv2=nn.Conv2d(c,c,3,padding=1,bias=False);self.bn2=nn.BatchNorm2d(c)
    def forward(self,x):return (x+self.bn2(self.conv2(self.bn1(self.conv1(x)).relu()))).relu()

class BigCnn(nn.Module):
    def __init__(self,channels=256,blocks=8,head_width=256):
        super().__init__();self.channels=channels;self.blocks=blocks;self.head_width=head_width
        self.stem=nn.Sequential(nn.Conv2d(2,channels,3,padding=1,bias=False),nn.BatchNorm2d(channels),nn.ReLU())
        self.tower=nn.Sequential(*(Block(channels) for _ in range(blocks)))
        self.fc=nn.Linear(channels*64,head_width);self.value=nn.Linear(head_width,1);self.policy=nn.Linear(head_width,65)
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):nn.init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')
        for block in self.tower:nn.init.zeros_(block.bn2.weight)
    def forward(self,boards,turns):
        relative=boards*turns[:,None]
        x=torch.stack((relative==1,relative==-1),1).float().reshape(-1,2,8,8)
        x=self.fc(self.tower(self.stem(x)).flatten(1)).relu()
        return self.value(x).squeeze(1).tanh(),self.policy(x)

def predict(net,data,split,device,batch_size=256):
    net.eval();values=[];probs=[]
    with torch.inference_mode():
        for start in range(0,len(data[split+'_board']),batch_size):
            part=slice(start,start+batch_size)
            v,p=net(torch.from_numpy(data[split+'_board'][part]).to(device),torch.from_numpy(data[split+'_turn'][part]).to(device))
            mask=torch.from_numpy(data[split+'_legal'][part]).to(device)
            values.append(v.cpu().numpy());probs.append(p.masked_fill(~mask,-torch.inf).softmax(1).cpu().numpy())
    return np.concatenate(values),np.concatenate(probs)

def atomic_save(value,path):
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);temp.replace(path)

def main(args):
    args.output.mkdir(parents=True,exist_ok=True);published=ROOT/'models/v1.24-big-cnn.json';assert not published.exists()
    torch.set_num_threads(2);torch.manual_seed(20261027);torch.use_deterministic_algorithms(True)
    assert args.device!='mps' or torch.backends.mps.is_available()
    device=torch.device(args.device);started=time.monotonic()
    progress=dict(experiment='cnn-offline-training',complete=False,stage='preparing',currentName='C256 · 8 残差块 · H256',
        currentChannels=256,epoch=0,maxEpochs=30,completedModels=0,totalModels=1)
    def update(**kwargs):
        progress.update(kwargs,seconds=round(time.monotonic()-started,2));h.write_json(args.output/'progress.json',progress)
    update()
    cache=ROOT/'.build/training/v1/positions.npz';labels_path=ROOT/'.build/training/policy/labels.npz'
    meta=json.loads(cache.with_suffix('.json').read_text());labels_meta=json.loads(labels_path.with_suffix('.json').read_text())
    assert h.sha(cache)==meta['cacheSha256']==labels_meta['positionsSha256'] and h.sha(labels_path)==labels_meta['cacheSha256']
    previous=json.loads((ROOT/'models/v1.13-cnn2-tuning.json').read_text());assert previous['complete']
    baseline=json.loads((ROOT/'models/v1.9-default.json').read_text());assert h.sha(ROOT/baseline['defaultModel'])==baseline['modelSha256']
    data=dict(np.load(cache));labels=dict(np.load(labels_path))
    for split in ('train','validation','test'):
        for key in ('legal','move'):data[split+'_'+key]=labels[split+'_'+key]
    sources=['scripts/train-big-cnn.py','scripts/benchmark-big-cnn.py','scripts/verify-big-cnn.py','scripts/train-cnn-policy-suite.py',
        'scripts/train-mixed-5m.py','scripts/train-deep.py','scripts/train-outcome.py','scripts/train-policy.py','scripts/train-nnue.py']
    net=BigCnn().to(device)
    protocol=dict(version='v1.24-big-cnn',status='candidate',defaultPromoted=False,architecture='relative2x8x8 -> C256 Conv3x3/BN/ReLU -> 8 residual blocks each2x Conv3x3/BN -> Flatten16384 -> FC256/ReLU -> value1/tanh and policy65',
        channels=256,residualBlocks=8,headWidth=256,convolutionLayers=17,parameters=sum(p.numel() for p in net.parameters()),
        epochs=30,batchSize=args.batch_size,learningRate=.001,cosineEtaMin=.00001,weightDecay=.001,gradientNormClip=5.,valueWeight=4.,policyWeight=1.,
        seed=20261027,dataSeed=20261028,precision='AMP FP16 training with FP32 master weights, batchnorm/optimizer state and FP32 evaluation' if args.amp else 'FP32',device=args.device,
        targets='recorded side-to-move terminal outcome +1/0/-1 and legal recorded move; no distillation labels or added games',
        initialization='from scratch; Kaiming conv initialization and zero final BN scale in each residual block',
        selection='save validation MSE+CE minimum and validation value-MSE minimum, evaluate held-out test only after all30 epochs',
        dataset=meta,labels=labels_meta,baselineModelSha256=baseline['modelSha256'],previousCnnReportSha256=h.sha(ROOT/'models/v1.13-cnn2-tuning.json'),
        preflight=json.loads((ROOT/'.build/training/big-cnn-v1.24-checks/report.json').read_text()),
        verification=json.loads((ROOT/'.build/training/big-cnn-v1.24-checks/verification.json').read_text()),
        torchVersion=torch.__version__,sourceSha256={p:h.sha(ROOT/p) for p in sources},
        limits='Offline FP32 CNN only; no native model/search integration or strength comparison; much larger model may need different optimization and data scale')
    protocol_path=args.output/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol,'Resume protocol changed'
    else:h.write_json(protocol_path,protocol)
    archive=ROOT/'models/archive/v1.24-big-cnn-sources.tar.gz'
    if not archive.exists():
        with tarfile.open(archive,'w:gz') as tar:
            for p in sources:tar.add(ROOT/p,arcname=p)
    monitor=ROOT/'models/current-experiment.json';old=json.loads(monitor.read_text())
    if not (args.output/'previous-monitor.json').exists():h.write_json(args.output/'previous-monitor.json',old)
    live=str((args.output/'progress.json').relative_to(ROOT));entries=[r for r in old['experiments'] if r.get('status')=='stopped' and r.get('games',0)>0]
    entries.append(dict(name='大 CNN · C256 / 8 残差块 / H256',focus='big CNN',games=0,runtimeReport=live,report=str(published.relative_to(ROOT))))
    h.write_json(monitor,dict(version='大 CNN 离线训练',experiments=entries))
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30,eta_min=.00001)
    scaler=torch.amp.GradScaler(args.device,init_scale=1024,enabled=args.amp)
    rng=torch.Generator().manual_seed(20261028);best=dict(joint=float('inf'),value=float('inf'));best_epoch={};history=[];first_epoch=1
    checkpoint=args.output/'resume.pt'
    if checkpoint.exists():
        resume=torch.load(checkpoint,map_location=device,weights_only=False);net.load_state_dict(resume['state_dict']);opt.load_state_dict(resume['optimizer']);scheduler.load_state_dict(resume['scheduler'])
        scaler.load_state_dict(resume['scaler']);rng.set_state(resume['data_rng'].cpu());best=resume['best'];best_epoch=resume['bestEpoch'];history=resume['history'];first_epoch=resume['epoch']+1
        update(resumedFromEpoch=resume['epoch'])
    boards=torch.from_numpy(data['train_board']);turns=torch.from_numpy(data['train_turn']);target=torch.from_numpy(np.sign(data['train_target']).astype(np.float32))
    moves=torch.from_numpy(data['train_move'].astype(np.int64));legal=torch.from_numpy(data['train_legal']);perms=torch.from_numpy(h.policy.value.symmetries());inverse=torch.argsort(perms,1)
    for epoch in range(first_epoch,31):
        net.train();order=torch.randperm(len(boards),generator=rng);epoch_started=time.monotonic();accum_loss=0.;count=0;skipped_steps=0
        update(stage='training',epoch=epoch)
        for offset in range(0,len(boards),args.batch_size):
            ids=order[offset:offset+args.batch_size];sym=torch.randint(8,(len(ids),),generator=rng);p=perms[sym]
            b=torch.gather(boards[ids],1,p).to(device);t=turns[ids].to(device)
            mask=torch.cat((torch.gather(legal[ids,:64],1,p),legal[ids,64:]),1).to(device)
            move=torch.where(moves[ids]==64,64,inverse[sym,moves[ids].clamp(max=63)]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=args.device,dtype=torch.float16,enabled=args.amp):
                v,logits=net(b,t);loss=4*(v.float()-target[ids].to(device)).square().mean()+nn.functional.cross_entropy(logits.masked_fill(~mask,-1e4 if args.amp else -1e9),move)
            if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
            old_scale=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=not args.amp)
            scaler.step(opt);scaler.update();skipped_steps+=scaler.get_scale()<old_scale
            accum_loss+=float(loss.detach().cpu())*len(ids);count+=len(ids)
            if offset%16384==0:update(processedPositions=(epoch-1)*len(boards)+count,epochPositions=count,epochPositionsTotal=len(boards),trainingLoss=accum_loss/count)
        score=h.metrics(*predict(net,data,'validation',device,args.batch_size),data,'validation');assert np.isfinite(score['selectionLoss'])
        row=dict(epoch=epoch,ampScale=scaler.get_scale(),ampSkippedSteps=skipped_steps,learningRate=opt.param_groups[0]['lr'],epochSeconds=round(time.monotonic()-epoch_started,2),trainingLoss=accum_loss/count,**score);history.append(row)
        for criterion,metric in [('joint',score['selectionLoss']),('value',score['value']['outcomeMse'])]:
            if metric<best[criterion]:
                best[criterion]=metric;best_epoch[criterion]=epoch
                atomic_save(dict(channels=256,blocks=8,headWidth=256,epoch=epoch,criterion=criterion,state_dict={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}),args.output/(criterion+'.pt'))
        scheduler.step();h.write_json(args.output/'history.json',history)
        atomic_save(dict(epoch=epoch,state_dict=net.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),data_rng=rng.get_state(),best=best,bestEpoch=best_epoch,history=history),checkpoint)
        update(epoch=epoch,validation=score,bestEpoch=best_epoch,epochPositions=len(boards));print(json.dumps(row),flush=True)
    update(stage='frozen-evaluation',completedModels=1);candidates=[]
    for criterion in ('joint','value'):
        path=args.output/(criterion+'.pt');bundle=torch.load(path,map_location='cpu',weights_only=True)
        candidate=BigCnn().eval();candidate.load_state_dict(bundle['state_dict']);candidate.to(device)
        record=dict(name='big-cnn-'+criterion,channels=256,criterion=criterion,epoch=bundle['epoch'],parameters=protocol['parameters'],metrics={})
        for split in ('validation','test'):
            fp=predict(candidate,data,split,device,args.batch_size);record['metrics'][split]=h.metrics(*fp,data,split)
            if split=='test':np.savez_compressed(args.output/(criterion+'-test-predictions.npz'),value=fp[0],policy=fp[1],board=data['test_board'],turn=data['test_turn'])
        candidate.cpu();sample_b=torch.from_numpy(data['validation_board'][:8]);sample_t=torch.from_numpy(data['validation_turn'][:8])
        script_path=args.output/(criterion+'.torchscript.pt');traced=torch.jit.trace(candidate,(sample_b,sample_t));traced.save(str(script_path));reloaded=torch.jit.load(str(script_path)).eval()
        with torch.inference_mode():
            a=candidate(sample_b,sample_t);b=reloaded(sample_b,sample_t)
            error=max(float(abs(x-y).max()) for x,y in zip(a,b));assert error<1e-6
        record['torchscriptParity']=dict(positions=8,maxError=error)
        for src,suffix in ((path,'.pt'),(script_path,'.torchscript.pt')):
            dst=ROOT/'models'/('cnn-c256-r8-h256-'+criterion+'-v1.24'+suffix)
            if dst.exists():assert h.sha(dst)==h.sha(src)
            else:shutil.copy2(src,dst)
        record['model']='models/cnn-c256-r8-h256-'+criterion+'-v1.24.pt';record['sha256']=h.sha(ROOT/record['model']);candidates.append(record)
        del candidate,reloaded,traced
    report=dict(**protocol,complete=True,experiment='cnn-offline-training',candidates=candidates,history=history,bestEpoch=best_epoch,
        baselineMetrics=previous['baselineMetrics'],smallCnnReference=[r for r in previous['candidates'] if r['name'] in ('cosine-joint','cosine-v4-value')],seconds=round(time.monotonic()-started,2))
    assert all(h.sha(ROOT/p)==digest for p,digest in protocol['sourceSha256'].items())
    h.write_json(args.output/'report.json',report);h.write_json(published,report)
    lines=['# 大 CNN 离线训练','','C256、8个残差块（17层卷积）、共享H256、价值/策略双头，从头FP32训练30轮。原百万训练局面，同一验证/测试集；不使用蒸馏标签。',
        '验证集分别选择联合损失与价值MSE最优权重，全部训练后评估测试集。未接入原生搜索，尚未对战，默认保持v1.9。','',
        '| 模型 | 胜负MSE | 非和局准确率 | 走法命中率 |','|---|---:|---:|---:|']
    scores=[('v1.9',previous['baselineMetrics']['test'])]+[(r['name'],r['metrics']['test']) for r in report['smallCnnReference']+candidates]
    for name,s in scores:lines.append(f"| {name} | {s['value']['outcomeMse']:.6f} | {s['value']['nonDrawSignAccuracy']:.2%} | {s['policy']['top1']:.2%} |")
    doc=ROOT/'models/v1.24-big-cnn-comparison.md';doc.write_text('\n'.join(lines)+'\n')
    artifacts=[published,doc]+list((ROOT/'models').glob('cnn-c256-r8-h256-*-v1.24*.pt'))
    h.write_json(ROOT/'models/archive/v1.24-big-cnn-sources.json',dict(archive=str(archive.relative_to(ROOT)),sha256=h.sha(archive),files=protocol['sourceSha256'],artifactSha256={str(p.relative_to(ROOT)):h.sha(p) for p in artifacts}))
    update(stage='complete',complete=True,candidates=candidates,baselineMetrics=previous['baselineMetrics']);print(json.dumps(dict(complete=True)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'.build/training/big-cnn-v1.24-run1')
    p.add_argument('--device',choices=['cpu','mps'],default='mps');p.add_argument('--batch-size',type=int,default=256);p.add_argument('--amp',action='store_true');args=p.parse_args()
    assert args.batch_size>0
    try:main(args)
    except Exception as exc:
        path=args.output/'progress.json'
        if path.exists():d=json.loads(path.read_text());d.update(stage='failed',error=str(exc));h.write_json(path,d)
        raise
