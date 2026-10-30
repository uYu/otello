#!/usr/bin/env python3
"""Train one D4-tied global38->16->64 policy head on frozen approved features."""
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
import time
import numpy as np
import torch
from torch.nn import functional as F
from pattern_model import encode,legal_mask,rounded
from pattern_structure import export as export_baseline
from pattern_global_policy import GlobalPolicyNet,permutations,pattern_permutations

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/pattern-global-policy-v1.49'
REPORT=ROOT/'models/v1.49-pattern-global-policy.json'
PROTOCOL=ROOT/'models/v1.49-global-policy-protocol.json'
MANIFEST=ROOT/'models/archive/v1.49-global-policy-sources.json'
BASE=ROOT/'models/pattern32-symmetry-v1.37.otpat'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def write(path,value):
    temp=Path(path).with_suffix(Path(path).suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def frozen_fingerprint(net):
    h=hashlib.sha256()
    for name,tensor in net.state_dict().items():
        if not name.startswith('global_'):h.update(name.encode());h.update(tensor.detach().numpy().tobytes())
    return h.hexdigest()


def numeric_head(net,features):
    """Independent INT64 reference for the quantized global policy correction."""
    def integer(t,scale):
        x=t.detach().numpy().astype(np.float64)*scale
        return (np.sign(x)*np.floor(np.abs(x)+.5)).astype(np.int64)
    def scale256(x,lo,hi):
        y=np.sign(x)*((np.abs(x)+128)//256)
        return np.clip(y,lo,hi)
    first,last=net.global_down,net.global_up
    x=np.asarray(features,dtype=np.int64)
    hidden=scale256(x@integer(first.weight,256).T+integer(first.bias,65536),0,256)
    correction=scale256(hidden@integer(last.weight,256).T+integer(last.bias,65536),-1024,1024)
    return correction.astype(np.float32)/256


def export_head(net,path,baseline_sha,base_model='models/pattern32-symmetry-v1.37.otpat'):
    def integer(t,scale,bits):
        x=t.detach().numpy().astype(np.float64)*scale
        x=np.sign(x)*np.floor(np.abs(x)+.5);bounds=np.iinfo(np.int16 if bits==16 else np.int32)
        assert np.isfinite(x).all() and x.min()>=bounds.min and x.max()<=bounds.max
        return x.astype(np.int64).tolist()
    layers=[]
    for layer in [net.global_down,net.global_up]:
        layers.append(dict(weight=integer(layer.weight,256,16),bias=integer(layer.bias,65536,32),
            sharedWeight=integer(layer.coefficients,256,16),sharedBias=integer(layer.biases,65536,32)))
    write(path,dict(format='OTGLOBALPOLICY1',baseModel=base_model,baseModelSha256=baseline_sha,
        architecture=[38,16,64],channels=1,weightScale=256,biasScale=65536,input='Q(clamp(ordered38 conditional pattern projections,-4,4))',
        output='Residual correction added to the original64 policy logits before legal mask/softmax',
        activations=['quantized clip0..1','quantized clip-4..4'],layers=layers,nativeEngineSupported=False))


def main():
    assert not REPORT.exists(),'Completed reports are immutable'
    WORK.mkdir(parents=True,exist_ok=True);started=time.perf_counter();torch.set_num_threads(2)
    active=read(ROOT/'models/default-model.json');assert active['version']=='v1.37'
    preserve={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'models/default-model.json',ROOT/'models/v1.37-default.json',BASE.with_suffix('.otpat'),BASE.with_suffix('.pt'),ROOT/'src/game/wasm/othello.mjs']}
    cache=ROOT/'.build/training/v1/positions.npz';policyfile=ROOT/'.build/training/v1.8/policy.npz'
    metadata=read(cache.with_suffix('.json'));provenance=read(policyfile.with_suffix('.json'))
    assert sha(cache)==metadata['cacheSha256']==provenance['cacheSha256']
    assert sha(policyfile)==provenance['policySha256']
    protocol=dict(version='v1.49-global-policy',status='candidate',defaultPromoted=False,userRequest='训一版看看：全局38→16→64策略头',
        baseline='models/v1.37-default.json',architecture='Frozen v1.37 ordered38 projections -> D4-tied38->16->64 -> additive64 policy correction',
        channels=1,trainableParameters=237,denseMultiplyAddsPerBoard=1632,freeze='All baseline parameters: table,trunk,value,WDL,conditional gates and original policy biases',
        epochs=15,batchSize=2048,seed=20261029,dataSeed=20261030,threads=2,optimizer='AdamW',learningRate=.001,weightDecay=.0001,
        loss='Legal-next-move cross entropy only',selection='Minimum validation legal-policy CE among identity epoch0 and trained epochs1..15; no test-based selection',
        initialization='Approved trained v1.37 frozen weights; random down layer; zero output correction starts at exact baseline',
        data=dict(positions=1000000,sources=['WTHOR','Logistello'],labels='Original hard next-move labels; no reweighting or relabeling',positionsSha256=sha(cache),policySha256=sha(policyfile)),
        augmentation='Same shuffled rows and uniform D4 choices as v1.37; cached ordered projections and policy/mask permuted consistently',
        preservedSha256=preserve,sourceArchiveManifest=str(MANIFEST.relative_to(ROOT)),
        scope='One offline training experiment. No native implementation, speed/strength match, default promotion or browser change.',
        limits='Single seed, existing historical held-out splits. Frozen scalar projections may limit recoverable local detail. CE selection may differ from top1 selection.')
    assert not PROTOCOL.exists();write(PROTOCOL,protocol)
    config=read(ROOT/'models/current-experiment.json');archive=config.setdefault('archivedExperiments',[])
    for item in config.get('experiments',[]):
        if not any(x.get('report')==item.get('report') for x in archive):archive.insert(0,item)
    config.update(version='v1.49 全局8×8策略头 · 仅策略CE训练',experiments=[dict(name='全局38→16→64策略头 · 冻结价值',focus='global-policy',games=0,runtimeReport=str((WORK/'progress.json').relative_to(ROOT)),report='models/v1.49-global-policy-monitor.json')])
    write(ROOT/'models/current-experiment.json',config)
    history=[];baseline_metrics={};latest=None;best_metrics=None
    def progress(stage,epoch=0,summary='',complete=False):
        write(WORK/'progress.json',dict(experiment='global-policy-training',stage=stage,epoch=epoch,epochs=15,summary=summary,complete=complete,seconds=time.perf_counter()-started,
            validation=latest,bestValidation=best_metrics,baselineValidation=baseline_metrics.get('validation'),selectionMetric='validation policyCrossEntropy'))
    progress('核验数据和冻结参数',summary='准备缓存冻结特征；随后只训练237个策略参数')
    torch.manual_seed(20261029);np.random.seed(20261029)
    net=GlobalPolicyNet(1,torch.load(BASE.with_suffix('.pt'),weights_only=True))
    before=frozen_fingerprint(net);assert sum(p.numel() for p in net.parameters() if p.requires_grad)==237
    reexport=WORK/'baseline-reexport.otpat';export_baseline(net,reexport);assert sha(reexport)==preserve[str(BASE.with_suffix('.otpat').relative_to(ROOT))]
    paths=[ROOT/p for p in ['scripts/train-pattern-global-policy.py','scripts/pattern_global_policy.py','scripts/pattern_model.py','scripts/pattern_structure.py','scripts/watch-match.sh','scripts/watch-matches.py','tests/pattern_global_policy_test.py','models/v1.49-global-policy-protocol.json']]
    assert not MANIFEST.exists()
    archivepath=MANIFEST.with_suffix('.tar.gz')
    with tarfile.open(archivepath,'w:gz') as t:
        for p in paths:t.add(p,arcname=str(p.relative_to(ROOT)))
    write(MANIFEST,dict(archive=str(archivepath.relative_to(ROOT)),sha256=sha(archivepath),files={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    data=np.load(cache);labels=np.load(policyfile);features={};policies={};values={};wdls={};masks={};cache_metadata={}
    net.eval()
    for split in ['validation','train','test']:
        board,turn=data[split+'_board'],data[split+'_turn'];n=len(board)
        features[split]=np.lib.format.open_memmap(WORK/f'{split}-features.npy',mode='w+',dtype=np.int16,shape=(n,38))
        policies[split]=np.lib.format.open_memmap(WORK/f'{split}-policy.npy',mode='w+',dtype=np.float32,shape=(n,64))
        values[split]=np.empty(n,np.float32);wdls[split]=np.empty((n,3),np.float32)
        masks[split]=legal_mask(board,turn);assert masks[split][np.arange(n),labels[split]].all()
        with torch.no_grad():
            for start in range(0,n,2048):
                stop=min(start+2048,n);b=board[start:stop];actor=turn[start:stop]
                codes=torch.from_numpy(encode(b,actor).astype(np.int64));phase=torch.from_numpy((64-np.count_nonzero(b,axis=1)).astype(np.float32)/60)
                value,wdl,projection,policy=net.frozen_features(codes,phase)
                features[split][start:stop]=rounded(projection.clamp(-4,4)*256).numpy().astype(np.int16)
                policies[split][start:stop]=policy.numpy();values[split][start:stop]=value.numpy();wdls[split][start:stop]=wdl.numpy()
                if start%(2048*32)==0:progress('缓存冻结编码器特征',summary=f'{split}：{stop:,}/{n:,} 局面')
        features[split].flush();policies[split].flush()
        cache_metadata[split]=dict(positions=n,featuresSha256=sha(WORK/f'{split}-features.npy'),basePolicySha256=sha(WORK/f'{split}-policy.npy'))
    write(WORK/'cache-provenance.json',dict(baselineSha256=sha(BASE.with_suffix('.otpat')),protocolSha256=sha(PROTOCOL),splits=cache_metadata))
    def evaluate(split):
        logits=[];net.eval()
        with torch.no_grad():
            for start in range(0,len(features[split]),2048):
                x=torch.from_numpy(features[split][start:start+2048].astype(np.float32)/256)
                logits.append(policies[split][start:start+2048]+net.policy_correction(x).numpy())
        logits=np.concatenate(logits);mask=masks[split];target=labels[split]
        masked=torch.from_numpy(np.where(mask,logits,-1e9))
        return dict(policyCrossEntropy=float(F.cross_entropy(masked,torch.from_numpy(target.astype(np.int64)))),policyTop1=float(np.mean(masked.numpy().argmax(1)==target)),
            mse=float(np.mean((values[split]-np.sign(data[split+'_target']))**2)))
    baseline_metrics['validation']=evaluate('validation')
    latest=dict(baseline_metrics['validation'],epoch=0);best_metrics=dict(latest);best=latest['policyCrossEntropy'];best_epoch=0
    state={k:v.detach().clone() for k,v in net.state_dict().items()};history.append(dict(epoch=0,**baseline_metrics['validation']))
    pp=pattern_permutations();perm=permutations(8);inverse=np.argsort(perm,axis=1)
    # Cache augmentation must match computing the full transformed frozen board.
    with torch.no_grad():
        indices=np.arange(256);sym=np.arange(256)%8;b=np.take_along_axis(data['validation_board'][indices],perm[sym],axis=1)
        codes=torch.from_numpy(encode(b,data['validation_turn'][indices]).astype(np.int64));phase=torch.from_numpy((64-np.count_nonzero(b,axis=1)).astype(np.float32)/60)
        v,w,projection,policy=net.frozen_features(codes,phase)
        np.testing.assert_array_equal(rounded(projection.clamp(-4,4)*256).numpy().astype(np.int16),np.take_along_axis(features['validation'][indices],pp[sym],axis=1))
        np.testing.assert_allclose(policy.numpy(),np.take_along_axis(policies['validation'][indices],perm[sym],axis=1),atol=2e-6,rtol=0)
    order=torch.Generator().manual_seed(20261030);augmentation=np.random.default_rng(20261030);sequence=hashlib.sha256()
    optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
    for epoch in range(1,16):
        progress('训练全局策略头',epoch-1,'按验证策略CE选模；上一轮指标持续保留');net.train();losses=[]
        for rows in torch.randperm(len(features['train']),generator=order).split(2048):
            indices=rows.numpy();sym=augmentation.integers(8,size=len(indices),dtype=np.int8);sequence.update(indices.tobytes());sequence.update(sym.tobytes())
            x=np.take_along_axis(features['train'][indices],pp[sym],axis=1).astype(np.float32)/256
            base=np.take_along_axis(policies['train'][indices],perm[sym],axis=1)
            mask=np.take_along_axis(masks['train'][indices],perm[sym],axis=1)
            target=inverse[sym,labels['train'][indices]]
            logits=torch.from_numpy(base)+net.policy_correction(torch.from_numpy(x))
            loss=F.cross_entropy(logits.masked_fill(~torch.from_numpy(mask),-1e9),torch.from_numpy(target.astype(np.int64)))
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],5);optimizer.step();losses.append(float(loss.detach()))
        score=evaluate('validation');latest=dict(score,epoch=epoch)
        history.append(dict(epoch=epoch,trainPolicyCE=float(np.mean(losses)),**score))
        if score['policyCrossEntropy']<best:
            best=score['policyCrossEntropy'];best_epoch=epoch;best_metrics=dict(latest);state={k:v.detach().clone() for k,v in net.state_dict().items()}
        write(WORK/'history.json',history);progress('训练全局策略头',epoch)
        print(json.dumps(dict(seconds=time.perf_counter()-started,**history[-1],bestEpoch=best_epoch)),flush=True)
    assert sequence.hexdigest()==read(BASE.with_suffix('.json'))['dataOrderAndAugmentationSha256']
    assert frozen_fingerprint(net)==before
    net.load_state_dict(state);net.eval()
    model=ROOT/'models/pattern32-global-policy16-v1.49.pt';assert not model.exists();torch.save(state,model)
    head=ROOT/'models/pattern32-global-policy16-v1.49.head.json';export_head(net,head,sha(BASE.with_suffix('.otpat')))
    selected_metrics={};numeric_checks={}
    for split in ['validation','test']:
        selected_metrics[split]=evaluate(split)
        reference=numeric_head(net,features[split])
        with torch.no_grad():actual=net.policy_correction(torch.from_numpy(features[split].astype(np.float32)/256)).numpy()
        np.testing.assert_array_equal(actual,reference)
        numeric_checks[split]=dict(positions=len(actual),all64CorrectionsExactlyEqual=True)
    # Baseline test policy is evaluated only after candidate selection is frozen.
    mask=masks['test'];target=labels['test'];base_logits=torch.from_numpy(np.where(mask,np.asarray(policies['test']),-1e9))
    baseline_metrics['test']=dict(policyCrossEntropy=float(F.cross_entropy(base_logits,torch.from_numpy(target.astype(np.int64)))),policyTop1=float(np.mean(base_logits.numpy().argmax(1)==target)),mse=float(np.mean((values['test']-np.sign(data['test_target']))**2)))
    with torch.no_grad():
        codes=torch.from_numpy(encode(data['test_board'][:512],data['test_turn'][:512]).astype(np.int64));phase=torch.from_numpy((64-np.count_nonzero(data['test_board'][:512],axis=1)).astype(np.float32)/60)
        full=net(codes,phase)
        np.testing.assert_array_equal(full[0].numpy(),values['test'][:512]);np.testing.assert_array_equal(full[1].numpy(),wdls['test'][:512])
        cached=np.asarray(policies['test'][:512])+numeric_head(net,features['test'][:512])
        full_cached_error=float(np.max(np.abs(full[2].numpy()-cached)))
        np.testing.assert_allclose(full[2].numpy(),cached,atol=2e-6,rtol=0)
    for name,digest in preserve.items():assert sha(ROOT/name)==digest,name
    result=dict(version='v1.49-global-policy',complete=True,status='candidate',defaultPromoted=False,baseline='models/v1.37-default.json',bestEpoch=best_epoch,
        architecture=[38,16,64],trainableParameters=237,totalParameters=sum(p.numel() for p in net.parameters()),denseMultiplyAddsPerBoard=1632,
        metrics=selected_metrics,baselineMetrics=baseline_metrics,history=history,frozenParametersIdentical=True,frozenParametersSha256=before,valueAndWDLUnchanged=True,
        cachedAugmentationParityPositions=256,fullCachedPredictionParityPositions=512,fullCachedPredictionMaxPolicyLogitError=full_cached_error,integerHeadReference=numeric_checks,
        dataOrderAndAugmentationSha256=sequence.hexdigest(),model=str(model.relative_to(ROOT)),modelSha256=sha(model),headExport=str(head.relative_to(ROOT)),headExportSha256=sha(head),
        protocolSha256=sha(PROTOCOL),sourceArchiveManifest=str(MANIFEST.relative_to(ROOT)),sourceArchiveManifestSha256=sha(MANIFEST),preservedSha256=preserve,
        seconds=time.perf_counter()-started,nativeImplemented=False,matchesRun=False,limits=protocol['limits'])
    write(REPORT,result);write(model.with_suffix('.json'),result)
    latest=dict(selected_metrics['validation'],epoch=best_epoch)
    progress('训练完成：按验证策略CE选定',15,f'选第{best_epoch}轮 · 测试策略 CE {selected_metrics["test"]["policyCrossEntropy"]:.6f} · 命中率 {selected_metrics["test"]["policyTop1"]:.2%}',True)
    write(ROOT/'models/v1.49-global-policy-monitor.json',read(WORK/'progress.json'))
    print(json.dumps(dict(complete=True,bestEpoch=best_epoch,baseline=baseline_metrics,metrics=selected_metrics,seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
