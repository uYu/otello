#!/usr/bin/env python3
"""Train the entire global38->16->64 Pattern model from random initialization."""
import hashlib,importlib.util,json,tarfile,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pattern_model import encode,legal_mask,rounded
from pattern_structure import StructureNet,export as export_base
from pattern_global_policy import GlobalPolicyNet,permutations
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.build/training/pattern-global-policy-full-v1.49'
REPORT=ROOT/'models/v1.49-pattern-global-policy-full.json'
PROTOCOL=ROOT/'models/v1.49-global-policy-full-protocol.json'
MANIFEST=ROOT/'models/archive/v1.49-global-policy-full-sources.json'
BASE=ROOT/'models/pattern32-symmetry-v1.37.otpat'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
spec=importlib.util.spec_from_file_location('global_policy_export_helpers',ROOT/'scripts/train-pattern-global-policy.py')
exports=importlib.util.module_from_spec(spec);spec.loader.exec_module(exports)
write=exports.write
spec=importlib.util.spec_from_file_location('global_policy_full_metrics',ROOT/'scripts/train-pattern-outcome.py')
metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)


def main():
    assert not REPORT.exists(),'Completed reports are immutable'
    WORK.mkdir(parents=True,exist_ok=True);started=time.perf_counter();torch.set_num_threads(2)
    assert read(ROOT/'models/default-model.json')['version']=='v1.37'
    preserve={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'models/default-model.json',ROOT/'models/v1.37-default.json',BASE,BASE.with_suffix('.pt'),ROOT/'src/game/wasm/othello.mjs']}
    cache=ROOT/'.build/training/v1/positions.npz';policyfile=ROOT/'.build/training/v1.8/policy.npz'
    metadata=read(cache.with_suffix('.json'));provenance=read(policyfile.with_suffix('.json'))
    assert sha(cache)==metadata['cacheSha256']==provenance['cacheSha256']
    assert sha(policyfile)==provenance['policySha256']
    baseline=read(BASE.with_suffix('.json'))
    protocol=dict(version='v1.49-global-policy-full',status='candidate',defaultPromoted=False,userRequest='训一版看看；不只更新新增237参数，改为全网从头训练',
        baseline='models/v1.37-default.json',architecture='D4-tied ordered38 pattern projections -> shared38->16->64 additive policy correction; original value and conditional policy branches',
        trainableParameters=916987,additionalPolicyParameters=237,freezeBaseline=False,pretrainedWeightsLoaded=False,
        epochs=15,trainPositions=1000000,batchSize=2048,seed=20261029,dataSeed=20261030,threads=2,optimizer='AdamW',learningRate=.001,weightDecay=.0001,
        loss='outcome MSE +0.1 WDL CE +0.1 legal-next-move CE; same coefficients as v1.37',
        selection='Minimum original validation legal-policy CE among epochs1..15. Report top1 and value MSE; no test-based selection.',
        initialization='Random full network; shared table/trunk/original head initial values identical to v1.37. Extra global down random, global up zero.',
        data=dict(sources=['WTHOR','Logistello'],labels='Original hard next-move labels and outcome labels, no reweighting/relabeling',positionsSha256=sha(cache),policySha256=sha(policyfile)),
        augmentation='Same row shuffle and uniform D4 board/label/mask transformations as v1.37',
        preservedSha256=preserve,sourceArchiveManifest=str(MANIFEST.relative_to(ROOT)),
        scope='One offline full retraining run; no native deployment, throughput or strength match and no default promotion.',
        limits='Single seed and historical held-out splits. Validation selection differs from original v1.37 value-MSE selection; results cannot isolate architecture alone.')
    assert not PROTOCOL.exists();write(PROTOCOL,protocol)
    config=read(ROOT/'models/current-experiment.json');archive=config.setdefault('archivedExperiments',[])
    for item in config.get('experiments',[]):
        if not any(x.get('report')==item.get('report') for x in archive):archive.insert(0,item)
    config.update(version='v1.49 全局8×8策略头 · 全网从头训练',experiments=[dict(name='全局38→16→64 · 全网91.7万参数重训',focus='global-policy-full',games=0,runtimeReport=str((WORK/'progress.json').relative_to(ROOT)),report='models/v1.49-global-policy-full-monitor.json')])
    write(ROOT/'models/current-experiment.json',config)
    latest=None;best_metrics=None;history=[]
    def progress(stage,epoch=0,summary='',complete=False):
        write(WORK/'progress.json',dict(experiment='global-policy-training',stage=stage,epoch=epoch,epochs=15,summary=summary,complete=complete,freezeBaseline=False,
            seconds=time.perf_counter()-started,validation=latest,bestValidation=best_metrics,baselineValidation=baseline['metrics']['validation'],selectionMetric='validation policyCrossEntropy'))
    progress('核验数据并初始化全网',summary='全部916,987个参数可训练；不加载已训练权重')
    torch.manual_seed(20261029);np.random.seed(20261029);net=GlobalPolicyNet(1,freeze_baseline=False)
    assert all(p.requires_grad for p in net.parameters()) and sum(p.numel() for p in net.parameters())==916987
    torch.manual_seed(20261029);initial_base=StructureNet('symmetry')
    for key,value in initial_base.state_dict().items():assert torch.equal(value,net.state_dict()[key]),key
    initial_embedding=sha_bytes(net.embedding.weight.detach().numpy().tobytes());assert initial_embedding==baseline['initialEmbeddingSha256']
    initial_trunk=sha_bytes(net.trunk.weight.detach().numpy().tobytes())
    files=[ROOT/p for p in ['scripts/train-pattern-global-policy-full.py','scripts/train-pattern-global-policy.py','scripts/pattern_global_policy.py','scripts/pattern_model.py','scripts/pattern_structure.py','scripts/train-pattern-outcome.py','scripts/train-pattern.py','scripts/watch-match.sh','scripts/watch-matches.py','tests/pattern_global_policy_test.py','tests/global_policy_monitor_test.py','models/v1.49-global-policy-full-protocol.json']]
    archivepath=MANIFEST.with_suffix('.tar.gz');assert not MANIFEST.exists()
    with tarfile.open(archivepath,'w:gz') as out:
        for p in files:out.add(p,arcname=str(p.relative_to(ROOT)))
    write(MANIFEST,dict(archive=str(archivepath.relative_to(ROOT)),sha256=sha(archivepath),files={str(p.relative_to(ROOT)):sha(p) for p in files}))
    data=dict(np.load(cache));labels=dict(np.load(policyfile));masks={k:legal_mask(data[k+'_board'],data[k+'_turn']) for k in labels}
    assert all(masks[k][np.arange(len(labels[k])),labels[k]].all() for k in labels)
    board,turn,target=data['train_board'],data['train_turn'],np.sign(data['train_target']);n=len(board);assert n==1000000
    phase=(64-np.count_nonzero(board,axis=1)).astype(np.float32)/60
    perm=permutations(8);inverse=np.argsort(perm,axis=1)
    order=torch.Generator().manual_seed(20261030);augmentation=np.random.default_rng(20261030);sequence=hashlib.sha256()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    best=float('inf');best_epoch=0;best_state=None
    for epoch in range(1,16):
        net.train();losses=[];progress('全网训练中',epoch-1,'按验证策略CE选模；保留最近完成轮指标')
        for rows in torch.randperm(n,generator=order).split(2048):
            indices=rows.numpy();sym=augmentation.integers(8,size=len(indices),dtype=np.int8);sequence.update(indices.tobytes());sequence.update(sym.tobytes());p=perm[sym]
            batch=np.take_along_axis(board[indices],p,axis=1);codes=torch.from_numpy(encode(batch,turn[indices]).astype(np.int64))
            value,wdl,policy=net(codes,torch.from_numpy(phase[indices]));truth=torch.from_numpy(target[indices]);next_move=torch.from_numpy(inverse[sym,labels['train'][indices]])
            mask=torch.from_numpy(np.take_along_axis(masks['train'][indices],p,axis=1));outcome=torch.where(truth>0,0,torch.where(truth<0,1,2)).long()
            loss=F.mse_loss(value,truth)+.1*F.cross_entropy(wdl,outcome)+.1*F.cross_entropy(policy.masked_fill(~mask,-1e9),next_move)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        prediction=metrics.predict(net,data['validation_board'],data['validation_turn'])
        score=metrics.metric(prediction,data['validation_target'],labels['validation'],masks['validation']);latest=dict(score,epoch=epoch)
        history.append(dict(epoch=epoch,trainLoss=float(np.mean(losses)),**score))
        if score['policyCrossEntropy']<best:
            best=score['policyCrossEntropy'];best_epoch=epoch;best_metrics=dict(latest)
            best_state={k:v.detach().clone() for k,v in net.state_dict().items()};torch.save(best_state,WORK/'best-state.pt')
        write(WORK/'history.json',history);progress('全网训练中',epoch)
        print(json.dumps(dict(seconds=time.perf_counter()-started,**history[-1],bestEpoch=best_epoch)),flush=True)
    assert sequence.hexdigest()==baseline['dataOrderAndAugmentationSha256']
    net.load_state_dict(best_state);net.eval()
    assert sha_bytes(net.embedding.weight.detach().numpy().tobytes())!=initial_embedding
    assert sha_bytes(net.trunk.weight.detach().numpy().tobytes())!=initial_trunk
    model=ROOT/'models/pattern32-global-policy16-full-v1.49.pt';assert not model.exists();torch.save(best_state,model)
    component=model.with_name(model.stem+'.base.otpat');export_base(net,component)
    head=model.with_suffix('.head.json');exports.export_head(net,head,sha(component),str(component.relative_to(ROOT)))
    selected={};integer_checks={}
    progress('权重已选定：验证/测试及整数参考核验',15,f'选第{best_epoch}轮')
    with torch.no_grad():
        for split in ['validation','test']:
            prediction=metrics.predict(net,data[split+'_board'],data[split+'_turn'])
            selected[split]=metrics.metric(prediction,data[split+'_target'],labels[split],masks[split])
            maximum=0.0
            for start in range(0,len(data[split+'_board']),2048):
                b=data[split+'_board'][start:start+2048];t=data[split+'_turn'][start:start+2048]
                codes=torch.from_numpy(encode(b,t).astype(np.int64));ph=torch.from_numpy((64-np.count_nonzero(b,axis=1)).astype(np.float32)/60)
                _,_,projection,_=net.frozen_features(codes,ph)
                integer_features=rounded(projection.clamp(-4,4)*256).numpy().astype(np.int16)
                reference=exports.numeric_head(net,integer_features);actual=net.policy_correction(projection).numpy()
                maximum=max(maximum,float(np.max(np.abs(reference-actual))))
            assert maximum==0,(split,maximum)
            integer_checks[split]=dict(positions=len(data[split+'_board']),all64CorrectionsExactlyEqual=True)
        b=data['validation_board'][:256];t=data['validation_turn'][:256];ph=torch.from_numpy((64-np.count_nonzero(b,axis=1)).astype(np.float32)/60)
        original=net(torch.from_numpy(encode(b,t).astype(np.int64)),ph);symmetry_max=0.0
        for p in perm:
            actual=net(torch.from_numpy(encode(b[:,p],t).astype(np.int64)),ph)
            np.testing.assert_array_equal(actual[0].numpy(),original[0].numpy());np.testing.assert_array_equal(actual[1].numpy(),original[1].numpy())
            error=float(np.max(np.abs(actual[2].numpy()[:,np.argsort(p)]-original[2].numpy())));symmetry_max=max(symmetry_max,error)
        assert symmetry_max<2e-6
    for name,digest in preserve.items():assert sha(ROOT/name)==digest,name
    result=dict(version='v1.49-global-policy-full',complete=True,status='candidate',defaultPromoted=False,baseline='models/v1.37-default.json',
        architecture=[38,16,64],bestEpoch=best_epoch,trainableParameters=916987,totalParameters=916987,addedPolicyParameters=237,
        freezeBaseline=False,pretrainedWeightsLoaded=False,sharedInitialParametersIdentical=True,encoderAndTrunkChanged=True,initialEmbeddingSha256=initial_embedding,
        dataOrderAndAugmentationSha256=sequence.hexdigest(),metrics=selected,baselineMetrics=baseline['metrics'],history=history,
        integerHeadReference=integer_checks,symmetry=dict(positions=256,transforms=8,maximumPolicyLogitError=symmetry_max),
        model=str(model.relative_to(ROOT)),modelSha256=sha(model),baseComponent=str(component.relative_to(ROOT)),baseComponentSha256=sha(component),
        headExport=str(head.relative_to(ROOT)),headExportSha256=sha(head),baseComponentIsCompleteModel=False,nativeImplemented=False,matchesRun=False,
        protocolSha256=sha(PROTOCOL),sourceArchiveManifest=str(MANIFEST.relative_to(ROOT)),sourceArchiveManifestSha256=sha(MANIFEST),
        preservedSha256=preserve,seconds=time.perf_counter()-started,limits=protocol['limits'])
    write(REPORT,result);write(model.with_suffix('.json'),result)
    latest=dict(selected['validation'],epoch=best_epoch)
    progress('全网训练完成：按验证策略CE选定',15,f'选第{best_epoch}轮 · 测试策略 CE {selected["test"]["policyCrossEntropy"]:.6f} · 命中率 {selected["test"]["policyTop1"]:.2%}',True)
    write(ROOT/'models/v1.49-global-policy-full-monitor.json',read(WORK/'progress.json'))
    print(json.dumps(dict(complete=True,bestEpoch=best_epoch,baseline=baseline['metrics'],metrics=selected,seconds=result['seconds'])),flush=True)


def sha_bytes(value):return hashlib.sha256(value).hexdigest()


if __name__=='__main__':main()
