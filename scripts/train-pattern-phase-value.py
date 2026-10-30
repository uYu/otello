#!/usr/bin/env python3
"""Matched mixed hard-label training with two interpolated outcome readouts."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pattern_model import encode,legal_mask,parity
from pattern_structure import StructureNet
from pattern_phase_value import PhaseValueNet,export
from training_consensus import PERMS,INVERSE
from wthor_label_ablation import parameter_fingerprint,fingerprint
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'.build/training/pattern-phase-value-v1.46'
REPORT=ROOT/'models/v1.46-pattern-phase-value.json';PROTOCOL=ROOT/'models/v1.46-phase-value-protocol.json'
LIB=ROOT/('.build/native-pattern-phase-value-v1.46/libothello.'+('dylib' if sys.platform=='darwin' else 'so'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,r:Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('phase_value_metrics',ROOT/'scripts/train-pattern-outcome.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
EPOCHS=15;SEED=20261029;DATA_SEED=20261030


def main():
    assert not REPORT.exists(),'Completed reports are immutable'
    protocol=json.loads(PROTOCOL.read_text());WORK.mkdir(parents=True,exist_ok=True);started=time.perf_counter();variants=[]
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    assert json.loads((ROOT/'models/v1.45-soft-vs-v1.37-2s-stopped.json').read_text())['userStopped']
    torch.set_num_threads(2)
    def progress(stage,epoch=0,validation=None,complete=False):
        r=dict(experiment='pattern-structure-training',variant='phase-value',stage=stage,epoch=epoch,epochs=EPOCHS,seconds=time.perf_counter()-started,variants=variants,complete=complete)
        if validation is not None:r['validation']=validation
        write(WORK/'progress.json',r)
    config=json.loads((ROOT/'models/current-experiment.json').read_text());config['version']='v1.46 分阶段价值头 · 原混合语料硬标签'
    config['experiments'].append(dict(name='分阶段价值头 · 策略结构沿用 v1.37',focus='phase-value',games=0,runtimeReport=str((WORK/'progress.json').relative_to(ROOT)),report='models/v1.46-phase-value-monitor.json'));write(ROOT/'models/current-experiment.json',config)
    progress('核验固定数据和原生推理')
    data=dict(np.load(ROOT/'.build/training/v1/positions.npz'));labels=dict(np.load(ROOT/'.build/training/v1.8/policy.npz'))
    masks={k:legal_mask(data[k+'_board'],data[k+'_turn']) for k in labels}
    assert all(masks[k][np.arange(len(labels[k])),labels[k]].all() for k in labels)
    baseline=json.loads((ROOT/'models/pattern32-symmetry-v1.37.json').read_text());base=StructureNet('symmetry');base.load_state_dict(torch.load(ROOT/'models/pattern32-symmetry-v1.37.pt',weights_only=True));base.eval()
    legacy_parity={}
    baseline_predictions={}
    for split in ['validation','test']:
        # Baseline is already frozen. Candidate test evaluation occurs only after selection.
        pred=helper.predict(base,data[split+'_board'],data[split+'_turn']);baseline_predictions[split]=pred
        legacy_parity[split]=parity(LIB,ROOT/baseline['model'],data[split+'_board'],data[split+'_turn'],pred)
    torch.manual_seed(SEED);net=PhaseValueNet();initial=parameter_fingerprint(net)
    torch.manual_seed(SEED);initial_base=StructureNet('symmetry')
    for key,value in initial_base.state_dict().items():assert torch.equal(value,net.state_dict()[key]),key
    with torch.no_grad():
        sample=data['validation_board'][:256];sample_turn=data['validation_turn'][:256];code=torch.from_numpy(encode(sample,sample_turn).astype(np.int64));phase_sample=torch.from_numpy((64-np.count_nonzero(sample,axis=1)).astype(np.float32)/60)
        assert all(torch.equal(a,b) for a,b in zip(initial_base(code,phase_sample),net(code,phase_sample)))
    board,turn,target=data['train_board'],data['train_turn'],np.sign(data['train_target']);n=len(board);assert n==1000000
    phase=(64-np.count_nonzero(board,axis=1)).astype(np.float32)/60
    order=torch.Generator().manual_seed(DATA_SEED);augmentation=np.random.default_rng(DATA_SEED);sequence=hashlib.sha256()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);best=float('inf');state=None;history=[]
    progress('训练中：两阶段价值头；原硬标签')
    for epoch in range(EPOCHS):
        net.train();losses=[]
        for rows in torch.randperm(n,generator=order).split(2048):
            indices=rows.numpy();sym=augmentation.integers(8,size=len(indices),dtype=np.int8);sequence.update(indices.tobytes());sequence.update(sym.tobytes());perm=PERMS[sym]
            batch=np.take_along_axis(board[indices],perm,axis=1);code=torch.from_numpy(encode(batch,turn[indices]).astype(np.int64))
            value,wdl,policy=net(code,torch.from_numpy(phase[indices]));truth=torch.from_numpy(target[indices]);next_move=torch.from_numpy(INVERSE[sym,labels['train'][indices]]);mask=torch.from_numpy(np.take_along_axis(masks['train'][indices],perm,axis=1))
            outcome=torch.where(truth>0,0,torch.where(truth<0,1,2)).long()
            loss=F.mse_loss(value,truth)+.1*F.cross_entropy(wdl,outcome)+.1*F.cross_entropy(policy.masked_fill(~mask,-1e9),next_move)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        pred=helper.predict(net,data['validation_board'],data['validation_turn']);score=helper.metric(pred,data['validation_target'],labels['validation'],masks['validation'])
        if score['mse']<best:best=score['mse'];state={k:v.detach().clone() for k,v in net.state_dict().items()}
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**score));progress('训练中：两阶段价值头；原硬标签',epoch+1,score);print(json.dumps(dict(seconds=time.perf_counter()-started,**history[-1])),flush=True)
    assert sequence.hexdigest()==baseline['dataOrderAndAugmentationSha256']
    net.load_state_dict(state);net.eval();path=ROOT/'models/pattern32-symmetry-phase-value-v1.46.otpat';assert not path.exists()
    binary=export(net,path);torch.save(state,path.with_suffix('.pt'))
    variant=dict(name='phase-value',model=str(path.relative_to(ROOT)),modelSha256=sha(path),formatMagic='OTPAT6',parameters=sum(p.numel() for p in net.parameters()),bytes=len(binary),initialParametersSha256=initial,sharedInitialParametersIdentical=True,initialPredictionsIdentical=True,dataOrderAndAugmentationSha256=sequence.hexdigest(),history=history,bestEpoch=min(history,key=lambda x:x['mse'])['epoch'],metrics={},phaseMetrics={})
    variants.append(variant);progress('权重已选定；固定验证/测试及原生导出核验',EPOCHS)
    for split in ['validation','test']:
        pred=helper.predict(net,data[split+'_board'],data[split+'_turn']);variant['metrics'][split]=helper.metric(pred,data[split+'_target'],labels[split],masks[split]);checks=parity(LIB,path,data[split+'_board'],data[split+'_turn'],pred);checks['maxOutcomeError']=checks.pop('maxDiscError');variant[split+'Parity']=checks
        placed=np.count_nonzero(data[split+'_board'],axis=1)-4;phases=[]
        for low,high in [(9,18),(21,36),(39,54)]:
            selected=(placed>=low)&(placed<=high)
            metric=lambda predictions:helper.metric(tuple(p[selected] for p in predictions),data[split+'_target'][selected],labels[split][selected],masks[split][selected])
            phases.append(dict(placements=[low,high],positions=int(selected.sum()),baseline=metric(baseline_predictions[split]),candidate=metric(pred)))
        variant['phaseMetrics'][split]=phases
    write(path.with_suffix('.json'),dict(version='v1.46-phase-value',status='candidate',defaultPromoted=False,**variant))
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    report=dict(version='v1.46-phase-value',status='candidate',complete=True,defaultPromoted=False,matchesRun=False,baselineManifest='models/v1.37-default.json',baselineMetrics=baseline['metrics'],architecture=protocol['architecture'],phase=protocol['phase'],policy='Original conditional policy structure and loss; jointly trained with shared trunk, weights not frozen',dataset=dict(sources=['wthor','logistello'],positions=n,labels='Original hard labels; no aggregation',eloFiltering=False,eloWeighting=False,positionsSha256=sha(ROOT/'.build/training/v1/positions.npz'),policySha256=sha(ROOT/'.build/training/v1.8/policy.npz')),variants=variants,epochs=EPOCHS,batchSize=2048,seed=SEED,dataSeed=DATA_SEED,threads=2,learningRate=.001,weightDecay=.0001,loss='outcome MSE + 0.1 WLD CE + 0.1 legal next-move CE',selection='Minimum unchanged original 6000-position validation outcome MSE; candidate test only after selected model frozen',initialization='Random same original shared parameters; duplicate original first four head rows into opening head; not a warm start',baselineInferenceRegression=legacy_parity,protocolSha256=sha(PROTOCOL),sourceArchiveManifest=protocol['sourceArchiveManifest'],librarySha256=sha(LIB),preservedSha256=protocol['preservedSha256'],seconds=time.perf_counter()-started,limits='One seed, offline comparison; no strength claim, no default promotion. Phase thresholds fixed before training; no stage-specific target changes.')
    write(REPORT,report);progress('分阶段价值头训练及导出完成',EPOCHS,variant['metrics']['validation'],True);write(ROOT/'models/v1.46-phase-value-monitor.json',json.loads((WORK/'progress.json').read_text()));print(json.dumps(dict(complete=True,bestEpoch=variant['bestEpoch'],test=variant['metrics']['test'],seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
