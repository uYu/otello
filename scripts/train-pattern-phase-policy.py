#!/usr/bin/env python3
"""Single policy versus phase policy with matched middle-validation selection."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pattern_model import encode,legal_mask,parity
from pattern_structure import StructureNet,export as export_control
from pattern_phase_policy import PhasePolicyNet,export as export_phase
from training_consensus import PERMS,INVERSE
from wthor_label_ablation import parameter_fingerprint
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'.build/training/pattern-phase-policy-v1.47'
REPORT=ROOT/'models/v1.47-pattern-phase-policy.json';PROTOCOL=ROOT/'models/v1.47-phase-policy-protocol.json'
LIB=ROOT/('.build/native-pattern-phase-policy-v1.47/libothello.'+('dylib' if sys.platform=='darwin' else 'so'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,r:Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('phase_policy_metrics',ROOT/'scripts/train-pattern-outcome.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
EPOCHS=15;SEED=20261029;DATA_SEED=20261030


def main():
    assert not REPORT.exists(),'Completed reports are immutable'
    protocol=json.loads(PROTOCOL.read_text());WORK.mkdir(parents=True,exist_ok=True);started=time.perf_counter();variants=[];networks=[]
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    torch.set_num_threads(2)
    def progress(stage,name='control',epoch=0,validation=None,complete=False):
        r=dict(experiment='pattern-structure-training',variant=name,stage=stage,epoch=epoch,epochs=EPOCHS,seconds=time.perf_counter()-started,variants=variants,complete=complete)
        if validation is not None:r['validation']=validation
        write(WORK/'progress.json',r)
    config=json.loads((ROOT/'models/current-experiment.json').read_text());config['version']='v1.47 分阶段策略头 · 中盘策略对照'
    config['experiments'].append(dict(name='单头 / 分阶段策略头 · 按相同中盘验证 CE 选模',focus='phase-policy',games=0,runtimeReport=str((WORK/'progress.json').relative_to(ROOT)),report='models/v1.47-phase-policy-monitor.json'));write(ROOT/'models/current-experiment.json',config)
    progress('固定原百万硬标签；核验中盘验证分区')
    data=dict(np.load(ROOT/'.build/training/v1/positions.npz'));labels=dict(np.load(ROOT/'.build/training/v1.8/policy.npz'))
    masks={k:legal_mask(data[k+'_board'],data[k+'_turn']) for k in labels}
    assert all(masks[k][np.arange(len(labels[k])),labels[k]].all() for k in labels)
    baseline=json.loads((ROOT/'models/pattern32-symmetry-v1.37.json').read_text())
    board,turn,target=data['train_board'],data['train_turn'],np.sign(data['train_target']);n=len(board);assert n==1000000
    phase=(64-np.count_nonzero(board,axis=1)).astype(np.float32)/60
    placed={split:np.count_nonzero(data[split+'_board'],axis=1)-4 for split in ['train','validation','test']}
    middle=(placed['validation']>=21)&(placed['validation']<=36);assert middle.sum()==2463
    def metric_subset(pred,split,selected):
        return helper.metric(tuple(p[selected] for p in pred),data[split+'_target'][selected],labels[split][selected],masks[split][selected])
    torch.manual_seed(SEED);initial_base=StructureNet('symmetry');reference_initial=parameter_fingerprint(initial_base)
    for name in ['control','phase-policy']:
        path=ROOT/f'models/pattern32-symmetry-{name}-v1.47.otpat';assert not path.exists()
        torch.manual_seed(SEED);np.random.seed(SEED);net=StructureNet('symmetry') if name=='control' else PhasePolicyNet()
        for key,value in initial_base.state_dict().items():assert torch.equal(value,net.state_dict()[key]),key
        order=torch.Generator().manual_seed(DATA_SEED);augmentation=np.random.default_rng(DATA_SEED);sequence=hashlib.sha256()
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);best=float('inf');state=None;global_best=float('inf');global_state=None;history=[]
        progress('训练中：'+name,name)
        for epoch in range(EPOCHS):
            net.train();losses=[]
            for rows in torch.randperm(n,generator=order).split(2048):
                indices=rows.numpy();sym=augmentation.integers(8,size=len(indices),dtype=np.int8);sequence.update(indices.tobytes());sequence.update(sym.tobytes());perm=PERMS[sym]
                batch=np.take_along_axis(board[indices],perm,axis=1);code=torch.from_numpy(encode(batch,turn[indices]).astype(np.int64))
                value,wdl,policy=net(code,torch.from_numpy(phase[indices]));truth=torch.from_numpy(target[indices]);next_move=torch.from_numpy(INVERSE[sym,labels['train'][indices]]);mask=torch.from_numpy(np.take_along_axis(masks['train'][indices],perm,axis=1))
                outcome=torch.where(truth>0,0,torch.where(truth<0,1,2)).long()
                loss=F.mse_loss(value,truth)+.1*F.cross_entropy(wdl,outcome)+.1*F.cross_entropy(policy.masked_fill(~mask,-1e9),next_move)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
            pred=helper.predict(net,data['validation_board'],data['validation_turn']);score=helper.metric(pred,data['validation_target'],labels['validation'],masks['validation']);midscore=metric_subset(pred,'validation',middle)
            if midscore['policyCrossEntropy']<best:best=midscore['policyCrossEntropy'];state={k:v.detach().clone() for k,v in net.state_dict().items()}
            if score['mse']<global_best:global_best=score['mse'];global_state={k:v.detach().clone() for k,v in net.state_dict().items()}
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**score,middle=midscore));progress('训练中：'+name,name,epoch+1,score);print(json.dumps(dict(name=name,seconds=time.perf_counter()-started,**history[-1])),flush=True)
        assert sequence.hexdigest()==baseline['dataOrderAndAugmentationSha256']
        replay=None
        if name=='control':
            net.load_state_dict(global_state);export_control(net,WORK/'control-global-mse-reference.otpat')
            replay=dict(originalGlobalMseSelectionModelIdentical=sha(WORK/'control-global-mse-reference.otpat')==baseline['modelSha256'],allOriginalEpochMetricsIdentical=[{k:v for k,v in row.items() if k!='middle'} for row in history]==baseline['history'])
            assert all(replay.values()),replay
        net.load_state_dict(state);net.eval();binary=(export_control if name=='control' else export_phase)(net,path);torch.save(state,path.with_suffix('.pt'))
        variant=dict(name=name,model=str(path.relative_to(ROOT)),modelSha256=sha(path),formatMagic='OTPAT4' if name=='control' else 'OTPAT7',parameters=sum(p.numel() for p in net.parameters()),bytes=len(binary),sharedInitialParametersSha256=reference_initial,sharedInitialParametersIdentical=True,dataOrderAndAugmentationSha256=sequence.hexdigest(),history=history,bestEpoch=min(history,key=lambda x:x['middle']['policyCrossEntropy'])['epoch'],bestGlobalMseEpoch=min(history,key=lambda x:x['mse'])['epoch'],epoch15Validation=history[-1],metrics={},phaseMetrics={},baselineReplay=replay)
        variants.append(variant);networks.append(net)
    progress('两份中盘验证最佳权重已冻结；测试与原生导出核验','control/phase-policy',EPOCHS)
    base=StructureNet('symmetry');base.load_state_dict(torch.load(ROOT/'models/pattern32-symmetry-v1.37.pt',weights_only=True));base.eval();baseline_phases={};baseline_parity={}
    for split in ['validation','test']:
        prediction=helper.predict(base,data[split+'_board'],data[split+'_turn']);baseline_parity[split]=parity(LIB,ROOT/baseline['model'],data[split+'_board'],data[split+'_turn'],prediction)
        baseline_phases[split]=[]
        for low,high in [(9,18),(21,36),(39,54)]:
            selected=(placed[split]>=low)&(placed[split]<=high);baseline_phases[split].append(dict(placements=[low,high],positions=int(selected.sum()),metrics=metric_subset(prediction,split,selected)))
    for variant,net in zip(variants,networks):
        for split in ['validation','test']:
            pred=helper.predict(net,data[split+'_board'],data[split+'_turn']);variant['metrics'][split]=helper.metric(pred,data[split+'_target'],labels[split],masks[split]);checks=parity(LIB,ROOT/variant['model'],data[split+'_board'],data[split+'_turn'],pred);checks['maxOutcomeError']=checks.pop('maxDiscError');variant[split+'Parity']=checks
            variant['phaseMetrics'][split]=[]
            for low,high in [(9,18),(21,36),(39,54)]:
                selected=(placed[split]>=low)&(placed[split]<=high);variant['phaseMetrics'][split].append(dict(placements=[low,high],positions=int(selected.sum()),metrics=metric_subset(pred,split,selected)))
        write((ROOT/variant['model']).with_suffix('.json'),dict(version='v1.47-phase-policy',status='candidate',defaultPromoted=False,**variant))
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    report=dict(version='v1.47-phase-policy',status='candidate',complete=True,defaultPromoted=False,matchesRun=False,baselineManifest='models/v1.37-default.json',baselineMetrics=baseline['metrics'],baselinePhaseMetrics=baseline_phases,architecture=protocol['architecture'],phase=protocol['phase'],value='Original single value/WLD structure and loss; jointly trained, weights not frozen; separate from v1.46 phase-value model',dataset=dict(sources=['wthor','logistello'],positions=n,labels='Original hard labels; no aggregation',eloFiltering=False,eloWeighting=False,positionsSha256=sha(ROOT/'.build/training/v1/positions.npz'),policySha256=sha(ROOT/'.build/training/v1.8/policy.npz')),middleValidationPositions=int(middle.sum()),stageTrainPositions=[dict(placements=[lo,hi],positions=int(((placed['train']>=lo)&(placed['train']<=hi)).sum())) for lo,hi in [(9,18),(21,36),(39,54)]],variants=variants,epochs=EPOCHS,batchSize=2048,seed=SEED,dataSeed=DATA_SEED,threads=2,learningRate=.001,weightDecay=.0001,loss='outcome MSE + 0.1 WLD CE + 0.1 legal next-move CE; no phase reweighting',selection=protocol['selection'],initialization='Random same original shared parameters; duplicate original policy readout into opening policy head; initial predictions identical',baselineInferenceRegression=baseline_parity,protocolSha256=sha(PROTOCOL),sourceArchiveManifest=protocol['sourceArchiveManifest'],librarySha256=sha(LIB),preservedSha256=protocol['preservedSha256'],seconds=time.perf_counter()-started,limits='One seed offline ablation. Both control and phase candidate selected on same middle-policy metric; frozen approved default additionally reported under its original selection. Phase policy not combined with phase value. No match or promotion; middle test never used to choose epoch or phase thresholds.')
    write(REPORT,report);progress('单头 / 分阶段策略头训练及导出完成','control/phase-policy',EPOCHS,variants[1]['metrics']['validation'],True);write(ROOT/'models/v1.47-phase-policy-monitor.json',json.loads((WORK/'progress.json').read_text()));print(json.dumps(dict(complete=True,seconds=report['seconds'],summary=[dict(name=v['name'],bestEpoch=v['bestEpoch'],middleTest=v['phaseMetrics']['test'][1]['metrics'],test=v['metrics']['test']) for v in variants])),flush=True)


if __name__=='__main__':main()
