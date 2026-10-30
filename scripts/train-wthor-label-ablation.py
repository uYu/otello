#!/usr/bin/env python3
"""Controlled hard-first vs empirical-soft target experiment on identical WTHOR rows."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pattern_model import encode,legal_mask,parity
from pattern_structure import StructureNet,export
from training_consensus import boards_from_keys,PERMS
from wthor_label_ablation import first_labels,policy_targets,parameter_fingerprint,fingerprint
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'.build/training/wthor-label-ablation-v1.44';REPORT=ROOT/'models/v1.44-wthor-label-ablation.json';PROTOCOL=ROOT/'models/v1.44-wthor-label-ablation-protocol.json'
LIB=ROOT/('.build/native-v1.37-reference/libothello.'+('dylib' if sys.platform=='darwin' else 'so'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,r:Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('label_ablation_metrics',ROOT/'scripts/train-pattern-outcome.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
EPOCHS=15;SEED=20261029;DATA_SEED=20261030


def main():
    assert not REPORT.exists(),'Completed reports are immutable'
    protocol=json.loads(PROTOCOL.read_text());WORK.mkdir(parents=True,exist_ok=True);started=time.perf_counter();variants=[];networks=[]
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    torch.set_num_threads(2)
    def progress(stage,mode='hard',epoch=0,validation=None,complete=False):
        r=dict(experiment='pattern-structure-training',variant=mode,stage=stage,epoch=epoch,epochs=EPOCHS,seconds=time.perf_counter()-started,variants=variants,complete=complete)
        if validation is not None:r['validation']=validation
        write(WORK/'progress.json',r)
    config=json.loads((ROOT/'models/current-experiment.json').read_text());archived=config.get('archivedExperiments',[])
    for old in config['experiments']:
        if old.get('games',0)==0 and not any(x.get('report')==old.get('report') for x in archived):archived.insert(0,old)
    config.update(version='v1.44 相同 WTHOR 局面 · 硬/软标签对照',archivedExperiments=archived,experiments=[e for e in config['experiments'] if e.get('games',0)>0]+[dict(name='相同 WTHOR 局面：首条硬标签 / 聚合软标签',focus='wthor-label-ablation',games=0,runtimeReport=str((WORK/'progress.json').relative_to(ROOT)),report='models/v1.44-wthor-label-ablation-monitor.json')]);write(ROOT/'models/current-experiment.json',config)
    progress('固定局面与首条记录选择')
    data=dict(np.load(ROOT/'.build/training/pattern-wthor-v1.42/training-data.npz'))
    with np.load(ROOT/'.build/training/consensus-elo-v1.40/occurrences.npz') as z:occ={k:z[k] for k in ['keys','source','outcome','action_bits','game_index']}
    hard=first_labels(occ,data['keys']);assert not occ['source'][hard['first_occurrence']].any();np.savez_compressed(WORK/'first-labels.npz',**hard);del occ
    board=boards_from_keys(data['keys']);n=len(board);assert n==1054132;turn=np.ones(n,np.int8);phase=(64-np.count_nonzero(board,axis=1)).astype(np.float32)/60;mask=legal_mask(board,turn)
    assert mask[np.arange(n),hard['action']].all()
    frozen=dict(np.load(ROOT/'.build/training/v1/positions.npz'));labels=dict(np.load(ROOT/'.build/training/v1.8/policy.npz'))
    validation_mask=legal_mask(frozen['validation_board'],frozen['validation_turn']);test_mask=legal_mask(frozen['test_board'],frozen['test_turn'])
    inputs=dict(positionKeysSha256=fingerprint(data['keys']),canonicalBoardSha256=fingerprint(board),turnSha256=fingerprint(turn),phaseSha256=fingerprint(phase),legalMaskSha256=fingerprint(mask),positions=n)
    label_changes=dict(valueRows=int((data['value_mean']!=hard['value']).sum()),wdlRows=int(np.any(data['wdl_probability']!=hard['wdl'],axis=1).sum()))
    soft_groups=np.repeat(np.arange(n),np.diff(data['policy_offsets']).astype(np.int64));at_first=data['policy_action']==hard['action'][soft_groups];first_mass=np.bincount(soft_groups[at_first],weights=data['policy_probability'][at_first],minlength=n);label_changes['policyRowsIncludingCanonicalSymmetry']=int((first_mass<1-1e-7).sum())
    baseline=json.loads((ROOT/'models/pattern32-symmetry-v1.37.json').read_text())
    for mode in ['hard','soft']:
        path=ROOT/f'models/pattern32-symmetry-wthor-{mode}-v1.44.otpat';assert not path.exists()
        value_target=hard['value'] if mode=='hard' else data['value_mean'];wdl_target=hard['wdl'] if mode=='hard' else data['wdl_probability'];policy_target=policy_targets(data,hard,mode)
        assert not policy_target[~mask].any();np.testing.assert_allclose(policy_target.sum(1),1,atol=1e-6,rtol=0)
        target_hashes=dict(value=fingerprint(value_target),wdl=fingerprint(wdl_target),policy=fingerprint(policy_target))
        torch.manual_seed(SEED);np.random.seed(SEED);net=StructureNet('symmetry');initial=parameter_fingerprint(net);embedding=hashlib.sha256(net.embedding.weight.detach().numpy().tobytes()).hexdigest();assert embedding==baseline['initialEmbeddingSha256']
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);order=torch.Generator().manual_seed(DATA_SEED);augmentation=np.random.default_rng(DATA_SEED);sequence=hashlib.sha256();best=float('inf');best_state=None;history=[]
        progress('训练中：'+('首条硬标签' if mode=='hard' else '聚合软标签'),mode)
        for epoch in range(EPOCHS):
            net.train();losses=[]
            for rows in torch.randperm(n,generator=order).split(2048):
                indices=rows.numpy();sym=augmentation.integers(8,size=len(indices),dtype=np.int8);sequence.update(indices.tobytes());sequence.update(sym.tobytes());perm=PERMS[sym]
                batch=np.take_along_axis(board[indices],perm,axis=1);code=torch.from_numpy(encode(batch,turn[indices]).astype(np.int64));value,wdl,policy=net(code,torch.from_numpy(phase[indices]))
                probabilities=torch.from_numpy(np.take_along_axis(policy_target[indices],perm,axis=1));legal=torch.from_numpy(np.take_along_axis(mask[indices],perm,axis=1))
                loss=F.mse_loss(value,torch.from_numpy(value_target[indices]))+.1*F.cross_entropy(wdl,torch.from_numpy(wdl_target[indices]))+.1*-(probabilities*F.log_softmax(policy.masked_fill(~legal,-1e9),dim=1)).sum(1).mean()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
            pred=helper.predict(net,frozen['validation_board'],frozen['validation_turn']);score=helper.metric(pred,frozen['validation_target'],labels['validation'],validation_mask)
            if score['mse']<best:best=score['mse'];best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**score));progress('训练中：'+('首条硬标签' if mode=='hard' else '聚合软标签'),mode,epoch+1,score);print(json.dumps(dict(mode=mode,seconds=time.perf_counter()-started,**history[-1])),flush=True)
        net.load_state_dict(best_state);net.eval();binary=export(net,path);torch.save(best_state,path.with_suffix('.pt'))
        variant=dict(name=mode,model=str(path.relative_to(ROOT)),modelSha256=sha(path),formatMagic='OTPAT4',parameters=sum(p.numel() for p in net.parameters()),bytes=len(binary),initialParametersSha256=initial,initialEmbeddingSha256=embedding,dataOrderAndAugmentationSha256=sequence.hexdigest(),inputFingerprints=inputs,targetFingerprints=target_hashes,history=history,bestEpoch=min(history,key=lambda r:r['mse'])['epoch'],epoch15Validation={k:v for k,v in history[-1].items() if k not in ['loss','epoch']},metrics={})
        variants.append(variant);networks.append(net)
    assert variants[0]['initialParametersSha256']==variants[1]['initialParametersSha256']
    assert variants[0]['dataOrderAndAugmentationSha256']==variants[1]['dataOrderAndAugmentationSha256']==json.loads((ROOT/'models/v1.42-pattern-wthor.json').read_text())['variants'][0]['dataOrderAndAugmentationSha256']
    progress('两个模型均已选定；统一冻结验证/测试与 C++ 核验','hard/soft',EPOCHS)
    for variant,net in zip(variants,networks):
        for split,m in [('validation',validation_mask),('test',test_mask)]:
            pred=helper.predict(net,frozen[split+'_board'],frozen[split+'_turn']);variant['metrics'][split]=helper.metric(pred,frozen[split+'_target'],labels[split],m);checks=parity(LIB,ROOT/variant['model'],frozen[split+'_board'],frozen[split+'_turn'],pred);checks['maxOutcomeError']=checks.pop('maxDiscError');variant[split+'Parity']=checks
        write((ROOT/variant['model']).with_suffix('.json'),dict(version='v1.44-wthor-label-ablation',status='candidate',defaultPromoted=False,**variant))
    previous=json.loads((ROOT/'models/v1.42-pattern-wthor.json').read_text())['variants'][0];soft=variants[1];same_history=soft['history']==previous['history'];same_binary=soft['modelSha256']==previous['modelSha256']
    replay=dict(referenceModel=previous['model'],referenceSha256=previous['modelSha256'],modelBytesIdentical=same_binary,allEpochMetricsIdentical=same_history,maxHistoryMseDifference=max(abs(a['mse']-b['mse']) for a,b in zip(soft['history'],previous['history'])))
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    report=dict(version='v1.44-wthor-label-ablation',status='candidate',complete=True,defaultPromoted=False,matchesRun=False,architecture='Exact approved v1.37 symmetry-sharing model, original conditional policy, no direct MLP',baselineManifest='models/v1.37-default.json',baselineMetrics=baseline['metrics'],dataset=dict(sources=['wthor'],positions=n,occurrences=int(data['occurrence_count'].sum()),eloFiltering=False,eloWeighting=False,allPositionWeights=1,firstRecordOrder='First WTHOR row in frozen occurrence file; same observation supplies outcome and move; canonical stabilizer tie uses smallest aligned square',firstLabelsSha256=sha(WORK/'first-labels.npz')),controlledInputs=inputs,labelChanges=label_changes,variants=variants,softReferenceReplay=replay,epochs=EPOCHS,batchSize=2048,seed=SEED,dataSeed=DATA_SEED,learningRate=.001,weightDecay=.0001,threads=2,selection='Each minimum frozen validation original-label outcome MSE; both checkpoints selected/frozen before either test evaluation; also retain epoch15 validation for equal-update comparison',loss='Same implementation: value MSE +0.1 probability-target W/L/D CE +0.1 probability-target legal policy CE. Only target arrays differ.',protocolSha256=sha(PROTOCOL),librarySha256=sha(LIB),preservedSha256=protocol['preservedSha256'],scriptSha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'scripts/wthor_label_ablation.py',ROOT/'scripts/pattern_model.py',ROOT/'scripts/pattern_structure.py',ROOT/'scripts/training_consensus.py',ROOT/'scripts/train-pattern-outcome.py',ROOT/'scripts/train-pattern.py']},seconds=time.perf_counter()-started,limits='One seed, same WTHOR-only rows and model. This separates joint hard-vs-soft label construction, not policy-vs-value aggregation individually. It does not isolate WTHOR vs mixed-source corpus effects. No strength match or default promotion.')
    write(REPORT,report);progress('硬/软标签对照训练和验证完成','hard/soft',EPOCHS,complete=True);write(ROOT/'models/v1.44-wthor-label-ablation-monitor.json',json.loads((WORK/'progress.json').read_text()));print(json.dumps(dict(complete=True,seconds=report['seconds'],replay=replay,results=[dict(name=v['name'],bestEpoch=v['bestEpoch'],test=v['metrics']['test']) for v in variants]),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
