#!/usr/bin/env python3
"""Original million mixed-corpus hard baseline versus empirical soft labels."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pattern_model import encode,legal_mask,parity
from pattern_structure import StructureNet,export
from training_consensus import PERMS,INVERSE
from mixed_label_targets import canonical_transforms,soft_targets
from wthor_label_ablation import fingerprint,parameter_fingerprint
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'.build/training/mixed-label-ablation-v1.45';REPORT=ROOT/'models/v1.45-mixed-label-ablation.json';PROTOCOL=ROOT/'models/v1.45-mixed-label-ablation-protocol.json'
LIB=ROOT/('.build/native-v1.37-reference/libothello.'+('dylib' if sys.platform=='darwin' else 'so'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,r:Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('mixed_label_metrics',ROOT/'scripts/train-pattern-outcome.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
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
    config.update(version='v1.45 原百万混合语料 · 基线硬标签/聚合软标签',archivedExperiments=archived,experiments=[e for e in config['experiments'] if e.get('games',0)>0]+[dict(name='WTHOR + Logistello 原百万局面：硬/软对照',focus='mixed-label-ablation',games=0,runtimeReport=str((WORK/'progress.json').relative_to(ROOT)),report='models/v1.45-mixed-label-ablation-monitor.json')]);write(ROOT/'models/current-experiment.json',config)
    progress('固定原基线百万局面；对齐两库经验软标签')
    frozen=dict(np.load(ROOT/'.build/training/v1/positions.npz'));labels=dict(np.load(ROOT/'.build/training/v1.8/policy.npz'));mixed=dict(np.load(ROOT/'.build/training/consensus-elo-v1.40/consensus.npz'))
    board,turn=frozen['train_board'],frozen['train_turn'];n=len(board);assert n==1000000
    keys=[];sym=[]
    for start in range(0,n,16384):
        k,s=canonical_transforms(board[start:start+16384],turn[start:start+16384]);keys.append(k);sym.append(s)
    keys=np.concatenate(keys);sym=np.concatenate(sym);mapping=mixed['matched_train_group'];np.testing.assert_array_equal(keys,mixed['keys'][mapping])
    soft=soft_targets(mixed,mapping,sym);hard_value=np.sign(frozen['train_target']);hard_category=np.where(hard_value>0,0,np.where(hard_value<0,1,2));mask=legal_mask(board,turn)
    assert not soft['policy'][~mask].any();np.testing.assert_allclose(soft['policy'].sum(1),1,atol=1e-6,rtol=0)
    assert (soft['policy'][np.arange(n),labels['train']]>0).all() and (soft['wdl'][np.arange(n),hard_category]>0).all()
    np.savez_compressed(WORK/'aligned-soft-labels.npz',**soft,canonical_group=mapping,canonical_symmetry=sym)
    phase=(64-np.count_nonzero(board,axis=1)).astype(np.float32)/60;vmask=legal_mask(frozen['validation_board'],frozen['validation_turn']);tmask=legal_mask(frozen['test_board'],frozen['test_turn'])
    inputs=dict(positions=n,board=fingerprint(board),turn=fingerprint(turn),phase=fingerprint(phase),legalMask=fingerprint(mask),canonicalKeys=fingerprint(keys),rowOrder='Unmodified frozen v1 training rows')
    changes=dict(valueRows=int((soft['value']!=hard_value).sum()),wdlRows=int((soft['wdl'][np.arange(n),hard_category]<1-1e-7).sum()),policyRows=int((soft['policy'][np.arange(n),labels['train']]<1-1e-7).sum()),selectedOccurrenceCount=int(mixed['occurrence_count'][mapping].sum()),sourceOccurrenceCounts=mixed['source_count'][mapping].sum(0).astype(int).tolist())
    del mixed,keys,sym
    baseline=json.loads((ROOT/'models/pattern32-symmetry-v1.37.json').read_text())
    for mode in ['hard','soft']:
        path=ROOT/f'models/pattern32-symmetry-mixed-{mode}-v1.45.otpat';assert not path.exists()
        value_target=hard_value if mode=='hard' else soft['value']
        target_hashes=dict(value=fingerprint(value_target),wdl=fingerprint(hard_category if mode=='hard' else soft['wdl']),policy=fingerprint(labels['train'] if mode=='hard' else soft['policy']))
        torch.manual_seed(SEED);np.random.seed(SEED);order=torch.Generator().manual_seed(DATA_SEED);augmentation=np.random.default_rng(DATA_SEED);sequence=hashlib.sha256();net=StructureNet('symmetry');initial=parameter_fingerprint(net);embedding=hashlib.sha256(net.embedding.weight.detach().numpy().tobytes()).hexdigest();assert embedding==baseline['initialEmbeddingSha256']
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);best=float('inf');state=None;history=[];progress('训练中：'+('原基线硬标签复现' if mode=='hard' else '两库聚合软标签'),mode)
        for epoch in range(EPOCHS):
            net.train();losses=[]
            for rows in torch.randperm(n,generator=order).split(2048):
                indices=rows.numpy();symmetry=augmentation.integers(8,size=len(indices),dtype=np.int8);sequence.update(indices.tobytes());sequence.update(symmetry.tobytes());perm=PERMS[symmetry]
                batch=np.take_along_axis(board[indices],perm,axis=1);code=torch.from_numpy(encode(batch,turn[indices]).astype(np.int64));value,wdl,policy=net(code,torch.from_numpy(phase[indices]));truth=torch.from_numpy(value_target[indices]);legal=torch.from_numpy(np.take_along_axis(mask[indices],perm,axis=1))
                if mode=='hard':
                    outcome=torch.where(truth>0,0,torch.where(truth<0,1,2)).long();next_move=torch.from_numpy(INVERSE[symmetry,labels['train'][indices]])
                    loss=F.mse_loss(value,truth)+.1*F.cross_entropy(wdl,outcome)+.1*F.cross_entropy(policy.masked_fill(~legal,-1e9),next_move)
                else:
                    outcome=torch.from_numpy(soft['wdl'][indices]);next_move=torch.from_numpy(np.take_along_axis(soft['policy'][indices],perm,axis=1))
                    loss=F.mse_loss(value,truth)+.1*F.cross_entropy(wdl,outcome)+.1*F.cross_entropy(policy.masked_fill(~legal,-1e9),next_move)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
            pred=helper.predict(net,frozen['validation_board'],frozen['validation_turn']);score=helper.metric(pred,frozen['validation_target'],labels['validation'],vmask)
            if score['mse']<best:best=score['mse'];state={k:v.detach().clone() for k,v in net.state_dict().items()}
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**score));progress('训练中：'+('原基线硬标签复现' if mode=='hard' else '两库聚合软标签'),mode,epoch+1,score);print(json.dumps(dict(mode=mode,seconds=time.perf_counter()-started,**history[-1])),flush=True)
        net.load_state_dict(state);net.eval();binary=export(net,path);torch.save(state,path.with_suffix('.pt'))
        variant=dict(name=mode,model=str(path.relative_to(ROOT)),modelSha256=sha(path),formatMagic='OTPAT4',parameters=sum(p.numel() for p in net.parameters()),bytes=len(binary),initialParametersSha256=initial,initialEmbeddingSha256=embedding,dataOrderAndAugmentationSha256=sequence.hexdigest(),inputFingerprints=inputs,targetFingerprints=target_hashes,history=history,bestEpoch=min(history,key=lambda x:x['mse'])['epoch'],epoch15Validation={k:v for k,v in history[-1].items() if k not in ['epoch','loss']},metrics={})
        variants.append(variant);networks.append(net)
    assert variants[0]['initialParametersSha256']==variants[1]['initialParametersSha256']
    assert variants[0]['dataOrderAndAugmentationSha256']==variants[1]['dataOrderAndAugmentationSha256']==baseline['dataOrderAndAugmentationSha256']
    progress('两份权重已选定；统一冻结验证/测试与导出核验','hard/soft',EPOCHS)
    for variant,net in zip(variants,networks):
        for split,m in [('validation',vmask),('test',tmask)]:
            pred=helper.predict(net,frozen[split+'_board'],frozen[split+'_turn']);variant['metrics'][split]=helper.metric(pred,frozen[split+'_target'],labels[split],m);checks=parity(LIB,ROOT/variant['model'],frozen[split+'_board'],frozen[split+'_turn'],pred);checks['maxOutcomeError']=checks.pop('maxDiscError');variant[split+'Parity']=checks
        write((ROOT/variant['model']).with_suffix('.json'),dict(version='v1.45-mixed-label-ablation',status='candidate',defaultPromoted=False,**variant))
    replay=dict(referenceModel=baseline['model'],referenceSha256=baseline['modelSha256'],modelBytesIdentical=variants[0]['modelSha256']==baseline['modelSha256'],allEpochMetricsIdentical=variants[0]['history']==baseline['history'],maxHistoryMseDifference=max(abs(a['mse']-b['mse']) for a,b in zip(variants[0]['history'],baseline['history'])))
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    report=dict(version='v1.45-mixed-label-ablation',status='candidate',complete=True,defaultPromoted=False,matchesRun=False,architecture='Approved v1.37 symmetry original conditional-policy architecture',baselineManifest='models/v1.37-default.json',baselineMetrics=baseline['metrics'],controlledInputs=inputs,dataset=dict(sources=['wthor','logistello'],positions=n,eloFiltering=False,eloWeighting=False,allPositionWeights=1,alignedSoftLabelsSha256=sha(WORK/'aligned-soft-labels.npz'),hardLabels='Exact original frozen baseline labels; no first-record reconstruction',softLabels='Unweighted counts from both complete original-train corpora, restricted and aligned to exact original million rows'),labelChanges=changes,variants=variants,hardBaselineReplay=replay,epochs=EPOCHS,batchSize=2048,seed=SEED,dataSeed=DATA_SEED,threads=2,learningRate=.001,weightDecay=.0001,loss='Same mathematical objective: value MSE+0.1 W/L/D CE+0.1 legal policy CE. Hard mode uses original integer-class CE API for exact baseline replay; soft mode uses probability-target CE.',selection='Each minimum original frozen validation MSE; freeze both selected models before test evaluation; retain epoch15 validation for equal-update comparison',protocolSha256=sha(PROTOCOL),librarySha256=sha(LIB),preservedSha256=protocol['preservedSha256'],scriptSha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'scripts/mixed_label_targets.py',ROOT/'scripts/wthor_label_ablation.py',ROOT/'scripts/pattern_model.py',ROOT/'scripts/pattern_structure.py',ROOT/'scripts/training_consensus.py',ROOT/'scripts/train-pattern-outcome.py',ROOT/'scripts/train-pattern.py']},seconds=time.perf_counter()-started,limits='One seed, fixed original million mixed-source positions. Switch all value/WLD/policy targets jointly; does not separate policy vs value aggregation. No strength match, no default promotion. v1.44 WTHOR-only is a separate exploratory result, not this requested primary comparison.')
    write(REPORT,report);progress('原百万混合语料硬/软标签对照完成','hard/soft',EPOCHS,complete=True);write(ROOT/'models/v1.45-mixed-label-ablation-monitor.json',json.loads((WORK/'progress.json').read_text()));print(json.dumps(dict(complete=True,seconds=report['seconds'],replay=replay,results=[dict(name=v['name'],bestEpoch=v['bestEpoch'],test=v['metrics']['test']) for v in variants]),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
