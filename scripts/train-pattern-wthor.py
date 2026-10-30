#!/usr/bin/env python3
"""Train the frozen symmetry architecture on WTHOR-only empirical soft targets."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pattern_model import encode,legal_mask,parity
from pattern_structure import StructureNet,export
from training_consensus import aggregate,boards_from_keys,PERMS
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'.build/training/pattern-wthor-v1.42';BASE=ROOT/'.build/training/consensus-elo-v1.40'
REPORT=ROOT/'models/v1.42-pattern-wthor.json';MODEL=ROOT/'models/pattern32-symmetry-wthor-v1.42.otpat';PROTOCOL=ROOT/'models/v1.42-pattern-wthor-protocol.json'
LIB=ROOT/('.build/native-v1.37-reference/libothello.'+('dylib' if sys.platform=='darwin' else 'so'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,r:Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('wthor_outcome_metrics',ROOT/'scripts/train-pattern-outcome.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
EPOCHS=15;SEED=20261029;DATA_SEED=20261030


def main():
    assert not REPORT.exists() and not MODEL.exists(),'Completed candidates are immutable'
    protocol=json.loads(PROTOCOL.read_text());started=time.perf_counter();WORK.mkdir(parents=True,exist_ok=True);history=[]
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    def progress(stage,epoch=0,validation=None,complete=False):
        r=dict(experiment='pattern-structure-training',variant='symmetry-wthor',stage=stage,epoch=epoch,epochs=EPOCHS,seconds=time.perf_counter()-started,variants=[],complete=complete)
        if validation is not None:r['validation']=validation
        write(WORK/'progress.json',r)
    config=json.loads((ROOT/'models/current-experiment.json').read_text());old=config.get('archivedExperiments',[])
    for item in config['experiments']:
        if not any(x.get('report')==item.get('report') for x in old):old.insert(0,item)
    config.update(version='v1.42 WTHOR 全量 · 对称共享原结构',archivedExperiments=old,experiments=[dict(name='WTHOR 全量 · 对称共享原结构训练',focus='symmetry-wthor',games=0,runtimeReport=str((WORK/'progress.json').relative_to(ROOT)),report='models/v1.42-pattern-wthor-monitor.json')]);write(ROOT/'models/current-experiment.json',config)
    progress('准备 WTHOR 全量等权聚合；不作 Elo 筛选')
    with np.load(BASE/'occurrences.npz') as z:occ={k:z[k] for k in ['keys','action_bits','outcome','source','game_index']}
    keep=occ['source']==0;occ={k:v[keep] for k,v in occ.items()};assert len(np.unique(occ['game_index']))<=80561
    data=aggregate(occ['keys'],occ['action_bits'],occ['outcome'],np.ones(len(occ['keys'])),occ['source']);del occ
    np.testing.assert_array_equal(data.pop('policy_quality_mass'),data['policy_mass']);np.testing.assert_array_equal(data.pop('policy_quality_total'),data['occurrence_count']);assert not data['source_count'][:,1].any()
    n=len(data['keys']);count=data['occurrence_count'];data['wdl_probability']=(data['wdl_count']/count[:,None]).astype(np.float32);data['value_mean']=(data['wdl_probability'][:,0]-data['wdl_probability'][:,1]).astype(np.float32)
    groups=np.repeat(np.arange(n),np.diff(data['policy_offsets']).astype(np.int64));data['policy_probability']=(data['policy_mass']/count[groups]).astype(np.float32)
    np.savez_compressed(WORK/'training-data.npz',**data)
    frozen=dict(np.load(ROOT/'.build/training/v1/positions.npz'));labels=dict(np.load(ROOT/'.build/training/v1.8/policy.npz'))
    board=boards_from_keys(data['keys']);turn=np.ones(n,np.int8);target=data['value_mean'];wdl_target=data['wdl_probability'];phase=(64-np.count_nonzero(board,axis=1)).astype(np.float32)/60
    policy_target=np.zeros((n,64),np.float32);policy_target[groups,data['policy_action']]=data['policy_probability'];mask=legal_mask(board,turn)
    assert not policy_target[~mask].any();np.testing.assert_allclose(policy_target.sum(1),1,atol=1e-6,rtol=0)
    for split in ['validation','test']:
        key=np.concatenate([__import__('training_consensus').canonical(frozen[split+'_board'][a:a+2048],frozen[split+'_turn'][a:a+2048])[0] for a in range(0,6000,2048)])
        assert not np.isin(data['keys'],key).any()
    validation_mask=legal_mask(frozen['validation_board'],frozen['validation_turn']);test_mask=legal_mask(frozen['test_board'],frozen['test_turn'])
    write(WORK/'data.json',dict(complete=True,trainGames=80561,trainPositions=n,retainedOccurrences=int(count.sum()),sources=['wthor'],eloFiltering=False,eloWeighting=False,allOccurrencesEqual=True,trainingDataSha256=sha(WORK/'training-data.npz'),frozenCacheSha256=sha(ROOT/'.build/training/v1/positions.npz'),frozenPolicySha256=sha(ROOT/'.build/training/v1.8/policy.npz')))
    torch.set_num_threads(2);torch.manual_seed(SEED);np.random.seed(SEED);net=StructureNet('symmetry')
    initial_embedding_sha=hashlib.sha256(net.embedding.weight.detach().numpy().tobytes()).hexdigest();baseline_model=json.loads((ROOT/'models/pattern32-symmetry-v1.37.json').read_text());assert initial_embedding_sha==baseline_model['initialEmbeddingSha256']
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);order=torch.Generator().manual_seed(DATA_SEED);augmentation=np.random.default_rng(DATA_SEED);sequence=hashlib.sha256();best=float('inf');best_state=None
    progress('训练中：对称共享原结构')
    for epoch in range(EPOCHS):
        net.train();losses=[]
        for rows in torch.randperm(n,generator=order).split(2048):
            indices=rows.numpy();sym=augmentation.integers(8,size=len(indices),dtype=np.int8);sequence.update(indices.tobytes());sequence.update(sym.tobytes());perm=PERMS[sym]
            batch=np.take_along_axis(board[indices],perm,axis=1);code=torch.from_numpy(encode(batch,turn[indices]).astype(np.int64));value,wdl,policy=net(code,torch.from_numpy(phase[indices]))
            probabilities=torch.from_numpy(np.take_along_axis(policy_target[indices],perm,axis=1));legal=torch.from_numpy(np.take_along_axis(mask[indices],perm,axis=1))
            loss=F.mse_loss(value,torch.from_numpy(target[indices]))+.1*F.cross_entropy(wdl,torch.from_numpy(wdl_target[indices]))+.1*-(probabilities*F.log_softmax(policy.masked_fill(~legal,-1e9),dim=1)).sum(1).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        prediction=helper.predict(net,frozen['validation_board'],frozen['validation_turn']);score=helper.metric(prediction,frozen['validation_target'],labels['validation'],validation_mask)
        if score['mse']<best:best=score['mse'];best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**score));progress('训练中：对称共享原结构',epoch+1,score);print(json.dumps(dict(seconds=time.perf_counter()-started,**history[-1])),flush=True)
    net.load_state_dict(best_state);net.eval();binary=export(net,MODEL);torch.save(best_state,MODEL.with_suffix('.pt'))
    metrics={};checks={};progress('权重已选定；冻结验证/测试与 C++ 导出核验',EPOCHS)
    for split,m in [('validation',validation_mask),('test',test_mask)]:
        pred=helper.predict(net,frozen[split+'_board'],frozen[split+'_turn']);metrics[split]=helper.metric(pred,frozen[split+'_target'],labels[split],m);checks[split]=parity(LIB,MODEL,frozen[split+'_board'],frozen[split+'_turn'],pred);checks[split]['maxOutcomeError']=checks[split].pop('maxDiscError')
    variant=dict(name='symmetry-wthor',model=str(MODEL.relative_to(ROOT)),modelSha256=sha(MODEL),formatMagic='OTPAT4',parameters=sum(p.numel() for p in net.parameters()),bytes=len(binary),initialEmbeddingSha256=initial_embedding_sha,dataOrderAndAugmentationSha256=sequence.hexdigest(),history=history,bestEpoch=min(history,key=lambda r:r['mse'])['epoch'],metrics=metrics,validationParity=checks['validation'],testParity=checks['test'])
    write(MODEL.with_suffix('.json'),dict(version='v1.42-pattern-wthor',status='candidate',defaultPromoted=False,**variant))
    for name,digest in protocol['preservedSha256'].items():assert sha(ROOT/name)==digest,name
    report=dict(version='v1.42-pattern-wthor',status='candidate',defaultPromoted=False,architecture='Exact approved v1.37 D4 symmetry model:38 patterns,embedding32,three means+phase97->32,46 outputs with original conditional policy; no direct MLP',baselineManifest='models/v1.37-default.json',baselineMetrics=baseline_model['metrics'],baselineModelSha256=baseline_model['modelSha256'],variants=[variant],dataset=json.loads((WORK/'data.json').read_text()),epochs=EPOCHS,batchSize=2048,seed=SEED,dataSeed=DATA_SEED,threads=2,learningRate=.001,weightDecay=.0001,loss='soft recorded-outcome mean MSE +0.1 soft W/L/D CE +0.1 unweighted empirical legal-move CE',initialization='random; identical shared table/trunk/full46-head seed as v1.37, no trained-weight warm start',selection='minimum unchanged 6000-position validation original-label outcome MSE; test once after checkpoint selection',protocolSha256=sha(PROTOCOL),librarySha256=sha(LIB),scriptSha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'scripts/pattern_model.py',ROOT/'scripts/pattern_structure.py',ROOT/'scripts/training_consensus.py',ROOT/'scripts/train-pattern-outcome.py',ROOT/'scripts/train-pattern.py']},seconds=time.perf_counter()-started,limits='WTHOR-only source and full unweighted empirical multi-label aggregation change together; this experiment cannot attribute their effects separately. Frozen validation/test unchanged; no Elo threshold or weighting. Single training seed; offline metrics do not establish strength; default not promoted.')
    write(REPORT,report);progress('训练与冻结验证/测试完成',EPOCHS,metrics['validation'],True);write(ROOT/'models/v1.42-pattern-wthor-monitor.json',json.loads((WORK/'progress.json').read_text()));print(json.dumps(dict(complete=True,bestEpoch=variant['bestEpoch'],metrics=metrics,seconds=report['seconds']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
