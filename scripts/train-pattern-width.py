#!/usr/bin/env python3
"""Compare only H32/H64 Pattern trunks with identical random D4 training order."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from pattern_model import PatternNet,encode,legal_mask,export,parity
spec=importlib.util.spec_from_file_location('pattern_width_helpers',ROOT/'scripts/train-pattern-outcome.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
WORK=ROOT/'.build/training/pattern-width-v1.29'
REPORT=ROOT/'models/v1.29-pattern-width.json'
LIB=ROOT/('.build/native-pattern-h64-v1.29/libothello.'+('dylib' if sys.platform=='darwin' else 'so'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,r:Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
EPOCHS=15;SEED=20261029;DATA_SEED=20261030


def main():
    assert not REPORT.exists(),'Choose a fresh experiment; reports are never overwritten'
    active= json.loads((ROOT/'models/default-model.json').read_text());baseline=ROOT/active['baselineManifest']
    assert active['version']=='v1.25'
    active_sha=sha(ROOT/'models/default-model.json');baseline_sha=sha(baseline)
    cache=ROOT/'.build/training/v1/positions.npz';metadata=json.loads(cache.with_suffix('.json').read_text())
    assert sha(cache)==metadata['cacheSha256']==json.loads((ROOT/'models/nnue256-v1.json').read_text())['dataset']['cacheSha256']
    data=dict(np.load(cache));policy_dir=ROOT/'.build/training/pattern-outcome-v1.25';provenance=json.loads((policy_dir/'policy.json').read_text())
    assert sha(policy_dir/'policy.npz')==provenance['policySha256'] and provenance['cacheSha256']==metadata['cacheSha256']
    labels=dict(np.load(policy_dir/'policy.npz'));masks={k:legal_mask(data[k+'_board'],data[k+'_turn']) for k in labels}
    assert all(masks[k][np.arange(len(labels[k])),labels[k]].all() for k in labels)
    torch.set_num_threads(2);started=time.perf_counter();variants=[];networks=[]
    grid=np.arange(64).reshape(8,8);permutations=np.stack([np.rot90(np.fliplr(grid) if mirror else grid,rotation).reshape(64) for mirror in (False,True) for rotation in range(4)]);inverse=np.argsort(permutations,axis=1)
    board,turn,target=data['train_board'],data['train_turn'],np.sign(data['train_target']);phase=(64-np.count_nonzero(board,axis=1)).astype(np.float32)/60
    def progress(stage,name,epoch=0,validation=None,complete=False):
        value=dict(experiment='pattern-width-training',complete=complete,stage=stage,variant=name,epoch=epoch,epochs=EPOCHS,seconds=time.perf_counter()-started,variants=variants)
        if validation is not None:value['validation']=validation
        write(WORK/'progress.json',value)
    for width in [32,64]:
        name=f'H{width}'+('-control' if width==32 else '')
        path=ROOT/f'models/pattern32-h{width}-v1.29.otpat';assert not path.exists()
        torch.manual_seed(SEED);np.random.seed(SEED)
        order=torch.Generator().manual_seed(DATA_SEED);augmentation=np.random.default_rng(DATA_SEED);sequence=hashlib.sha256()
        net=PatternNet(hidden=width);initial_embedding_sha=hashlib.sha256(net.embedding.weight.detach().numpy().tobytes()).hexdigest()
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        best=float('inf');state=None;history=[];progress('训练中',name)
        for epoch in range(EPOCHS):
            net.train();losses=[]
            for rows in torch.randperm(len(board),generator=order).split(2048):
                indices=rows.numpy();sym=augmentation.integers(8,size=len(indices),dtype=np.int8);sequence.update(indices.tobytes());sequence.update(sym.tobytes());perm=permutations[sym]
                batch=np.take_along_axis(board[indices],perm,axis=1);code=torch.from_numpy(encode(batch,turn[indices]).astype(np.int64))
                value,wdl,policy=net(code,torch.from_numpy(phase[indices]));truth=torch.from_numpy(target[indices]);next_move=torch.from_numpy(inverse[sym,labels['train'][indices]]);mask=torch.from_numpy(np.take_along_axis(masks['train'][indices],perm,axis=1))
                outcome=torch.where(truth>0,0,torch.where(truth<0,1,2)).long()
                loss=F.mse_loss(value,truth)+.1*F.cross_entropy(wdl,outcome)+.1*F.cross_entropy(policy.masked_fill(~mask,-1e9),next_move)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
            pred=helper.predict(net,data['validation_board'],data['validation_turn']);score=helper.metric(pred,data['validation_target'],labels['validation'],masks['validation'])
            if score['mse']<best:best=score['mse'];state={k:v.detach().clone() for k,v in net.state_dict().items()}
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**score));progress('训练中',name,epoch+1,score)
            print(json.dumps(dict(variant=name,seconds=time.perf_counter()-started,**history[-1])),flush=True)
        net.load_state_dict(state);net.eval();binary=export(net,path,outcome=True);torch.save(state,path.with_suffix('.pt'))
        pred=helper.predict(net,data['validation_board'],data['validation_turn']);validation=helper.metric(pred,data['validation_target'],labels['validation'],masks['validation'])
        checks=parity(LIB,path,data['validation_board'],data['validation_turn'],pred);checks['maxOutcomeError']=checks.pop('maxDiscError')
        variant=dict(name=name,width=width,model=str(path.relative_to(ROOT)),modelSha256=sha(path),formatMagic='OTPAT2',parameters=sum(p.numel() for p in net.parameters()),bytes=len(binary),initialEmbeddingSha256=initial_embedding_sha,dataOrderAndAugmentationSha256=sequence.hexdigest(),history=history,bestEpoch=min(history,key=lambda row:row['mse'])['epoch'],metrics=dict(validation=validation),validationParity=checks)
        variants.append(variant);networks.append(net)
    # Evaluate test only after both validation-selected models have been frozen.
    progress('两组权重已选定；测试与 C++ 对照','H32/H64',EPOCHS)
    for net,variant in zip(networks,variants):
        pred=helper.predict(net,data['test_board'],data['test_turn']);variant['metrics']['test']=helper.metric(pred,data['test_target'],labels['test'],masks['test'])
        checks=parity(LIB,ROOT/variant['model'],data['test_board'],data['test_turn'],pred);checks['maxOutcomeError']=checks.pop('maxDiscError');variant['testParity']=checks
        write((ROOT/variant['model']).with_suffix('.json'),dict(version='v1.29-pattern-width',status='candidate',defaultPromoted=False,sha256=variant['modelSha256'],**variant))
    assert variants[0]['dataOrderAndAugmentationSha256']==variants[1]['dataOrderAndAugmentationSha256']
    assert variants[0]['initialEmbeddingSha256']==variants[1]['initialEmbeddingSha256']
    assert sha(ROOT/'models/default-model.json')==active_sha and sha(baseline)==baseline_sha
    report=dict(version='v1.29-pattern-width',status='candidate',defaultPromoted=False,baselineManifest=active['baselineManifest'],baselineManifestSha256=baseline_sha,baselineModelSha256=active['modelSha256'],trainPositions=len(board),epochs=EPOCHS,batchSize=2048,seed=SEED,dataSeed=DATA_SEED,learningRate=.001,weightDecay=.0001,loss='outcome MSE + 0.1 WDL CE + 0.1 legal next-move CE',initialization='random; identical embedding initialization; no warm start',augmentation='uniform random choice among 4 rotations x mirror/no mirror per sampled position; board, next move and legal mask transform together; data order and symmetry sequence identical for both widths',datasetSha256=metadata['cacheSha256'],policyDataset=provenance,selection='minimum original-label validation outcome MSE; all test evaluations after both selected models frozen',variants=variants,seconds=time.perf_counter()-started,librarySha256=sha(LIB),scriptSha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'scripts/pattern_model.py',ROOT/'scripts/train-pattern-outcome.py',ROOT/'scripts/train-pattern.py']},limits='Offline architecture comparison only; no match-strength claim; width increases arithmetic despite small parameter-byte increase')
    write(REPORT,report);progress('已完成','H32/H64',EPOCHS,complete=True)
    print(json.dumps(dict(complete=True,seconds=report['seconds'],summary=[dict(name=v['name'],validation=v['metrics']['validation'],test=v['metrics']['test']) for v in variants])),flush=True)


if __name__=='__main__':main()
