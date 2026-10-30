#!/usr/bin/env python3
"""Train the speed-selected shared residual candidate with frozen v1.37 data."""
import hashlib,importlib.util,json,time
import numpy as np
import torch
from torch.nn import functional as F
from pattern_model import encode,legal_mask,parity
from pattern_structure import StructureNet
from pattern_shared_residual import SharedResidualNet,export
from training_consensus import PERMS,INVERSE
import shared_residual_experiment as e
REPORT=e.ROOT/'models/v1.48-pattern-shared-residual.json'
spec=importlib.util.spec_from_file_location('shared_residual_metrics',e.ROOT/'scripts/train-pattern-outcome.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def main():
    assert not REPORT.exists(),'Reports are immutable'
    protocol=e.read(e.PROTOCOL);e.verify_preserved(protocol)
    gate=e.read(e.ROOT/'models/v1.48-shared-residual-speed-gate.json');width=gate['selectedResidualWidth']
    assert gate['complete'] and width in [8,16],'Candidate failed the preregistered speed gate'
    archive=e.read(e.ROOT/protocol['sourceArchiveManifest']);assert e.sha(e.ROOT/archive['archive'])==archive['sha256']
    assert e.sha(e.ROOT/'.build/training/v1/positions.npz')==protocol['datasetSha256']
    assert e.sha(e.ROOT/'.build/training/v1.8/policy.npz')==protocol['policySha256']
    torch.set_num_threads(2);started=time.perf_counter()
    data=dict(np.load(e.ROOT/'.build/training/v1/positions.npz'));labels=dict(np.load(e.ROOT/'.build/training/v1.8/policy.npz'))
    masks={k:legal_mask(data[k+'_board'],data[k+'_turn']) for k in labels}
    assert all(masks[k][np.arange(len(labels[k])),labels[k]].all() for k in labels)
    baseline=e.read(e.ROOT/'models/pattern32-symmetry-v1.37.json')
    torch.manual_seed(20261029);net=SharedResidualNet(width)
    torch.manual_seed(20261029);base=StructureNet('symmetry')
    for key,value in base.state_dict().items():assert torch.equal(value,net.state_dict()[key]),key
    initial_embedding_sha=hashlib.sha256(net.embedding.weight.detach().numpy().tobytes()).hexdigest()
    assert initial_embedding_sha==baseline['initialEmbeddingSha256']
    with torch.no_grad():
        sample=data['validation_board'][:256];turn=data['validation_turn'][:256]
        codes=torch.from_numpy(encode(sample,turn).astype(np.int64));phase=torch.from_numpy((64-np.count_nonzero(sample,axis=1)).astype(np.float32)/60)
        assert all(torch.equal(a,b) for a,b in zip(base(codes,phase),net(codes,phase)))
    board,turn,target=data['train_board'],data['train_turn'],np.sign(data['train_target']);n=len(board);assert n==1000000
    phase=(64-np.count_nonzero(board,axis=1)).astype(np.float32)/60
    order=torch.Generator().manual_seed(20261030);augmentation=np.random.default_rng(20261030);sequence=hashlib.sha256()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    best=float('inf');state=None;history=[]
    for epoch in range(15):
        net.train();losses=[]
        e.progress('训练共享主干残差',epoch,15,f'32→{width}→32 · 原百万硬标签')
        for rows in torch.randperm(n,generator=order).split(2048):
            indices=rows.numpy();sym=augmentation.integers(8,size=len(indices),dtype=np.int8);sequence.update(indices.tobytes());sequence.update(sym.tobytes());perm=PERMS[sym]
            batch=np.take_along_axis(board[indices],perm,axis=1);codes=torch.from_numpy(encode(batch,turn[indices]).astype(np.int64))
            value,wdl,policy=net(codes,torch.from_numpy(phase[indices]));truth=torch.from_numpy(target[indices])
            next_move=torch.from_numpy(INVERSE[sym,labels['train'][indices]]);mask=torch.from_numpy(np.take_along_axis(masks['train'][indices],perm,axis=1))
            outcome=torch.where(truth>0,0,torch.where(truth<0,1,2)).long()
            loss=F.mse_loss(value,truth)+.1*F.cross_entropy(wdl,outcome)+.1*F.cross_entropy(policy.masked_fill(~mask,-1e9),next_move)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        prediction=helper.predict(net,data['validation_board'],data['validation_turn']);score=helper.metric(prediction,data['validation_target'],labels['validation'],masks['validation'])
        if score['mse']<best:best=score['mse'];state={k:v.detach().clone() for k,v in net.state_dict().items()}
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**score))
        e.progress('训练共享主干残差',epoch+1,15,f'验证 MSE {score["mse"]:.6f} · 策略命中率 {score["policyTop1"]:.2%}',validation=score)
        print(json.dumps(dict(seconds=time.perf_counter()-started,**history[-1])),flush=True)
    assert sequence.hexdigest()==baseline['dataOrderAndAugmentationSha256']
    net.load_state_dict(state);net.eval()
    path=e.ROOT/f'models/pattern32-symmetry-residual{width}-v1.48.otpat';assert not path.exists()
    binary=export(net,path);torch.save(state,path.with_suffix('.pt'))
    variant=dict(name=f'shared-residual{width}',residualWidth=width,model=str(path.relative_to(e.ROOT)),modelSha256=e.sha(path),formatMagic='OTPAT8',parameters=sum(p.numel() for p in net.parameters()),bytes=len(binary),initialEmbeddingSha256=initial_embedding_sha,sharedInitialParametersIdentical=True,initialPredictionsIdentical=True,dataOrderAndAugmentationSha256=sequence.hexdigest(),history=history,bestEpoch=min(history,key=lambda x:x['mse'])['epoch'],metrics={},nativeParity={})
    e.progress('权重已冻结：测试集及原生导出核验')
    for split in ['validation','test']:
        prediction=helper.predict(net,data[split+'_board'],data[split+'_turn'])
        variant['metrics'][split]=helper.metric(prediction,data[split+'_target'],labels[split],masks[split])
        checks=parity(e.LIB,path,data[split+'_board'],data[split+'_turn'],prediction);checks['maxOutcomeError']=checks.pop('maxDiscError');variant['nativeParity'][split]=checks
    e.write(path.with_suffix('.json'),dict(version='v1.48-shared-residual',status='candidate',defaultPromoted=False,**variant))
    e.verify_preserved(protocol)
    report=dict(version='v1.48-shared-residual',complete=True,status='candidate',defaultPromoted=False,matchesRun=False,baselineManifest='models/v1.37-default.json',baselineMetrics=baseline['metrics'],variant=variant,
        protocolSha256=e.sha(e.PROTOCOL),speedGateSha256=e.sha(e.ROOT/'models/v1.48-shared-residual-speed-gate.json'),sourceArchiveManifest=protocol['sourceArchiveManifest'],sourceArchiveManifestSha256=e.sha(e.ROOT/protocol['sourceArchiveManifest']),librarySha256=e.sha(e.LIB),portableLibrarySha256=e.sha(e.PORTABLE),preservedSha256=protocol['preservedSha256'],training=protocol['training'],seconds=time.perf_counter()-started,limits='One seed, same original data/order/augmentation/selection. Model capacity added only in shared trunk. Offline metrics do not establish strength; pending time-budget match.')
    e.write(REPORT,report);e.progress('训练与导出核验完成',15,15,f'选第{variant["bestEpoch"]}轮 · 测试MSE {variant["metrics"]["test"]["mse"]:.6f}')
    print(json.dumps(dict(complete=True,variant=variant['name'],bestEpoch=variant['bestEpoch'],test=variant['metrics']['test'])),flush=True)


if __name__=='__main__':main()
