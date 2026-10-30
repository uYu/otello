#!/usr/bin/env python3
"""Distill a frozen large CNN into unchanged OTPAT2 Pattern outcome students."""
import argparse,hashlib,importlib.util,json,shutil,sys,tarfile,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from pattern_model import PatternNet,encode,legal_mask,export,parity
spec=importlib.util.spec_from_file_location('outcome_training',ROOT/'scripts/train-pattern-outcome.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
spec_reader=importlib.util.spec_from_file_location('teacher_targets',ROOT/'scripts/teacher-targets.py')
targets=importlib.util.module_from_spec(spec_reader);spec_reader.loader.exec_module(targets)
CONFIGS=[dict(name='finetune-control',alpha=0.),dict(name='cnn-distill25',alpha=.25),dict(name='cnn-distill50',alpha=.5)]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');temp.replace(path)
def save(path,data):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');torch.save(data,temp);temp.replace(path)

def loss_fn(value,wdl,logits,truth,move,mask,teacher_value,teacher_policy,alpha,temperature):
    hard=F.mse_loss(value,truth);soft=F.mse_loss(value,teacher_value)
    hard_policy=F.cross_entropy(logits.masked_fill(~mask,-1e9),move)
    soft_policy=-(teacher_policy*F.log_softmax(logits.masked_fill(~mask,-1e9)/temperature,1)).sum(1).mean()*temperature**2
    wdl_truth=torch.where(truth>0,0,torch.where(truth<0,1,2)).long()
    return (1-alpha)*hard+alpha*soft+.1*F.cross_entropy(wdl,wdl_truth)+.1*((1-alpha)*hard_policy+alpha*soft_policy)

def main(args):
    args.work.mkdir(parents=True,exist_ok=True);assert not (args.work/'protocol.json').exists() and not (ROOT/'models/v1.27-pattern-cnn-distill.json').exists()
    torch.set_num_threads(2);torch.manual_seed(20261029);torch.use_deterministic_algorithms(True)
    started=time.monotonic();progress=dict(experiment='cnn-offline-training',complete=False,stage='preparing',currentName='Pattern outcome · 读取CNN软标签',currentChannels=32,epoch=0,maxEpochs=args.epochs,completedModels=0,totalModels=3)
    def update(**kwargs):
        progress.update(kwargs,seconds=round(time.monotonic()-started,2));write(args.work/'progress.json',progress)
    update()
    cache=targets.TeacherCache(args.teacher_cache);manifest=cache.manifest
    assert manifest['inputs']['positionsSha256']==sha(ROOT/'.build/training/v1/positions.npz')
    parent=ROOT/'models/pattern32-outcome-v1.25.pt';parent_binary=parent.with_suffix('.otpat');parent_report=json.loads(parent.with_suffix('.json').read_text())
    parent_experiment=json.loads((ROOT/'models/v1.25-pattern-outcome-experiment.json').read_text())
    assert sha(parent)==parent_experiment['artifactSha256'][str(parent.relative_to(ROOT))] and sha(parent_binary)==parent_report['sha256']
    source_library=ROOT/'.build/native/libothello.dylib';assert sha(source_library)==parent_experiment['librarySha256']
    library=args.work/'libothello-pattern-frozen.dylib';shutil.copy2(source_library,library)
    baseline=json.loads((ROOT/'models/v1.9-default.json').read_text());assert sha(ROOT/baseline['defaultModel'])==baseline['modelSha256']
    preserved={str(p.relative_to(ROOT)):sha(p) for p in (parent,parent_binary,parent.with_suffix('.json'),ROOT/'models/default-model.json',ROOT/'models/v1.9-default.json',ROOT/baseline['defaultModel'])}
    data=dict(np.load(ROOT/'.build/training/v1/positions.npz'));labels=dict(np.load(ROOT/'.build/training/pattern-outcome-v1.25/policy.npz'))
    label_meta=json.loads((ROOT/'.build/training/pattern-outcome-v1.25/policy.json').read_text());assert sha(ROOT/'.build/training/pattern-outcome-v1.25/policy.npz')==label_meta['policySha256']
    masks={split:legal_mask(data[split+'_board'],data[split+'_turn']) for split in ('train','validation','test')}
    n=len(data['train_board']);teacher_value=np.empty(n,np.float32);teacher_policy=np.empty((n,64),np.float32);cursor=0;different_moves=0
    for row in cache.iter_train():
        ids=row['position_index'];count=len(ids);assert np.array_equal(ids,np.arange(cursor,cursor+count))
        assert np.array_equal(row['board'],data['train_board'][ids]) and np.array_equal(row['turn'],data['train_turn'][ids])
        assert not row['legal'][:,64].any() and np.array_equal(row['legal'][:,:64],masks['train'][ids])
        assert np.array_equal(row['recorded_outcome'],np.sign(data['train_target'][ids]))
        different_moves+=int((row['recorded_move']!=labels['train'][ids]).sum())
        teacher_value[ids]=row['outcome_value'];teacher_policy[ids]=targets.probabilities(row['policy_logits'],row['legal'],args.temperature)[:,:64].astype(np.float32)
        cursor+=count;update(currentName=f'加载CNN软标签 {cursor:,}/{n:,}',processedPositions=cursor)
    assert cursor==n and np.allclose(teacher_policy.sum(1),1)
    parent_state=torch.load(parent,map_location='cpu',weights_only=True)
    initial=PatternNet();initial.load_state_dict(parent_state);assert export(initial,args.work/'parent-roundtrip.otpat',outcome=True)==parent_binary.read_bytes()
    sources=['scripts/distill-pattern-outcome.py','scripts/verify-pattern-distill.py','scripts/teacher-targets.py','scripts/pattern_model.py','scripts/train-pattern-outcome.py','scripts/train-pattern.py']
    protocol=dict(version='v1.27-pattern-cnn-distill',status='candidate',defaultPromoted=False,architecture=parent_report['architecture'],formatMagic='OTPAT2',parameters=parent_report['parameters'],
        parentModel=str(parent_binary.relative_to(ROOT)),parentSha256=sha(parent_binary),parentStateSha256=sha(parent),configs=CONFIGS,
        teacherCache=str(args.teacher_cache.relative_to(ROOT)),teacherManifestSha256=sha(args.teacher_cache/'manifest.json'),teacher=manifest['teacher'],
        trainingSplitOnly=True,trainPositions=n,epochs=args.epochs,batchSize=args.batch_size,seed=20261029,dataSeed=20261030,learningRate=.0003,cosineEtaMin=.000003,weightDecay=.0001,gradientClip=5.,
        temperature=args.temperature,valueLoss='(1-alpha)*MSE(recordedOutcome)+alpha*MSE(CNNoutcome)',
        policyLoss='0.1*((1-alpha)*recordedMoveCE+alpha*T^2*CE(CNNsoftmax(logits/T),studentSoftmax(logits/T)))',
        wdlLoss='0.1*recordedWdlCE; CNN has no WDL distribution, no fabricated WDL soft targets',
        initialization='all variants warm-start from identical frozen v1.25 outcome Pattern; all parameters jointly fine-tuned with existing STE quantization',
        augmentation='identical seeded sample order and D4 sequence; cached policy targets permuted with board; original-orientation CNN supervision is augmented, not rerun',
        selection='minimum original hard-label validation outcome MSE, including epoch0 parent; test all students only after all freeze',
        datasetSha256=manifest['inputs']['positionsSha256'],recordedPolicySource=label_meta,teacherRecordedMoveDifferences=different_moves,
        library=str(library.relative_to(ROOT)),librarySha256=sha(library),parentSourceManifestSha256=sha(ROOT/parent_experiment['sourceManifest']),
        sources={p:sha(ROOT/p) for p in sources},preservedSha256=preserved,
        verification=json.loads((ROOT/'.build/training/pattern-cnn-distill-v1.27-checks/report.json').read_text()),
        torchVersion=torch.__version__,numpyVersion=np.__version__,device='cpu',threads=2)
    write(args.work/'protocol.json',protocol)
    archive=ROOT/'models/archive/v1.27-pattern-cnn-distill-sources.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'w:gz') as tar:
        for p in sources:tar.add(ROOT/p,arcname=p)
    monitor=ROOT/'models/current-experiment.json';config=json.loads(monitor.read_text());runtime=str((args.work/'progress.json').relative_to(ROOT))
    if not any(x.get('runtimeReport')==runtime for x in config['experiments']):config['experiments'].append(dict(name='CNN → outcome Pattern 蒸馏 · 3组离线对照',focus='pattern distillation',games=0,runtimeReport=runtime,report='models/v1.27-pattern-cnn-distill.json'))
    # Preserve any other ongoing comparison's monitor rather than taking ownership of it.
    other=ROOT/'models/v1.25-pattern-outcome-2s-20-run1-experiments.json'
    if other.exists():
        for item in json.loads(other.read_text()).get('experiments',[]):
            if not any(r.get('runtimeReport')==item.get('runtimeReport') for r in config['experiments']):config['experiments'].insert(0,item)
    write(monitor,config)
    board,turn=data['train_board'],data['train_turn'];truth=np.sign(data['train_target']).astype(np.float32);phase=(64-np.count_nonzero(board,axis=1)).astype(np.float32)/60
    grid=np.arange(64).reshape(8,8);perms=np.stack([np.rot90(np.fliplr(grid) if mirror else grid,rotation).reshape(64) for mirror in (False,True) for rotation in range(4)]);inverse=np.argsort(perms,1)
    training=[]
    for conf in CONFIGS:
        torch.manual_seed(20261029);torch_rng=torch.Generator().manual_seed(20261030);rng=np.random.default_rng(20261030)
        net=PatternNet();net.load_state_dict(parent_state);opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=.000003);best=float('inf');best_epoch=0;history=[]
        update(stage='training',currentName=conf['name'],epoch=0,validation=None)
        def evaluate(epoch):
            nonlocal best,best_epoch
            scores=old.metric(old.predict(net,data['validation_board'],data['validation_turn']),data['validation_target'],labels['validation'],masks['validation'])
            assert np.isfinite(scores['mse'])
            history.append(dict(epoch=epoch,**scores))
            if scores['mse']<best:
                best=scores['mse'];best_epoch=epoch;save(args.work/(conf['name']+'.pt'),dict(config=conf,epoch=epoch,state_dict={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}))
            update(epoch=epoch,validation=dict(value=dict(outcomeMse=scores['mse']),policy=dict(top1=scores['policyTop1'])),bestEpoch=best_epoch)
            print(json.dumps(dict(name=conf['name'],**history[-1])),flush=True)
        evaluate(0)
        for epoch in range(1,args.epochs+1):
            net.train();order=torch.randperm(n,generator=torch_rng).numpy();loss_sum=0.;done=0
            for start in range(0,n,args.batch_size):
                ids=order[start:start+args.batch_size];sym=rng.integers(8,size=len(ids));p=perms[sym]
                b=np.take_along_axis(board[ids],p,1);code=torch.from_numpy(encode(b,turn[ids]).astype(np.int64))
                value,wdl,logits=net(code,torch.from_numpy(phase[ids]));mask=torch.from_numpy(np.take_along_axis(masks['train'][ids],p,1))
                move=torch.from_numpy(inverse[sym,labels['train'][ids]].astype(np.int64));soft_policy=torch.from_numpy(np.take_along_axis(teacher_policy[ids],p,1))
                loss=loss_fn(value,wdl,logits,torch.from_numpy(truth[ids]),move,mask,torch.from_numpy(teacher_value[ids]),soft_policy,conf['alpha'],args.temperature)
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
                done+=len(ids);loss_sum+=float(loss.detach())*len(ids)
                if start%65536==0:update(epoch=epoch,processedPositions=(epoch-1)*n+done,trainingLoss=loss_sum/done)
            scheduler.step();evaluate(epoch)
        training.append(dict(**conf,bestEpoch=best_epoch,bestValidationMse=best,history=history));write(args.work/'training-checkpoints.json',training);update(completedModels=len(training))
    update(stage='frozen-evaluation');evaluated=[]
    for conf in CONFIGS:
        checkpoint=args.work/(conf['name']+'.pt');bundle=torch.load(checkpoint,map_location='cpu',weights_only=True);net=PatternNet().eval();net.load_state_dict(bundle['state_dict'])
        model_path=ROOT/'models'/('pattern32-'+conf['name']+'-v1.27.otpat');assert not model_path.exists();binary=export(net,model_path,outcome=True)
        state_path=model_path.with_suffix('.pt');save(state_path,net.state_dict())
        row=dict(**conf,epoch=bundle['epoch'],model=str(model_path.relative_to(ROOT)),sha256=sha(model_path),parameters=sum(p.numel() for p in net.parameters()),bytes=len(binary),metrics={},cppParity={})
        for split in ('validation','test'):
            pred=old.predict(net,data[split+'_board'],data[split+'_turn']);row['metrics'][split]=old.metric(pred,data[split+'_target'],labels[split],masks[split])
            checked=parity(library,model_path,data[split+'_board'],data[split+'_turn'],pred);checked['maxOutcomeError']=checked.pop('maxDiscError');row['cppParity'][split]=checked
            if split=='test':np.savez_compressed(args.work/(conf['name']+'-test-predictions.npz'),value=pred[0],wdl_logits=pred[1],policy_logits=pred[2])
        write(model_path.with_suffix('.json'),row);evaluated.append(row)
    assert all(sha(ROOT/p)==digest for p,digest in preserved.items()) and all(sha(ROOT/p)==digest for p,digest in protocol['sources'].items())
    reference=json.loads((ROOT/'models/v1.25-pattern-outcome-offline.json').read_text())
    report=dict(**protocol,complete=True,experiment='cnn-offline-training',candidates=evaluated,trainingReports=training,parentMetrics=parent_report['metrics'],baselineMetrics=reference['baseline'],seconds=round(time.monotonic()-started,2))
    write(args.work/'report.json',report);write(ROOT/'models/v1.27-pattern-cnn-distill.json',report)
    lines=['# 大 CNN → outcome Pattern：离线蒸馏','','学生结构、OTPAT2推理格式和参数量保持不变；从同一v1.25权重微调。普通微调与25%/50%蒸馏各10轮，相同数据顺序、D4增强和学习率；按验证胜负MSE选权重，包含epoch0父模型。',
        '教师为固定的大CNN第11轮，仅读取训练集缓存；温度2，策略KD乘T²，WDL辅助头只使用原始赛果。测试集不用于选模型；全部训练后统一测试，所有验证/测试局面检查C++导出。默认保持v1.9；本轮不对战。','',
        '| 模型 | 胜负MSE ↓ | 非和局准确率 ↑ | 走法命中率 ↑ | 走法交叉熵 ↓ |','|---|---:|---:|---:|---:|']
    base=reference['baseline']['test'];lines.append(f"| v1.9 | {base['outcomeMse']:.6f} | {base['nonDrawSignAccuracy']:.2%} | {base['policyTop1']:.2%} | {base['policyCrossEntropy']:.6f} |")
    for name,s in [('原 outcome Pattern',parent_report['metrics']['test'])]+[(r['name'],r['metrics']['test']) for r in evaluated]:lines.append(f"| {name} | {s['mse']:.6f} | {s['nonDrawSignAccuracy']:.2%} | {s['policyTop1']:.2%} | {s['policyCrossEntropy']:.6f} |")
    doc=ROOT/'models/v1.27-pattern-cnn-distill-comparison.md';doc.write_text('\n'.join(lines)+'\n')
    artifacts=[ROOT/'models/v1.27-pattern-cnn-distill.json',doc]+list((ROOT/'models').glob('pattern32-*-v1.27.*'))
    write(ROOT/'models/archive/v1.27-pattern-cnn-distill-sources.json',dict(archive=str(archive.relative_to(ROOT)),sha256=sha(archive),sources=protocol['sources'],artifactSha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts},preservedSha256=preserved))
    displayed=[dict(r,channels=32,metrics={split:dict(value=dict(outcomeMse=s['mse'],nonDrawSignAccuracy=s['nonDrawSignAccuracy']),policy=dict(top1=s['policyTop1'])) for split,s in r['metrics'].items()}) for r in evaluated]
    update(stage='complete',complete=True,candidates=displayed);print(json.dumps(dict(complete=True,seconds=report['seconds'])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,default=ROOT/'.build/training/pattern-cnn-distill-v1.27-run1')
    p.add_argument('--teacher-cache',type=Path,default=ROOT/'.build/training/teacher-cache/big-cnn-v1.24-joint-snapshot1')
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--batch-size',type=int,default=2048);p.add_argument('--temperature',type=float,default=2.);args=p.parse_args()
    assert args.epochs>0 and args.batch_size>0 and args.temperature>0
    try:main(args)
    except Exception as exc:
        path=args.work/'progress.json'
        if path.exists():d=json.loads(path.read_text());d.update(stage='failed',error=str(exc));write(path,d)
        raise
