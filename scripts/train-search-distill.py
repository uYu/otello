#!/usr/bin/env python3
"""Pilot search distillation into unchanged v1.9 NNUE, with matched finetuning control."""
import argparse,ctypes,importlib.util,json,shutil,tarfile,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('product_training',ROOT/'scripts/train-product-nnue.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
h,suite=m.h,m.suite
spec2=importlib.util.spec_from_file_location('engineering',ROOT/'scripts/check-engineering.py')
e=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(e)
CONFIGS=[dict(name='finetune-control',alpha=0.),dict(name='search-distilled',alpha=.5)]
LIBRARY=ROOT/'.build/native-product-nnue-v1.22/libothello.dylib'

def parent_net(binary):
    assert binary[:8]==b'OTNNUE4\0' and len(binary)==104936
    net=m.ProductNet(False);offset=24
    with torch.no_grad():
        a=np.frombuffer(binary,dtype='<i2',count=128*256,offset=offset).reshape(128,256).T.copy();offset+=a.nbytes
        net.transform.weight.copy_(torch.from_numpy(a)/256)
        for tensor,dtype,scale in [(net.transform.bias,'<i4',256),(net.value.weight,'<i2',256),(net.value.bias,'<i4',65536)]:
            a=np.frombuffer(binary,dtype=dtype,count=tensor.numel(),offset=offset).copy();offset+=a.nbytes
            tensor.copy_(torch.from_numpy(a).reshape(tensor.shape)/scale)
        assert binary[offset:offset+8]==b' \x00\x00\x00A\x00\x00\x00';offset+=8
        for layer in (net.policy_hidden,net.policy_out):
            for tensor,dtype,scale in [(layer.weight,'<i2',256),(layer.bias,'<i4',65536)]:
                a=np.frombuffer(binary,dtype=dtype,count=tensor.numel(),offset=offset).copy();offset+=a.nbytes
                tensor.copy_(torch.from_numpy(a).reshape(tensor.shape)/scale)
    assert offset==len(binary) and m.export(net)==binary
    return net

class Teacher:
    def __init__(self,library,model):
        self.lib=e.load(str(library));self.lib.ot_nnue_error.restype=ctypes.c_char_p
        self.lib.ot_cnn_set_native_batch_evaluator.argtypes=[ctypes.c_int]
        assert self.lib.ot_cnn_set_native_batch_evaluator(32)==1
        self.handle=self.lib.ot_nnue_load(str(model).encode());assert self.handle
    def close(self):self.lib.ot_nnue_destroy(self.handle)
    def search(self,board,turn,legal,simulations,seed):
        cells=(ctypes.c_int*64)(*map(int,board));out=(ctypes.c_int*136)();policy=(ctypes.c_int*6)();visits=(ctypes.c_int*64)();stats=(ctypes.c_double*72)()
        self.lib.ot_mcts_clear_cache()
        ok=self.lib.ot_nnue_mcts_play(self.handle,cells,int(turn),0,simulations,seed,0,0,0,out,policy,visits,stats)
        if ok!=1:raise ValueError(self.lib.ot_nnue_error())
        probs=np.zeros(65,dtype=np.float64);qs=np.zeros(65,dtype=np.float64);counts=np.zeros(65,dtype=np.int64)
        if out[6]:
            for i in range(out[6]):
                move=out[7+2*i];assert 0<=move<64 and legal[move]
                counts[move]=visits[i];qs[move]=stats[8+i]
                probs[move]=visits[i]
                if out[5] and out[8+2*i]!=out[1]:probs[move]=0
            if probs.sum()==0:
                assert out[0]>=0;probs[out[0]]=1
            value=float(np.sign(out[1])) if out[5] else float(qs[out[0]])
        else:
            assert legal[64] and not legal[:64].any(),'Only forced-pass training positions supported without candidates'
            probs[64]=1;value=float(np.sign(out[1])) if out[5] else out[1]/1000
        probs/=probs.sum()
        assert abs(value)<=1 and np.isfinite(probs).all() and (probs[~legal]==0).all() and abs(probs.sum()-1)<1e-12
        assert 0<stats[0]<=simulations and not stats[7]
        return dict(value=value,policy=probs.astype(np.float32),visits=counts,childValues=qs.astype(np.float32),
                    exact=bool(out[5]),simulations=int(stats[0]),bestMove=out[0])

def choose_indices(data,count):
    rng=np.random.default_rng(20261028);seen=set();ids=[]
    for i in rng.permutation(len(data['train_board'])):
        key=data['train_board'][i].tobytes()+data['train_turn'][i].tobytes()
        if key in seen:continue
        seen.add(key);ids.append(int(i))
        if len(ids)==count:break
    assert len(ids)==count
    return np.array(ids,dtype=np.int64)

def make_teacher(args,data,model,update):
    ids=choose_indices(data,args.positions);folder=args.output/'teacher-shards';folder.mkdir(exist_ok=True)
    teacher=Teacher(LIBRARY,model);shards=[]
    try:
        for start in range(0,len(ids),128):
            selected=ids[start:start+128];path=folder/f'{start:06d}.npz'
            if path.exists():
                with np.load(path) as z:assert np.array_equal(z['indices'],selected)
            else:
                rows=[]
                for j,i in enumerate(selected):
                    rows.append(teacher.search(data['train_board'][i],data['train_turn'][i],data['train_legal'][i],args.simulations,(20261028^int(i))&0x7fffffff))
                    if j%16==0:update(stage='teacher-search',currentName=f'v1.9 搜索软标签 {start+j+1}/{len(ids)}',teacherCompleted=start+j+1,teacherPlanned=len(ids))
                tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,indices=selected,**{key:np.array([r[key] for r in rows]) for key in rows[0]});tmp.replace(path)
            shards.append(dict(path=str(path.relative_to(ROOT)),sha256=h.sha(path)));update(teacherCompleted=start+len(selected))
    finally:teacher.close()
    loaded=[dict(np.load(ROOT/s['path'])) for s in shards]
    result={key:np.concatenate([z[key] for z in loaded]) for key in loaded[0]}
    assert np.array_equal(result['indices'],ids)
    report=dict(complete=True,positions=len(ids),sourceSplit='train only',selection='seeded uniform rows, skip duplicate board+turn; no validation/test selection',
        requestedSimulations=args.simulations,totalSimulations=int(result['simulations'].sum()),exactPositions=int(result['exact'].sum()),
        valueTarget='selected max-visit root child mean side-to-move outcome; solved root uses proven outcome',
        policyTarget='root visit distribution at temperature1; proven roots keep only proven optimal moves; forced pass maps to65th entry',
        librarySha256=h.sha(LIBRARY),teacherModel=str(model.relative_to(ROOT)),teacherModelSha256=h.sha(model),shards=shards)
    h.write_json(args.output/'teacher-report.json',report)
    return result,report

def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'models/v1.23-search-distill.json').exists()
    started=time.monotonic();progress=dict(experiment='cnn-offline-training',complete=False,stage='teacher-search',currentChannels=256,
        currentName='准备搜索教师',epoch=0,maxEpochs=args.epochs,completedModels=0,totalModels=2)
    def update(**kwargs):
        progress.update(kwargs,seconds=round(time.monotonic()-started,2));h.write_json(args.output/'progress.json',progress)
    update()
    monitor=ROOT/'models/current-experiment.json';config=json.loads(monitor.read_text())
    runtime=str((args.output/'progress.json').relative_to(ROOT))
    if not any(x.get('runtimeReport')==runtime for x in config['experiments']):
        config['experiments'].append(dict(name='搜索蒸馏 NNUE · 普通微调对照',focus='distillation',games=0,runtimeReport=runtime,report='models/v1.23-search-distill.json'))
    h.write_json(monitor,config)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    cache=ROOT/'.build/training/v1/positions.npz';label_path=ROOT/'.build/training/policy/labels.npz'
    meta=json.loads(cache.with_suffix('.json').read_text());labels_meta=json.loads(label_path.with_suffix('.json').read_text())
    assert h.sha(cache)==meta['cacheSha256']==labels_meta['positionsSha256'] and h.sha(label_path)==labels_meta['cacheSha256']
    baseline=json.loads((ROOT/'models/v1.9-default.json').read_text());model=ROOT/baseline['defaultModel'];assert h.sha(model)==baseline['modelSha256']
    verified=json.loads((ROOT/'models/v1.22-product-nnue.json').read_text());assert h.sha(LIBRARY)==verified['librarySha256']
    source_files=['scripts/train-search-distill.py','scripts/verify-search-distill.py','scripts/benchmark-search-distill.py',
        'scripts/train-product-nnue.py','scripts/train-cnn-policy-suite.py','scripts/train-mixed-5m.py','scripts/train-deep.py',
        'scripts/train-outcome.py','scripts/train-policy.py','scripts/train-nnue.py','scripts/check-engineering.py']
    source_files += [str(p.relative_to(ROOT)) for p in (ROOT/'experiments/product-nnue-v1.22').rglob('*') if p.is_file()]
    protocol=dict(version='v1.23-search-distill',status='candidate',defaultPromoted=False,architecture=baseline['architecture'],parameters=52098,
        modelFormat='OTNNUE4',configs=CONFIGS,epochs=args.epochs,seed=20261027,dataSeed=20261028,
        batchSize=1024,supplementalBatchSize=256,learningRate=.0001,cosineEtaMin=.000001,weightDecay=.001,valueWeight=4.,policyWeight=1.,
        initialization='both start with exact approved v1.9 bytes, all layers jointly fine-tuned with QAT',
        training='original million supervised rows plus supplemental256 sampled from same teacher subset for BOTH variants; supplemental target alpha0 control vs alpha0.5 search-value/visit-policy; combined loss weighted by1024/256 counts',
        teacherPositions=args.positions,teacherSimulationBudget=args.simulations,teacherUsesBook=False,teacherModelSha256=h.sha(model),
        librarySha256=h.sha(LIBRARY),dataset=meta,labels=labels_meta,device=args.device,
        selection='save validation MSE+CE minimum and value MSE minimum, including epoch0 baseline; test after both freeze',
        validationAndTestUnchanged=True,verification=json.loads((ROOT/'.build/training/search-distill-v1.23-checks/report.json').read_text()),sourceSha256={p:h.sha(ROOT/p) for p in source_files})
    protocol_path=args.output/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol,'Refusing different resume protocol'
    else:h.write_json(protocol_path,protocol)
    archive=ROOT/'models/archive/v1.23-search-distill-sources.tar.gz'
    if not archive.exists():
        with tarfile.open(archive,'w:gz') as tar:
            for p in source_files:tar.add(ROOT/p,arcname=p)
    data=dict(np.load(cache));labels=dict(np.load(label_path))
    for split in ('train','validation','test'):
        for key in ('legal','move'):data[split+'_'+key]=labels[split+'_'+key]
    teacher,teacher_report=make_teacher(args,data,model,update)
    assert args.device!='mps' or torch.backends.mps.is_available();device=torch.device(args.device)
    ids=teacher['indices'];boards=torch.from_numpy(data['train_board']);turns=torch.from_numpy(data['train_turn'])
    target=torch.from_numpy(np.sign(data['train_target']).astype(np.float32));moves=torch.from_numpy(data['train_move'].astype(np.int64));legal=torch.from_numpy(data['train_legal'])
    softvalue=torch.from_numpy(teacher['value'].astype(np.float32));softpolicy=torch.from_numpy(teacher['policy'])
    perms=torch.from_numpy(h.policy.value.symmetries());inverse=torch.argsort(perms,1);teacher_ids=torch.from_numpy(ids)
    training=[]
    for conf in CONFIGS:
        torch.manual_seed(20261027);rng=torch.Generator().manual_seed(20261028);net=parent_net(model.read_bytes()).to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=.0001,weight_decay=.001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=.000001)
        best=dict(joint=float('inf'),value=float('inf'));history=[];best_epoch={}
        def score_and_save(epoch):
            score=h.metrics(*suite.predict(net,data,'validation',device),data,'validation');assert np.isfinite(score['selectionLoss'])
            history.append(dict(epoch=epoch,**score))
            for criterion,metric in [('joint',score['selectionLoss']),('value',score['value']['outcomeMse'])]:
                if metric<best[criterion]:
                    best[criterion]=metric;best_epoch[criterion]=epoch
                    torch.save(dict(config=conf,epoch=epoch,criterion=criterion,state_dict={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}),args.output/f"{conf['name']}-{criterion}.pt")
            update(validation=score,bestEpoch=best_epoch,epoch=epoch)
            print(json.dumps(dict(name=conf['name'],**history[-1])),flush=True)
        update(stage='training',currentName=conf['name'],epoch=0);score_and_save(0)
        for epoch in range(1,args.epochs+1):
            net.train();order=torch.randperm(len(boards),generator=rng)
            for offset in range(0,len(boards),1024):
                original=order[offset:offset+1024];teacher_select=torch.randint(len(ids),(256,),generator=rng)
                chosen=torch.cat((original,teacher_ids[teacher_select]));sym=torch.randint(8,(len(chosen),),generator=rng);p=perms[sym]
                b=torch.gather(boards[chosen],1,p).to(device);t=turns[chosen].to(device)
                mask=torch.cat((torch.gather(legal[chosen,:64],1,p),legal[chosen,64:]),1).to(device)
                mm=torch.where(moves[chosen]==64,64,inverse[sym,moves[chosen].clamp(max=63)])
                vt=target[chosen].clone();pi=nn.functional.one_hot(mm,65).float();n=len(original);alpha=conf['alpha']
                vt[n:]=(1-alpha)*vt[n:]+alpha*softvalue[teacher_select]
                teacher_pi=softpolicy[teacher_select];tp=torch.cat((torch.gather(teacher_pi[:,:64],1,p[n:]),teacher_pi[:,64:]),1)
                pi[n:]=(1-alpha)*pi[n:]+alpha*tp
                v,logits=net(b,t);logp=nn.functional.log_softmax(logits.masked_fill(~mask,-1e9),1)
                loss=4*(v-vt.to(device)).square().mean()-(pi.to(device)*logp).sum(1).mean()
                opt.zero_grad();loss.backward();opt.step()
                if offset%65536==0:update(epoch=epoch,processedPositions=(epoch-1)*len(boards)+offset+len(original))
            scheduler.step();score_and_save(epoch)
        training.append(dict(**conf,bestEpoch=best_epoch,bestValidation=best,history=history));h.write_json(args.output/'training-checkpoints.json',training);update(completedModels=len(training))
    update(stage='frozen-evaluation');evaluated=[]
    for conf in CONFIGS:
        for criterion in ('joint','value'):
            name=conf['name']+'-'+criterion;path=args.output/(name+'.pt');bundle=torch.load(path,map_location='cpu',weights_only=True)
            net=m.ProductNet(False).double().eval();net.load_state_dict(bundle['state_dict'])
            native_path=args.output/(name+'.otnnue');native_path.write_bytes(m.export(net))
            row=dict(name=name,epoch=bundle['epoch'],parameters=52098,bytes=native_path.stat().st_size,sha256=h.sha(native_path),metrics={},nativeParity={})
            for split in ('validation','test'):
                py=suite.predict(net,data,split,torch.device('cpu'));native=h.native_predictions(native_path,data,split,LIBRARY)
                ve,pe=float(abs(py[0]-native[0]).max()),float(abs(py[1]-native[1]).max());assert ve<=.000501 and pe<1e-10,(name,split,ve,pe)
                row['metrics'][split]=h.metrics(*native,data,split);row['nativeParity'][split]=dict(positions=len(py[0]),maxValueError=ve,maxPolicyError=pe)
            for suffix in ('.pt','.otnnue'):
                dst=ROOT/'models'/('nnue256-'+name+'-v1.23'+suffix);assert not dst.exists();shutil.copy2(args.output/(name+suffix),dst)
            row['model']='models/nnue256-'+name+'-v1.23.otnnue';evaluated.append(row)
    h.write_json(args.output/'evaluated.json',evaluated);update(stage='speed-benchmark')
    import subprocess,sys
    subprocess.run([sys.executable,str(ROOT/'scripts/benchmark-search-distill.py'),'--output',str(args.output/'speed.json')],check=True)
    report=dict(**protocol,complete=True,experiment='cnn-offline-training',teacher=teacher_report,trainingReports=training,candidates=evaluated,
        baselineMetrics=verified['baselineMetrics'],nativeSpeed=json.loads((args.output/'speed.json').read_text()),seconds=round(time.monotonic()-started,2))
    assert all(h.sha(ROOT/p)==v for p,v in protocol['sourceSha256'].items())
    h.write_json(args.output/'report.json',report);h.write_json(ROOT/'models/v1.23-search-distill.json',report)
    lines=['# 搜索蒸馏 NNUE 小样实验','','教师v1.9，只使用训练集1万个不同局面，每局面最多32768次模拟，无开局库。普通微调和蒸馏均从v1.9开始，全层QAT微调15轮；参数量和推理结构不变。',
        '两组使用相同原百万局面及额外256子集样本；控制组额外样本仍用原标签，蒸馏组使用原标签与搜索标签各50%。验证与测试集不变。',
        '验证选择包含epoch0；每组保存联合损失和价值MSE最优权重。教师预算是固定模拟数，不等于每步2秒，也不保证教师每个局面都更准确。默认保持v1.9，尚未对战。','',
        '| 模型 | 胜负MSE | 非和局准确率 | 走法命中率 |','|---|---:|---:|---:|']
    for name,s in [('v1.9',verified['baselineMetrics']['test'])]+[(c['name'],c['metrics']['test']) for c in evaluated]:
        lines.append(f"| {name} | {s['value']['outcomeMse']:.6f} | {s['value']['nonDrawSignAccuracy']:.2%} | {s['policy']['top1']:.2%} |")
    doc=ROOT/'models/v1.23-search-distill-comparison.md';doc.write_text('\n'.join(lines)+'\n')
    artifacts=[ROOT/'models/v1.23-search-distill.json',doc]+list((ROOT/'models').glob('nnue256-*-v1.23.*'))
    h.write_json(ROOT/'models/archive/v1.23-search-distill-sources.json',dict(archive=str(archive.relative_to(ROOT)),sha256=h.sha(archive),files=protocol['sourceSha256'],artifactSha256={str(p.relative_to(ROOT)):h.sha(p) for p in artifacts}))
    update(stage='complete',complete=True,candidates=[dict(c,channels=256) for c in evaluated],baselineMetrics=verified['baselineMetrics'])
    print(json.dumps(dict(complete=True)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'.build/training/search-distill-v1.23-run1')
    p.add_argument('--positions',type=int,default=10000);p.add_argument('--simulations',type=int,default=32768);p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--device',choices=['cpu','mps'],default='mps');args=p.parse_args()
    assert min(args.positions,args.simulations,args.epochs)>0
    try:main(args)
    except Exception as exc:
        path=args.output/'progress.json'
        if path.exists():d=json.loads(path.read_text());d.update(stage='failed',error=str(exc));h.write_json(path,d)
        raise
