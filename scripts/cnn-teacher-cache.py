#!/usr/bin/env python3
"""Freeze a CNN checkpoint and cache portable, split-separated distillation targets."""
import argparse,hashlib,importlib.util,json,shutil,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('big_cnn_training',ROOT/'scripts/train-big-cnn.py')
big=importlib.util.module_from_spec(spec);spec.loader.exec_module(big)
h=big.h


spec_reader=importlib.util.spec_from_file_location('teacher_targets',ROOT/'scripts/teacher-targets.py')
reader=importlib.util.module_from_spec(spec_reader);spec_reader.loader.exec_module(reader)
probabilities,TeacherCache=reader.probabilities,reader.TeacherCache


def write_shard(path,data):
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**data);temp.replace(path)


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get('complete'):raise ValueError('Use existing complete cache or a new output directory')
    progress_path=args.output/'progress.json';started=time.monotonic()
    progress=dict(experiment='cnn-offline-training',complete=False,stage='prediction-cache',currentName='固定 CNN 权重',currentChannels=256,
        epoch=0,maxEpochs=1,completedModels=0,totalModels=1)
    def update(**kwargs):
        progress.update(kwargs,seconds=round(time.monotonic()-started,2));h.write_json(progress_path,progress)
    update()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    assert args.device!='mps' or torch.backends.mps.is_available()
    device=torch.device(args.device)
    frozen=args.output/'teacher.pt'
    if not frozen.exists():
        temp=frozen.with_suffix('.tmp');shutil.copyfile(args.checkpoint,temp);temp.replace(frozen)
    bundle=torch.load(frozen,map_location='cpu',weights_only=True)
    net=big.BigCnn(bundle['channels'],bundle['blocks'],bundle['headWidth']).eval();net.load_state_dict(bundle['state_dict']);net.to(device)
    cache=ROOT/'.build/training/v1/positions.npz';labels_path=ROOT/'.build/training/policy/labels.npz'
    meta=json.loads(cache.with_suffix('.json').read_text());labels_meta=json.loads(labels_path.with_suffix('.json').read_text())
    assert h.sha(cache)==meta['cacheSha256']==labels_meta['positionsSha256'] and h.sha(labels_path)==labels_meta['cacheSha256']
    training_protocol=json.loads((ROOT/'.build/training/big-cnn-v1.24-run1/protocol.json').read_text())
    assert all(h.sha(ROOT/p)==digest for p,digest in training_protocol['sourceSha256'].items())
    source_files=['scripts/cnn-teacher-cache.py','scripts/verify-cnn-teacher-cache.py','scripts/teacher-targets.py']+list(training_protocol['sourceSha256'])
    sources={p:h.sha(ROOT/p) for p in source_files}
    schema=dict(version='v1.25-cnn-teacher-cache',complete=False,defaultPromoted=False,teacher=dict(
        checkpoint='teacher.pt',sha256=h.sha(frozen),epoch=bundle['epoch'],criterion=bundle['criterion'],
        channels=bundle['channels'],blocks=bundle['blocks'],headWidth=bundle['headWidth'],
        selection='frozen checkpoint chosen by validation before preprocessing; independent of later training checkpoints'),
        inputs=dict(positionsSha256=h.sha(cache),labelsSha256=h.sha(labels_path),positionsMetadata=meta,labelsMetadata=labels_meta),
        format=dict(board='int8[N,64], black1 white-1 empty0',turn='int8[N], black1 white-1',position_index='int64[N], original within-split row index',
            outcome_value='float32[N], expected side-to-move terminal outcome win1 draw0 loss-1; (value+1)/2 is expected score',
            policy_logits='float32[N,65], raw finite logits; entry64 is pass; apply legal mask before softmax',
            legal='bool[N,65]',recorded_outcome='int8[N], sign of original recorded outcome',recorded_move='uint8[N], original move0..64'),
        temperature='raw logits support arbitrary positive temperature when consumed; no temperature baked in',
        augmentation='original orientation only; board/logits/legal/move must receive the same D4 permutation during student augmentation; such augmented targets impose symmetry and need not equal a fresh nonequivariant CNN prediction',
        heldoutPolicy='train is the only training-target split; validation/test are separate inference caches, no held-out metrics or checkpoint selection is performed here',
        device=args.device,precision='FP32 eval, batchnorm frozen',batchSize=args.batch_size,shardSize=args.shard_size,
        sources=sources,requestedSplits=args.splits,splits={})
    known_shards={}
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text());assert old['teacher']==schema['teacher'] and old['inputs']==schema['inputs'] and old['sources']==sources
        assert all(old[k]==schema[k] for k in ('batchSize','shardSize','device','precision','requestedSplits'))
        known_shards={r['path']:r['sha256'] for split in old['splits'].values() for r in split['shards']}
    h.write_json(manifest_path,schema)
    monitor_path=ROOT/'models/current-experiment.json';config=json.loads(monitor_path.read_text());runtime=str(progress_path.relative_to(ROOT))
    if not any(r.get('runtimeReport')==runtime for r in config['experiments']):
        config['experiments'].append(dict(name=f"CNN 软标签缓存 · 冻结第 {bundle['epoch']} 轮",focus='teacher cache',games=0,runtimeReport=runtime,report='models/v1.25-cnn-teacher-cache.json'))
        h.write_json(monitor_path,config)
    with np.load(cache,allow_pickle=False) as d:data={k:d[k] for k in d.files}
    with np.load(labels_path,allow_pickle=False) as d:labels={k:d[k] for k in d.files}
    total=sum(len(data[s+'_board']) for s in args.splits);done=0;parity=[]
    for split in args.splits:
        rows=len(data[split+'_board']);split_dir=args.output/split;split_dir.mkdir(exist_ok=True)
        shards=[]
        for start in range(0,rows,args.shard_size):
            stop=min(start+args.shard_size,rows);path=split_dir/f'{start:07d}-{stop:07d}.npz'
            relative=str(path.relative_to(args.output))
            if path.exists() and relative in known_shards:assert h.sha(path)==known_shards[relative], 'Existing teacher shard changed'
            if not path.exists() or relative not in known_shards:
                values=[];logits=[]
                with torch.inference_mode():
                    for offset in range(start,stop,args.batch_size):
                        end=min(offset+args.batch_size,stop)
                        b=torch.from_numpy(data[split+'_board'][offset:end]).to(device);t=torch.from_numpy(data[split+'_turn'][offset:end]).to(device)
                        v,p=net(b,t);values.append(v.cpu().numpy());logits.append(p.cpu().numpy())
                        update(currentName=f'{split} · {end:,}/{rows:,}',processedPositions=done+end-start,totalPositions=total,split=split)
                v=np.concatenate(values);p=np.concatenate(logits);legal=labels[split+'_legal'][start:stop]
                assert v.dtype==np.float32 and p.dtype==np.float32 and np.isfinite(v).all() and np.isfinite(p).all() and (abs(v)<=1).all()
                probs=probabilities(p,legal);assert np.allclose(probs.sum(1),1) and (probs[~legal]==0).all()
                move=labels[split+'_move'][start:stop];assert legal[np.arange(len(move)),move].all()
                write_shard(path,dict(position_index=np.arange(start,stop,dtype=np.int64),board=data[split+'_board'][start:stop],turn=data[split+'_turn'][start:stop],
                    outcome_value=v,policy_logits=p,legal=legal,recorded_outcome=np.sign(data[split+'_target'][start:stop]).astype(np.int8),recorded_move=move.astype(np.uint8)))
            with np.load(path,allow_pickle=False) as z:
                assert np.array_equal(z['position_index'],np.arange(start,stop)) and np.array_equal(z['board'],data[split+'_board'][start:stop])
                assert np.array_equal(z['turn'],data[split+'_turn'][start:stop]) and np.array_equal(z['legal'],labels[split+'_legal'][start:stop])
                # Replay one cached batch per split without reading labels into model selection.
                if start==0:
                    n=min(8,stop-start)
                    with torch.inference_mode():
                        v,p=net(torch.from_numpy(z['board'][:n]).to(device),torch.from_numpy(z['turn'][:n]).to(device))
                    ve=float(abs(v.cpu().numpy()-z['outcome_value'][:n]).max());pe=float(abs(p.cpu().numpy()-z['policy_logits'][:n]).max())
                    assert ve<.0001 and pe<.001,(ve,pe)
                    parity.append(dict(split=split,positions=n,maxValueError=ve,maxLogitError=pe))
            shards.append(dict(path=str(path.relative_to(args.output)),start=start,stop=stop,positions=stop-start,bytes=path.stat().st_size,sha256=h.sha(path)))
            done+=stop-start;schema['splits'][split]=dict(positions=rows,role='training targets' if split=='train' else 'held-out inference only',shards=shards)
            h.write_json(manifest_path,schema)
        update(processedPositions=done)
    assert h.sha(frozen)==schema['teacher']['sha256'] and all(h.sha(ROOT/p)==digest for p,digest in sources.items())
    schema.update(complete=True,seconds=round(time.monotonic()-started,2),positions=total,predictionReplay=parity)
    h.write_json(manifest_path,schema)
    descriptor=dict(version=schema['version'],complete=True,cache=str(args.output.relative_to(ROOT)),manifestSha256=h.sha(manifest_path),teacher=schema['teacher'],
        splits={k:dict(positions=v['positions'],role=v['role'],shards=len(v['shards'])) for k,v in schema['splits'].items()},sources=sources,defaultPromoted=False)
    h.write_json(ROOT/'models/v1.25-cnn-teacher-cache.json',descriptor)
    update(stage='complete',complete=True,epoch=1,completedModels=1,candidates=[],processedPositions=total)
    print(json.dumps(dict(complete=True,cache=str(args.output),positions=total,teacherEpoch=bundle['epoch'])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'.build/training/big-cnn-v1.24-run1/joint.pt')
    p.add_argument('--output',type=Path,default=ROOT/'.build/training/teacher-cache/big-cnn-v1.24-joint-snapshot1')
    p.add_argument('--device',choices=['cpu','mps'],default='mps');p.add_argument('--batch-size',type=int,default=256);p.add_argument('--shard-size',type=int,default=16384)
    p.add_argument('--splits',nargs='+',choices=['train','validation','test'],default=['train','validation','test']);args=p.parse_args()
    assert args.batch_size>0 and args.shard_size>0 and len(set(args.splits))==len(args.splits)
    try:main(args)
    except Exception as exc:
        path=args.output/'progress.json'
        if path.exists():d=json.loads(path.read_text());d.update(stage='failed',error=str(exc));h.write_json(path,d)
        raise
