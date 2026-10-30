#!/usr/bin/env python3
"""Finish v1.20 evaluation from frozen checkpoints; no retraining or weight changes."""
import argparse, importlib.util, json, shutil, tarfile, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('light_training',ROOT/'scripts/train-light-cnn.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
suite,h,exporter,CONFIGS=m.suite,m.h,m.exporter,m.CONFIGS

def main(args):
    torch.set_num_threads(2)
    protocol=json.loads((args.output/'protocol.json').read_text())
    assert all(h.sha(ROOT/p)==digest for p,digest in protocol['sourceSha256'].items())
    assert h.sha(args.library)==protocol['librarySha256']
    old_progress=json.loads((args.output/'progress.json').read_text())
    h.write_json(args.output/'original-evaluation-failure.json',old_progress)
    protocol['evaluationRecovery']=dict(reason=old_progress.get('error'),
        trainerUnchanged=True,weightsUnchanged=True,valueTolerance=.00051,policyTolerance=.00002,
        reasonForTolerance='FP32 convolution accumulation order differs between PyTorch and native SIMD; probability error is measured on every held-out position; exact selected moves and native metrics also recorded',
        finalizerSha256=h.sha(Path(__file__)))
    source_files=list(protocol['sourceSha256'])+['scripts/finalize-light-cnn.py']
    protocol['sourceSha256']['scripts/finalize-light-cnn.py']=h.sha(Path(__file__))
    data=dict(np.load(ROOT/'.build/training/v1/positions.npz'))
    labels=dict(np.load(ROOT/'.build/training/policy/labels.npz'))
    for split in ('train','validation','test'):
        for key in ('legal','move'):data[split+'_'+key]=labels[split+'_'+key]
    assert h.sha(ROOT/'.build/training/v1/positions.npz')==protocol['dataset']['cacheSha256']
    assert h.sha(ROOT/'.build/training/policy/labels.npz')==protocol['labels']['cacheSha256']
    previous=json.loads((ROOT/'models/v1.13-cnn2-tuning.json').read_text())
    assert h.sha(ROOT/'models/v1.13-cnn2-tuning.json')==protocol['previousReportSha256']
    original32=next(r for r in previous['candidates'] if r['name']=='cosine-v4-value')
    training_reports=json.loads((args.output/'training-checkpoints.json').read_text())
    assert len(training_reports)==3 and all(len(r['history'])==30 for r in training_reports)
    started=time.monotonic()-old_progress['seconds']
    progress=dict(old_progress);progress.pop('error',None)
    def update(**kwargs):
        progress.update(kwargs,seconds=round(time.monotonic()-started,2))
        h.write_json(args.output/'progress.json',progress)
    winner = min(training_reports, key=lambda r: r['bestValidation']['joint'])['name']
    value_winner = min(training_reports, key=lambda r: r['bestValidation']['value'])['name']
    update(stage='frozen-evaluation', winnerName=winner, valueWinnerName=value_winner)
    evaluated = []
    for report in training_reports:
        for criterion in ('joint', 'value'):
            name = report['name']+'-'+criterion
            path = args.output/(name+'.pt')
            bundle = torch.load(path, map_location='cpu', weights_only=True)
            net = suite.CnnPolicyValue(bundle['channels'], bundle['headWidth']).eval()
            net.load_state_dict(bundle['state_dict'])
            traced = torch.jit.trace(net, (torch.from_numpy(data['validation_board'][:2]), torch.from_numpy(data['validation_turn'][:2])))
            export = args.output/(name+'.torchscript.pt')
            traced.save(str(export))
            reloaded = torch.jit.load(str(export)).eval()
            native_path = args.output/(name+'.otcnn')
            exporter.export(net, native_path)
            candidate = dict(model='models/cnn2-'+name+'-v1.20.otcnn', nativeSha256=h.sha(native_path), name=name, channels=bundle['channels'], headWidth=bundle['headWidth'],
                             parameters=sum(p.numel() for p in net.parameters()), criterion=criterion, config=report,
                             epoch=bundle['epoch'], modelSha256=h.sha(path), torchscriptSha256=h.sha(export), metrics={})
            for split in ('validation', 'test'):
                fp = suite.predict(net, data, split, torch.device('cpu'))
                loaded = suite.predict(reloaded, data, split, torch.device('cpu'))
                error = max(float(abs(a-b).max()) for a, b in zip(fp, loaded))
                assert error < 1e-6
                native = h.native_predictions(native_path, data, split, args.library)
                verr, perr = float(abs(fp[0]-native[0]).max()), float(abs(fp[1]-native[1]).max())
                assert verr < .00051 and perr < .00002, (name,split,verr,perr)
                candidate.setdefault('nativeParity', {})[split] = dict(positions=len(fp[0]), maxValueError=verr, maxPolicyError=perr)
                candidate.setdefault('nativePredictionAgreement', {})[split] = dict(policyTop1Agreement=float((fp[1].argmax(1)==native[1].argmax(1)).mean()), nativeMetrics=h.metrics(*native,data,split))
                candidate['metrics'][split] = dict(**h.metrics(*fp, data, split), maxExportError=error, exportParityPositions=len(fp[0]))
                if split == 'test':
                    np.savez_compressed(args.output/(name+'-test-predictions.npz'), board=data['test_board'], turn=data['test_turn'],
                                        target=np.sign(data['test_target']), recorded_move=data['test_move'], value=fp[0], policy=fp[1])
            evaluated.append(candidate)
            for src, suffix in [(path, '.pt'), (export, '.torchscript.pt'), (native_path, '.otcnn')]:
                dst = ROOT/'models'/('cnn2-'+name+'-v1.20'+suffix)
                assert not dst.exists()
                shutil.copy2(src, dst)
    h.write_json(args.output/'evaluated.json', evaluated)
    update(stage='speed-benchmark')
    import subprocess, sys
    subprocess.run([sys.executable, str(ROOT/'scripts/benchmark-light-cnn.py'), '--output', str(args.output/'speed.json')], check=True)
    speed = json.loads((args.output/'speed.json').read_text())
    report = dict(**protocol, experiment='cnn-offline-training', complete=True,
                  baselineMetrics=previous['baselineMetrics'], originalC32=original32, nativeSpeed=speed,
                  trainingReports=training_reports, candidates=evaluated, winnerName=winner, valueWinnerName=value_winner,
                  completedModels=len(CONFIGS), totalModels=len(CONFIGS), currentChannels=32, epoch=args.epochs,
                  maxEpochs=args.epochs, seconds=time.monotonic()-started)
    h.write_json(args.output/'report.json', report)
    h.write_json(ROOT/'models/v1.20-light-cnn.json', report)
    update(stage='complete', complete=True, candidates=evaluated, baselineMetrics=previous['baselineMetrics'])
    rows = []
    for name, score in [('v1.9', previous['baselineMetrics']['test']), ('原 32 通道', original32['metrics']['test'])]+[(r['name'], r['metrics']['test']) for r in evaluated]:
        rows.append(f"| {name} | {score['value']['outcomeMse']:.6f} | {score['value']['nonDrawSignAccuracy']:.2%} | {score['policy']['top1']:.2%} | {score['policy']['crossEntropy']:.6f} |")
    doc = ROOT/'models/v1.20-light-cnn-comparison.md'
    doc.write_text('# 轻量两层 CNN 离线实验\n\n原百万训练局面、相同验证/测试集，各从头训练30轮；C32/H32、C16/H32、C8/H32。'
                   '固定 AdamW、batch1024、学习率0.001余弦衰减至0.00001、价值/策略损失权重4:1；与v1.13 cosine-v4-value采用相同配置。'
                   '各保留验证joint/value最优检查点，全部训练后统一测试。\n\n'
                   '| 模型 | 胜负MSE ↓ | 非和局准确率 ↑ | 走法命中率 ↑ | 走法交叉熵 ↓ |\n|---|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n\n'
                   '原生OTCNN3格式按真实H32计算；验证和测试全量检查导出误差。单线程、相同原生batch32后端速度记录见JSON nativeSpeed；速度不代表棋力。尚未对战，默认保持v1.9。\n')
    archive = ROOT/'models/archive/v1.20-light-cnn-sources.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as tar:
        for p in source_files:
            tar.add(ROOT/p, arcname=p)
    artifacts = [ROOT/'models/v1.20-light-cnn.json', doc]+list((ROOT/'models').glob('cnn2-*-v1.20.*'))
    assert all(h.sha(ROOT/p) == digest for p, digest in protocol['sourceSha256'].items())
    h.write_json(ROOT/'models/archive/v1.20-light-cnn-sources.json', dict(
        archive=str(archive.relative_to(ROOT)), sha256=h.sha(archive), files=protocol['sourceSha256'],
        artifactSha256={str(p.relative_to(ROOT)): h.sha(p) for p in artifacts}))
    print(json.dumps(dict(complete=True, winnerName=winner, valueWinnerName=value_winner, seconds=report['seconds'])), flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'.build/training/light-cnn-v1.20-run1')
    p.add_argument('--library',type=Path,default=ROOT/'.build/native-light-cnn-v1.20/libothello.dylib')
    args=p.parse_args();args.epochs=30
    try:main(args)
    except Exception as exc:
        path=args.output/'progress.json';d=json.loads(path.read_text());d.update(stage='failed',error=str(exc));h.write_json(path,d);raise
