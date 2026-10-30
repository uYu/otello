#!/usr/bin/env python3
"""Train compact two-layer CNNs on the original frozen data, then export and benchmark."""
import argparse
import importlib.util
import json
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cnn_suite', ROOT/'scripts/train-cnn-policy-suite.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
h = suite.helpers

CONFIGS = [dict(name=f'c{c}-h32', channels=c, headWidth=32, learningRate=.001,
                schedule='cosine', valueWeight=4.) for c in (32,16,8)]
spec_export = importlib.util.spec_from_file_location('light_export', ROOT/'scripts/export-light-cnn.py')
exporter = importlib.util.module_from_spec(spec_export)
spec_export.loader.exec_module(exporter)



def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'protocol.json').exists() or (ROOT/'models/v1.20-light-cnn.json').exists():
        raise ValueError('Use a fresh experiment; refusing overwrite')
    progress = dict(experiment='cnn-offline-training', complete=False, stage='queued',
                    currentName='等待 2 秒对局结束', channels=[32,16,8], currentChannels=32,
                    epoch=0, maxEpochs=args.epochs, completedModels=0, totalModels=3)
    h.write_json(args.output/'progress.json', progress)
    monitor = ROOT/'models/current-experiment.json'
    old = json.loads(monitor.read_text())
    old['experiments'].append(dict(name='轻量 CNN · C32/C16/C8 + H32', focus='light CNN', games=0,
        runtimeReport=str((args.output/'progress.json').relative_to(ROOT)), report='models/v1.20-light-cnn.json'))
    h.write_json(monitor, old)
    while True:
        match = json.loads(args.wait_match.read_text()) if args.wait_match.exists() else {}
        if match.get('complete'):
            break
        time.sleep(5)
    previous = json.loads(args.previous_report.read_text())
    assert previous['complete']
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable')
    device = torch.device(args.device)
    cache = ROOT/'.build/training/v1/positions.npz'
    labels_path = ROOT/'.build/training/policy/labels.npz'
    meta = json.loads(cache.with_suffix('.json').read_text())
    label_meta = json.loads(labels_path.with_suffix('.json').read_text())
    assert h.sha(cache) == meta['cacheSha256'] == label_meta['positionsSha256'] == previous['dataset']['cacheSha256']
    assert h.sha(labels_path) == label_meta['cacheSha256'] == previous['labels']['cacheSha256']
    data, labels = dict(np.load(cache)), dict(np.load(labels_path))
    for split in ('train', 'validation', 'test'):
        for key in ('legal', 'move'):
            data[split+'_'+key] = labels[split+'_'+key]
    original32 = next(r for r in previous['candidates'] if r['name'] == 'cosine-v4-value')
    source_files = ['scripts/train-light-cnn.py', 'scripts/export-light-cnn.py', 'scripts/benchmark-light-cnn.py', 'scripts/check-engineering.py', 'scripts/verify-light-cnn.py', 'scripts/train-cnn-policy-suite.py',
                    'scripts/train-mixed-5m.py', 'scripts/train-deep.py', 'scripts/train-outcome.py',
                    'scripts/train-policy.py', 'scripts/train-nnue.py']
    source_files += [str(p.relative_to(ROOT)) for p in (ROOT/'experiments/light-cnn-v1.20').rglob('*') if p.is_file()]
    protocol = dict(version='v1.20-light-cnn', status='candidate', defaultPromoted=False,
                    configs=CONFIGS, epochs=args.epochs, batchSize=1024,
                    weightDecay=.001, seed=20261027, device=args.device,
                    targets='original recorded side-to-move outcome and recorded move; no added games or book labels',
                    dataset=meta, labels=label_meta, originalC32Validation=original32['metrics']['validation'],
                    nativeSmokeChecks=json.loads((ROOT/'.build/training/light-cnn-v1.20-checks/report.json').read_text()),
                    librarySha256=h.sha(args.library), library=str(args.library.relative_to(ROOT)),
                    previousReportSha256=h.sha(args.previous_report),
                    training='from scratch per configuration; same seed and separate identical sample/augmentation RNG as width sweep; full epochs, no early stopping',
                    selection='save both fixed validation MSE+CE minimum and validation value MSE minimum; use joint minimum for primary selection across configurations; all test evaluation after all training',
                    cosineEtaMin=.00001, torchVersion=torch.__version__, numpyVersion=np.__version__,
                    sourceSha256={p: h.sha(ROOT/p) for p in source_files})
    h.write_json(args.output/'protocol.json', protocol)
    started = time.monotonic()
    progress.update(stage='training')

    def update(**kwargs):
        progress.update(kwargs, seconds=round(time.monotonic()-started, 2))
        h.write_json(args.output/'progress.json', progress)

    boards, turns = torch.from_numpy(data['train_board']), torch.from_numpy(data['train_turn'])
    target = torch.from_numpy(np.sign(data['train_target']))
    move = torch.from_numpy(data['train_move'].astype(np.int64))
    legal = torch.from_numpy(data['train_legal'])
    perms = torch.from_numpy(h.policy.value.symmetries())
    inverse = torch.argsort(perms, 1)
    training_reports = []
    for config in CONFIGS:
        torch.manual_seed(20261027)
        rng = torch.Generator().manual_seed(20261028)
        net = suite.CnnPolicyValue(config['channels'], config['headWidth']).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=config['learningRate'], weight_decay=.001)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=.00001)
                     if config['schedule'] == 'cosine' else None)
        best = {'joint': float('inf'), 'value': float('inf')}
        best_epoch, history = {}, []
        update(currentName=config['name'], currentChannels=config['channels'], epoch=0, validation=None)
        for epoch in range(1, args.epochs+1):
            net.train()
            order = torch.randperm(len(boards), generator=rng)
            lr = opt.param_groups[0]['lr']
            for offset in range(0, len(boards), 1024):
                ids = order[offset:offset+1024]
                sym = torch.randint(8, (len(ids),), generator=rng)
                p = perms[sym]
                b = torch.gather(boards[ids], 1, p).to(device)
                mask = torch.cat((torch.gather(legal[ids, :64], 1, p), legal[ids, 64:]), 1).to(device)
                m = torch.where(move[ids] == 64, 64, inverse[sym, move[ids].clamp(max=63)]).to(device)
                value, logits = net(b, turns[ids].to(device))
                loss = config['valueWeight']*(value-target[ids].to(device)).square().mean()
                loss += torch.nn.functional.cross_entropy(logits.masked_fill(~mask, -1e9), m)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if offset % 32768 == 0:
                    update(epoch=epoch, processedPositions=(epoch-1)*len(boards)+offset+len(ids))
            score = h.metrics(*suite.predict(net, data, 'validation', device), data, 'validation')
            assert np.isfinite(score['selectionLoss'])
            row = dict(epoch=epoch, learningRate=lr, **score)
            history.append(row)
            for criterion, metric in [('joint', score['selectionLoss']), ('value', score['value']['outcomeMse'])]:
                if metric < best[criterion]:
                    best[criterion], best_epoch[criterion] = metric, epoch
                    state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    torch.save(dict(channels=config['channels'], headWidth=config['headWidth'], config=config, epoch=epoch, criterion=criterion,
                                    state_dict=state), args.output/f"{config['name']}-{criterion}.pt")
            if scheduler:
                scheduler.step()
            print(json.dumps(dict(name=config['name'], **row)), flush=True)
            update(epoch=epoch, validation=score, bestEpoch=best_epoch, processedPositions=epoch*len(boards))
        training_reports.append(dict(**config, history=history, bestEpoch=best_epoch, bestValidation=best))
        h.write_json(args.output/'training-checkpoints.json', training_reports)
        update(completedModels=len(training_reports))
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
                assert verr < .00051 and perr < .00001, (name,split,verr,perr)
                candidate.setdefault('nativeParity', {})[split] = dict(positions=len(fp[0]), maxValueError=verr, maxPolicyError=perr)
                candidate['metrics'][split] = dict(**h.metrics(*fp, data, split), maxExportError=error, exportParityPositions=len(fp[0]))
                if split == 'test':
                    np.savez_compressed(args.output/(name+'-test-predictions.npz'), board=data['test_board'], turn=data['test_turn'],
                                        target=np.sign(data['test_target']), recorded_move=data['test_move'], value=fp[0], policy=fp[1])
            evaluated.append(candidate)
            for src, suffix in [(path, '.pt'), (export, '.torchscript.pt'), (native_path, '.otcnn')]:
                dst = ROOT/'models'/('cnn2-'+name+'-v1.20'+suffix)
                assert not dst.exists()
                shutil.copy2(src, dst)
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


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'.build/training/light-cnn-v1.20-run1')
    p.add_argument('--previous-report', type=Path, default=ROOT/'models/v1.13-cnn2-tuning.json')
    p.add_argument('--wait-match', type=Path, default=ROOT/'.build/training/common-batch-match-v1.19/cnn-vs-v1.9-b32-2s-20-run1.json')
    p.add_argument('--library', type=Path, default=ROOT/'.build/native-light-cnn-v1.20/libothello.dylib')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    args = p.parse_args()
    if args.epochs < 1:
        p.error('Positive epoch count required')
    try:
        main(args)
    except Exception as exc:
        progress_path = args.output/'progress.json'
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            progress.update(stage='failed', error=str(exc))
            h.write_json(progress_path, progress)
        raise
