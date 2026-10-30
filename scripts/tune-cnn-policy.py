#!/usr/bin/env python3
"""Controlled four-configuration tuning of the original-data two-layer CNN."""
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

CONFIGS = [
    dict(name='low-lr', learningRate=.0003, schedule='constant', valueWeight=1.),
    dict(name='cosine', learningRate=.001, schedule='cosine', valueWeight=1.),
    dict(name='cosine-v2', learningRate=.001, schedule='cosine', valueWeight=2.),
    dict(name='cosine-v4', learningRate=.001, schedule='cosine', valueWeight=4.),
]


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'protocol.json').exists() or (ROOT/'models/v1.13-cnn2-tuning.json').exists():
        raise ValueError('Use a fresh experiment; refusing overwrite')
    # Serialize GPU work with the preceding width sweep, without changing it.
    while not args.previous_report.exists():
        time.sleep(2)
    previous = json.loads(args.previous_report.read_text())
    assert previous['complete'] and {r['channels'] for r in previous['candidates']} == {8, 16, 32}
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
    original32 = next(r for r in previous['candidates'] if r['channels'] == 32)
    source_files = ['scripts/tune-cnn-policy.py', 'scripts/train-cnn-policy-suite.py',
                    'scripts/train-mixed-5m.py', 'scripts/train-deep.py', 'scripts/train-outcome.py',
                    'scripts/train-policy.py', 'scripts/train-nnue.py']
    protocol = dict(version='v1.13-cnn2-tuning', status='candidate', defaultPromoted=False,
                    channels=32, headWidth=64, configs=CONFIGS, epochs=args.epochs, batchSize=1024,
                    weightDecay=.001, seed=20261027, device=args.device,
                    targets='original recorded side-to-move outcome and recorded move; no added games or book labels',
                    dataset=meta, labels=label_meta, originalC32Validation=original32['metrics']['validation'],
                    previousReportSha256=h.sha(args.previous_report),
                    training='from scratch per configuration; same seed and separate identical sample/augmentation RNG as width sweep; full epochs, no early stopping',
                    selection='save both fixed validation MSE+CE minimum and validation value MSE minimum; use joint minimum for primary selection across configurations; all test evaluation after all training',
                    cosineEtaMin=.00001, torchVersion=torch.__version__, numpyVersion=np.__version__,
                    sourceSha256={p: h.sha(ROOT/p) for p in source_files})
    h.write_json(args.output/'protocol.json', protocol)
    # Switch the single monitor only after the preceding sweep has finished.
    old_config = ROOT/'models/current-experiment.json'
    (args.output/'previous-monitor-config.json').write_text(old_config.read_text())
    h.write_json(old_config, dict(version='两层 CNN · 超参数对照', experiments=[dict(
        name='32 通道 CNN · 学习率／损失权重', focus='CNN tuning', games=0,
        runtimeReport=str((args.output/'progress.json').relative_to(ROOT)), report='models/v1.13-cnn2-tuning.json')]))
    started = time.monotonic()
    progress = dict(experiment='cnn-offline-training', complete=False, stage='training',
                    channels=[32]*len(CONFIGS), currentChannels=32, epoch=0, maxEpochs=args.epochs,
                    completedModels=0, totalModels=len(CONFIGS))

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
        net = suite.CnnPolicyValue(32).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=config['learningRate'], weight_decay=.001)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=.00001)
                     if config['schedule'] == 'cosine' else None)
        best = {'joint': float('inf'), 'value': float('inf')}
        best_epoch, history = {}, []
        update(currentName=config['name'], epoch=0)
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
                    torch.save(dict(channels=32, headWidth=64, config=config, epoch=epoch, criterion=criterion,
                                    state_dict=state), args.output/f"{config['name']}-{criterion}.pt")
            if scheduler:
                scheduler.step()
            print(json.dumps(dict(name=config['name'], **row)), flush=True)
            update(epoch=epoch, validation=score, bestEpoch=best_epoch, processedPositions=epoch*len(boards))
        training_reports.append(dict(**config, history=history, bestEpoch=best_epoch, bestValidation=best))
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
            net = suite.CnnPolicyValue(32).eval()
            net.load_state_dict(bundle['state_dict'])
            traced = torch.jit.trace(net, (torch.from_numpy(data['validation_board'][:2]), torch.from_numpy(data['validation_turn'][:2])))
            export = args.output/(name+'.torchscript.pt')
            traced.save(str(export))
            reloaded = torch.jit.load(str(export)).eval()
            candidate = dict(name=name, channels=32, criterion=criterion, config=report,
                             epoch=bundle['epoch'], modelSha256=h.sha(path), torchscriptSha256=h.sha(export), metrics={})
            for split in ('validation', 'test'):
                fp = suite.predict(net, data, split, torch.device('cpu'))
                loaded = suite.predict(reloaded, data, split, torch.device('cpu'))
                error = max(float(abs(a-b).max()) for a, b in zip(fp, loaded))
                assert error < 1e-6
                candidate['metrics'][split] = dict(**h.metrics(*fp, data, split), maxExportError=error, exportParityPositions=len(fp[0]))
                if split == 'test':
                    np.savez_compressed(args.output/(name+'-test-predictions.npz'), board=data['test_board'], turn=data['test_turn'],
                                        target=np.sign(data['test_target']), recorded_move=data['test_move'], value=fp[0], policy=fp[1])
            evaluated.append(candidate)
            for src, suffix in [(path, '.pt'), (export, '.torchscript.pt')]:
                dst = ROOT/'models'/('cnn2-c32-'+name+'-v1.13'+suffix)
                assert not dst.exists()
                shutil.copy2(src, dst)
    report = dict(**protocol, experiment='cnn-offline-training', complete=True,
                  baselineMetrics=previous['baselineMetrics'], originalC32=original32,
                  trainingReports=training_reports, candidates=evaluated, winnerName=winner, valueWinnerName=value_winner,
                  completedModels=len(CONFIGS), totalModels=len(CONFIGS), currentChannels=32, epoch=args.epochs,
                  maxEpochs=args.epochs, seconds=time.monotonic()-started)
    h.write_json(args.output/'report.json', report)
    h.write_json(ROOT/'models/v1.13-cnn2-tuning.json', report)
    update(stage='complete', complete=True, candidates=evaluated, baselineMetrics=previous['baselineMetrics'])
    rows = []
    for name, score in [('v1.9', previous['baselineMetrics']['test']), ('原 32 通道', original32['metrics']['test'])]+[(r['name'], r['metrics']['test']) for r in evaluated]:
        rows.append(f"| {name} | {score['value']['outcomeMse']:.6f} | {score['value']['nonDrawSignAccuracy']:.2%} | {score['policy']['top1']:.2%} | {score['policy']['crossEntropy']:.6f} |")
    doc = ROOT/'models/v1.13-cnn2-tuning-comparison.md'
    doc.write_text('# 两层 CNN 超参数对照\n\n固定原百万训练局面、32 通道两层 CNN、batch 1024、AdamW weight decay 0.001、相同种子及 D4 增强序列；各从头训练 30 轮。\n'
                   '四组：学习率 0.0003 恒定；0.001 余弦衰减至 0.00001；在衰减组上将价值损失权重设为 2 或 4。策略损失权重均为 1。\n'
                   '每组保留验证 MSE+CE 最低（joint）和胜负 MSE 最低（value）两套检查点；选择依据不使用测试集。\n'
                   f'主候选由验证联合损失选为 {winner}，纯价值标准选为 {value_winner}。测试集在四组全部训练结束后统一评估。\n\n'
                   '| 模型 | 胜负 MSE ↓ | 非和局准确率 ↑ | 走法命中率 ↑ | 走法交叉熵 ↓ |\n|---|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n\n'
                   '测试集沿用原 6,000 局面（其中 5,192 非和局）。CNN 为 CPU FP32 推理；v1.9 为原生量化推理。新权重和 TorchScript 均保存，逐条预测位于本次 `.build/training/cnn2-v1.13-tuning-run1/`。\n'
                   '尚未接入原生搜索；未对战，默认保持 v1.9。开局库蒸馏标签未加入本轮。\n')
    archive = ROOT/'models/archive/v1.13-cnn2-tuning-sources.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as tar:
        for p in source_files:
            tar.add(ROOT/p, arcname=p)
    artifacts = [ROOT/'models/v1.13-cnn2-tuning.json', doc]+list((ROOT/'models').glob('cnn2-c32-*-v1.13*.pt'))
    assert all(h.sha(ROOT/p) == digest for p, digest in protocol['sourceSha256'].items())
    h.write_json(ROOT/'models/archive/v1.13-cnn2-tuning-sources.json', dict(
        archive=str(archive.relative_to(ROOT)), sha256=h.sha(archive), files=protocol['sourceSha256'],
        artifactSha256={str(p.relative_to(ROOT)): h.sha(p) for p in artifacts}))
    print(json.dumps(dict(complete=True, winnerName=winner, valueWinnerName=value_winner, seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'.build/training/cnn2-v1.13-tuning-run1')
    p.add_argument('--previous-report', type=Path, default=ROOT/'.build/training/cnn2-v1.12-run1/report.json')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    args = p.parse_args()
    if args.epochs < 1:
        p.error('Positive epoch count required')
    main(args)
