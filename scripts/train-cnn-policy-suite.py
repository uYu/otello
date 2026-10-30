#!/usr/bin/env python3
"""Offline two-convolution policy/outcome candidates on the original v1 cache."""
import argparse
import importlib.util
import json
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mixed_helpers', ROOT/'scripts/train-mixed-5m.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


class CnnPolicyValue(nn.Module):
    def __init__(self, channels, head_width=64):
        super().__init__()
        self.conv1 = nn.Conv2d(2, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.fc = nn.Linear(channels*64, head_width)
        self.value = nn.Linear(head_width, 1)
        self.policy = nn.Linear(head_width, 65)

    def forward(self, board, turn):
        relative = board*turn[:, None]
        x = torch.stack((relative == 1, relative == -1), 1).float().reshape(-1, 2, 8, 8)
        x = self.conv2(self.conv1(x).relu()).relu()
        x = self.fc(x.flatten(1)).relu()
        return self.value(x).squeeze(1).tanh(), self.policy(x)


def predict(net, data, split, device):
    net.eval()
    values, probs = [], []
    with torch.no_grad():
        for start in range(0, len(data[split+'_board']), 1024):
            part = slice(start, start+1024)
            board = torch.from_numpy(data[split+'_board'][part]).to(device)
            turn = torch.from_numpy(data[split+'_turn'][part]).to(device)
            mask = torch.from_numpy(data[split+'_legal'][part]).to(device)
            v, logits = net(board, turn)
            values.append(v.cpu().numpy())
            probs.append(logits.masked_fill(~mask, -torch.inf).softmax(1).cpu().numpy())
    return np.concatenate(values), np.concatenate(probs)


def train(args):
    if (args.output/'protocol.json').exists():
        raise ValueError('Choose a fresh suite output directory')
    if args.publish and (ROOT/'models/v1.12-cnn2-offline.json').exists():
        raise ValueError('Published suite already exists; use a new experiment version')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('Requested MPS unavailable; refusing silent CPU fallback')
    cache = ROOT/'.build/training/v1/positions.npz'
    labels_path = ROOT/'.build/training/policy/labels.npz'
    meta = json.loads(cache.with_suffix('.json').read_text())
    label_meta = json.loads(labels_path.with_suffix('.json').read_text())
    original = json.loads((ROOT/'models/nnue256-outcome-v1.9.json').read_text())
    assert helpers.sha(cache) == meta['cacheSha256'] == label_meta['positionsSha256'] == original['dataset']['cacheSha256']
    assert helpers.sha(labels_path) == label_meta['cacheSha256']
    data, labels = dict(np.load(cache)), dict(np.load(labels_path))
    for split in ('train', 'validation', 'test'):
        for key in ('move', 'legal'):
            data[split+'_'+key] = labels[split+'_'+key]
    assert [len(data[s+'_board']) for s in ('train', 'validation', 'test')] == [1000000, 6000, 6000]
    for split in ('test', 'validation'):
        with np.load(ROOT/'.build/training/egaroucid-v0002/full-5m-v1/evaluation/frozen.npz') as frozen:
            for key in ('board', 'turn', 'target', 'move', 'legal'):
                assert np.array_equal(frozen[split+'_'+key], data[split+'_'+key])
    active_hash = helpers.sha(ROOT/'models/default-model.json')
    baseline_path = ROOT/'models/nnue256-outcome-v1.9.otnnue'
    baseline_manifest = json.loads((ROOT/'models/v1.9-default.json').read_text())
    assert helpers.sha(baseline_path) == baseline_manifest['modelSha256']
    started = time.monotonic()
    protocol = dict(version='v1.12-cnn2-offline', status='candidate', defaultPromoted=False,
                    channels=args.channels, headWidth=64, maxEpochs=args.epochs, patience=args.patience,
                    minEpochs=args.min_epochs, batchSize=args.batch_size, learningRate=.001, weightDecay=.001,
                    optimizer='AdamW; constant learning rate', loss='outcome MSE + legal-mask recorded-move cross entropy, equal coefficients',
                    target='side-to-move recorded outcome +1/0/-1; recorded move, including pass token 64 where applicable',
                    seed=args.seed, threads=args.threads, device=str(device),
                    architecture='own/opponent planes -> Conv3x3 C/ReLU -> Conv3x3 C/ReLU -> Flatten -> Linear64/ReLU -> value Linear1/tanh and policy Linear65; no BN, pooling, residual or frozen weights',
                    dataset=meta, labels=label_meta, baselineModelSha256=helpers.sha(baseline_path),
                    baselineLibrarySha256=helpers.sha(args.library),
                    selection='each model and overall winner by frozen validation MSE+CE; test evaluated only after all training is finished; no test tuning',
                    trainingScriptSha256=helpers.sha(__file__),
                    helperSha256={p: helpers.sha(ROOT/p) for p in ['scripts/train-mixed-5m.py', 'scripts/train-deep.py', 'scripts/train-outcome.py', 'scripts/train-policy.py', 'scripts/train-nnue.py']},
                    torchVersion=torch.__version__, numpyVersion=np.__version__,
                    export='PyTorch state_dict and CPU TorchScript; offline only, no native search loader')
    helpers.write_json(args.output/'protocol.json', protocol)
    progress = dict(experiment='cnn-offline-training', complete=False, channels=args.channels,
                    completedModels=0, totalModels=len(args.channels), epoch=0, maxEpochs=args.epochs)

    def update(**kwargs):
        progress.update(kwargs, seconds=round(time.monotonic()-started, 2))
        helpers.write_json(args.output/'progress.json', progress)

    permutations = torch.from_numpy(helpers.policy.value.symmetries())
    inverse = torch.argsort(permutations, 1)
    boards = torch.from_numpy(data['train_board'])
    turns = torch.from_numpy(data['train_turn'])
    targets = torch.from_numpy(np.sign(data['train_target']))
    moves = torch.from_numpy(data['train_move'].astype(np.int64))
    legal = torch.from_numpy(data['train_legal'])
    reports = []
    for c in args.channels:
        # Same initialization seed and separate, identical data/augmentation RNG for all widths.
        torch.manual_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed+1)
        net = CnnPolicyValue(c).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.001)
        best, state, best_epoch, history = float('inf'), None, 0, []
        model_started = time.monotonic()
        update(stage='training', currentChannels=c, epoch=0, processedPositions=0)
        for epoch in range(1, args.epochs+1):
            net.train()
            order = torch.randperm(len(boards), generator=generator)
            for offset in range(0, len(boards), args.batch_size):
                ids = order[offset:offset+args.batch_size]
                symmetry = torch.randint(8, (len(ids),), generator=generator)
                p = permutations[symmetry]
                batch = torch.gather(boards[ids], 1, p).to(device)
                mask = torch.cat((torch.gather(legal[ids, :64], 1, p), legal[ids, 64:]), 1).to(device)
                augmented_move = torch.where(moves[ids] == 64, 64, inverse[symmetry, moves[ids].clamp(max=63)]).to(device)
                v, logits = net(batch, turns[ids].to(device))
                loss = ((v-targets[ids].to(device))**2).mean()+nn.functional.cross_entropy(logits.masked_fill(~mask, -1e9), augmented_move)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if offset % (args.batch_size*32) == 0:
                    update(epoch=epoch, processedPositions=(epoch-1)*len(boards)+offset+len(ids))
            score = helpers.metrics(*predict(net, data, 'validation', device), data, 'validation')
            row = dict(epoch=epoch, **score, seconds=time.monotonic()-model_started)
            history.append(row)
            if not np.isfinite(score['selectionLoss']):
                raise ValueError('Nonfinite validation loss')
            if score['selectionLoss'] < best:
                best, best_epoch = score['selectionLoss'], epoch
                state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                torch.save(dict(channels=c, headWidth=64, state_dict=state), args.output/f'cnn2-c{c}-best.pt')
            print(json.dumps(dict(channels=c, **row)), flush=True)
            update(epoch=epoch, processedPositions=epoch*len(boards), validation=score, bestEpoch=best_epoch)
            if epoch >= args.min_epochs and epoch-best_epoch >= args.patience:
                break
        reports.append(dict(channels=c, parameters=sum(p.numel() for p in net.parameters()),
                            epochs=len(history), bestEpoch=best_epoch, history=history,
                            validationSelectionLoss=best, trainingSeconds=time.monotonic()-model_started))
        update(completedModels=len(reports))
    winner = min(reports, key=lambda r: r['validationSelectionLoss'])['channels']
    update(stage='frozen-evaluation', winnerChannels=winner)
    baseline = {}
    baseline_predictions = {}
    for split in ('validation', 'test'):
        baseline_predictions[split] = helpers.native_predictions(baseline_path, data, split, args.library)
        baseline[split] = helpers.metrics(*baseline_predictions[split], data, split)
    for report in reports:
        c = report['channels']
        checkpoint = args.output/f'cnn2-c{c}-best.pt'
        bundle = torch.load(checkpoint, map_location='cpu', weights_only=True)
        net = CnnPolicyValue(c).eval()
        net.load_state_dict(bundle['state_dict'])
        traced = torch.jit.trace(net, (torch.from_numpy(data['validation_board'][:2]), torch.from_numpy(data['validation_turn'][:2])))
        export_path = args.output/f'cnn2-c{c}.torchscript.pt'
        torch.jit.save(traced, export_path)
        reloaded = torch.jit.load(export_path).eval()
        report.update(status='candidate', modelSha256=helpers.sha(checkpoint), torchscriptSha256=helpers.sha(export_path), metrics={})
        for split in ('validation', 'test'):
            fp = predict(net, data, split, torch.device('cpu'))
            loaded = predict(reloaded, data, split, torch.device('cpu'))
            value_error = float(abs(fp[0]-loaded[0]).max())
            policy_error = float(abs(fp[1]-loaded[1]).max())
            assert value_error < 1e-6 and policy_error < 1e-6
            report['metrics'][split] = dict(**helpers.metrics(*fp, data, split),
                                                  exportParityPositions=len(fp[0]), maxExportValueError=value_error, maxExportPolicyError=policy_error)
            if split == 'test':
                np.savez_compressed(args.output/f'cnn2-c{c}-test-predictions.npz',
                                    board=data['test_board'], turn=data['test_turn'], target=np.sign(data['test_target']),
                                    recorded_move=data['test_move'], value=fp[0], policy=fp[1],
                                    baseline_value=baseline_predictions[split][0], baseline_policy=baseline_predictions[split][1])
        helpers.write_json(args.output/f'cnn2-c{c}-report.json', report)
    report = dict(**protocol, experiment='cnn-offline-training', complete=True,
                  completedModels=len(reports), totalModels=len(reports),
                  epoch=reports[-1]['epochs'], currentChannels=args.channels[-1],
                  baselineMetrics=baseline, candidates=reports,
                  winnerChannels=winner, seconds=time.monotonic()-started)
    assert helpers.sha(ROOT/'models/default-model.json') == active_hash
    helpers.write_json(args.output/'report.json', report)
    update(stage='complete', complete=True, candidates=reports, baselineMetrics=baseline)
    if args.publish:
        artifacts = []
        for c in args.channels:
            for src, dst in [(f'cnn2-c{c}-best.pt', f'cnn2-c{c}-v1.12.pt'),
                             (f'cnn2-c{c}.torchscript.pt', f'cnn2-c{c}-v1.12.torchscript.pt'),
                             (f'cnn2-c{c}-report.json', f'cnn2-c{c}-v1.12.json')]:
                target = ROOT/'models'/dst
                if target.exists():
                    raise ValueError('Refusing to overwrite '+str(target))
                shutil.copy2(args.output/src, target)
                artifacts.append(str(target.relative_to(ROOT)))
        helpers.write_json(ROOT/'models/v1.12-cnn2-offline.json', report)
        rows = []
        for name, score in [('v1.9', baseline['test'])]+[(f"CNN {r['channels']} 通道", r['metrics']['test']) for r in reports]:
            rows.append(f"| {name} | {score['value']['outcomeMse']:.6f} | {score['value']['nonDrawSignAccuracy']:.2%} | {score['policy']['top1']:.2%} | {score['policy']['crossEntropy']:.6f} |")
        doc = ROOT/'models/v1.12-cnn2-offline-comparison.md'
        doc.write_text('# 两层 CNN：原始训练集离线实验\n\n'
                       '使用原有 1,000,000 个训练局面、6,000 个验证局面、6,000 个测试局面。未加入 500 万局新增数据。\n'
                       '两层 3×3 卷积，分别为 8、16、32 通道；Flatten 后共享 Linear64，再接胜负价值与 65 维走法头。全部参数从头训练。\n'
                       'AdamW，学习率 0.001，weight decay 0.001，batch 1024，随机 D4；各版相同数据排列和增强随机数序列。\n'
                       '最多 30 轮，至少 10 轮；连续 5 轮验证总损失未改善则早停。按冻结验证胜负 MSE＋走法交叉熵选择权重和整体候选。\n'
                       f'验证集选择 {winner} 通道；测试集只在所有训练完成后统一评估，没有根据测试结果调参。\n\n'
                       '| 模型 | 胜负 MSE ↓ | 非和局准确率 ↑ | 记录走法命中率 ↑ | 走法交叉熵 ↓ |\n'
                       '|---|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n\n'
                       'MSE 包含全部 6,000 个测试局面；非和局准确率统计其中 5,192 个。基线为 v1.9 原生量化推理；CNN 为 CPU FP32 推理。\n'
                       '三份检查点和 CPU TorchScript 均保留，各在验证与测试全集验证导出一致性。\n'
                       f'训练日志、逐轮进度和完整逐条测试预测保存在 `{args.output.relative_to(ROOT)}`。\n'
                       '本次仅做离线训练，未实现这些新 CNN 的 C++／Wasm 推理和搜索接线，未进行对战，默认仍为 v1.9。\n')
        artifacts += ['models/v1.12-cnn2-offline.json', str(doc.relative_to(ROOT))]
        source_files = ['scripts/train-cnn-policy-suite.py', 'scripts/watch-matches.py', 'scripts/watch-match.sh',
                        'tests/cnn_policy_training_test.py']+list(protocol['helperSha256'])
        archive = ROOT/'models/archive/v1.12-cnn2-offline-sources.tar.gz'
        if archive.exists():
            raise ValueError('Source archive already exists')
        with tarfile.open(archive, 'w:gz') as tar:
            for path in source_files:
                tar.add(ROOT/path, arcname=path)
        helpers.write_json(ROOT/'models/archive/v1.12-cnn2-offline-sources.json',
                           dict(archive=str(archive.relative_to(ROOT)), sha256=helpers.sha(archive),
                                files={p: helpers.sha(ROOT/p) for p in source_files},
                                artifactSha256={p: helpers.sha(ROOT/p) for p in artifacts}))
    print(json.dumps(dict(complete=True, winnerChannels=winner, seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'.build/training/cnn2-v1.12-run1')
    p.add_argument('--channels', type=int, nargs='+', default=[8, 16, 32])
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--min-epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261027)
    p.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    p.add_argument('--publish', action='store_true', help='Save unique v1.12 candidate artifacts and reports under models/')
    p.add_argument('--library', type=Path, default=ROOT/'.build/native-engineering/libothello.dylib')
    args = p.parse_args()
    if min(args.channels+[args.epochs, args.min_epochs, args.patience, args.batch_size, args.threads]) < 1 or len(args.channels) != len(set(args.channels)):
        p.error('Positive, unique model channels and positive training counts required')
    train(args)
