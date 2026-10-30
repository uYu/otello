#!/usr/bin/env python3
"""Replay v1.25 once, retain epoch checkpoints, and evaluate equal-weight tail averages."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from pattern_model import PatternNet, encode, legal_mask, export, parity

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'.build/training/pattern-average-v1.30'
REPORT = ROOT/'models/v1.30-pattern-average.json'
LIB = ROOT/('.build/native-v1.25-reference/libothello.' + ('dylib' if sys.platform == 'darwin' else 'so'))
SPEC = importlib.util.spec_from_file_location('average_helpers', ROOT/'scripts/train-pattern-outcome.py')
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def average_checkpoints(paths):
    """Average trainable parameters in float64; require identical fixed buffers."""
    paths = list(paths)
    if not paths:
        raise ValueError('At least one checkpoint required')
    template = PatternNet()
    parameters = set(dict(template.named_parameters()))
    expected = template.state_dict()
    sums = {}
    first = None
    for path in paths:
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state.keys() != expected.keys():
            raise ValueError('Checkpoint keys differ from Pattern H32')
        for key, value in state.items():
            if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
                raise ValueError('Checkpoint shape/dtype mismatch: '+key)
            if not torch.isfinite(value).all():
                raise ValueError('Non-finite checkpoint tensor: '+key)
            if key in parameters:
                if key not in sums:
                    sums[key] = torch.zeros_like(value, dtype=torch.float64)
                sums[key].add_(value.to(torch.float64))
            elif not torch.equal(value, expected[key]):
                raise ValueError('Fixed Pattern buffer differs: '+key)
        if first is None:
            first = state
    return {key: (sums[key]/len(paths)).to(value.dtype) if key in parameters else value.clone()
            for key, value in first.items()}


def main():
    if WORK.exists() or REPORT.exists():
        raise ValueError('Choose a fresh run; never overwrite checkpoints or reports')
    active_path = ROOT/'models/default-model.json'
    active = json.loads(active_path.read_text())
    manifest = ROOT/active['baselineManifest']
    baseline = json.loads(manifest.read_text())
    assert active['version'] == 'v1.25' and baseline['status'] == 'frozen' and baseline['userApproved']
    baseline_model = ROOT/active['defaultModel']
    assert sha(baseline_model) == active['modelSha256'] == baseline['modelSha256']
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in [active_path, manifest, baseline_model, baseline_model.with_suffix('.pt')]}
    source_paths = [Path(__file__).resolve(), ROOT/'scripts/pattern_model.py', ROOT/'scripts/train-pattern-outcome.py', ROOT/'scripts/train-pattern.py']
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    cache = ROOT/'.build/training/v1/positions.npz'
    metadata = json.loads(cache.with_suffix('.json').read_text())
    assert sha(cache) == metadata['cacheSha256'] == json.loads((ROOT/'models/nnue256-v1.json').read_text())['dataset']['cacheSha256']
    policy_dir = ROOT/'.build/training/pattern-outcome-v1.25'
    provenance = json.loads((policy_dir/'policy.json').read_text())
    assert sha(policy_dir/'policy.npz') == provenance['policySha256'] and provenance['cacheSha256'] == metadata['cacheSha256']
    assert sha(ROOT/'.build/training/games.jsonl') == provenance['gamesSha256']
    data = dict(np.load(cache)); labels = dict(np.load(policy_dir/'policy.npz'))
    masks = {s: legal_mask(data[s+'_board'], data[s+'_turn']) for s in labels}
    assert all(masks[s][np.arange(len(labels[s])), labels[s]].all() for s in labels)
    config = json.loads(baseline_model.with_suffix('.json').read_text())
    assert config['epochs'] == 15 and config['trainPositions'] == len(data['train_board'])
    windows = [3, 5, 10]
    WORK.mkdir(parents=True); (WORK/'checkpoints').mkdir()
    started = time.perf_counter(); variants = []; history = []; checkpoints = []

    def progress(stage, epoch=0, validation=None, complete=False):
        write(WORK/'progress.json', dict(experiment='checkpoint-average-training', stage=stage,
              complete=complete, variant='单次训练 → 最后3/5/10轮等权平均', epoch=epoch, epochs=config['epochs'],
              seconds=time.perf_counter()-started, validation=validation, variants=variants))

    write(WORK/'protocol.json', dict(version='v1.30', windows=windows, epochs=config['epochs'],
          selection='Minimum validation outcome MSE among the predeclared averages; test evaluated only afterward',
          initialization='Random, exact v1.25 seed and training recipe; no mixing separate runs or variants',
          frozenArtifacts=frozen, sourceSha256=source_hashes))
    progress('读取并验证数据完成；准备重跑')
    monitor_path = ROOT/'models/current-experiment.json'
    monitor = json.loads(monitor_path.read_text())
    monitor['experiments'].insert(0, dict(name='Pattern 单次训练 checkpoint 平均 · v1.30', focus='checkpoint average', games=0,
          runtimeReport=str((WORK/'progress.json').relative_to(ROOT)), report=str(REPORT.relative_to(ROOT))))
    write(monitor_path, monitor)
    torch.set_num_threads(config['threads']); torch.manual_seed(config['seed']); np.random.seed(config['seed'])
    net = PatternNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=config['learningRate'], weight_decay=.0001)
    board, turn, target = [data['train_'+k] for k in ('board', 'turn', 'target')]
    phase = (64-np.count_nonzero(board, axis=1)).astype(np.float32)/60
    grid = np.arange(64).reshape(8, 8)
    permutations = np.stack([np.rot90(np.fliplr(grid) if mirror else grid, rotation).reshape(64)
                             for mirror in (False, True) for rotation in range(4)])
    inverse = np.argsort(permutations, axis=1)
    for epoch in range(1, config['epochs']+1):
        net.train(); losses = []
        for rows in torch.randperm(len(board)).split(config['batchSize']):
            indices = rows.numpy(); sym = np.random.randint(8, size=len(indices)); perm = permutations[sym]
            batch = np.take_along_axis(board[indices], perm, axis=1)
            codes = torch.from_numpy(encode(batch, turn[indices]).astype(np.int64))
            value, wdl, policy = net(codes, torch.from_numpy(phase[indices]))
            truth = torch.from_numpy(np.sign(target[indices]))
            next_move = torch.from_numpy(inverse[sym, labels['train'][indices]])
            mask = torch.from_numpy(np.take_along_axis(masks['train'][indices], perm, axis=1))
            outcome = torch.where(truth > 0, 0, torch.where(truth < 0, 1, 2)).long()
            loss = F.mse_loss(value, truth)+.1*F.cross_entropy(wdl, outcome)+.1*F.cross_entropy(policy.masked_fill(~mask, -1e9), next_move)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5); optimizer.step()
            losses.append(float(loss.detach()))
        path = WORK/'checkpoints'/f'epoch-{epoch:02d}.pt'
        torch.save(net.state_dict(), path)
        checkpoints.append(dict(epoch=epoch, path=str(path.relative_to(ROOT)), sha256=sha(path)))
        pred = helper.predict(net, data['validation_board'], data['validation_turn'])
        score = helper.metric(pred, data['validation_target'], labels['validation'], masks['validation'])
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), **score))
        write(WORK/'history.json', dict(history=history, checkpoints=checkpoints))
        progress('训练中；每轮保存真实权重', epoch, score)
        print(json.dumps(dict(seconds=time.perf_counter()-started, **history[-1])), flush=True)
    best_epoch = min(history, key=lambda r: r['mse'])['epoch']
    all_paths = [ROOT/c['path'] for c in checkpoints]
    replay_path = WORK/'replayed-best.otpat'
    net.load_state_dict(torch.load(all_paths[best_epoch-1], map_location='cpu', weights_only=True))
    export(net, replay_path, outcome=True)
    replay_equal = sha(replay_path) == frozen[str(baseline_model.relative_to(ROOT))]
    print(json.dumps(dict(replayedBestEpoch=best_epoch, exactBaselineBinaryReplay=replay_equal)), flush=True)
    recipe = [('single-best', [best_epoch]), *[(f'avg-last-{n}', list(range(config['epochs']-n+1, config['epochs']+1))) for n in windows]]
    for name, epochs in recipe:
        state = average_checkpoints([all_paths[e-1] for e in epochs])
        net.load_state_dict(state)
        path = ROOT/f'models/pattern32-{name}-v1.30.otpat'
        if path.exists() or path.with_suffix('.pt').exists():
            raise ValueError('Refusing to overwrite '+str(path))
        binary = export(net, path, outcome=True); torch.save(state, path.with_suffix('.pt'))
        pred = helper.predict(net, data['validation_board'], data['validation_turn'])
        score = helper.metric(pred, data['validation_target'], labels['validation'], masks['validation'])
        checks = parity(LIB, path, data['validation_board'], data['validation_turn'], pred)
        checks['maxOutcomeError'] = checks.pop('maxDiscError')
        variants.append(dict(name=name, epochs=epochs, model=str(path.relative_to(ROOT)), sha256=sha(path),
              checkpointSha256=[checkpoints[e-1]['sha256'] for e in epochs], bytes=len(binary),
              parameters=sum(p.numel() for p in net.parameters()), metrics=dict(validation=score), validationParity=checks))
        progress('平均并评估验证集', config['epochs'], score)
    selected = min(variants[1:], key=lambda v: v['metrics']['validation']['mse'])['name']
    # Freeze the choice before reading test predictions; evaluate all predeclared variants for transparency.
    write(WORK/'selection.json', dict(selectedAverage=selected, criterion='validation outcome MSE', variants=variants))
    progress('已按验证集选定平均窗口；统一测试与 C++ 检查', config['epochs'])
    for variant in variants:
        path = ROOT/variant['model']; net.load_state_dict(torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True))
        pred = helper.predict(net, data['test_board'], data['test_turn'])
        variant['metrics']['test'] = helper.metric(pred, data['test_target'], labels['test'], masks['test'])
        checks = parity(LIB, path, data['test_board'], data['test_turn'], pred)
        checks['maxOutcomeError'] = checks.pop('maxDiscError'); variant['testParity'] = checks
        write(path.with_suffix('.json'), dict(version='v1.30', status='candidate', defaultPromoted=False, **variant))
    baseline_metrics = {}
    net.load_state_dict(torch.load(baseline_model.with_suffix('.pt'), map_location='cpu', weights_only=True))
    for split in ('validation', 'test'):
        baseline_metrics[split] = helper.metric(helper.predict(net, data[split+'_board'], data[split+'_turn']), data[split+'_target'], labels[split], masks[split])
    assert all(sha(ROOT/p) == s for p, s in frozen.items()), 'Frozen baseline changed during experiment'
    assert all(sha(ROOT/p) == s for p, s in source_hashes.items()), 'Training sources changed during experiment'
    report = dict(version='v1.30-pattern-average', status='candidate', defaultPromoted=False, baselineVersion='v1.25',
          baselineManifest=active['baselineManifest'], frozenArtifacts=frozen, baselineMetrics=baseline_metrics,
          exactBaselineBinaryReplay=replay_equal, bestSingleEpoch=best_epoch, selectedAverage=selected,
          selection='Predeclared last 3/5/10 epochs; minimum validation outcome MSE; no test-based selection',
          averaging='Equal-weight float64 mean of float checkpoint trainable parameters, cast to float32, then standard OTPAT2 quantization; fixed buffers verified identical',
          initialization='random, v1.25 recipe', seed=config['seed'], epochs=config['epochs'], trainPositions=len(board),
          batchSize=config['batchSize'], threads=config['threads'], learningRate=config['learningRate'], weightDecay=.0001,
          datasetSha256=metadata['cacheSha256'], policyDataset=provenance, sourceSha256=source_hashes,
          librarySha256=sha(LIB), history=history, checkpoints=checkpoints, variants=variants,
          seconds=time.perf_counter()-started, limitations='Offline only; no strength match or default promotion')
    write(REPORT, report)
    lines = ['# 同一次训练的 checkpoint 等权平均（v1.30）', '',
             '按 v1.25 原配置随机初始化训练15轮，每轮保留浮点权重；固定比较最后3/5/10轮等权平均。',
             '仅平均可训练参数，固定棋盘 buffer 校验一致；平均后统一量化导出 OTPAT2。模型大小与单次推理结构不变。',
             f'最佳单轮为第{best_epoch}轮；其导出是否逐字节复现冻结基线：{replay_equal}。',
             f'仅按验证胜负 MSE 选中的平均模型：{selected}。测试集在选定后统一评估，不用于选择窗口。', '',
             '| 模型 | 验证胜负 MSE ↓ | 测试胜负 MSE ↓ | 测试非和局准确率 ↑ | 测试走法命中率 ↑ |',
             '|---|---:|---:|---:|---:|']
    for name, metrics in [('冻结 v1.25', baseline_metrics), *[(v['name'], v['metrics']) for v in variants]]:
        t = metrics['test']; lines.append(f"| {name} | {metrics['validation']['mse']:.6f} | {t['mse']:.6f} | {t['nonDrawSignAccuracy']:.2%} | {t['policyTop1']:.2%} |")
    lines += ['', '每个候选在验证/测试各6000局面均检查 Python/C++ 导出一致性。', '这是离线实验，未测对战棋力；默认与冻结基线保持不变。', '', '复现：`.build/venv/bin/python scripts/train-pattern-average.py`（已存在结果时拒绝覆盖）。', '监控：`sh scripts/watch-match.sh`。']
    (ROOT/'models/v1.30-pattern-average-comparison.md').write_text('\n'.join(lines)+'\n')
    progress('已完成；仅离线评估', config['epochs'], complete=True)
    print(json.dumps(dict(complete=True, selectedAverage=selected, exactReplay=replay_equal, variants=variants)), flush=True)


if __name__ == '__main__':
    main()
