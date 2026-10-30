#!/usr/bin/env python3
"""Recover completed v1.12 outputs after its report-only duplicate-key error."""
import importlib.util
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('suite', ROOT/'scripts/train-cnn-policy-suite.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
h = suite.helpers


def main():
    torch.set_num_threads(2)
    run = ROOT/'.build/training/cnn2-v1.12-run1'
    protocol = json.loads((run/'protocol.json').read_text())
    original_source = run/'original-source/train-cnn-policy-suite.py'
    assert h.sha(original_source) == protocol['trainingScriptSha256']
    assert not (run/'report.json').exists()
    reports = [json.loads((run/f'cnn2-c{c}-report.json').read_text()) for c in protocol['channels']]
    frozen = dict(np.load(ROOT/'.build/training/egaroucid-v0002/full-5m-v1/evaluation/frozen.npz'))
    model = ROOT/'models/nnue256-outcome-v1.9.otnnue'
    library = ROOT/'.build/native-engineering/libothello.dylib'
    assert h.sha(model) == protocol['baselineModelSha256'] and h.sha(library) == protocol['baselineLibrarySha256']
    baseline = {split: h.metrics(*h.native_predictions(model, frozen, split, library), frozen, split)
                for split in ('validation', 'test')}
    for report in reports:
        c = report['channels']
        assert h.sha(run/f'cnn2-c{c}-best.pt') == report['modelSha256']
        assert h.sha(run/f'cnn2-c{c}.torchscript.pt') == report['torchscriptSha256']
    progress = json.loads((run/'progress.json').read_text())
    report = dict(protocol)
    report.update(experiment='cnn-offline-training', complete=True, completedModels=3, totalModels=3,
                  epoch=reports[-1]['epochs'], currentChannels=32, baselineMetrics=baseline, candidates=reports,
                  winnerChannels=min(reports, key=lambda r: r['validationSelectionLoss'])['channels'], seconds=progress['seconds'],
                  recovery=dict(reason='Training/evaluation finished; report assembly raised duplicate maxEpochs keyword',
                                trainingRepeated=False, originalTrainingSourceSha256=h.sha(original_source),
                                finalizerSha256=h.sha(__file__), repairedTrainerSha256=h.sha(ROOT/'scripts/train-cnn-policy-suite.py')))
    artifacts = []
    for c in protocol['channels']:
        for src, dst in [(f'cnn2-c{c}-best.pt', f'cnn2-c{c}-v1.12.pt'),
                         (f'cnn2-c{c}.torchscript.pt', f'cnn2-c{c}-v1.12.torchscript.pt'),
                         (f'cnn2-c{c}-report.json', f'cnn2-c{c}-v1.12.json')]:
            target = ROOT/'models'/dst
            assert not target.exists()
            shutil.copy2(run/src, target)
            artifacts.append(target)
    target_report = ROOT/'models/v1.12-cnn2-offline.json'
    h.write_json(target_report, report)
    rows = []
    for name, score in [('v1.9', baseline['test'])]+[(f"CNN {r['channels']} 通道", r['metrics']['test']) for r in reports]:
        rows.append(f"| {name} | {score['value']['outcomeMse']:.6f} | {score['value']['nonDrawSignAccuracy']:.2%} | {score['policy']['top1']:.2%} | {score['policy']['crossEntropy']:.6f} |")
    doc = ROOT/'models/v1.12-cnn2-offline-comparison.md'
    doc.write_text('# 两层 CNN：原始训练集离线实验\n\n'
                   '原百万训练局面，验证与测试各 6,000 局面。两层 3×3 卷积，8/16/32 通道；共享 Linear64，胜负价值头和走法策略头。全部参数从头训练。\n'
                   'AdamW 学习率 0.001、weight decay 0.001、batch 1024、随机 D4；各版相同数据及增强序列。最多 30 轮，按验证 MSE+CE 选择权重及整体候选。\n'
                   f"验证选择 {report['winnerChannels']} 通道。所有训练结束后统一评估测试集，无测试调参。\n\n"
                   '| 模型 | 胜负 MSE ↓ | 非和局准确率 ↑ | 走法命中率 ↑ | 走法交叉熵 ↓ |\n|---|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n\n'
                   'CNN 为 CPU FP32，v1.9 为原生量化推理；非和局准确率仅统计 5,192 个非和局。\n'
                   '各候选在验证／测试全集通过 TorchScript 保存重载一致性检查。原训练及预测已全部完成；汇总时因 maxEpochs 字段重复报错，修复后从已有权重与报告恢复，没有重训。\n'
                   '原执行脚本及修复脚本均保留在源码归档；逐条预测在 `.build/training/cnn2-v1.12-run1/`。未接入原生搜索、未对战，默认仍为 v1.9。\n')
    artifacts += [target_report, doc]
    sources = ['scripts/train-cnn-policy-suite.py', 'scripts/finalize-cnn-policy-suite.py', 'tests/cnn_policy_training_test.py']+list(protocol['helperSha256'])
    archive = ROOT/'models/archive/v1.12-cnn2-offline-sources.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as tar:
        for p in sources:
            tar.add(ROOT/p, arcname=p)
        tar.add(original_source, arcname='original-executed/train-cnn-policy-suite.py')
    hashes = {p: h.sha(ROOT/p) for p in sources}
    hashes['original-executed/train-cnn-policy-suite.py'] = h.sha(original_source)
    h.write_json(ROOT/'models/archive/v1.12-cnn2-offline-sources.json', dict(archive=str(archive.relative_to(ROOT)), sha256=h.sha(archive),
                 files=hashes, artifactSha256={str(p.relative_to(ROOT)): h.sha(p) for p in artifacts}))
    progress.update(complete=True, stage='complete', candidates=reports, baselineMetrics=baseline)
    h.write_json(run/'progress.json', progress)
    # Release the next GPU job only after publication and recovery checks complete.
    h.write_json(run/'report.json', report)
    print('Recovered and published completed v1.12 models without retraining')


if __name__ == '__main__':
    main()
