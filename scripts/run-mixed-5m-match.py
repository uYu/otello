#!/usr/bin/env python3
"""Run and publish the requested full-epoch 5m vs v1.9, 1s/50 games."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = '.build/training/mixed-5m-v1/v1.11-mixed-5m-vs-v1.9-1s-50-run1.json'
REPORT = 'models/v1.11-mixed-5m-vs-v1.9-1s-50-run1.json'
MANIFEST = 'models/v1.11-mixed-5m-1s-50-run1-experiment.json'
DOC = 'models/v1.11-mixed-5m-1s-50-run1-comparison.md'
MODEL = 'models/nnue256-mixed-5m-full-v1.11.otnnue'
LIBRARY = '.build/native-engineering/libothello.dylib'


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def save(path, value):
    target = ROOT/path
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(target)


def main():
    for path in (RUNTIME, REPORT, MANIFEST, DOC):
        if (ROOT/path).exists():
            raise ValueError('Refusing to overwrite '+path)
    baseline = json.loads((ROOT/'models/v1.9-default.json').read_text())
    training = json.loads((ROOT/'models/v1.11-mixed-5m-full.json').read_text())
    engineering = json.loads((ROOT/'models/engineering-rerun1.json').read_text())
    sources = json.loads((ROOT/engineering['sourceManifest']).read_text())
    assert sha(MODEL) == training['sha256']
    assert training['newPositions'] == 104627926 and training['oldPositions'] == 1000000
    assert sha(baseline['defaultModel']) == baseline['modelSha256']
    assert sha(LIBRARY) == engineering['optimizedLibrarySha256']
    assert all(sha(p) == h for p, h in sources['files'].items() if p.startswith('cpp/'))
    active_hash = sha('models/default-model.json')
    files = sorted(str(p.relative_to(ROOT)) for p in (ROOT/'cpp').rglob('*') if p.is_file())
    files += ['scripts/run-mixed-5m-match.py', 'scripts/compare-nnue-candidate.py',
              'scripts/compare-nnue-search.py', 'scripts/compare-resnet.py', 'scripts/opening_policy.py',
              'scripts/watch-match.sh', 'scripts/watch-matches.py', 'models/current-experiment.json']
    archive = 'models/archive/v1.11-mixed-5m-1s-50-run1-sources.tar.gz'
    source_manifest = 'models/archive/v1.11-mixed-5m-1s-50-run1-sources.json'
    assert not (ROOT/archive).exists() and not (ROOT/source_manifest).exists()
    with tarfile.open(ROOT/archive, 'w:gz') as tar:
        for path in files:
            tar.add(ROOT/path, arcname=path)
    save(source_manifest, dict(archive=archive, sha256=sha(archive), files={p: sha(p) for p in files}))
    subprocess.run([sys.executable, 'scripts/compare-nnue-candidate.py', '--model', MODEL,
                    '--label', 'v1.11-mixed-5m-full-1s-50-run1', '--baseline-manifest', 'models/v1.9-default.json',
                    '--library', LIBRARY, '--baseline-library', LIBRARY,
                    '--games', '50', '--milliseconds', '1000', '--seed', '20261028', '--output', RUNTIME],
                   cwd=ROOT, check=True)
    report = json.loads((ROOT/RUNTIME).read_text())
    assert report['complete'] and report['games'] == report['plannedGames'] == len(report['records']) == 50
    assert report['millisecondsPerMove'] == 1000 and not report['useBook']
    assert report['librarySha256'] == report['baselineLibrarySha256'] == sha(LIBRARY)
    assert sha('models/default-model.json') == active_hash
    spec = importlib.util.spec_from_file_location('candidate_match', ROOT/'scripts/compare-nnue-candidate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.match.CONFIGS = report['configs']
    actions = 0
    for i, game in enumerate(report['records']):
        assert game['number'] == i+1
        assert (game['black'], game['white']) == (('baseline', 'candidate') if i % 2 == 0 else ('candidate', 'baseline'))
        assert game['seed'] == (20261028 ^ ((i//2+1)*0x9e3779b9)) & 0x7fffffff
        actions += module.match.replay(game)
    results, matrix = module.match.aggregate(report['records'])
    assert results == report['results'] and matrix == report['matrix'] and actions == report['replayedActions']
    snapshot = json.loads((ROOT/source_manifest).read_text())
    assert all(snapshot['files'][p] == h for p, h in report['sourceSha256'].items())
    report['publication'] = dict(runtimeReportSha256=sha(RUNTIME), sourceManifest=source_manifest,
                                 modelSelection='full epoch, all eligible 5m-game positions plus original training set',
                                 auditedActions=actions, defaultPromoted=False)
    save(REPORT, report)
    c, b = results['candidate'], results['baseline']
    (ROOT/DOC).write_text(f'''# 500 万局混合模型 vs v1.9：1 秒，50 局

双方每步各 1 秒，共 50 局，交替执黑白，从标准初始局面开始，使用配对随机种子和同一优化版 PUCT MCTS 引擎，不使用开局库。
候选是完整遍历一轮的 `nnue256-mixed-5m-full-v1.11.otnnue`，不是第 8 分片的验证最佳检查点。
它在 v1.9 的冻结特征上联合训练价值和策略头，使用 104,627,926 个新增训练聚合局面和原有 1,000,000 个局面。

候选 **{c['win']} 胜、{c['draw']} 平、{c['loss']} 负，得分率 {c['scoreRate']:.1%}，平均子差 {c['meanDiscDifference']:+.2f}**。

| 指标 | 500 万局混合候选 | v1.9 |
|---|---:|---:|
| 胜／平／负 | {c['win']}／{c['draw']}／{c['loss']} | {b['win']}／{b['draw']}／{b['loss']} |
| 得分率 | {c['scoreRate']:.1%} | {b['scoreRate']:.1%} |
| 平均子差 | {c['meanDiscDifference']:+.2f} | {b['meanDiscDifference']:+.2f} |
| 平均每步耗时 | {c['meanWallMs']:.1f}ms | {b['meanWallMs']:.1f}ms |
| 平均每步模拟数 | {c['meanSearchUnits']:,.0f} | {b['meanSearchUnits']:,.0f} |

完整记录共 {actions} 步，已逐步重放审计，并重新计算战绩。双方各执黑白 25 局。
初始局面相同，可能重复路径，不推算 Elo。预测测试与本次对战分别记录；默认仍为 v1.9。
模型和训练预测见 [混合训练报告](v1.11-mixed-5m-comparison.md)。
统一监控：`sh scripts/watch-match.sh`，完成后保留最后终盘。
''')
    artifacts = [MODEL, 'models/v1.11-mixed-5m-full.json', 'models/v1.9-default.json',
                 'models/engineering-rerun1.json', REPORT, DOC, source_manifest, archive]
    save(MANIFEST, dict(version='v1.11-mixed-5m-1s-50-run1', status='candidate', defaultPromoted=False,
                         protocol=dict(games=50, completedGames=50, millisecondsPerMove=1000),
                         model=MODEL, baselineManifest='models/v1.9-default.json', matchReport=REPORT,
                         sourceManifest=source_manifest, artifactSha256={p: sha(p) for p in artifacts},
                         results=results, replayedActions=actions, runtimeReportSha256=sha(RUNTIME)))
    print(json.dumps(dict(complete=True, results=c, report=REPORT)), flush=True)


if __name__ == '__main__':
    main()
