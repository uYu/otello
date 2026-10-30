#!/usr/bin/env python3
"""Two validation-selected CNNs vs approved v1.9; 2s/20 games each, serial."""
import hashlib
import importlib.util
import json
import signal
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'.build/training/cnn-policy-matches-v1.14'
LIBRARY = '.build/native-cnnpolicy/libothello.dylib'
NAMES = [('cosine-joint', 'CNN 综合版'), ('cosine-v4-value', 'CNN 价值版')]


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def main():
    RUN.mkdir(exist_ok=True)
    assert not (RUN/'protocol.json').exists(), 'Use a fresh run, never overwrite'
    tuning = json.loads((ROOT/'models/v1.13-cnn2-tuning.json').read_text())
    assert tuning['complete'] and tuning['winnerName'] == 'cosine' and tuning['valueWinnerName'] == 'cosine-v4'
    verification = json.loads((ROOT/'models/v1.14-cnn-policy-native-verification.json').read_text())
    assert verification['verified'] and verification['baselineFixedWorkParity']
    assert verification['librarySha256'] == sha(LIBRARY)
    baseline = json.loads((ROOT/'models/v1.9-default.json').read_text())
    assert sha(baseline['defaultModel']) == baseline['modelSha256']
    default_hash = sha('models/default-model.json')
    configs = []
    for name, label in NAMES:
        prefix = f'v1.14-cnn-{name}-vs-v1.9-2s-20-run1'
        model = f'models/cnn2-c32-{name}-v1.13.otcnn'
        export = json.loads((ROOT/model).with_suffix('.export.json').read_text())
        assert sha(model) == export['sha256']
        candidate = next(c for c in tuning['candidates'] if c['name'] == name)
        assert sha(export['checkpoint']) == candidate['modelSha256']
        runtime, report = str((RUN/(prefix+'.json')).relative_to(ROOT)), 'models/'+prefix+'.json'
        assert not (ROOT/runtime).exists() and not (ROOT/report).exists()
        configs.append(dict(name=name, label=label, prefix=prefix, model=model, runtime=runtime, report=report,
                            epoch=candidate['epoch'], validation=candidate['metrics']['validation']))
    source_files = sorted(str(p.relative_to(ROOT)) for p in (ROOT/'cpp').rglob('*') if p.is_file())
    source_files += ['CMakeLists.txt', 'tests/native.cpp', 'scripts/run-cnn-policy-matches.py',
                     'scripts/export-cnn-policy.py', 'scripts/verify-cnn-policy-native.py', 'scripts/check-engineering.py',
                     'scripts/compare-nnue-candidate.py', 'scripts/compare-nnue-search.py',
                     'scripts/compare-resnet.py', 'scripts/opening_policy.py', 'scripts/watch-match.sh', 'scripts/watch-matches.py']
    archive = 'models/archive/v1.14-cnn-policy-match-sources.tar.gz'
    source_manifest = 'models/archive/v1.14-cnn-policy-match-sources.json'
    assert not (ROOT/archive).exists()
    with tarfile.open(ROOT/archive, 'w:gz') as tar:
        for p in source_files:
            tar.add(ROOT/p, arcname=p)
    save(ROOT/source_manifest, dict(archive=archive, sha256=sha(archive), files={p: sha(p) for p in source_files}))
    protocol = dict(version='v1.14-cnn-policy-matches', status='candidate', defaultPromoted=False,
                    gamesPerCandidate=20, millisecondsPerMove=2000, seed=20261028, configs=configs,
                    modelSelection='validation-only: minimum joint loss and minimum value MSE from v1.13',
                    sameLibraryOnBothSides=True, librarySha256=sha(LIBRARY),
                    sourceManifest=source_manifest, nativeVerificationSha256=sha('models/v1.14-cnn-policy-native-verification.json'),
                    baselineManifest='models/v1.9-default.json', defaultManifestSha256=default_hash,
                    sequential=True, useBook=False, start='initial')
    save(RUN/'protocol.json', protocol)
    monitor = dict(version='两层 CNN vs v1.9 · 每步2秒 · 各20局', experiments=[
        dict(name=c['label']+' vs v1.9 · 2秒/步', shortName=c['label']+' vs v1.9 · 2秒/步', focus=c['label'],
             displayNames=dict(baseline='v1.9', candidate=c['label']), displayOrder=['candidate', 'baseline'],
             games=20, runtimeReport=c['runtime'], report=c['report']) for c in configs])
    save(ROOT/'models/current-experiment.json', monitor)
    spec = importlib.util.spec_from_file_location('match', ROOT/'scripts/compare-nnue-candidate.py')
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    published = []
    for config in configs:
        command = [sys.executable, 'scripts/compare-nnue-candidate.py', '--model', config['model'],
                   '--label', config['name'], '--baseline-manifest', 'models/v1.9-default.json',
                   '--library', LIBRARY, '--baseline-library', LIBRARY,
                   '--games', '20', '--milliseconds', '2000', '--seed', '20261028', '--output', config['runtime']]
        with (RUN/(config['name']+'.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            (RUN/'current-match.pid').write_text(str(child.pid)+'\n')
            try:
                code = child.wait()
            except KeyboardInterrupt:
                child.send_signal(signal.SIGINT)
                child.wait()
                raise
        if code:
            raise subprocess.CalledProcessError(code, command)
        report = json.loads((ROOT/config['runtime']).read_text())
        assert report['complete'] and report['games'] == report['plannedGames'] == len(report['records']) == 20
        assert report['millisecondsPerMove'] == 2000 and not report['useBook']
        assert report['librarySha256'] == report['baselineLibrarySha256'] == protocol['librarySha256']
        assert all(c['prior'] == 'learned' and c['valueTarget'] == 'outcome' for c in report['configs'])
        comparison.match.CONFIGS = report['configs']
        actions = 0
        for i, game in enumerate(report['records']):
            assert game['number'] == i+1
            assert (game['black'], game['white']) == (('baseline', 'candidate') if i % 2 == 0 else ('candidate', 'baseline'))
            assert game['seed'] == (20261028 ^ ((i//2+1)*0x9e3779b9)) & 0x7fffffff
            actions += comparison.match.replay(game)
        totals, matrix = comparison.match.aggregate(report['records'])
        assert totals == report['results'] and matrix == report['matrix'] and actions == report['replayedActions']
        assert sha('models/default-model.json') == default_hash
        report['publication'] = dict(sourceManifest=source_manifest, defaultPromoted=False,
                                     runtimeReportSha256=sha(config['runtime']), auditedActions=actions,
                                     validationSelection=config['validation'], modelCheckpointEpoch=config['epoch'])
        save(ROOT/config['report'], report)
        c, b = totals['candidate'], totals['baseline']
        doc = 'models/'+config['prefix']+'-comparison.md'
        (ROOT/doc).write_text(f'''# {config['label']} vs v1.9：2 秒，20 局

从 v1.13 调参候选按验证集选出 `{config['name']}`，第 {config['epoch']} 轮权重。两层 32 通道卷积，共享 64 维全连接，胜负价值与走法策略双头。
直接导出原 FP32 权重，未重训。验证／测试全集各 6,000 局面通过 Python 与原生输出对照；新库在 v1.9 上通过固定模拟冷／热搜索一致性检查。

双方同一原生引擎，每步各 2 秒，20 局各执黑白 10 次，配对随机种子，从标准初始局面开始，不用开局库。两组顺序运行。

候选 **{c['win']} 胜、{c['draw']} 平、{c['loss']} 负，得分率 {c['scoreRate']:.1%}，平均子差 {c['meanDiscDifference']:+.2f}**。

| 指标 | {config['label']} | v1.9 |
|---|---:|---:|
| 平均耗时／步 | {c['meanWallMs']:.1f}ms | {b['meanWallMs']:.1f}ms |
| 平均模拟／步 | {c['meanSearchUnits']:,.0f} | {b['meanSearchUnits']:,.0f} |

完整棋谱共 {actions} 步，逐步重放并重新汇总战绩。相同初始局面可能重复路径，不推算 Elo。默认仍为 v1.9。
统一监控：`sh scripts/watch-match.sh`，完成后保留终盘。
''')
        artifacts = [config['model'], config['report'], doc, source_manifest, archive,
                     'models/v1.14-cnn-policy-native-verification.json', 'models/v1.13-cnn2-tuning.json']
        manifest = dict(version=config['prefix'], status='candidate', defaultPromoted=False,
                        protocol=dict(games=20, completedGames=20, millisecondsPerMove=2000),
                        model=config['model'], matchReport=config['report'], baselineManifest='models/v1.9-default.json',
                        sourceManifest=source_manifest, artifactSha256={p: sha(p) for p in artifacts},
                        results=totals, replayedActions=actions)
        save(ROOT/('models/'+config['prefix']+'-experiment.json'), manifest)
        published.append(dict(name=config['name'], report=config['report'], results=c))
        save(RUN/'summary.json', dict(complete=len(published)==2, matches=published))
        print(json.dumps(published[-1], ensure_ascii=False), flush=True)
    save(ROOT/'models/v1.14-cnn-policy-matches.json', dict(**protocol, complete=True, matches=published))
    print('Both 2s/20-game comparisons completed and audited', flush=True)


if __name__ == '__main__':
    main()
