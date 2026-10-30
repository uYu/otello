#!/usr/bin/env python3
"""CNN vs v1.9, identical native batch32 backend, 2s/20 games."""
import argparse
import ctypes
import importlib.util
import json
import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('candidate_compare', ROOT/'scripts/compare-nnue-candidate.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
match = comparison.match
BASE_WORKER = match.worker
BASE_ENGINE = match.Engine
BASE_WRITE = match.rules.write_json
RUNTIME = '.build/training/common-batch-match-v1.19/cnn-vs-v1.9-b32-2s-20-run1.json'
REPORT = 'models/v1.19-cnn-vs-v1.9-b32-2s-20-run1.json'
LIBRARY = ROOT/'.build/native-common-batch-v1.18/libothello.dylib'
MODEL = ROOT/'models/cnn2-c32-cosine-joint-v1.13.otcnn'


def worker(connection, library, config):
    try:
        assert os.environ.get('VECLIB_MAXIMUM_THREADS') == '1'
        lib = ctypes.CDLL(library)
        lib.ot_cnn_set_native_batch_evaluator.argtypes = [ctypes.c_int]
        lib.ot_cnn_batch_stats.argtypes = [ctypes.POINTER(ctypes.c_uint)]*2
        assert lib.ot_cnn_set_native_batch_evaluator(config['nativeBatch']) == 1

        class Connection:
            def __init__(self):
                self.last_calls = self.last_leaves = 0

            def send(self, message):
                if message.get('ready'):
                    message.update(nativeBatch=config['nativeBatch'], blasThreads=1)
                if message.get('mcts'):
                    calls, leaves = ctypes.c_uint(), ctypes.c_uint()
                    lib.ot_cnn_batch_stats(ctypes.byref(calls), ctypes.byref(leaves))
                    c, n = calls.value-self.last_calls, leaves.value-self.last_leaves
                    message['mcts'].update(batchCalls=c, batchLeaves=n, actualBatchMean=n/c if c else 0)
                    self.last_calls, self.last_leaves = calls.value, leaves.value
                connection.send(message)

            def recv(self):
                return connection.recv()

            def close(self):
                connection.close()

        BASE_WORKER(Connection(), library, config)
    except BaseException as error:
        try:
            connection.send(dict(error=repr(error)))
        finally:
            connection.close()


class Engine(BASE_ENGINE):
    def __init__(self, library, config):
        self.batch = 32
        super().__init__(library, dict(config, nativeBatch=self.batch))

    def receive(self):
        response = super().receive()
        if response.get('ready'):
            assert response['nativeBatch'] == self.batch and response['blasThreads'] == 1
        return response


def setup():
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    match.worker = worker
    match.Engine = Engine


def main(args):
    setup()
    checks = json.loads((ROOT/'models/v1.18-common-native-batch-checks.json').read_text())
    diagnostic = json.loads((ROOT/'models/v1.18-common-native-batch-diagnostic.json').read_text())
    assert checks['releaseTestsPassed'] == checks['asanUbsanTestsPassed'] == 2
    assert match.sha(LIBRARY) == diagnostic['librarySha256']
    if args.smoke:
        rows = []
        for label, path in [('baseline', ROOT/'models/nnue256-outcome-v1.9.otnnue'), ('candidate', MODEL)]:
            engine = Engine(LIBRARY, dict(id=label, name=label, search='mcts', model=str(path)))
            try:
                board = match.rules.initial()
                engine.request(kind='clear')
                decision = engine.request(kind='search', board=board, turn=1, seed=20261028, ms=20)
                assert decision['move'] in match.rules.legal(board, 1)
                assert decision['mcts']['batchCalls'] > 0 and decision['mcts']['batchLeaves'] > 0
                rows.append(dict(model=label, **decision['mcts']))
            finally:
                engine.close()
        print(json.dumps(dict(workerSmokePassed=True, rows=rows)), flush=True)
        return
    runtime = ROOT/RUNTIME
    runtime.parent.mkdir(exist_ok=True)
    assert not runtime.exists() and not (ROOT/REPORT).exists()
    default_hash = match.sha(ROOT/'models/default-model.json')
    source_manifest = 'models/archive/v1.19-common-batch-match-sources.json'
    archive = ROOT/'models/archive/v1.19-common-batch-match-sources.tar.gz'
    files = sorted(str(p.relative_to(ROOT)) for p in (ROOT/'cpp').rglob('*') if p.is_file())
    files += sorted(str(p.relative_to(ROOT)) for p in (ROOT/'experiments/common-native-batch-v1.18').glob('*') if p.is_file())
    files += ['scripts/run-common-batch-match.py', 'scripts/compare-nnue-candidate.py', 'scripts/compare-nnue-search.py',
              'scripts/compare-resnet.py', 'scripts/opening_policy.py', 'scripts/watch-matches.py', 'scripts/watch-match.sh']
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as tar:
        for p in files:
            tar.add(ROOT/p, arcname=p)
    BASE_WRITE(ROOT/source_manifest, dict(archive=str(archive.relative_to(ROOT)), sha256=match.sha(archive),
                                        files={p: match.sha(ROOT/p) for p in files}))
    protocol = dict(version='v1.19-common-batch-2s-20', batchSize=32, blasThreads=1,
                    bothSides=True, backend='C++ virtual-visit MCTS + Apple Accelerate, same library on both sides',
                    modelSelection='v1.13 validation joint-loss winner cosine-joint, original weights',
                    sourceManifest=source_manifest, adapterSha256=match.sha(Path(__file__)),
                    inferenceSourceOverride={p: match.sha(ROOT/p) for p in files if p.startswith('experiments/')},
                    checksSha256=match.sha(ROOT/'models/v1.18-common-native-batch-checks.json'), defaultPromoted=False)

    def write(path, value):
        if isinstance(value, dict) and value.get('experiment') == 'candidate-vs-approved-baseline':
            value = dict(value, batchProtocol=protocol)
        BASE_WRITE(path, value)

    match.rules.write_json = write
    monitor = dict(version='双方 batch32 · CNN vs v1.9 · 2秒/20局', experiments=[dict(
        name='CNN vs v1.9 · 双方 batch32 · 2秒/步', shortName='CNN vs v1.9 · batch32 · 2秒/步', focus='CNN batch32',
        displayNames=dict(baseline='v1.9 batch32', candidate='CNN batch32'), displayOrder=['candidate', 'baseline'],
        games=20, runtimeReport=RUNTIME, report=REPORT)])
    BASE_WRITE(ROOT/'models/current-experiment.json', monitor)
    comparison.compare(argparse.Namespace(model=MODEL, label='CNN-cosine-joint-batch32',
        baseline_manifest=ROOT/'models/v1.9-default.json', library=LIBRARY, baseline_library=LIBRARY,
        games=20, milliseconds=2000, seed=20261028, output=runtime, resume=False))
    report = json.loads(runtime.read_text())
    assert report['complete'] and report['games'] == report['plannedGames'] == len(report['records']) == 20
    assert report['millisecondsPerMove'] == 2000 and report['batchProtocol'] == protocol
    assert report['librarySha256'] == report['baselineLibrarySha256'] == match.sha(LIBRARY)
    assert match.sha(ROOT/'models/default-model.json') == default_hash
    match.CONFIGS = report['configs']
    actions = 0
    batches = {key: dict(calls=0, leaves=0) for key in ('baseline', 'candidate')}
    for i, game in enumerate(report['records']):
        assert (game['black'], game['white']) == (('baseline', 'candidate') if i % 2 == 0 else ('candidate', 'baseline'))
        actions += match.replay(game)
        for decision in game['decisions']:
            m = decision['mcts']
            batches[decision['engine']]['calls'] += m['batchCalls']
            batches[decision['engine']]['leaves'] += m['batchLeaves']
    totals, matrix = match.aggregate(report['records'])
    assert totals == report['results'] and matrix == report['matrix'] and actions == report['replayedActions']
    report['publication'] = dict(runtimeReportSha256=match.sha(runtime), auditedActions=actions, batches=batches, defaultPromoted=False)
    BASE_WRITE(ROOT/REPORT, report)
    c, b = totals['candidate'], totals['baseline']
    doc = ROOT/'models/v1.19-common-batch-2s-20-comparison.md'
    doc.write_text(f'''# 共同 batch 32：CNN vs v1.9，2 秒，20 局

双方同一原生库、单线程 Accelerate、batch 32、PUCT 与临时访问批次调度；每步各 2 秒，共 20 局交换颜色，不使用开局库，从标准初始局面开始，采用配对种子。
候选为原始训练集的 `cosine-joint` 两层 32 通道 CNN，基线为冻结 v1.9，均未重新训练。

CNN **{c['win']} 胜、{c['draw']} 平、{c['loss']} 负，得分率 {c['scoreRate']:.1%}，平均子差 {c['meanDiscDifference']:+.2f}**。

| 指标 | CNN batch32 | v1.9 batch32 |
|---|---:|---:|
| 平均模拟／步 | {c['meanSearchUnits']:,.0f} | {b['meanSearchUnits']:,.0f} |
| 平均耗时／步 | {c['meanWallMs']:.1f}ms | {b['meanWallMs']:.1f}ms |
| 实际平均批次 | {batches['candidate']['leaves']/max(1,batches['candidate']['calls']):.2f} | {batches['baseline']['leaves']/max(1,batches['baseline']['calls']):.2f} |

{actions} 步完整棋谱已重放审计，战绩重新汇总。每一步记录批次数与叶节点数；同协议之外的旧对战不混算。
同起点可能出现重复路径，小样本不推算 Elo。默认仍为 v1.9，统一监控保留终盘。
''')
    files = [REPORT, str(doc.relative_to(ROOT)), str(MODEL.relative_to(ROOT)), 'models/nnue256-outcome-v1.9.otnnue',
             source_manifest, str(archive.relative_to(ROOT)), 'models/v1.18-common-native-batch-checks.json']
    BASE_WRITE(ROOT/'models/v1.19-common-batch-2s-20-experiment.json', dict(
        status='candidate', defaultPromoted=False, batchProtocol=protocol,
        protocol=dict(games=20, completedGames=20, millisecondsPerMove=2000),
        model=str(MODEL.relative_to(ROOT)), baselineManifest='models/v1.9-default.json', matchReport=REPORT,
        artifactSha256={p: match.sha(ROOT/p) for p in files}, results=totals, replayedActions=actions))
    print(json.dumps(dict(complete=True, candidate=c, report=REPORT)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true')
    main(p.parse_args())
