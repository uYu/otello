#!/usr/bin/env python3
"""Check the published v1 model, sources, report checksums and W/D/L totals."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    active_path=ROOT/'models/default-model.json'
    if active_path.exists():
        active=json.loads(active_path.read_text())
        release_path=ROOT/active['baselineManifest']
        release=json.loads(release_path.read_text())
        assert release['status']=='frozen' and release['userApproved']
        assert active['version']==release['version']
        assert active['defaultModel']==release['defaultModel']
        assert active['modelSha256']==release['modelSha256']==sha(ROOT/release['defaultModel'])
        frozen=json.loads((ROOT/release['sourceManifest']).read_text())
        assert frozen['baselineSha256']==sha(release_path)
        assert sha(ROOT/frozen['archive'])==frozen['sha256']
        assert frozen['files']=={**release['sourceSha256'],**release['artifactSha256']}
        with tarfile.open(ROOT/frozen['archive'],'r:gz') as archive:
            assert len(archive.getmembers())==len(frozen['files'])
            for name,expected in frozen['files'].items():
                assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==expected,name
        assert all(sha(ROOT/name)==expected for name,expected in release['artifactSha256'].items())
        assert all(sha(ROOT/r['path'])==r['sha256'] for r in release['referenceExperiments'])
        print(f"{release['version']} active default verified: user-approved model, frozen sources and current Wasm artifact")
    # Previous defaults retain their own archived Wasm and sources after promotion.
    for previous_version in ('v1.8','v1.9','v1.25','v1.36','v1.37'):
        previous_manifest=ROOT/f'models/archive/{previous_version}-default-sources.json'
        if not previous_manifest.exists() or (active_path.exists() and active['version']==previous_version):
            continue
        frozen=json.loads(previous_manifest.read_text())
        release_path=ROOT/f'models/{previous_version}-default.json'
        release=json.loads(release_path.read_text())
        assert sha(release_path)==frozen['baselineSha256']
        assert sha(ROOT/frozen['archive'])==frozen['sha256']
        assert frozen['files']=={**release['sourceSha256'],**release['artifactSha256']}
        with tarfile.open(ROOT/frozen['archive'],'r:gz') as archive:
            assert len(archive.getmembers())==len(frozen['files'])
            for name,expected in frozen['files'].items():
                assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==expected,name
        assert sha(ROOT/release['defaultModel'])==release['modelSha256']
        assert all(sha(ROOT/r['path'])==r['sha256'] for r in release['referenceExperiments'])
        print(f'{previous_version} previous default verified: original source/artifact snapshot and unchanged frozen weights')
    v17_archive = ROOT / 'models/archive/v1.7-sources.json'
    if v17_archive.exists():
        frozen = json.loads(v17_archive.read_text())
        default = json.loads((ROOT / frozen['baseline']).read_text())
        assert sha(ROOT / frozen['baseline']) == frozen['baselineSha256']
        assert sha(ROOT / frozen['archive']) == frozen['sha256']
        assert sha(ROOT / default['defaultModel']) == default['modelSha256']
        assert frozen['files'] == {**default['sourceSha256'], **default['artifactSha256'], **frozen['buildSupportSha256']}
        with tarfile.open(ROOT / frozen['archive'], 'r:gz') as archive:
            assert len(archive.getmembers()) == len(frozen['files'])
            for name, expected in frozen['files'].items():
                assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected, name
        print('v1.7 default verified: original source/artifact snapshot and unchanged frozen weights')
    baseline = json.loads((ROOT / 'models/v1-baseline.json').read_text())
    metadata = json.loads((ROOT / 'models/nnue256-v1.json').read_text())
    # v1 measurements stay frozen when the working tree advances to v1.1. Check
    # their original source bytes from the checksummed, reproducible snapshot.
    snapshot_path = ROOT / 'models/archive/v1-sources.json'
    sources = {}
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        assert sha(ROOT / 'models/v1-baseline.json') == snapshot['baselineSha256']
        archive_path = ROOT / snapshot['archive']
        assert sha(archive_path) == snapshot['sha256']
        with tarfile.open(archive_path, 'r:gz') as archive:
            assert len(archive.getmembers()) == len(snapshot['files'])
            for name, expected in snapshot['files'].items():
                sources[name] = archive.extractfile(name).read()
                assert hashlib.sha256(sources[name]).hexdigest() == expected, name

    def source_sha(name):
        return hashlib.sha256(sources[name]).hexdigest() if sources else sha(ROOT / name)
    assert baseline['status'] == 'frozen'
    assert sha(ROOT / baseline['model']) == baseline['modelSha256'] == metadata['sha256']
    assert sha(ROOT / baseline['protocol']) == baseline['protocolSha256']
    assert source_sha('CMakeLists.txt') == baseline['cmakeSha256']
    assert all(source_sha(name) == value for name, value in baseline['sourceSha256'].items())
    assert source_sha('scripts/train-nnue.py') == metadata['trainingScriptSha256']
    total_games = book_hits = book_misses = 0
    for experiment in baseline['experiments']:
        path = ROOT / experiment['report']
        assert sha(path) == experiment['reportSha256']
        report = json.loads(path.read_text())
        assert report['complete'] and report['games'] == len(report['records']) == experiment['games']
        assert report['modelSha256'] == baseline['modelSha256']
        assert report['sourceSha256'] == baseline['sourceSha256']
        assert source_sha('scripts/compare-evaluators.py') == report['comparisonScriptSha256']
        counts = Counter()
        for game in report['records']:
            difference = game['modelDiscDifference']
            outcome = 'win' if difference > 0 else 'loss' if difference < 0 else 'draw'
            assert outcome == game['outcome']
            counts[outcome] += 1
            counts['discDifference'] += difference
            for decision in game['decisions']:
                if decision['bookProbe'] == 'hit':
                    assert decision['provider'] == 'book' and decision['nodes'] == 0
                    book_hits += 1
                elif decision['bookProbe'] == 'miss':
                    assert decision['provider'] == 'nnue'
                    book_misses += 1
        assert all(counts[k] == report['results'][k] == experiment['results'][k]
                   for k in ('win', 'draw', 'loss', 'discDifference'))
        total_games += report['games']
    assert total_games == 200 and book_hits == 893 and book_misses == 2107
    print(f'v1 verified: {total_games} games, frozen model/source snapshot, {book_hits} book hits, {book_misses} NNUE fallbacks')
    current_path = ROOT / 'models/v1.1-search-baseline.json'
    v11_sources = {}
    archive_manifest = ROOT / 'models/archive/v1.1-sources.json'
    if archive_manifest.exists():
        snapshot = json.loads(archive_manifest.read_text())
        assert sha(ROOT / 'models/v1.1-baseline.json') == snapshot['baselineSha256']
        assert sha(current_path) == snapshot['searchBaselineSha256']
        archive_path = ROOT / snapshot['archive']
        assert sha(archive_path) == snapshot['sha256']
        with tarfile.open(archive_path, 'r:gz') as archive:
            assert len(archive.getmembers()) == len(snapshot['files'])
            for name, expected in snapshot['files'].items():
                v11_sources[name] = archive.extractfile(name).read()
                assert hashlib.sha256(v11_sources[name]).hexdigest() == expected, name

    def v11_sha(name):
        return hashlib.sha256(v11_sources[name]).hexdigest() if v11_sources else sha(ROOT / name)

    if current_path.exists():
        current = json.loads(current_path.read_text())
        assert current['modelSha256'] == sha(ROOT / current['model'])
        assert all(v11_sha(name) == value for name, value in current['sourceSha256'].items())
        assert all(v11_sha(name) == value for name, value in current['artifactSha256'].items())
        assert all(row['allCandidateScoresEqual'] for row in current['benchmark']['results'])
        print('v1.1 search verified: frozen source snapshot and model match the performance baseline')
    release_path = ROOT / 'models/v1.1-baseline.json'
    if release_path.exists():
        release = json.loads(release_path.read_text())
        assert release['version'] == 'v1.1' and release['status'] == 'frozen'
        assert release['modelSha256'] == baseline['modelSha256']
        assert all(v11_sha(name) == value for name, value in release['sourceSha256'].items())
        assert sha(current_path) == release['searchBaselineSha256']
        entry = release['strengthReport']
        assert sha(ROOT / entry['path']) == entry['sha256']
        report = json.loads((ROOT / entry['path']).read_text())
        assert report['complete'] and report['version'] == 'v1.1'
        assert report['games'] == report['plannedGames'] == len(report['records']) == 100
        assert report['start'] == 'initial' and report['opening'] == [] and not report['useBook']
        assert report['millisecondsPerMove'] == 50 and report['maxDepth'] == 64
        assert report['modelSha256'] == release['modelSha256']
        assert report['librarySha256'] == release['librarySha256']
        assert report['sourceSha256']['v1'] == baseline['sourceSha256']
        assert all(v11_sha(name) == value for name, value in report['sourceSha256']['v1.1'].items())
        runner = ROOT / 'scripts/compare-versions.py'
        assert sha(runner) == report['runnerSha256']
        spec = importlib.util.spec_from_file_location('version_comparison', runner)
        comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparison)
        counts, actions = Counter(), 0
        for i, game in enumerate(report['records']):
            assert game['number'] == i + 1 and game['focusSide'] == (1 if i % 2 == 0 else -1)
            assert game['opening'] == []
            actions += comparison.replay(game)
            counts[game['outcome']] += 1
            counts['discDifference'] += game['focusDiscDifference']
        assert all(counts[k] == report['results'][k] == entry['results'][k]
                   for k in ('win', 'draw', 'loss', 'discDifference'))
        assert report['replayedGames'] == 100 and actions == report['replayedActions']
        assert report['uniqueTranscripts'] == len({tuple(game['moves']) for game in report['records']})
        assert report['scoreRate'] == entry['scoreRate'] == (counts['win'] + counts['draw'] / 2) / 100
        print(f'v1.1 strength verified: 100 initial-position games, {actions} replayed actions, equal 50 ms budget')
    new_path = ROOT / 'models/v1.2-baseline.json'
    v12_sources = {}
    archive_manifest = ROOT / 'models/archive/v1.2-sources.json'
    if archive_manifest.exists():
        snapshot = json.loads(archive_manifest.read_text())
        assert sha(new_path) == snapshot['baselineSha256']
        archive_path = ROOT / snapshot['archive']
        assert sha(archive_path) == snapshot['sha256']
        with tarfile.open(archive_path, 'r:gz') as archive:
            assert len(archive.getmembers()) == len(snapshot['files'])
            for name, expected in snapshot['files'].items():
                v12_sources[name] = archive.extractfile(name).read()
                assert hashlib.sha256(v12_sources[name]).hexdigest() == expected, name

    def v12_sha(name):
        return hashlib.sha256(v12_sources[name]).hexdigest() if name in v12_sources else sha(ROOT / name)

    if new_path.exists():
        release = json.loads(new_path.read_text())
        assert release['version'] == 'v1.2' and release['status'] == 'frozen'
        assert all(v12_sha(name) == value for name, value in release['sourceSha256'].items())
        assert all(v12_sha(name) == value for name, value in release['artifactSha256'].items())
        model = json.loads((ROOT / release['modelMetadata']).read_text())
        assert model['sha256'] == release['modelSha256'] == sha(ROOT / release['model'])
        assert model['modelFormat'] == 2 and model['parameters'] == 33537
        assert model['dataset']['cacheSha256'] == metadata['dataset']['cacheSha256']
        assert model['trainingScriptSha256'] == v12_sha('scripts/train-nnue.py')
        assert model['cppParityPositions'] == 1024
        entry = release['strengthReport']
        assert sha(ROOT / entry['path']) == entry['sha256']
        report = json.loads((ROOT / entry['path']).read_text())
        assert report['complete'] and report['version'] == 'v1.2'
        assert report['games'] == report['plannedGames'] == len(report['records']) == 100
        assert report['start'] == 'initial' and report['opening'] == [] and not report['useBook']
        assert report['millisecondsPerMove'] == 50 and report['maxDepth'] == 64
        assert report['modelSha256'] == {'v1.1': baseline['modelSha256'], 'v1.2': release['modelSha256']}
        assert report['librarySha256'] == release['librarySha256']
        old_release = json.loads((ROOT / 'models/v1.1-baseline.json').read_text())
        assert report['sourceSha256']['v1.1'] == old_release['sourceSha256']
        assert all(v12_sha(name) == value for name, value in report['sourceSha256']['v1.2'].items())
        runner = ROOT / 'scripts/compare-model-versions.py'
        assert sha(runner) == report['runnerSha256']
        spec = importlib.util.spec_from_file_location('model_comparison', runner)
        comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparison)
        counts, actions = Counter(), 0
        for i, game in enumerate(report['records']):
            assert game['number'] == i + 1 and game['focusSide'] == (1 if i % 2 == 0 else -1)
            assert game['opening'] == []
            actions += comparison.replay(game)
            counts[game['outcome']] += 1
            counts['discDifference'] += game['focusDiscDifference']
        assert all(counts[k] == report['results'][k] == entry['results'][k]
                   for k in ('win', 'draw', 'loss', 'discDifference'))
        assert report['replayedGames'] == 100 and actions == report['replayedActions']
        assert report['uniqueTranscripts'] == len({tuple(game['moves']) for game in report['records']})
        assert report['scoreRate'] == entry['scoreRate'] == (counts['win'] + counts['draw'] / 2) / 100
        print(f'v1.2 verified: turn-aware model, same frozen dataset, 100 games, {actions} replayed actions')
    latest_path = ROOT / 'models/v1.3-baseline.json'
    v13_sources = {}
    archive_manifest = ROOT / 'models/archive/v1.3-sources.json'
    if archive_manifest.exists():
        snapshot = json.loads(archive_manifest.read_text())
        assert sha(latest_path) == snapshot['baselineSha256']
        archive_path = ROOT / snapshot['archive']
        assert sha(archive_path) == snapshot['sha256']
        with tarfile.open(archive_path, 'r:gz') as archive:
            assert len(archive.getmembers()) == len(snapshot['files'])
            for name, expected in snapshot['files'].items():
                v13_sources[name] = archive.extractfile(name).read()
                assert hashlib.sha256(v13_sources[name]).hexdigest() == expected, name

    def v13_sha(name):
        return hashlib.sha256(v13_sources[name]).hexdigest() if name in v13_sources else sha(ROOT / name)

    if latest_path.exists():
        release = json.loads(latest_path.read_text())
        assert release['version'] == 'v1.3' and release['status'] == 'frozen'
        assert all(v13_sha(name) == value for name, value in release['sourceSha256'].items())
        assert all(v13_sha(name) == value for name, value in release['artifactSha256'].items())
        assert release['modelSha256'] == sha(ROOT / release['model'])
        entry = release['strengthReport']
        assert sha(ROOT / entry['path']) == entry['sha256']
        report = json.loads((ROOT / entry['path']).read_text())
        assert report['complete'] and report['version'] == 'v1.3'
        assert report['games'] == report['plannedGames'] == len(report['records']) == 100
        assert report['start'] == 'initial' and report['opening'] == [] and not report['useBook']
        assert not report['referenceVariety']
        assert report['openingPolicy'] == {'plies': 12, 'temperature': 32, 'max_loss': 64}
        assert report['millisecondsPerMove'] == 50 and report['maxDepth'] == 64
        assert report['modelSha256'] == {'v1.1': baseline['modelSha256'], 'v1.3': release['modelSha256']}
        assert report['librarySha256'] == release['librarySha256']
        assert report['sourceSha256']['v1.1'] == json.loads((ROOT / 'models/v1.1-baseline.json').read_text())['sourceSha256']
        assert all(v13_sha(name) == value for name, value in report['sourceSha256']['v1.3'].items())
        runner = ROOT / 'scripts/compare-opening-policy.py'
        assert sha(runner) == report['runnerSha256']
        assert sha(ROOT / 'scripts/opening_policy.py') == report['policyAuditSha256']
        spec = importlib.util.spec_from_file_location('opening_comparison', runner)
        comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparison)
        counts, actions = Counter(), 0
        for i, game in enumerate(report['records']):
            assert game['number'] == i + 1 and game['focusSide'] == (1 if i % 2 == 0 else -1)
            assert game['opening'] == [] and game['openingPolicy'] == report['openingPolicy']
            assert game['seed'] == (report['baseSeed'] ^ ((i + 1) * 0x9e3779b9)) & 0x7fffffff
            actions += comparison.replay(game)
            counts[game['outcome']] += 1
            counts['discDifference'] += game['focusDiscDifference']
        assert all(counts[k] == report['results'][k] == entry['results'][k]
                   for k in ('win', 'draw', 'loss', 'discDifference'))
        assert report['replayedGames'] == 100 and actions == report['replayedActions']
        assert report['uniqueTranscripts'] == len({tuple(g['moves']) for g in report['records']})
        canonical = comparison.opening_policy.canonical_moves
        assert report['symmetryUniqueTranscripts'] == len({canonical(g['moves']) for g in report['records']})
        assert report['scoreRate'] == entry['scoreRate'] == (counts['win'] + counts['draw'] / 2) / 100
        print(f'v1.3 verified: 100 games, {actions} replayed actions; seeds, weighted choices and symmetry dedup audited')
    v14_sources = {}
    archive_manifest = ROOT / 'models/archive/v1.4-sources.json'
    if archive_manifest.exists():
        snapshot = json.loads(archive_manifest.read_text())
        assert sha(ROOT / 'models/v1.4-baseline.json') == snapshot['baselineSha256']
        archive_path = ROOT / snapshot['archive']
        assert sha(archive_path) == snapshot['sha256']
        with tarfile.open(archive_path, 'r:gz') as archive:
            assert len(archive.getmembers()) == len(snapshot['files'])
            for name, expected in snapshot['files'].items():
                v14_sources[name] = archive.extractfile(name).read()
                assert hashlib.sha256(v14_sources[name]).hexdigest() == expected, name

    def v14_sha(name):
        return hashlib.sha256(v14_sources[name]).hexdigest() if name in v14_sources else sha(ROOT / name)

    cnn_path = ROOT / 'models/v1.4-baseline.json'
    if cnn_path.exists():
        release = json.loads(cnn_path.read_text())
        assert release['version'] == 'v1.4' and release['status'] == 'frozen'
        assert all(v14_sha(name) == value for name, value in release['sourceSha256'].items())
        assert all(v14_sha(name) == value for name, value in release['artifactSha256'].items())
        model = json.loads((ROOT / release['modelMetadata']).read_text())
        assert model['modelFormat'] == 3 and model['parameters'] == 17185
        assert model['sha256'] == release['modelSha256'] == sha(ROOT / release['model'])
        assert model['dataset']['cacheSha256'] == metadata['dataset']['cacheSha256']
        assert model['trainingScriptSha256'] == sha(ROOT / 'scripts/train-cnn.py')
        assert model['cppParityPositions'] == 1024 and model['cppMaxRoundingError'] <= 0.501
        entry = release['strengthReport']
        assert sha(ROOT / entry['path']) == entry['sha256']
        report = json.loads((ROOT / entry['path']).read_text())
        assert report['complete'] and report['version'] == 'v1.4'
        assert report['games'] == report['plannedGames'] == len(report['records']) == 100
        assert report['start'] == 'initial' and report['opening'] == [] and not report['useBook']
        assert report['referenceVariety']
        assert report['openingPolicy'] == {'plies': 12, 'temperature': 32, 'max_loss': 64}
        assert report['millisecondsPerMove'] == 50 and report['maxDepth'] == 64
        old_release = json.loads((ROOT / 'models/v1.3-baseline.json').read_text())
        assert report['modelSha256'] == {'v1.3': old_release['modelSha256'], 'v1.4': release['modelSha256']}
        assert report['librarySha256'] == release['librarySha256']
        assert report['sourceSha256']['v1.3'] == old_release['sourceSha256']
        assert all(v14_sha(name) == value for name, value in report['sourceSha256']['v1.4'].items())
        runner = ROOT / 'scripts/compare-cnn.py'
        assert sha(runner) == report['runnerSha256']
        assert sha(ROOT / 'scripts/opening_policy.py') == report['policyAuditSha256']
        spec = importlib.util.spec_from_file_location('cnn_comparison', runner)
        comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparison)
        counts, actions = Counter(), 0
        for i, game in enumerate(report['records']):
            assert game['number'] == i + 1 and game['focusSide'] == (1 if i % 2 == 0 else -1)
            assert game['opening'] == [] and game['openingPolicy'] == report['openingPolicy']
            assert game['seed'] == (report['baseSeed'] ^ ((i + 1) * 0x9e3779b9)) & 0x7fffffff
            actions += comparison.replay(game)
            counts[game['outcome']] += 1
            counts['discDifference'] += game['focusDiscDifference']
        assert all(counts[k] == report['results'][k] == entry['results'][k]
                   for k in ('win', 'draw', 'loss', 'discDifference'))
        assert actions == report['replayedActions'] and report['replayedGames'] == 100
        assert report['scoreRate'] == entry['scoreRate'] == (counts['win'] + counts['draw']/2) / 100
        canonical = comparison.opening_policy.canonical_moves
        assert report['symmetryUniqueTranscripts'] == len({canonical(g['moves']) for g in report['records']})
        benchmark = json.loads((ROOT / release['inferenceBenchmark']).read_text())
        assert sha(ROOT / release['inferenceBenchmark']) == release['inferenceBenchmarkSha256']
        assert benchmark['scriptSha256'] == sha(ROOT / 'scripts/benchmark-cnn.py')
        diagnostic = release['fixedDepthDiagnostic']
        assert sha(ROOT / diagnostic['path']) == diagnostic['sha256']
        fixed = json.loads((ROOT / diagnostic['path']).read_text())
        assert fixed['complete'] and fixed['games'] == len(fixed['records']) == diagnostic['games'] == 20
        assert fixed['maxDepth'] == diagnostic['depth'] == 4
        assert fixed['millisecondsPerMove'] == 60000
        assert fixed['start'] == 'initial' and fixed['opening'] == [] and not fixed['useBook']
        assert fixed['modelSha256'] == report['modelSha256']
        assert fixed['librarySha256'] == report['librarySha256']
        assert fixed['sourceSha256'] == report['sourceSha256']
        assert fixed['runnerSha256'] == report['runnerSha256']
        assert fixed['policyAuditSha256'] == report['policyAuditSha256']
        assert fixed['referenceVariety'] and fixed['openingPolicy'] == report['openingPolicy']
        fixed_counts, fixed_actions = Counter(), 0
        for i, game in enumerate(fixed['records']):
            assert game['number'] == i + 1 and game['focusSide'] == (1 if i % 2 == 0 else -1)
            assert game['seed'] == (fixed['baseSeed'] ^ ((i + 1) * 0x9e3779b9)) & 0x7fffffff
            assert game['opening'] == [] and game['openingPolicy'] == fixed['openingPolicy']
            fixed_actions += comparison.replay(game)
            assert all(d['depth'] == 4 for d in game['decisions'] if not d['exact'])
            fixed_counts[game['outcome']] += 1
            fixed_counts['discDifference'] += game['focusDiscDifference']
        assert all(fixed_counts[k] == fixed['results'][k] == diagnostic['results'][k]
                   for k in ('win', 'draw', 'loss', 'discDifference'))
        assert fixed_actions == fixed['replayedActions']
        assert fixed['scoreRate'] == diagnostic['scoreRate'] == (fixed_counts['win'] + fixed_counts['draw']/2)/20
        print(f'v1.4 CNN verified: same dataset, 1024 inference checks, 100 games, {actions} audited actions')

    v15_sources = {}
    archive_manifest = ROOT / 'models/archive/v1.5-sources.json'
    if archive_manifest.exists():
        snapshot = json.loads(archive_manifest.read_text())
        assert sha(ROOT / 'models/v1.5-baseline.json') == snapshot['baselineSha256']
        assert sha(ROOT / snapshot['archive']) == snapshot['sha256']
        with tarfile.open(ROOT / snapshot['archive'], 'r:gz') as archive:
            assert len(archive.getmembers()) == len(snapshot['files'])
            for name, expected in snapshot['files'].items():
                v15_sources[name] = archive.extractfile(name).read()
                assert hashlib.sha256(v15_sources[name]).hexdigest() == expected

    def v15_sha(name):
        return hashlib.sha256(v15_sources[name]).hexdigest() if name in v15_sources else sha(ROOT / name)

    residual_path = ROOT / 'models/v1.5-baseline.json'
    if residual_path.exists():
        release = json.loads(residual_path.read_text())
        assert release['version'] == 'v1.5' and release['status'] == 'frozen'
        assert all(v15_sha(name) == value for name, value in release['sourceSha256'].items())
        assert all(v15_sha(name) == value for name, value in release['artifactSha256'].items())
        model = json.loads((ROOT / release['modelMetadata']).read_text())
        assert model['modelFormat'] == 4 and model['channels'] == 32 and model['residualBlocks'] == 2
        assert model['sha256'] == release['modelSha256'] == sha(ROOT / release['model'])
        assert model['dataset']['cacheSha256'] == metadata['dataset']['cacheSha256']
        assert model['trainingScriptSha256'] == sha(ROOT / 'scripts/train-resnet.py')
        assert model['cppParityPositions'] == 6000 and model['cppMaxRoundingError'] <= 0.51
        entry = release['strengthReport']
        assert sha(ROOT / entry['path']) == entry['sha256']
        report = json.loads((ROOT / entry['path']).read_text())
        assert report['complete'] and report['version'] == 'v1.5'
        assert report['games'] == report['plannedGames'] == len(report['records']) == 100
        assert report['start'] == 'initial' and report['opening'] == [] and not report['useBook']
        assert report['referenceVariety']
        assert report['openingPolicy'] == {'plies': 12, 'temperature': 32, 'max_loss': 64}
        assert report['millisecondsPerMove'] == 50 and report['maxDepth'] == 64
        old_release = json.loads((ROOT / 'models/v1.3-baseline.json').read_text())
        assert report['modelSha256'] == {'v1.3': old_release['modelSha256'], 'v1.5': release['modelSha256']}
        assert report['librarySha256'] == release['librarySha256']
        assert report['sourceSha256']['v1.3'] == old_release['sourceSha256']
        assert all(v15_sha(name) == value for name, value in report['sourceSha256']['v1.5'].items())
        runner = ROOT / 'scripts/compare-resnet.py'
        assert sha(runner) == report['runnerSha256']
        assert sha(ROOT / 'scripts/opening_policy.py') == report['policyAuditSha256']
        spec = importlib.util.spec_from_file_location('resnet_comparison', runner)
        comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparison)
        counts, actions = Counter(), 0
        for i, game in enumerate(report['records']):
            assert game['number'] == i + 1 and game['focusSide'] == (1 if i % 2 == 0 else -1)
            assert game['opening'] == [] and game['openingPolicy'] == report['openingPolicy']
            assert game['seed'] == (report['baseSeed'] ^ ((i + 1) * 0x9e3779b9)) & 0x7fffffff
            actions += comparison.replay(game)
            counts[game['outcome']] += 1
            counts['discDifference'] += game['focusDiscDifference']
        assert all(counts[k] == report['results'][k] == entry['results'][k]
                   for k in ('win', 'draw', 'loss', 'discDifference'))
        assert actions == report['replayedActions'] and report['replayedGames'] == 100
        assert report['scoreRate'] == entry['scoreRate'] == (counts['win'] + counts['draw']/2) / 100
        canonical = comparison.opening_policy.canonical_moves
        assert report['symmetryUniqueTranscripts'] == len({canonical(g['moves']) for g in report['records']})
        benchmark = json.loads((ROOT / release['inferenceBenchmark']).read_text())
        assert sha(ROOT / release['inferenceBenchmark']) == release['inferenceBenchmarkSha256']
        assert benchmark['scriptSha256'] == sha(ROOT / 'scripts/benchmark-resnet.py')
        if 'fixedDepthDiagnostic' in release:
            diagnostic = release['fixedDepthDiagnostic']
            assert sha(ROOT / diagnostic['path']) == diagnostic['sha256']
            fixed = json.loads((ROOT / diagnostic['path']).read_text())
            assert fixed['complete'] and fixed['games'] == len(fixed['records']) == diagnostic['games'] == 20
            assert fixed['maxDepth'] == diagnostic['depth'] == 4
            assert fixed['millisecondsPerMove'] == 60000
            assert fixed['start'] == 'initial' and fixed['opening'] == [] and not fixed['useBook']
            assert fixed['modelSha256'] == report['modelSha256']
            assert fixed['librarySha256'] == report['librarySha256']
            assert fixed['sourceSha256'] == report['sourceSha256']
            assert fixed['runnerSha256'] == report['runnerSha256']
            assert fixed['policyAuditSha256'] == report['policyAuditSha256']
            assert fixed['referenceVariety'] and fixed['openingPolicy'] == report['openingPolicy']
            fixed_counts, fixed_actions = Counter(), 0
            for i, game in enumerate(fixed['records']):
                assert game['number'] == i + 1 and game['focusSide'] == (1 if i % 2 == 0 else -1)
                assert game['seed'] == (fixed['baseSeed'] ^ ((i + 1) * 0x9e3779b9)) & 0x7fffffff
                assert game['opening'] == [] and game['openingPolicy'] == fixed['openingPolicy']
                fixed_actions += comparison.replay(game)
                assert all(d['depth'] == 4 for d in game['decisions'] if not d['exact'])
                fixed_counts[game['outcome']] += 1
                fixed_counts['discDifference'] += game['focusDiscDifference']
            assert all(fixed_counts[k] == fixed['results'][k] == diagnostic['results'][k]
                       for k in ('win', 'draw', 'loss', 'discDifference'))
            assert fixed_actions == fixed['replayedActions']
            assert fixed['scoreRate'] == diagnostic['scoreRate'] == (fixed_counts['win'] + fixed_counts['draw']/2)/20
        pilot_entry = release['threeSecondPilot']
        assert sha(ROOT / pilot_entry['path']) == pilot_entry['sha256']
        pilot = json.loads((ROOT / pilot_entry['path']).read_text())
        assert pilot['games'] == len(pilot['records']) == pilot_entry['games']
        assert pilot['plannedGames'] == pilot_entry['plannedGames'] == 4
        assert not pilot['complete'] and pilot_entry['status'] == 'stopped'
        assert pilot['millisecondsPerMove'] == pilot_entry['millisecondsPerMove'] == 3000
        assert pilot['maxDepth'] == 64 and pilot['start'] == 'initial' and pilot['opening'] == [] and not pilot['useBook']
        assert pilot['modelSha256'] == report['modelSha256'] and pilot['librarySha256'] == report['librarySha256']
        assert pilot['sourceSha256'] == report['sourceSha256'] and pilot['runnerSha256'] == report['runnerSha256']
        assert pilot['policyAuditSha256'] == report['policyAuditSha256']
        assert pilot['referenceVariety'] and pilot['openingPolicy'] == report['openingPolicy']
        pilot_counts, pilot_actions = Counter(), 0
        for i, game in enumerate(pilot['records']):
            assert game['number'] == i + 1 and game['focusSide'] == (1 if i % 2 == 0 else -1)
            assert game['seed'] == report['records'][i]['seed']
            assert game['opening'] == [] and game['openingPolicy'] == pilot['openingPolicy']
            pilot_actions += comparison.replay(game)
            pilot_counts[game['outcome']] += 1
            pilot_counts['discDifference'] += game['focusDiscDifference']
        assert all(pilot_counts[k] == pilot['results'][k] == pilot_entry['results'][k]
                   for k in ('win', 'draw', 'loss', 'discDifference'))
        assert pilot_actions == pilot['replayedActions']
        assert pilot['scoreRate'] == pilot_entry['scoreRate'] == (pilot_counts['win'] + pilot_counts['draw']/2)/pilot['games']
        print(f'v1.5 residual verified: same dataset, 6000 inference checks, 100 games, {actions} audited actions; stopped 3s pilot replayed')


    v16_sources = {}
    archive_manifest = ROOT / 'models/archive/v1.6-sources.json'
    if archive_manifest.exists():
        snapshot = json.loads(archive_manifest.read_text())
        assert sha(ROOT / 'models/v1.6-baseline.json') == snapshot['baselineSha256']
        assert sha(ROOT / snapshot['archive']) == snapshot['sha256']
        with tarfile.open(ROOT / snapshot['archive'], 'r:gz') as archive:
            assert len(archive.getmembers()) == len(snapshot['files'])
            for name, expected in snapshot['files'].items():
                v16_sources[name] = archive.extractfile(name).read()
                assert hashlib.sha256(v16_sources[name]).hexdigest() == expected

    def v16_sha(name):
        return hashlib.sha256(v16_sources[name]).hexdigest() if name in v16_sources else sha(ROOT / name)

    matrix_path = ROOT / 'models/v1.6-baseline.json'
    if matrix_path.exists():
        release = json.loads(matrix_path.read_text())
        assert release['version'] == 'v1.6' and release['status'] == 'frozen'
        assert all(v16_sha(name) == value for name, value in release['sourceSha256'].items())
        assert all(v16_sha(name) == value for name, value in release['artifactSha256'].items())
        entry = release['strengthReport']
        assert sha(ROOT / entry['path']) == entry['sha256']
        report = json.loads((ROOT / entry['path']).read_text())
        assert report['complete'] and report['version'] == 'v1.6' and report['experiment'] == 'search-matrix'
        assert report['games'] == report['plannedGames'] == len(report['records']) == 24
        assert report['millisecondsPerMove'] == 1000 and report['gamesPerPair'] == 4
        assert report['start'] == 'initial' and report['opening'] == [] and not report['useBook']
        assert report['librarySha256'] == release['librarySha256']
        assert all(v16_sha(name) == value for name, value in report['sourceSha256'].items())
        runner = ROOT / 'scripts/compare-search-matrix.py'
        assert sha(runner) == report['runnerSha256']
        assert sha(ROOT / 'scripts/compare-resnet.py') == report['rulesSha256']
        assert sha(ROOT / 'scripts/opening_policy.py') == report['policyAuditSha256']
        spec = importlib.util.spec_from_file_location('search_matrix_comparison', runner)
        comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparison)
        assert report['openingPolicy'] == comparison.POLICY
        assert report['configs'] == [dict(c,modelSha256=sha(ROOT/c['model'])) for c in comparison.CONFIGS]
        assert report['mcts']['maxNodes'] == 2097152 and report['mcts']['exploration'] == 1.5
        assert not report['mcts']['rollout'] and not report['mcts']['policyHead'] and report['mcts']['subtreeReuse']
        import itertools
        pairs = list(itertools.combinations(comparison.CONFIGS,2))
        actions = 0
        for i, game in enumerate(report['records']):
            assert game['number'] == i+1 and game['pair'] == i//4 and game['round'] == i%4
            left,right = pairs[i//4]
            black,white = (left,right) if i%2==0 else (right,left)
            assert game['black'] == black['id'] and game['white'] == white['id']
            assert game['seed'] == (report['baseSeed'] ^ ((game['round']//2+1)*0x9e3779b9)) & 0x7fffffff
            assert game['opening'] == [] and game['openingPolicy'] == report['openingPolicy']
            assert all(d['search'] == comparison.BY_ID[d['engine']]['search'] for d in game['decisions'])
            actions += comparison.replay(game)
        assert actions == report['replayedActions'] and report['replayedGames'] == 24
        results,matrix = comparison.aggregate(report['records'])
        assert results == report['results'] == entry['results']
        assert matrix == report['matrix'] == entry['matrix']
        assert all(r['games'] == 12 for r in results.values())
        assert sum(r['win'] for r in results.values()) == sum(r['loss'] for r in results.values())
        assert sum(r['discDifference'] for r in results.values()) == 0
        print(f'v1.6 verified: four model/search combinations, equal 1s, 24 paired-color games, {actions} audited actions')

    v16_time_sources = {}
    time_archive = ROOT / 'models/archive/v1.6-100ms-sources.json'
    if time_archive.exists():
        snapshot = json.loads(time_archive.read_text())
        assert sha(ROOT / 'models/v1.6-100ms-baseline.json') == snapshot['baselineSha256']
        assert sha(ROOT / snapshot['archive']) == snapshot['sha256']
        with tarfile.open(ROOT / snapshot['archive'], 'r:gz') as archive:
            assert len(archive.getmembers()) == len(snapshot['files'])
            for name, expected in snapshot['files'].items():
                v16_time_sources[name] = archive.extractfile(name).read()
                assert hashlib.sha256(v16_time_sources[name]).hexdigest() == expected

    def v16_time_sha(name):
        return hashlib.sha256(v16_time_sources[name]).hexdigest() if name in v16_time_sources else sha(ROOT / name)

    nnue_time_path = ROOT / 'models/v1.6-100ms-baseline.json'
    if nnue_time_path.exists():
        release = json.loads(nnue_time_path.read_text())
        assert release['version'] == 'v1.6' and release['status'] == 'frozen'
        assert all(v16_time_sha(name) == value for name, value in release['sourceSha256'].items())
        assert all(v16_time_sha(name) == value for name, value in release['artifactSha256'].items())
        entry = release['strengthReport']
        assert sha(ROOT / entry['path']) == entry['sha256']
        report = json.loads((ROOT / entry['path']).read_text())
        assert report['complete'] and report['version'] == 'v1.6' and report['experiment'] == 'nnue-search-100ms'
        assert report['games'] == report['plannedGames'] == len(report['records']) == 100
        assert report['millisecondsPerMove'] == 100 and report['gamesPerPair'] == 100
        assert report['start'] == 'initial' and report['opening'] == [] and not report['useBook']
        assert report['librarySha256'] == release['librarySha256']
        assert all(v16_time_sha(name) == value for name, value in report['sourceSha256'].items())
        runner = ROOT / 'scripts/compare-nnue-search.py'
        assert sha(runner) == report['runnerSha256']
        assert sha(ROOT / 'scripts/compare-resnet.py') == report['rulesSha256']
        assert sha(ROOT / 'scripts/opening_policy.py') == report['policyAuditSha256']
        spec = importlib.util.spec_from_file_location('nnue_search_comparison', runner)
        comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparison)
        assert report['openingPolicy'] == comparison.POLICY
        assert report['configs'] == [dict(c,modelSha256=sha(ROOT/c['model'])) for c in comparison.CONFIGS]
        assert report['mcts']['maxNodes'] == 2097152 and report['mcts']['exploration'] == 1.5
        assert not report['mcts']['rollout'] and not report['mcts']['policyHead'] and report['mcts']['subtreeReuse']
        import itertools
        pairs = list(itertools.combinations(comparison.CONFIGS,2))
        actions = 0
        for i, game in enumerate(report['records']):
            assert game['number'] == i+1 and game['pair'] == 0 and game['round'] == i
            left,right = pairs[0]
            black,white = (left,right) if i%2==0 else (right,left)
            assert game['black'] == black['id'] and game['white'] == white['id']
            assert game['seed'] == (report['baseSeed'] ^ ((game['round']//2+1)*0x9e3779b9)) & 0x7fffffff
            assert game['opening'] == [] and game['openingPolicy'] == report['openingPolicy']
            assert all(d['search'] == comparison.BY_ID[d['engine']]['search'] for d in game['decisions'])
            actions += comparison.replay(game)
        assert actions == report['replayedActions'] and report['replayedGames'] == 100
        results,matrix = comparison.aggregate(report['records'])
        assert results == report['results'] == entry['results']
        assert matrix == report['matrix'] == entry['matrix']
        assert all(r['games'] == 100 for r in results.values())
        assert sum(r['win'] for r in results.values()) == sum(r['loss'] for r in results.values())
        assert sum(r['discDifference'] for r in results.values()) == 0
        print(f'v1.6 100ms verified: same NNUE, AB vs MCTS, 100 paired-color games, {actions} audited actions')


if __name__ == '__main__':
    verify()
