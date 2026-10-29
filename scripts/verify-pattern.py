#!/usr/bin/env python3
"""Verify the v1.8 artifacts, paired matches and frozen source snapshot."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pattern_comparison', ROOT/'scripts/compare-pattern.py')
comparison = importlib.util.module_from_spec(spec); sys.modules[spec.name] = comparison; spec.loader.exec_module(comparison)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    baseline = json.loads((ROOT/'models/v1.8-baseline.json').read_text())
    for name, digest in baseline['artifacts'].items():
        assert sha(ROOT/name) == digest, name
    archive = ROOT/baseline['sourceArchive']
    assert sha(archive) == baseline['sourceArchiveSha256']
    with tarfile.open(archive, 'r:gz') as stream:
        assert len(stream.getmembers()) == len(baseline['sourceSha256'])
        for name, digest in baseline['sourceSha256'].items():
            assert hashlib.sha256(stream.extractfile(name).read()).hexdigest() == digest, name
    model = json.loads((ROOT/'models/pattern32-v1.8.json').read_text())
    assert all(baseline['sourceSha256']['scripts/'+name] == digest for name, digest in model['scriptSha256'].items())
    assert sha(ROOT/'models/pattern32-v1.8.otpat') == model['sha256']
    assert model['trainPositions'] == 1000000 and model['cppParity']['positions'] == 6000
    assert model['dataset']['cacheSha256'] == json.loads((ROOT/'models/nnue256-v1.json').read_text())['dataset']['cacheSha256']
    assert model['policyDataset']['labels'] == dict(train=1000000, validation=6000, test=6000)
    report = json.loads((ROOT/'models/v1.8-pattern-matches.json').read_text())
    assert report['complete'] and report['games'] == report['plannedGames'] == len(report['records'])
    assert report['games'] == 3*report['gamesPerPair']
    for c in report['configs']:
        assert sha(ROOT/c['model']) == c['modelSha256']
    assert all(baseline['sourceSha256'][name] == digest for name, digest in report['sourceSha256'].items())
    assert all(baseline['sourceSha256'][name] == digest for name, digest in report['scriptSha256'].items())
    actions = 0
    for game in report['records']:
        actions += comparison.base.replay(game)
        assert game['seed'] == (report['baseSeed'] ^ ((game['round']//2+1)*0x9e3779b9)) & 0x7fffffff
        assert game['openingPolicy'] == report['openingPolicy']
    for pair in range(3):
        records = [r for r in report['records'] if r['pair'] == pair]
        assert len(records) == report['gamesPerPair']
        for i in range(0, len(records), 2):
            a, b = records[i:i+2]
            assert a['black'] == b['white'] and a['white'] == b['black'] and a['seed'] == b['seed']
    totals, matrix = comparison.base.aggregate(report['records'])
    assert totals == report['results'] and matrix == report['matrix']
    assert actions == report['replayedActions'] and report['replayedGames'] == report['games']
    print(f"v1.8 verified: {report['games']} paired-color games, {actions} independently replayed actions, all model/source hashes, frozen million-position cache and 6000 three-head inference checks")


if __name__ == '__main__':
    verify()
