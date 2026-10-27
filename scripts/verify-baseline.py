#!/usr/bin/env python3
"""Check the published v1 model, sources, report checksums and W/D/L totals."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    baseline = json.loads((ROOT / 'models/v1-baseline.json').read_text())
    metadata = json.loads((ROOT / 'models/nnue256-v1.json').read_text())
    assert baseline['status'] == 'frozen'
    assert sha(ROOT / baseline['model']) == baseline['modelSha256'] == metadata['sha256']
    assert sha(ROOT / baseline['protocol']) == baseline['protocolSha256']
    assert sha(ROOT / 'CMakeLists.txt') == baseline['cmakeSha256']
    assert all(sha(ROOT / name) == value for name, value in baseline['sourceSha256'].items())
    assert sha(ROOT / 'scripts/train-nnue.py') == metadata['trainingScriptSha256']
    total_games = book_hits = book_misses = 0
    for experiment in baseline['experiments']:
        path = ROOT / experiment['report']
        assert sha(path) == experiment['reportSha256']
        report = json.loads(path.read_text())
        assert report['complete'] and report['games'] == len(report['records']) == experiment['games']
        assert report['modelSha256'] == baseline['modelSha256']
        assert report['sourceSha256'] == baseline['sourceSha256']
        assert sha(ROOT / 'scripts/compare-evaluators.py') == report['comparisonScriptSha256']
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
    print(f'v1 verified: {total_games} games, unchanged model/sources, {book_hits} book hits, {book_misses} NNUE fallbacks')


if __name__ == '__main__':
    verify()
