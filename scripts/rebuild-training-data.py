#!/usr/bin/env python3
"""Rebuild v1 JSONL and caches from raw snapshots, without training a model."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def rebuild(args):
    metadata = json.loads((ROOT / 'models/v1-dataset.json').read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    games = output / 'games.jsonl'
    if args.force or not games.exists():
        subprocess.run([sys.executable, str(ROOT / 'scripts/prepare-training-data.py'),
                        '--raw', str(args.raw.resolve()), '--output', str(games),
                        '--converter', str(args.converter.resolve())], check=True)
    if digest(games) != metadata['inputSha256']:
        raise ValueError('Prepared games differ from the frozen v1 input')
    spec = importlib.util.spec_from_file_location('v1_trainer', ROOT / 'scripts/train-nnue.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    trainer.torch.set_num_threads(2)
    base = output / 'positions.npz'
    expected_base = metadata['frozenHoldout']['sha256']
    if args.force or not base.exists():
        trainer.prepare(SimpleNamespace(seed=20261027, max_games=6000, max_positions=60000,
            games=games, library=args.library, cache=base, holdout_cache=None,
            all_training_games=False, train_positions=0))
    if digest(base) != expected_base:
        raise ValueError('Rebuilt fixed holdout cache differs from v1; use the recorded NumPy/Python environment')
    expanded = output / 'v1/positions.npz'
    if args.force or not expanded.exists():
        trainer.prepare(SimpleNamespace(seed=20261027, max_games=6000, max_positions=60000,
            games=games, library=args.library, cache=expanded, holdout_cache=base,
            all_training_games=True, train_positions=1000000))
    if digest(expanded) != metadata['cacheSha256']:
        raise ValueError('Rebuilt million-position cache differs from v1')
    print(json.dumps(dict(verified=True, output=str(output), gamesSha256=digest(games),
                         holdoutCacheSha256=digest(base), v1CacheSha256=digest(expanded))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / '.build/training/raw')
    parser.add_argument('--output', type=Path, default=ROOT / '.build/training')
    parser.add_argument('--converter', type=Path, default=ROOT / '.build/native/otello_dataset')
    parser.add_argument('--library', type=Path, default=ROOT / ('.build/native/libothello.' + ('dylib' if sys.platform == 'darwin' else 'so')))
    parser.add_argument('--force', action='store_true', help='Regenerate existing derived JSONL/caches')
    rebuild(parser.parse_args())
