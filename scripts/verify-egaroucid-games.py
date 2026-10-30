#!/usr/bin/env python3
"""Verify every aggregated shard, with bounded memory and native rule spot checks."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def popcount(v):
    v = v - ((v >> 1) & np.uint64(0x5555555555555555))
    v = (v & np.uint64(0x3333333333333333)) + ((v >> 2) & np.uint64(0x3333333333333333))
    return (((v + (v >> 4)) & np.uint64(0x0f0f0f0f0f0f0f0f)) * np.uint64(0x0101010101010101)) >> 56


def verify(args):
    started = time.monotonic()

    def await_file(path):
        while not path.exists():
            if not args.wait or time.monotonic()-started > 1800:
                raise ValueError('Missing completed artifact: ' + str(path))
            time.sleep(1)

    signature = json.loads((args.input / 'run.json').read_text())
    frozen_raw = np.fromfile(args.input / 'holdout-keys.bin', dtype='<u8').reshape(-1, 2)
    keytype = np.dtype([('own', '<u8'), ('opponent', '<u8')])
    frozen = np.empty(len(frozen_raw), dtype=keytype)
    frozen['own'], frozen['opponent'] = frozen_raw[:, 0], frozen_raw[:, 1]
    frozen = np.unique(frozen)
    lib = ctypes.CDLL(str(args.library.resolve()))
    ptr = ctypes.POINTER(ctypes.c_int)
    lib.ot_legal.argtypes = [ptr, ctypes.c_int, ptr]
    Board, Moves = ctypes.c_int*64, ctypes.c_int*64
    rng = np.random.default_rng(20261027)
    positions = observations = train = native_checks = 0
    for bucket in range(signature['buckets']):
        checkpoint = args.input / f'positions/{bucket:03d}.json'
        await_file(checkpoint)
        meta = json.loads(checkpoint.read_text())
        path = checkpoint.with_suffix('.npz')
        if sha(path) != meta['sha256'] or meta['signature'] != signature:
            raise ValueError('Shard SHA-256/signature mismatch')
        # Load each array once, rather than repeatedly decompressing NPZ members.
        with np.load(path) as archive:
            data = {k: archive[k] for k in archive.files}
        own, opponent, legal, count = (data[k] for k in ('own', 'opponent', 'legal', 'count'))
        n = len(own)
        assert n == meta['positionSplitPairs'] and (count > 0).all()
        assert ((own & opponent) == 0).all() and ((legal & (own | opponent)) == 0).all()
        assert np.array_equal(popcount(own | opponent), data['ply'].astype(np.uint64)+4)
        assert np.array_equal(data['outcome_counts'].sum(1), count)
        assert (np.abs(data['disc_sum']) <= count.astype(np.uint64)*64).all()
        assert np.isin(data['split'], [0, 1, 2]).all()
        keys = np.empty(n, dtype=keytype)
        keys['own'], keys['opponent'] = own, opponent
        loc = np.searchsorted(frozen, keys)
        is_frozen = (loc < len(frozen)) & (frozen[np.minimum(loc, len(frozen)-1)] == keys)
        assert np.array_equal(is_frozen, data['frozen'])
        eligible = (data['split'] == 0) & ~is_frozen
        assert int(eligible.sum()) == meta['trainPositions']
        offsets, moves, policy_count = (data[k] for k in ('policy_offsets', 'policy_move', 'policy_count'))
        assert len(offsets) == n+1 and offsets[0] == 0 and offsets[-1] == len(moves)
        assert (np.diff(offsets) > 0).all() and len(moves) == len(policy_count)
        assert (moves < 64).all() and (policy_count > 0).all()
        assert np.array_equal(np.add.reduceat(policy_count.astype(np.uint64), offsets[:-1].astype(np.int64)), count)
        owner = np.repeat(np.arange(n), np.diff(offsets).astype(np.int64))
        assert ((((legal[owner] >> moves.astype(np.uint64)) & 1) != 0).all())
        assert (((np.diff(moves.astype(np.int16)) > 0) | (owner[1:] != owner[:-1])).all())
        for index in rng.choice(n, min(16, n), replace=False):
            a, b = int(own[index]), int(opponent[index])
            board = Board(*[((a >> i) & 1)-((b >> i) & 1) for i in range(64)])
            output = Moves()
            size = lib.ot_legal(board, 1, output)
            assert size > 0
            assert sum(1 << square for square in output[:size]) == int(legal[index])
            native_checks += 1
        positions += n
        observations += int(count.sum())
        train += int(eligible.sum())
        (args.input / 'verification-progress.json').write_text(json.dumps(dict(
            completedBuckets=bucket+1, totalBuckets=signature['buckets'], positionSplitPairs=positions,
            observations=observations, trainPositions=train, elapsedSeconds=round(time.monotonic()-started)))+'\n')
    await_file(args.input / 'report.json')
    report = json.loads((args.input / 'report.json').read_text())
    assert positions == report['totals']['positionSplitPairs']
    assert observations == report['selection']['eligibleOccurrencesAfterTranscriptDedup']
    assert observations == report['totals']['observations']
    assert train == report['totals']['trainPositions']
    with np.load(args.input / 'evaluation/frozen.npz') as result, \
         np.load(ROOT / '.build/training/v1/positions.npz') as old, \
         np.load(ROOT / '.build/training/policy/labels.npz') as labels:
        for split in ('test', 'validation'):
            for name in ('board', 'turn', 'target'):
                assert np.array_equal(result[split+'_'+name], old[split+'_'+name])
            for name in ('move', 'legal'):
                assert np.array_equal(result[split+'_'+name], labels[split+'_'+name])
    for name, path in [('script', ROOT / 'scripts/process-egaroucid-games.py'),
                       ('converterSource', ROOT / 'scripts/expand-egaroucid.cpp')]:
        assert sha(path) == signature[name]
    checks = dict(verified=True, allShardHashesMatch=True, allPositionCountsConserved=True,
                  allPolicyCountsConserved=True, allPolicyMovesLegalAccordingToMask=True,
                  allBoardOccupanciesMatchPlacement=True, frozenFlagsExactlyMatchHoldoutKeys=True,
                  frozenEvaluationArraysUnchanged=True, nativeRuleSpotChecks=native_checks,
                  positionSampling=False, selectedGames=report['selection']['selectedGames'],
                  observations=observations, positionSplitPairs=positions, trainPositions=train,
                  diskBytes=sum(p.stat().st_size for p in args.input.rglob('*') if p.is_file()),
                  elapsedSeconds=round(time.monotonic()-started))
    (args.input / 'verification.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / '.build/training/egaroucid-v0002/full-5m-v1')
    parser.add_argument('--library', type=Path, default=ROOT / ('.build/native/libothello.'+('dylib' if sys.platform=='darwin' else 'so')))
    parser.add_argument('--wait', action='store_true', help='Wait for in-progress shards, up to 30 minutes')
    verify(parser.parse_args())
