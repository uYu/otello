#!/usr/bin/env python3
"""Exhaustively expand and aggregate every non-random position in a fixed game subset."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GAME = np.dtype([('moves', 'u1', (60,)), ('random', 'u1'), ('length', 'u1'),
                 ('disc', 'i1'), ('split', 'u1')])
OBS = np.dtype([('own', '<u8'), ('opp', '<u8'), ('legal', '<u8'), ('move', 'u1'),
                ('outcome', 'i1'), ('disc', 'i1'), ('ply', 'u1'), ('split', 'u1'),
                ('frozen', 'u1'), ('padding', 'u1', (2,))])


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def select_games(args, signature):
    target = args.output / 'selection'
    target.mkdir(exist_ok=True)
    checkpoint = target / 'report.json'
    if checkpoint.exists():
        result = json.loads(checkpoint.read_text())
        if result['signature'] != signature:
            raise ValueError('Existing selection uses different inputs/options; choose a fresh output')
        for name, digest in result['hashes'].items():
            if sha(target / name) != digest:
                raise ValueError('Selection checksum mismatch: ' + name)
        return result
    arrays, source_groups, source_rows, source_hashes = [], [], [], {}
    quotas = {}
    for number in range(8, 60):
        source = args.input / f'groups/{number:02d}.games.bin'
        meta = json.loads(source.with_suffix('').with_suffix('.json').read_text())
        digest = sha(source)
        if digest != meta['hashes']['.games.bin']:
            raise ValueError('Source binary hash mismatch')
        source_hashes[str(number)] = digest
        data = np.memmap(source, mode='r', dtype=GAME)
        if len(data) != 1000000:
            raise ValueError('Expected one million source games per group')
        quota = args.games // 52 + (number - 8 < args.games % 52)
        # Sorted row indices avoid random disk reads; selection itself is uniform.
        indices = np.sort(np.random.default_rng(args.seed + number).choice(len(data), quota, replace=False))
        arrays.append(np.array(data[indices]))
        source_groups.append(np.full(quota, number, dtype=np.uint8))
        source_rows.append(indices.astype(np.uint32))
        quotas[str(number)] = quota
    records = np.concatenate(arrays)
    del arrays
    if len(records) != args.games:
        raise ValueError('Selection size mismatch')
    original_occurrences = int((records['length'].astype(np.int64) - records['random']).sum())
    np.savez_compressed(target / 'source-indices.npz', group=np.concatenate(source_groups),
                        row=np.concatenate(source_rows))
    # Compare all 60 actual move bytes, not a fingerprint. Identical transcripts
    # from different random-prefix folders contribute the union of eligible plies.
    keys = np.ascontiguousarray(records['moves']).view('V60').reshape(-1)
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    unique = records[first].copy()
    minimum = np.full(len(unique), 255, dtype=np.uint8)
    np.minimum.at(minimum, inverse, records['random'])
    unique['random'] = minimum
    for field in ('length', 'disc', 'split'):
        if np.any(records[field] != unique[field][inverse]):
            raise ValueError('Identical transcript has conflicting metadata')
    del inverse, keys, records
    group_counts = {}
    for number in range(8, 60):
        selected = unique[unique['random'] == number]
        selected.tofile(target / f'{number:02d}.games.bin')
        group_counts[str(number)] = len(selected)
    result = dict(signature=signature, requestedGames=args.games, selectedGames=args.games,
                  selection='Uniform without replacement within each of the 52 random-prefix groups; '
                            'NumPy default_rng(seed + group), exact row indices retained.',
                  quotas=quotas, inputGroupHashes=source_hashes, uniqueTranscripts=len(unique),
                  duplicateTranscriptsRemoved=args.games-len(unique),
                  eligibleOccurrencesBeforeTranscriptDedup=original_occurrences,
                  eligibleOccurrencesAfterTranscriptDedup=int((unique['length'].astype(np.int64)-unique['random']).sum()),
                  transcriptDedup='Exact 60-byte padded move sequence, no hash collisions; '
                                  'minimum random-prefix length across duplicate copies. No D4 game dedup.',
                  uniqueTranscriptsByRandomPrefix=group_counts,
                  hashes={p.name: sha(p) for p in sorted(target.iterdir()) if p.suffix in ('.bin', '.npz')})
    save(checkpoint, result)
    return result


def expand_group(args, number, selection, signature):
    target = args.output / f'observations/{number:02d}'
    checkpoint = target.with_suffix('.json')
    if checkpoint.exists():
        result = json.loads(checkpoint.read_text())
        if result['signature'] != signature:
            raise ValueError('Expansion signature mismatch')
        for name, digest in result['hashes'].items():
            if sha(target / name) != digest:
                raise ValueError('Observation checksum mismatch')
        return result
    partial = target.with_suffix('.partial')
    source = args.output / f'selection/{number:02d}.games.bin'
    result = subprocess.run([str(args.converter.resolve()), str(source), str(partial),
                             str(args.output / 'holdout-keys.bin'), str(args.buckets)],
                            capture_output=True, text=True, check=True)
    result = json.loads(result.stdout)
    if result['games'] != selection['uniqueTranscriptsByRandomPrefix'][str(number)]:
        raise ValueError('Expansion game count mismatch')
    actual = sum(p.stat().st_size for p in partial.glob('*.bin'))
    if actual != result['positions'] * OBS.itemsize:
        raise ValueError('Observation record length mismatch')
    partial.replace(target)
    result.update(signature=signature, randomPrefix=number,
                  hashes={p.name: sha(p) for p in target.glob('*.bin')})
    save(checkpoint, result)
    return result


def aggregate_rows(rows):
    """Exact position/split grouping with sparse policy counts; no position cap."""
    if not len(rows):
        raise ValueError('Empty observation bucket')
    if np.any(rows['move'] >= 64) or np.any(~np.isin(rows['outcome'], [-1, 0, 1])):
        raise ValueError('Invalid observation label')
    keys, first, inverse, counts = np.unique(rows[['split', 'own', 'opp']],
                                           return_index=True, return_inverse=True, return_counts=True)
    n = len(keys)
    representatives = rows[first]
    for field in ('legal', 'ply', 'frozen'):
        if np.any(rows[field] != representatives[field][inverse]):
            raise ValueError('Same position has inconsistent ' + field)
    observed_legal = ((rows['legal'] >> rows['move'].astype(np.uint64)) & 1) != 0
    if not observed_legal.all():
        raise ValueError('Illegal policy observation')
    outcomes = np.stack([np.bincount(inverse, weights=(rows['outcome'] == value), minlength=n)
                         for value in (-1, 0, 1)], axis=1).astype(np.uint32)
    disc_sum = np.bincount(inverse, weights=rows['disc'], minlength=n).astype(np.int64)
    policy_ids, policy_count = np.unique(inverse.astype(np.uint64)*64 + rows['move'], return_counts=True)
    policy_board = (policy_ids // 64).astype(np.int64)
    policy_n = np.bincount(policy_board, minlength=n)
    offsets = np.concatenate((np.zeros(1, dtype=np.uint64), np.cumsum(policy_n, dtype=np.uint64)))
    if not np.array_equal(np.bincount(policy_board, weights=policy_count, minlength=n), counts):
        raise ValueError('Policy count conservation failure')
    if not np.array_equal(outcomes.sum(1), counts):
        raise ValueError('Outcome count conservation failure')
    frozen = representatives['frozen'].astype(bool)
    train = (keys['split'] == 0) & ~frozen
    arrays = dict(own=keys['own'], opponent=keys['opp'], legal=representatives['legal'],
                  count=counts.astype(np.uint32), outcome_counts=outcomes, disc_sum=disc_sum,
                  ply=representatives['ply'], split=keys['split'], frozen=frozen,
                  policy_offsets=offsets, policy_move=(policy_ids % 64).astype(np.uint8),
                  policy_count=policy_count.astype(np.uint32))
    distinct_boards = len(np.unique(keys[['own', 'opp']]))
    stats = dict(observations=len(rows), positionSplitPairs=n, distinctBoards=distinct_boards,
                 trainPositions=int(train.sum()), trainObservations=int(counts[train].sum()),
                 frozenPositionSplitPairs=int(frozen.sum()), frozenObservations=int(counts[frozen].sum()),
                 splitPositions=[int((keys['split'] == s).sum()) for s in range(3)],
                 splitObservations=[int(counts[keys['split'] == s].sum()) for s in range(3)],
                 trainPositionsByPlacement={str(p): int((train & (representatives['ply'] == p)).sum())
                                            for p in np.unique(representatives['ply'])},
                 multipleOutcomePositionSplitPairs=int(((outcomes > 0).sum(1) > 1).sum()),
                 multipleMovePositionSplitPairs=int((policy_n > 1).sum()), policyEntries=len(policy_ids))
    return arrays, stats


def reduce_bucket(args, bucket, signature):
    folder = args.output / 'positions'
    checkpoint = folder / f'{bucket:03d}.json'
    path = folder / f'{bucket:03d}.npz'
    if checkpoint.exists():
        result = json.loads(checkpoint.read_text())
        if result['signature'] != signature or sha(path) != result['sha256']:
            raise ValueError('Aggregated shard checksum/signature mismatch')
        return result
    pieces = [np.fromfile(args.output / f'observations/{number:02d}/{bucket}.bin', dtype=OBS)
              for number in range(8, 60)]
    rows = np.concatenate(pieces)
    del pieces
    arrays, result = aggregate_rows(rows)
    temporary = folder / f'{bucket:03d}.tmp.npz'
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    result.update(signature=signature, bucket=bucket, bytes=path.stat().st_size, sha256=sha(path))
    save(checkpoint, result)
    return result


def run(args):
    if not 1 <= args.games <= 52000000 or not 1 <= args.buckets <= 128 or args.workers < 1:
        raise ValueError('Invalid size or worker settings')
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('observations', 'positions', 'evaluation'):
        (args.output / name).mkdir(exist_ok=True)
    source_report = json.loads((args.input / 'report.json').read_text())
    if source_report['totals']['accepted'] != 52000000:
        raise ValueError('Expected the verified complete source corpus')
    holdout = args.input / 'holdout-keys.bin'
    if sha(holdout) != source_report['signature']['holdoutKeys']:
        raise ValueError('Frozen holdout keys mismatch')
    shutil.copyfile(holdout, args.output / 'holdout-keys.bin')
    signature = dict(script=sha(Path(__file__)), converterSource=sha(ROOT / 'scripts/expand-egaroucid.cpp'),
                     converter=sha(args.converter), sourceReport=sha(args.input / 'report.json'),
                     frozenKeys=sha(holdout), games=args.games, seed=args.seed, buckets=args.buckets)
    existing = args.output / 'run.json'
    if existing.exists() and json.loads(existing.read_text()) != signature:
        raise ValueError('Existing output has different processing inputs; choose a fresh directory')
    save(existing, signature)
    started = time.monotonic()

    def progress(phase, **extra):
        state = dict(phase=phase, requestedGames=args.games, elapsedSeconds=round(time.monotonic()-started), **extra)
        save(args.output / 'progress.json', state)
        print(json.dumps(state), flush=True)

    progress('selecting-games')
    selection = select_games(args, signature)
    progress('expanding-all-positions', selection=selection['uniqueTranscripts'], completedGroups=0, totalGroups=52)
    expansion = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(expand_group, args, number, selection, signature): number for number in range(8, 60)}
        for future in as_completed(futures):
            expansion[futures[future]] = future.result()
            progress('expanding-all-positions', completedGroups=len(expansion), totalGroups=52,
                     games=sum(r['games'] for r in expansion.values()),
                     positions=sum(r['positions'] for r in expansion.values()))
    expected = selection['eligibleOccurrencesAfterTranscriptDedup']
    if sum(r['positions'] for r in expansion.values()) != expected:
        raise ValueError('Not all eligible occurrences were expanded')
    reduced = {}
    progress('aggregating-all-positions', observations=expected, completedBuckets=0, totalBuckets=args.buckets)
    # Limit aggregation memory independently of native expansion parallelism.
    with ThreadPoolExecutor(max_workers=min(2, args.workers)) as pool:
        futures = {pool.submit(reduce_bucket, args, bucket, signature): bucket for bucket in range(args.buckets)}
        for future in as_completed(futures):
            reduced[futures[future]] = future.result()
            progress('aggregating-all-positions', completedBuckets=len(reduced), totalBuckets=args.buckets,
                     observations=sum(r['observations'] for r in reduced.values()),
                     distinctBoards=sum(r['distinctBoards'] for r in reduced.values()),
                     trainPositions=sum(r['trainPositions'] for r in reduced.values()))
    if sum(r['observations'] for r in reduced.values()) != expected:
        raise ValueError('Not all observations were aggregated')
    # Keep the original evaluation labels unchanged in a small separate artifact.
    cache = ROOT / '.build/training/v1/positions.npz'
    labels = ROOT / '.build/training/policy/labels.npz'
    if sha(cache) != source_report['signature']['frozenPositions'] or sha(labels) != source_report['signature']['frozenLabels']:
        raise ValueError('Frozen evaluation source hashes changed')
    with np.load(cache) as data, np.load(labels) as policy:
        evaluation = {s+'_'+k: data[s+'_'+k] for s in ('test', 'validation') for k in ('board', 'turn', 'target')}
        evaluation.update({s+'_'+k: policy[s+'_'+k] for s in ('test', 'validation') for k in ('move', 'legal')})
    np.savez_compressed(args.output / 'evaluation/frozen.npz', **evaluation)
    totals = {k: sum(r[k] for r in reduced.values()) for k in
              ('observations', 'positionSplitPairs', 'distinctBoards', 'trainPositions', 'trainObservations',
               'frozenPositionSplitPairs', 'frozenObservations', 'multipleOutcomePositionSplitPairs',
               'multipleMovePositionSplitPairs', 'policyEntries', 'bytes')}
    totals['splitPositions'] = [sum(r['splitPositions'][s] for r in reduced.values()) for s in range(3)]
    totals['splitObservations'] = [sum(r['splitObservations'][s] for r in reduced.values()) for s in range(3)]
    totals['trainPositionsByPlacement'] = {str(p): sum(r['trainPositionsByPlacement'].get(str(p), 0) for r in reduced.values()) for p in range(8, 60)}
    result = dict(version='egaroucid-exhaustive-5m-v1', status='complete-not-trained', signature=signature,
                  selection=selection, totals=totals, positionSampling=False, positionCap=None,
                  source=source_report['source'],
                  aggregation='All non-random occurrences after exact transcript dedup. '
                              'Exact D4/side-to-move relative board keys. Separate counters for each opening split; '
                              'never pool labels across splits. No position or stage sampling/cap.',
                  trainingFilter='Use only split == 0 and frozen == false. Other splits retained for provenance '
                                 'and analysis, not as replacement evaluation sets.',
                  targets='Mean outcome=(win-loss)/count. Mean raw disc value=disc_sum/(64*count). '
                          'Sparse policy probabilities=policy_count/count with CSR policy_offsets.',
                  format='NPZ shards: own/opponent/ legal uint64 bitboards; count uint32; outcome_counts '
                         'Nx3 uint32 (loss/draw/win); disc_sum int64; ply/split uint8; frozen bool; '
                         'policy_offsets uint64 N+1; policy_move uint8; policy_count uint32.',
                  frozenEvaluationSha256=sha(args.output / 'evaluation/frozen.npz'),
                  completedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  elapsedSeconds=round(time.monotonic()-started))
    save(args.output / 'report.json', result)
    progress('complete', selectedGames=args.games, uniqueTranscripts=selection['uniqueTranscripts'], **totals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / '.build/training/egaroucid-v0002/clean-v1')
    parser.add_argument('--output', type=Path, default=ROOT / '.build/training/egaroucid-v0002/full-5m-v1')
    parser.add_argument('--converter', type=Path, default=ROOT / '.build/training/egaroucid-v0002/expand-egaroucid')
    parser.add_argument('--games', type=int, default=5000000)
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--buckets', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    run(parser.parse_args())
