#!/usr/bin/env python3
"""Validate all official v0002 games and build a bounded, separate training cache."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import time
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DTYPE = np.dtype([('own', '<u8'), ('opp', '<u8'), ('legal', '<u8'), ('game', '<u8'),
                  ('move', 'u1'), ('outcome', 'i1'), ('ply', 'u1'), ('random', 'u1'),
                  ('disc', 'i1'), ('padding', 'u1', (3,))])


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def canonical_keys(boards, turns):
    relative = boards * turns.astype(np.int8)[:, None]
    idx = np.arange(64).reshape(8, 8)
    best = None
    for mirror in (False, True):
        for rotation in range(4):
            # Same D4 set as the native validator; choose numeric (own, opponent).
            p = np.rot90(np.fliplr(idx) if mirror else idx, rotation).reshape(64)
            b = relative[:, p]
            own = np.packbits(b == 1, axis=1, bitorder='little').copy().view('<u8').reshape(-1)
            opp = np.packbits(b == -1, axis=1, bitorder='little').copy().view('<u8').reshape(-1)
            if best is None:
                best = np.stack((own, opp), axis=1)
            else:
                mask = (own < best[:, 0]) | ((own == best[:, 0]) & (opp < best[:, 1]))
                best[mask, 0], best[mask, 1] = own[mask], opp[mask]
    return best.astype('<u8')


def build_cache(args, groups, frozen):
    arrays = [np.fromfile(args.output / f'groups/{n:02d}.samples.bin', dtype=DTYPE) for n in sorted(groups)]
    rows = np.concatenate(arrays)
    del arrays
    if not len(rows):
        raise ValueError('No eligible sampled positions')
    if np.any(rows['ply'] < rows['random']):
        raise ValueError('Random-prefix labels reached cache')
    # A trajectory may appear in several random-prefix groups. Count it once per
    # board using its recorded 64-bit transcript fingerprint, not once per copy.
    trajectory_keys = rows[['own', 'opp', 'game']]
    _, indices = np.unique(trajectory_keys, return_index=True)
    original = len(rows)
    rows = rows[indices]
    keys, inverse, counts = np.unique(rows[['own', 'opp']], return_inverse=True, return_counts=True)
    n = len(keys)
    outcomes = np.zeros((n, 3), dtype=np.uint32)
    policy = np.zeros((n, 65), dtype=np.uint32)
    np.add.at(outcomes, (inverse, rows['outcome'].astype(int) + 1), 1)
    np.add.at(policy, (inverse, rows['move']), 1)
    disc_sum = np.bincount(inverse, weights=rows['disc'], minlength=n)
    first = np.zeros(n, dtype=np.int64)
    # Last row is sufficient: same canonical board implies same legal mask/stage.
    first[inverse] = np.arange(len(rows))
    legal_bits = rows['legal'][first]
    ply = rows['ply'][first]
    for bit in range(64):
        if np.any((policy[:, bit] > 0) & (((legal_bits >> np.uint64(bit)) & 1) == 0)):
            raise ValueError('Aggregated policy includes illegal moves')
    expected = np.concatenate([canonical_keys(frozen[s + '_board'], frozen[s + '_turn']) for s in ('validation', 'test')])
    blocked = {(int(a), int(b)) for a, b in expected}
    if any((int(k['own']), int(k['opp'])) in blocked for k in keys):
        raise ValueError('Frozen holdout overlap after canonicalization')
    # Reproducible uniform selection within each placement stage, independent of label.
    rng = np.random.default_rng(args.seed)
    selected = np.concatenate([rng.choice(np.flatnonzero(ply == p),
                               min(args.positions_per_stage, int((ply == p).sum())), replace=False)
                               for p in sorted(np.unique(ply))])
    selected = selected[rng.permutation(len(selected))]
    keys, counts, outcomes, policy = keys[selected], counts[selected], outcomes[selected], policy[selected]
    legal_bits, ply = legal_bits[selected], ply[selected]
    shifts = np.arange(64, dtype=np.uint64)
    board = (((keys['own'][:, None] >> shifts) & 1).astype(np.int8) -
             ((keys['opp'][:, None] >> shifts) & 1).astype(np.int8))
    legal = np.zeros((len(board), 65), dtype=np.bool_)
    legal[:, :64] = ((legal_bits[:, None] >> shifts) & 1).astype(bool)
    data = dict(train_board=board, train_turn=np.ones(len(board), dtype=np.float32),
                train_target=(disc_sum[selected] / counts / 64).astype(np.float32),
                train_outcome=((outcomes[:, 2].astype(float) - outcomes[:, 0]) / counts).astype(np.float32),
                train_count=counts.astype(np.uint32), train_outcome_counts=outcomes, train_ply=ply)
    labels = dict(train_move=policy.argmax(1).astype(np.int16), train_legal=legal,
                  train_policy=(policy / counts[:, None]).astype(np.float32))
    original_labels = np.load(args.frozen_labels)
    for s in ('validation', 'test'):
        for key in ('board', 'turn', 'target'):
            data[s + '_' + key] = frozen[s + '_' + key]
        data[s + '_outcome'] = np.sign(frozen[s + '_target']).astype(np.float32)
        for key in ('move', 'legal'):
            labels[s + '_' + key] = original_labels[s + '_' + key]
    args.output.joinpath('cache').mkdir(exist_ok=True)
    for name, values in (('positions', data), ('labels', labels)):
        temporary = args.output / f'cache/{name}.tmp.npz'
        np.savez_compressed(temporary, **values)
        temporary.replace(args.output / f'cache/{name}.npz')
    report = dict(positions=len(board), sampledOccurrences=original,
                  uniqueSampledTrajectoriesPerBoard=len(rows), uniqueSampledBoards=n,
                  trajectoryDuplicatesRemoved=original-len(rows),
                  positionsByPlacement={str(p): int((ply == p).sum()) for p in np.unique(ply)},
                  retainedObservations=int(counts.sum()),
                  boardsWithMultipleOutcomes=int(((outcomes > 0).sum(1) > 1).sum()),
                  boardsWithMultipleMoves=int(((policy > 0).sum(1) > 1).sum()),
                  holdoutOverlaps=0, randomPrefixViolations=0,
                  softOutcomeMean=float(data['train_outcome'].mean()),
                  hashes={name: sha(args.output / f'cache/{name}.npz') for name in ('positions', 'labels')})
    save(args.output / 'cache/report.json', report)
    return report


def prepare(args):
    if args.workers < 1 or args.sample_per_stage < 1 or args.positions_per_stage < 1:
        raise ValueError('Workers and sampling limits must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'groups').mkdir(exist_ok=True)
    manifest_path = args.raw / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['version'] != 'egaroucid-v0002':
        raise ValueError('Wrong source manifest')
    metadata = json.loads(args.frozen_cache.with_suffix('.json').read_text())
    if sha(args.frozen_cache) != metadata['cacheSha256']:
        raise ValueError('Frozen position cache checksum mismatch')
    label_meta = json.loads(args.frozen_labels.with_suffix('.json').read_text())
    if sha(args.frozen_labels) != label_meta['cacheSha256'] or label_meta['positionsSha256'] != sha(args.frozen_cache):
        raise ValueError('Frozen policy cache checksum mismatch')
    frozen = dict(np.load(args.frozen_cache))
    holdout = np.concatenate([canonical_keys(frozen[s + '_board'], frozen[s + '_turn']) for s in ('validation', 'test')])
    holdout_path = args.output / 'holdout-keys.bin'
    holdout_path.write_bytes(holdout.tobytes())
    signature = dict(converter=sha(args.converter), converterSource=sha(ROOT / 'scripts/prepare-egaroucid.cpp'),
                     script=sha(Path(__file__)), frozenPositions=sha(args.frozen_cache),
                     frozenLabels=sha(args.frozen_labels), holdoutKeys=sha(holdout_path),
                     samplePerStage=args.sample_per_stage, positionsPerStage=args.positions_per_stage,
                     seed=args.seed)
    started = time.monotonic()
    reports = {}
    tasks = []
    found_groups = set()
    for entry in manifest['files']:
        archive_path = args.raw / entry['file']
        if archive_path.stat().st_size != entry['bytes'] or sha(archive_path) != entry['sha256']:
            raise ValueError('Archive checksum mismatch: ' + entry['file'])
        with zipfile.ZipFile(archive_path) as archive:
            members = defaultdict(list)
            for info in archive.infolist():
                path = PurePosixPath(info.filename)
                if info.is_dir() or path.suffix != '.txt':
                    continue
                if not path.parent.name.isdigit() or not 8 <= int(path.parent.name) <= 59:
                    raise ValueError('Unexpected transcript path: ' + info.filename)
                members[int(path.parent.name)].append(info)
            for number in sorted(members):
                if number in found_groups or len(members[number]) != 100:
                    raise ValueError('Unexpected group layout')
                found_groups.add(number)
                names = [info.filename for info in sorted(members[number], key=lambda i: i.filename)]
                tasks.append((number, names, archive_path, entry))
            if 'README_EN.md' in archive.namelist():
                (args.output / 'source-README_EN.md').write_bytes(archive.read('README_EN.md'))

    def process_group(task):
        number, names, archive_path, entry = task
        output = args.output / f'groups/{number:02d}'
        checkpoint = output.with_suffix('.json')
        if checkpoint.exists():
            old = json.loads(checkpoint.read_text())
            if old['signature'] != signature or old['archiveSha256'] != entry['sha256']:
                raise ValueError('Existing group uses different inputs; choose a new output directory')
            for suffix, digest in old['hashes'].items():
                if sha(output.with_suffix(suffix)) != digest:
                    raise ValueError('Existing group checksum mismatch')
            return old
        print(json.dumps(dict(startGroup=number)), flush=True)
        clean = output.with_suffix('.games.bin.tmp')
        samples = output.with_suffix('.samples.bin.tmp')
        command = [str(args.converter.resolve()), str(number), str(clean), str(samples),
                   str(holdout_path), str(args.sample_per_stage), str(args.seed)]
        with zipfile.ZipFile(archive_path) as archive:
            with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
                try:
                    for name in names:
                        with archive.open(name) as stream:
                            while block := stream.read(1024 * 1024):
                                process.stdin.write(block)
                    process.stdin.close()
                    stdout = process.stdout.read()
                    stderr = process.stderr.read()
                    code = process.wait()
                except BaseException:
                    process.kill()
                    process.wait()
                    raise
        if code:
            raise RuntimeError(stderr.decode())
        result = json.loads(stdout)
        if result['records'] != 1000000:
            raise ValueError('Expected one million records per random-prefix group')
        if clean.stat().st_size != result['accepted'] * 64 or samples.stat().st_size != result['sampled'] * DTYPE.itemsize:
            raise ValueError('Binary record length mismatch')
        clean.replace(output.with_suffix('.games.bin'))
        samples.replace(output.with_suffix('.samples.bin'))
        result.update(signature=signature, archiveSha256=entry['sha256'], randomPlies=number,
                      hashes={suffix: sha(output.with_suffix(suffix)) for suffix in ('.games.bin', '.samples.bin')})
        save(checkpoint, result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('signature', 'hashes')}), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_group, task): task[0] for task in tasks}
        for future in as_completed(futures):
            number = futures[future]
            reports[number] = future.result()
            save(args.output / 'progress.json', dict(status='validating', lastCompletedGroup=number,
                 completedGroups=len(reports), totalGroups=52,
                 acceptedGames=sum(r['accepted'] for r in reports.values()),
                 elapsedSeconds=round(time.monotonic()-started)))
    if set(reports) != set(range(8, 60)):
        raise ValueError('Missing random-prefix groups')
    save(args.output / 'progress.json', dict(status='building-cache', completedGroups=52, totalGroups=52))
    cache = build_cache(args, reports, frozen)
    rejected = Counter()
    for r in reports.values():
        rejected.update(r['rejected'])
    totals = {key: sum(r[key] for r in reports.values()) for key in
              ('records', 'accepted', 'sampled', 'excludedHoldout', 'forcedPasses', 'earlyTerminal')}
    report = dict(version='egaroucid-v0002-clean-v1', status='prepared-not-trained',
                  source=manifest, signature=signature, totals=totals, rejected=dict(rejected), cache=cache,
                  split='Frozen v1 validation/test unchanged. New training uses only canonical-first-8-move '
                        'FNV train buckets; additionally exclude exact D4/color-relative holdout boards.',
                  targets='train_target: mean raw final disc difference /64; train_outcome: mean WDL '
                          '(+1/0/-1); train_policy: empirical next-move distribution among sampled '
                          'distinct transcript fingerprints. Never sign(train_target) for soft outcome training.',
                  sampling='Exclude first N random placements in group N. Bernoulli sample by full '
                           'transcript fingerprint and placement, then cap unique boards per placement. '
                           'Aggregates cover sampled observations, not every occurrence in all 52M games. '
                           'Only the bounded cache is deduplicated; full binary corpus retains repeated games.',
                  binaryFormat='games.bin: 60 uint8 squares (255 padding), uint8 random prefix, uint8 '
                               'length, int8 raw black disc difference, uint8 split (0 train/1 validation/2 test). '
                               'Passes implicit; use legality to infer. samples.bin: see DTYPE in script.',
                  finishedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  elapsedSeconds=round(time.monotonic()-started))
    save(args.output / 'report.json', report)
    save(args.output / 'cache/positions.json', dict(report, cacheSha256=cache['hashes']['positions']))
    save(args.output / 'cache/labels.json', dict(cacheSha256=cache['hashes']['labels'],
         positionsSha256=cache['hashes']['positions'], valueTarget='separate train_outcome soft label'))
    save(args.output / 'progress.json', dict(status='complete', completedGroups=52, totalGroups=52,
                                           totals=totals, positions=cache['positions']))
    print(json.dumps(dict(complete=True, totals=totals, cache=cache)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / '.build/training/egaroucid-v0002/raw')
    parser.add_argument('--output', type=Path, default=ROOT / '.build/training/egaroucid-v0002/clean-v1')
    parser.add_argument('--converter', type=Path, default=ROOT / '.build/training/egaroucid-v0002/prepare-egaroucid')
    parser.add_argument('--frozen-cache', type=Path, default=ROOT / '.build/training/v1/positions.npz')
    parser.add_argument('--frozen-labels', type=Path, default=ROOT / '.build/training/policy/labels.npz')
    parser.add_argument('--sample-per-stage', type=int, default=60000)
    parser.add_argument('--positions-per-stage', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--workers', type=int, default=4)
    prepare(parser.parse_args())
