#!/usr/bin/env python3
"""Restore causal internal Elo and collect only the low-rated actor's moves."""
import collections
import ctypes
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import numpy as np
from training_consensus import canonical, rate_records

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.build/training/beginner-low-elo1200-v1'
REPORT = ROOT / 'models/beginner-low-elo1200-v1-data.json'
THRESHOLD = 1200.
MIN_PRIOR = 20


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if REPORT.exists() or (WORK/'positions.npz').exists():
        raise ValueError('Preserve existing low-Elo data candidate')
    started = time.perf_counter()
    WORK.mkdir(parents=True, exist_ok=True)
    reference = json.loads((ROOT/'models/v1.40-training-consensus-elo.json').read_text())
    games_path = ROOT/'.build/training/games.jsonl'
    expected_games = reference['preservedSha256']['.build/training/games.jsonl']
    if sha(games_path) != expected_games:
        raise ValueError('Source games differ from the previous Elo experiment')
    manifest = json.loads((ROOT/'.build/training/raw/manifest.json').read_text())
    raw, raw_hashes = {}, {}
    for item in manifest['files']:
        path = ROOT/'.build/training/raw'/item['file']
        if path.suffix != '.wtb':
            continue
        if sha(path) != item['sha256']:
            raise ValueError('WTHOR raw checksum mismatch')
        raw_hashes[path.name] = item['sha256']
        binary = path.read_bytes()
        assert len(binary) == 16+struct.unpack_from('<I', binary, 4)[0]*68
        for index, offset in enumerate(range(16, len(binary), 68)):
            _, black, white = struct.unpack_from('<HHH', binary, offset)
            moves = bytes((v//10-1)*8+v%10-1 for v in binary[offset+8:offset+68] if v)
            raw.setdefault((path.name, moves), dict(year=int(path.stem[-4:]), rawRecordIndex=index,
                                                   blackPlayerIndex=black, whitePlayerIndex=white))
    records, games = [], {}
    for line in games_path.open():
        game = json.loads(line)
        if game['source'] != 'wthor':
            continue
        metadata = raw[(game['sourceFile'], bytes(m for m in game['moves'] if m >= 0))]
        records.append(dict(gameId=game['id'], split=game['split'], complete=game['complete'],
                            blackResult=game['blackResult'], **metadata))
        games[game['id']] = game
    # Same namespace size and exact unchanged Elo routine as v1.40; names are not needed.
    size = reference['elo']['knownNamesInitialized']+1
    ledger, full, train, _ = rate_records(records, size)
    assert len(records) == 137507
    assert sum(row['rated'] for row in ledger) == reference['elo']['ratedCompleteGames']
    assert int(train.games.sum()//2) == reference['elo']['trainOnlyRatedGames']
    expected_quantiles = list(reference['elo']['fullCorpusRatingQuantiles'].values())
    np.testing.assert_allclose(np.quantile(full.rating[full.games > 0], [0,.25,.5,.75,1]), expected_quantiles, atol=1e-10, rtol=0)
    ledger_path = WORK/'game-elo.jsonl'
    with ledger_path.open('w') as stream:
        for row in ledger:
            stream.write(json.dumps(row)+'\n')
    eligible = collections.Counter()
    for row in ledger:
        if row['split'] == 'train' and row['complete']:
            for color in ('black', 'white'):
                rating = row[color+'TrainYearStartElo']
                support = row[color+'TrainYearStartGames']
                if rating is not None and support >= MIN_PRIOR:
                    for threshold in (1200,1250,1300,1350,1400,1450,1500):
                        if rating < threshold:
                            eligible[threshold] += 1
    print(json.dumps(dict(stage='Elo exactly reproduced', qualifyingActorGamesByThreshold=eligible)), flush=True)
    suffix = 'dylib' if sys.platform == 'darwin' else 'so'
    library = ROOT/f'.build/native/libothello.{suffix}'
    lib = ctypes.CDLL(str(library))
    ptr = ctypes.POINTER(ctypes.c_int)
    lib.ot_play.argtypes = [ptr, ctypes.c_int, ctypes.c_int, ptr]
    lib.ot_legal.argtypes = [ptr, ctypes.c_int, ptr]
    cells_type, out_type = ctypes.c_int*64, ctypes.c_int*136
    rows = {split: [] for split in ('train', 'validation', 'test')}
    selected_actor_games = {split: set() for split in rows}
    provenance = []
    for row in ledger:
        if not row['complete']:
            continue
        qualifying = {}
        for turn, color in ((1,'black'),(-1,'white')):
            rating = row[color+'TrainYearStartElo']
            qualifying[turn] = rating is not None and rating < THRESHOLD and row[color+'TrainYearStartGames'] >= MIN_PRIOR
        if not any(qualifying.values()):
            continue
        game = games[row['gameId']]
        cells = cells_type(); cells[28] = cells[35] = 1; cells[27] = cells[36] = -1
        turn, ply = 1, 0
        for move in game['moves']:
            if move >= 0 and 8 <= ply <= 56 and ply % 3 == 0 and qualifying[turn]:
                output = out_type(); count = lib.ot_legal(cells, turn, output)
                mask = np.zeros(64, np.bool_); mask[list(output[:count])] = True
                assert mask[move]
                board = np.array(cells, dtype=np.int8)*turn
                color = 'black' if turn == 1 else 'white'
                actor = row[color+'PlayerIndex']
                rows[row['split']].append((board, move, mask, row['gameId'], actor,
                    row[color+'TrainYearStartElo'], row[color+'TrainYearStartGames']))
                selected_actor_games[row['split']].add((row['gameId'], turn))
            output = out_type()
            assert lib.ot_play(cells, turn, move, output) == 1
            cells = cells_type(*output[:64]); turn = output[64]; ply += move >= 0
    cache = np.load(ROOT/'.build/training/positions.npz')
    holdout = np.concatenate([canonical(cache[s+'_board'], cache[s+'_turn'])[0] for s in ('validation','test')])
    blocked = set(holdout.tobytes()[i:i+16] for i in range(0, holdout.nbytes, 16))
    dataset = {}; rejected = {}
    for split in ('test','validation','train'):
        candidates = rows[split]
        boards = np.stack([r[0] for r in candidates])
        keys = canonical(boards, np.ones(len(boards), np.int8))[0]
        # Keep repeated training occurrences/moves, but never cross held-out canonical positions.
        own_keys = set(); kept = []
        for i, key in enumerate(keys):
            token = key.tobytes()
            if token in blocked and split == 'train':
                continue
            if token in own_keys and split != 'train':
                continue
            if split == 'validation' and token in test_keys:
                continue
            own_keys.add(token); kept.append(i)
        if split == 'test':
            test_keys = own_keys
        blocked.update(own_keys)
        rejected[split] = len(candidates)-len(kept)
        dataset[split+'_board'] = boards[kept]
        dataset[split+'_move'] = np.array([candidates[i][1] for i in kept], np.int64)
        dataset[split+'_legal'] = np.stack([candidates[i][2] for i in kept])
        for i in kept:
            r = candidates[i]
            provenance.append(dict(split=split, gameId=r[3], actor=r[4], elo=r[5], priorGames=r[6], move=r[1]))
    output = WORK/'positions.npz'; np.savez_compressed(output, **dataset)
    provenance_path = WORK/'positions-provenance.jsonl'
    with provenance_path.open('w') as stream:
        for row in provenance:
            stream.write(json.dumps(row)+'\n')
    report = dict(version='beginner-low-elo1200-v1', complete=True, threshold=THRESHOLD, comparison='strictly below threshold',
                  minPriorTrainGames=MIN_PRIOR, playerNamespace='WTHOR indices; no names or official WOF rating required',
                  rating='Exact v1.40 internal logistic Elo: initial1500,K32,scale400; train-only year-start',
                  restoredReference='models/v1.40-training-consensus-elo.json', restoredReferenceSha256=sha(ROOT/'models/v1.40-training-consensus-elo.json'),
                  exactReferenceQuantiles=True, ratedCompleteGames=sum(r['rated'] for r in ledger),
                  trainOnlyRatedGames=int(train.games.sum()//2), qualifyingActorGamesByThreshold=eligible,
                  selection='Only moves made by the qualified low-Elo actor; never include stronger opponent moves merely because the game qualifies; no winner-only filtering',
                  unknownOrUnestablished='Excluded; never treat missing identity or the initial1500 default as low skill',
                  target='Recorded legal next move; no stronger engine relabeling',
                  phases='placements 9..54, every third placement; original opening-group splits',
                  positions={s:len(dataset[s+'_board']) for s in rows}, rejectedCanonicalHoldoutOrDuplicates=rejected,
                  crossSplitCanonicalOverlap=0, selectedActorGames={s:len(v) for s,v in selected_actor_games.items()},
                  eloDistribution=dict(zip(['min','p25','median','p75','max'],map(float,np.quantile([r['elo'] for r in provenance if r['split']=='train'],[0,.25,.5,.75,1])))),
                  gamesSha256=sha(games_path), rawSourceSha256=raw_hashes, cacheSha256=sha(output),
                  provenanceSha256=sha(provenance_path), ledgerSha256=sha(ledger_path),
                  sourceSha256=sha(Path(__file__)), eloSourceSha256=sha(ROOT/'scripts/training_consensus.py'),
                  dataNotices='data/README.md', seconds=time.perf_counter()-started,
                  limits='Internal relative Elo is not official rating or proof of novice strength; low-rated tournament players may still beat new players. Match strength must be measured.')
    REPORT.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
