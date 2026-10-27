#!/usr/bin/env python3
"""Paired-color heuristic/model or model/model+book experiments."""
import argparse
from collections import Counter, defaultdict
import ctypes
import hashlib
import json
import platform
from pathlib import Path
import random
import time


def compare(args):
    lib = ctypes.CDLL(str(args.library.resolve()))
    ptr = ctypes.POINTER(ctypes.c_int)
    cells_type, output_type = ctypes.c_int * 64, ctypes.c_int * 136
    lib.ot_legal.argtypes = [ptr, ctypes.c_int, ptr]
    lib.ot_play.argtypes = [ptr, ctypes.c_int, ctypes.c_int, ptr]
    lib.ot_search.argtypes = [ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ptr]
    lib.ot_nnue_load.argtypes = [ctypes.c_char_p]
    lib.ot_nnue_load.restype = ctypes.c_void_p
    lib.ot_nnue_destroy.argtypes = [ctypes.c_void_p]
    lib.ot_nnue_search.argtypes = [ctypes.c_void_p, *lib.ot_search.argtypes]
    lib.ot_nnue_error.restype = ctypes.c_char_p
    rng = random.Random(args.seed)
    records, stats, totals = [], defaultdict(Counter), Counter()
    started = time.perf_counter()
    current_game = None
    book_handle = None
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    library_hash = hashlib.sha256(args.library.read_bytes()).hexdigest()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    book_hash = hashlib.sha256((args.resources / 'book.egbk3').read_bytes()).hexdigest() if args.experiment == 'book' else None
    source_root = Path(__file__).resolve().parents[1]
    sources = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted((source_root / 'cpp').glob('*')) if path.suffix in ('.cpp', '.hpp', '.h')}
    openings, opening_keys = [], set()

    def play(cells, turn, move):
        out = output_type()
        if lib.ot_play(cells, turn, move, out) != 1:
            raise ValueError('Native search returned an illegal move')
        return cells_type(*out[:64]), out[64], bool(out[65])

    def initial():
        cells = cells_type()
        cells[28] = cells[35] = 1
        cells[27] = cells[36] = -1
        return cells

    if args.start == 'initial':
        openings = [[] for _ in range(args.games // 2)]
    elif args.openings and args.openings.exists():
        bank = json.loads(args.openings.read_text())
        if bank['seed'] != args.seed or bank['plies'] != args.opening_plies:
            raise ValueError('Opening bank uses a different seed or opening length')
        openings = bank['openings']
        if len(openings) < args.games // 2:
            raise ValueError('Opening bank is too small')
    else:
        while len(openings) < args.games // 2:
            cells, turn, opening = initial(), 1, []
            for _ in range(args.opening_plies):
                legal = output_type()
                n = lib.ot_legal(cells, turn, legal)
                move = legal[rng.randrange(n)] if n else -1
                opening.append(move)
                cells, turn, over = play(cells, turn, move)
                if over:
                    raise ValueError('Opening ended the game')
            variants = []
            for symmetry in range(8):
                transformed = [0] * 64
                for m, value in enumerate(cells):
                    x, y = m % 8, m // 8
                    if symmetry & 4:
                        x = 7 - x
                    for _ in range(symmetry & 3):
                        x, y = 7 - y, x
                    transformed[y * 8 + x] = value
                variants.append(tuple(transformed) + (turn,))
            key = min(variants)
            if key not in opening_keys:
                opening_keys.add(key)
                openings.append(opening)
        if args.openings:
            args.openings.parent.mkdir(parents=True, exist_ok=True)
            args.openings.write_text(json.dumps(dict(version=1, seed=args.seed, plies=args.opening_plies,
                                                     symmetryDistinct=True, openings=openings), indent=2) + '\n')
    opening_hash = hashlib.sha256(json.dumps(openings[:args.games // 2], separators=(',', ':')).encode()).hexdigest()

    def report(complete=False):
        detail = {}
        for name, values in stats.items():
            detail[name] = dict(moves=values['moves'], nodes=values['nodes'],
                                searchMoves=values['searchMoves'], bookHits=values['bookHits'], bookProbes=values['bookProbes'],
                                meanDepth=values['depth'] / max(1, values['searchMoves']),
                                meanWallMs=values['wallMs'] / values['moves'],
                                nodesPerSecond=values['nodes'] / (values['wallMs'] / 1000))
        total = len(records)
        return dict(version=1, complete=complete, games=total, plannedGames=args.games,
                    experiment=args.experiment, start=args.start,
                    focus='nnue256+book' if args.experiment == 'book' else 'nnue256',
                    randomOpeningPlies=args.opening_plies, seed=args.seed,
                    millisecondsPerMove=args.milliseconds, maxDepth=args.depth,
                    model=str(args.model), modelSha256=model_hash, librarySha256=library_hash,
                    sourceSha256=sources, comparisonScriptSha256=script_hash,
                    bookSha256=book_hash, bookVariety=args.book_variety if book_hash else None,
                    platform=platform.platform(), machine=platform.machine(),
                    openingBankSha256=opening_hash, scoreRate=(totals['win'] + totals['draw'] / 2) / total if total else 0,
                    results={key: totals[key] for key in ('win', 'draw', 'loss', 'discDifference')},
                    meanDiscDifference=totals['discDifference'] / total if total else 0,
                    engines=detail, seconds=time.perf_counter() - started, records=records,
                    limits='One frozen checkpoint, paired colors, same search budget. Initial-position repetitions are not independent opening samples; timing may change moves. Book lookup never falls back to Egaroucid search. Not an Elo estimate.')

    def save(value):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(args.output)

    def live(complete=False):
        value = report(complete)
        value.pop('records')
        value['currentGame'] = current_game
        path = args.output.with_suffix('.live.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)

    handle = lib.ot_nnue_load(str(args.model.resolve()).encode())
    if not handle:
        raise ValueError(lib.ot_nnue_error().decode())
    try:
        if args.experiment == 'book':
            lib.ot_egaroucid_create.argtypes = [ctypes.c_char_p]
            lib.ot_egaroucid_create.restype = ctypes.c_void_p
            lib.ot_egaroucid_destroy.argtypes = [ctypes.c_void_p]
            lib.ot_egaroucid_book.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ptr]
            lib.ot_egaroucid_last_error.restype = ctypes.c_char_p
            book_handle = lib.ot_egaroucid_create(str(args.resources.resolve()).encode())
            if not book_handle:
                raise ValueError(lib.ot_egaroucid_last_error().decode())
        save(report())
        live()
        for pair in range(args.games // 2):
            cells, turn, opening = initial(), 1, openings[pair]
            for move in opening:
                cells, turn, over = play(cells, turn, move)
                if over:
                    raise ValueError('Opening ended the game')
            opening_cells, opening_turn = tuple(cells), turn
            for model_side in (1, -1):
                cells, turn = cells_type(*opening_cells), opening_turn
                moves, decisions = [], []
                current_game = dict(number=len(records) + 1, ply=len(opening), black=tuple(cells).count(1), white=tuple(cells).count(-1), focusSide=model_side)
                live()
                while True:
                    legal = output_type()
                    n = lib.ot_legal(cells, turn, legal)
                    if n < 0:
                        raise ValueError('Invalid board')
                    if not n:
                        if lib.ot_legal(cells, -turn, legal) == 0:
                            break
                        move = -1
                    else:
                        result = output_type()
                        focused = turn == model_side
                        model = focused or args.experiment == 'book'
                        book_hit = False
                        probed = bool(focused and book_handle)
                        tick = time.perf_counter()
                        if probed:
                            seed = (args.seed ^ ((len(records) + 1) * 0x9e3779b9) ^ (len(moves) * 0x85ebca6b)) & 0xffffffff or 1
                            status = lib.ot_egaroucid_book(book_handle, cells, turn, args.book_variety, seed, result)
                            if status < 0:
                                raise ValueError(lib.ot_egaroucid_last_error().decode())
                            book_hit = status == 1
                        if not book_hit:
                            spent = (time.perf_counter() - tick) * 1000
                            budget = max(1, int(args.milliseconds - spent)) if probed and args.milliseconds else args.milliseconds
                            common = (cells, turn, budget, args.depth, 2, 1, result)
                            success = lib.ot_nnue_search(handle, *common) if model else lib.ot_search(*common)
                            if success != 1:
                                raise ValueError('Search failed: ' + lib.ot_nnue_error().decode())
                        wall = (time.perf_counter() - tick) * 1000
                        name = 'nnue256+book' if focused and args.experiment == 'book' else ('nnue256' if model else 'heuristic')
                        stats[name].update(moves=1, nodes=result[4], depth=max(0, result[2]),
                                           searchMoves=not book_hit, bookHits=book_hit, bookProbes=probed,
                                           elapsedMs=result[3], wallMs=wall)
                        move = result[0]
                        decisions.append(dict(ply=len(opening) + len(moves), turn=turn, move=move,
                                              provider='book' if book_hit else ('nnue' if model else 'heuristic'),
                                              bookProbe=('hit' if book_hit else 'miss') if probed else None,
                                              depth=result[2], nodes=result[4], wallMs=wall))
                    moves.append(move)
                    cells, turn, over = play(cells, turn, move)
                    current_game.update(ply=len(opening) + len(moves), black=tuple(cells).count(1), white=tuple(cells).count(-1))
                    live()
                    if over:
                        break
                black_diff = sum(cells)
                model_diff = black_diff * model_side
                outcome = 'win' if model_diff > 0 else 'loss' if model_diff < 0 else 'draw'
                totals[outcome] += 1
                totals['discDifference'] += model_diff
                records.append(dict(pair=pair, modelSide=model_side, opening=opening, moves=moves,
                                    modelDiscDifference=model_diff, outcome=outcome, decisions=decisions))
                current_game = None
                save(report())
                live()
            print(json.dumps(dict(completed=len(records), **totals)), flush=True)
            save(report())
    finally:
        if book_handle:
            lib.ot_egaroucid_destroy(book_handle)
        lib.ot_nnue_destroy(handle)
    summary = report(complete=True)
    save(summary)
    live(complete=True)
    print(json.dumps({key: value for key, value in summary.items() if key != 'records'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=Path('.build/native/libothello.dylib'))
    parser.add_argument('--model', type=Path, default=Path('.build/training/nnue-256.otnnue'))
    parser.add_argument('--output', type=Path, default=Path('.build/training/comparison.json'))
    parser.add_argument('--openings', type=Path, help='Read or create a frozen symmetric-distinct opening bank')
    parser.add_argument('--experiment', choices=('heuristic', 'book'), default='heuristic')
    parser.add_argument('--start', choices=('initial', 'random'), default='random')
    parser.add_argument('--resources', type=Path, default=Path('.build/deps/Egaroucid/bin/resources'))
    parser.add_argument('--book-variety', type=int, choices=(0, 1, 2, 3), default=1)
    parser.add_argument('--games', type=int, default=40)
    parser.add_argument('--milliseconds', type=int, default=50)
    parser.add_argument('--depth', type=int, default=64)
    parser.add_argument('--opening-plies', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261027)
    args = parser.parse_args()
    if args.games < 2 or args.games % 2:
        parser.error('--games must be a positive even number')
    if not 0 <= args.milliseconds <= 60000 or not 1 <= args.depth <= 64 or not 1 <= args.opening_plies <= 20:
        parser.error('Invalid search budget or opening length')
    if args.start == 'initial':
        if args.openings:
            parser.error('--start initial does not use an opening bank')
        args.opening_plies = 0
    compare(args)
