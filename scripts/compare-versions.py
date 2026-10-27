#!/usr/bin/env python3
"""Compare v1.1 NNUE against v1 NNUE from the standard starting position.

Both versions use the same frozen model and per-move time limit. Each native
library runs in its own persistent process; games run sequentially and start
with cleared caches. No opening book or prescribed moves are used.
"""
import argparse
from collections import Counter, defaultdict
import ctypes
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial():
    board = [0] * 64
    board[28] = board[35] = 1
    board[27] = board[36] = -1
    return board


def flips(board, turn, move):
    if not 0 <= move < 64 or board[move]:
        return []
    result = []
    for dx, dy in DIRECTIONS:
        x, y = move % 8 + dx, move // 8 + dy
        ray = []
        while 0 <= x < 8 and 0 <= y < 8 and board[y * 8 + x] == -turn:
            ray.append(y * 8 + x)
            x, y = x + dx, y + dy
        if 0 <= x < 8 and 0 <= y < 8 and board[y * 8 + x] == turn:
            result.extend(ray)
    return result


def legal(board, turn):
    return [m for m in range(64) if flips(board, turn, m)]


def play(board, turn, move):
    if move == -1:
        if legal(board, turn) or not legal(board, -turn):
            raise ValueError('Invalid forced pass')
        return list(board), -turn
    captured = flips(board, turn, move)
    if not captured:
        raise ValueError('Illegal search move')
    after = list(board)
    for m in [move, *captured]:
        after[m] = turn
    return after, -turn


def native_worker(connection, library, model):
    handle = None
    try:
        lib = ctypes.CDLL(library)
        ptr = ctypes.POINTER(ctypes.c_int)
        lib.ot_nnue_load.argtypes = [ctypes.c_char_p]
        lib.ot_nnue_load.restype = ctypes.c_void_p
        lib.ot_nnue_destroy.argtypes = [ctypes.c_void_p]
        lib.ot_nnue_search.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_uint, ptr]
        lib.ot_nnue_error.restype = ctypes.c_char_p
        if hasattr(lib, 'ot_search_clear_cache'):
            lib.ot_search_clear_cache.restype = None
        handle = lib.ot_nnue_load(model.encode())
        if not handle:
            raise RuntimeError(lib.ot_nnue_error().decode())
        connection.send(dict(ready=True))
        while True:
            command = connection.recv()
            if command['kind'] == 'quit':
                break
            if command['kind'] == 'clear':
                if hasattr(lib, 'ot_search_clear_cache'):
                    lib.ot_search_clear_cache()
                connection.send(dict(cleared=True))
                continue
            board = (ctypes.c_int * 64)(*command['board'])
            output = (ctypes.c_int * 136)()
            tick = time.perf_counter_ns()
            success = lib.ot_nnue_search(handle, board, command['turn'], command['ms'],
                                         command['depth'], 2, 1, output)
            wall = (time.perf_counter_ns() - tick) / 1e6
            if success != 1:
                raise RuntimeError(lib.ot_nnue_error().decode())
            connection.send(dict(move=output[0], score=output[1], depth=output[2],
                                 elapsedMs=output[3], nodes=output[4], exact=bool(output[5]),
                                 ttHits=output[135], wallMs=wall))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        connection.send(dict(error=str(error)))
    finally:
        if handle:
            lib.ot_nnue_destroy(handle)
        connection.close()


class Engine:
    def __init__(self, library, model):
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=native_worker,
                                       args=(child, str(library.resolve()), str(model.resolve())))
        self.process.start()
        child.close()
        self.receive()

    def receive(self):
        if not self.connection.poll(60):
            raise RuntimeError('Native worker timed out')
        result = self.connection.recv()
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def request(self, **message):
        self.connection.send(message)
        return self.receive()

    def close(self):
        if self.process.is_alive():
            self.connection.send(dict(kind='quit'))
            self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self.connection.close()


def replay(record):
    board, turn = initial(), 1
    decisions = iter(record['decisions'])
    for move in record['moves']:
        if move >= 0:
            decision = next(decisions)
            assert decision['turn'] == turn and decision['move'] == move
            expected = 'v1.1' if turn == record['focusSide'] else 'v1'
            assert decision['engine'] == expected
        board, turn = play(board, turn, move)
    assert next(decisions, None) is None
    assert not legal(board, turn) and not legal(board, -turn)
    difference = sum(board) * record['focusSide']
    assert difference == record['focusDiscDifference']
    assert record['outcome'] == ('win' if difference > 0 else 'loss' if difference < 0 else 'draw')
    assert record['finalBoard'] == board
    return len(record['moves'])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def compare(args):
    if args.output.exists() and not args.overwrite:
        raise ValueError('Report exists; choose another --output or explicitly use --overwrite')
    v1 = json.loads((ROOT / 'models/v1-baseline.json').read_text())
    metadata = json.loads((ROOT / 'models/nnue256-v1.json').read_text())
    assert sha(args.model) == v1['modelSha256'] == metadata['sha256']
    assert all(sha(args.before_source / name) == value for name, value in v1['sourceSha256'].items())
    current_sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'cpp').rglob('*'))
                       if p.suffix in ('.cpp', '.hpp', '.h')}
    provenance = dict(version='v1.1', experiment='version-nnue', focus='v1.1 NNUE', start='initial',
                      opening=[], useBook=False, model='models/nnue256-v1.otnnue', modelSha256=sha(args.model),
                      millisecondsPerMove=args.milliseconds, maxDepth=args.depth, plannedGames=args.games,
                      processes='One persistent native worker per version, sequential moves/games',
                      cachePolicy='Clear both at each game; retain v1.1 TT between its own moves',
                      searchSeed=1, colorOrder='v1.1 black on odd games, white on even games',
                      librarySha256={'v1': sha(args.before), 'v1.1': sha(args.after)},
                      sourceSha256={'v1': v1['sourceSha256'], 'v1.1': current_sources},
                      runnerSha256=sha(Path(__file__)), platform=platform.platform(), machine=platform.machine(),
                      limits='Repeated standard initial position; games can repeat and are not independent opening samples. No Elo or statistical significance claim. Time-limited decisions depend on system load.')
    records, totals, stats = [], Counter(win=0, draw=0, loss=0, discDifference=0), defaultdict(Counter)
    started = time.perf_counter()
    current = None

    def report(complete=False):
        engines = {}
        for name, counter in stats.items():
            n = counter['moves']
            engines[name] = dict(moves=n, nodes=counter['nodes'], ttHits=counter['ttHits'],
                                 meanDepth=counter['depth'] / n, meanWallMs=counter['wallMs'] / n,
                                 nodesPerSecond=counter['nodes'] / (counter['wallMs'] / 1000),
                                 meanNonExactDepth=counter['nonExactDepth'] / max(1, counter['nonExactMoves']),
                                 nonExactMoves=counter['nonExactMoves'], exactMoves=counter['exactMoves'])
        return dict(**provenance, complete=complete, games=len(records), results=dict(totals),
                    scoreRate=(totals['win'] + totals['draw'] / 2) / len(records) if records else 0,
                    meanDiscDifference=totals['discDifference'] / len(records) if records else 0,
                    uniqueTranscripts=len({tuple(r['moves']) for r in records}),
                    seconds=time.perf_counter() - started, engines=engines, records=records,
                    replayedGames=len(records), replayedActions=sum(len(r['moves']) for r in records))

    def save(complete=False, full=False):
        value = report(complete)
        if full:
            write_json(args.output, value)
        value.pop('records')
        value['currentGame'] = current
        write_json(args.output.with_suffix('.live.json'), value)

    engines = {}
    try:
        for name, path in [('v1', args.before), ('v1.1', args.after)]:
            engines[name] = Engine(path, args.model)
        save(full=True)
        for game in range(args.games):
            for engine in engines.values():
                engine.request(kind='clear')
            focus_side = 1 if game % 2 == 0 else -1
            board, turn, moves, decisions = initial(), 1, [], []
            current = dict(number=game + 1, ply=0, black=2, white=2, focusSide=focus_side)
            save()
            while True:
                available = legal(board, turn)
                if not available:
                    if not legal(board, -turn):
                        break
                    move = -1
                else:
                    name = 'v1.1' if turn == focus_side else 'v1'
                    result = engines[name].request(kind='search', board=board, turn=turn,
                                                   ms=args.milliseconds, depth=args.depth)
                    move = result['move']
                    if move not in available:
                        raise ValueError(f'{name} returned illegal move {move}')
                    decisions.append(dict(ply=len(moves), turn=turn, engine=name, **result))
                    stats[name].update(moves=1, nodes=result['nodes'], ttHits=result['ttHits'],
                                       depth=result['depth'], wallMs=result['wallMs'],
                                       nonExactMoves=not result['exact'], exactMoves=result['exact'],
                                       nonExactDepth=result['depth'] if not result['exact'] else 0)
                board, turn = play(board, turn, move)
                moves.append(move)
                if len(moves) > 120:
                    raise ValueError('Game did not terminate')
                current.update(ply=len(moves), black=board.count(1), white=board.count(-1))
                save()
            difference = sum(board) * focus_side
            outcome = 'win' if difference > 0 else 'loss' if difference < 0 else 'draw'
            record = dict(number=game + 1, pair=game // 2, focusSide=focus_side, opening=[],
                          moves=moves, decisions=decisions, finalBoard=board,
                          focusDiscDifference=difference, outcome=outcome)
            replay(record)
            records.append(record)
            totals.update([outcome])
            totals['discDifference'] += difference
            current = None
            save(full=True)
            print(json.dumps(dict(completed=len(records), **totals)), flush=True)
        assert sha(args.before) == provenance['librarySha256']['v1']
        assert sha(args.after) == provenance['librarySha256']['v1.1']
        assert sha(args.model) == provenance['modelSha256']
        save(complete=True, full=True)
    finally:
        for engine in engines.values():
            engine.close()
    summary = report(complete=True)
    summary.pop('records')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    suffix = 'dylib' if sys.platform == 'darwin' else 'so'
    parser.add_argument('--before', type=Path, default=ROOT / f'.build/v1.1-evaluation/v1-native/libothello.{suffix}')
    parser.add_argument('--before-source', type=Path, default=ROOT / '.build/v1.1-evaluation/v1-source')
    parser.add_argument('--after', type=Path, default=ROOT / f'.build/native-v1.1/libothello.{suffix}')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/nnue256-v1.otnnue')
    parser.add_argument('--output', type=Path, default=ROOT / 'models/v1.1-vs-v1-nnue.json')
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--milliseconds', type=int, default=50)
    parser.add_argument('--depth', type=int, default=64)
    parser.add_argument('--overwrite', action='store_true')
    options = parser.parse_args()
    if options.games < 2 or options.games % 2:
        parser.error('--games must be positive and even')
    if not 1 <= options.milliseconds <= 60000 or not 1 <= options.depth <= 64:
        parser.error('Invalid search budget')
    compare(options)
