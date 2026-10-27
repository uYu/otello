#!/usr/bin/env python3
"""Compare the v1.4 8x8 CNN against frozen v1.3 NNUE with the same opening policy from the standard starting position.

Both versions use their frozen models and per-move time limit. Each native
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import opening_policy

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
        if hasattr(lib, 'ot_nnue_play'):
            lib.ot_nnue_play.argtypes = [ctypes.c_void_p, ptr] + [ctypes.c_int] * 4 + [ctypes.c_uint] + [ctypes.c_int] * 3 + [ptr, ptr]
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
            policy = (ctypes.c_int * 6)()
            seed = command.get('seed', 1)
            varied = command.get('variety', False)
            if varied:
                success = lib.ot_nnue_play(handle, board, command['turn'], command['ms'], command['depth'], 2, seed,
                                           command['openingPlies'], command['temperature'], command['maxLoss'], output, policy)
            else:
                success = lib.ot_nnue_search(handle, board, command['turn'], command['ms'], command['depth'], 2, seed, output)
                policy[0], policy[3] = 1, output[1]
            wall = (time.perf_counter_ns() - tick) / 1e6
            if success != 1:
                raise RuntimeError(lib.ot_nnue_error().decode())
            connection.send(dict(move=output[0], score=output[1], depth=output[2],
                                 elapsedMs=output[3], nodes=output[4], exact=bool(output[5]),
                                 ttHits=output[135], wallMs=wall,
                                 candidates=[dict(move=output[7+i*2], score=output[8+i*2]) for i in range(output[6])] if varied and sum(bool(c) for c in board)-4 < command['openingPlies'] else [],
                                 selection=dict(seed=seed, eligible=policy[0], temperatureMilli=policy[1], lossLimit=policy[2],
                                                bestScore=policy[3], scoreLoss=policy[4], sampled=bool(policy[5]))))
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
            expected = 'v1.4' if turn == record['focusSide'] else 'v1.3'
            assert decision['engine'] == expected
            selection = decision['selection']
            if expected in ('v1.3', 'v1.4'):
                assert selection['seed'] == record['seed']
                if decision['candidates']:
                    assert {c['move'] for c in decision['candidates']} == set(legal(board, turn))
                    chosen, score, audited = opening_policy.choose(board, turn, decision['candidates'], record['seed'],
                        **record['openingPolicy'], exact=decision['exact'])
                    assert chosen == move and score == decision['score'] and audited == selection
                else:
                    assert not selection['sampled'] and selection['temperatureMilli'] == 0 and selection['scoreLoss'] == 0
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
    v1 = json.loads((ROOT / 'models/v1.3-baseline.json').read_text())
    metadata = json.loads(args.after_model.with_suffix('.json').read_text())
    assert sha(args.before_model) == v1['modelSha256']
    assert sha(args.after_model) == metadata['sha256']
    assert metadata['modelFormat'] == 3
    assert all(sha(args.before_source / name) == value for name, value in v1['sourceSha256'].items())
    current_sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'cpp').rglob('*'))
                       if p.suffix in ('.cpp', '.hpp', '.h')}
    provenance = dict(version='v1.4', experiment='cnn8x8', focus='v1.4 CNN 8x8', start='initial',
                      opening=[], useBook=False, models={'v1.3': 'models/nnue256-v1.2.otnnue', 'v1.4': 'models/cnn8x8-v1.4.otcnn'},
                      modelSha256={'v1.3': sha(args.before_model), 'v1.4': sha(args.after_model)},
                      millisecondsPerMove=args.milliseconds, maxDepth=args.depth, plannedGames=args.games,
                      processes='One persistent native worker per version, sequential moves/games',
                      cachePolicy='Clear both at each game; retain each TT between its own moves',
                      openingPolicy=dict(plies=args.opening_plies, temperature=args.temperature, max_loss=args.max_loss),
                      baseSeed=args.seed, referenceVariety=True, colorOrder='v1.4 black on odd games, white on even games',
                      librarySha256={'v1.3': sha(args.before), 'v1.4': sha(args.after)},
                      sourceSha256={'v1.3': v1['sourceSha256'], 'v1.4': current_sources},
                      runnerSha256=sha(Path(__file__)), policyAuditSha256=sha(ROOT / 'scripts/opening_policy.py'),
                      platform=platform.platform(), machine=platform.machine(),
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
                    symmetryUniqueTranscripts=len({opening_policy.canonical_moves(r['moves']) for r in records}),
                    symmetryUniqueOpenings=len({opening_policy.canonical_moves([m for m in r['moves'] if m >= 0][:12]) for r in records}),
                    sampledDecisions=sum(d['selection']['sampled'] for r in records for d in r['decisions']),
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
        for name, path in [('v1.3', args.before), ('v1.4', args.after)]:
            engines[name] = Engine(path, args.before_model if name == 'v1.3' else args.after_model)
        save(full=True)
        for game in range(args.games):
            for engine in engines.values():
                engine.request(kind='clear')
            focus_side = 1 if game % 2 == 0 else -1
            game_seed = (args.seed ^ ((game + 1) * 0x9e3779b9)) & 0x7fffffff
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
                    name = 'v1.4' if turn == focus_side else 'v1.3'
                    result = engines[name].request(kind='search', board=board, turn=turn,
                                                   ms=args.milliseconds, depth=args.depth, seed=game_seed,
                                                   variety=True, openingPlies=args.opening_plies,
                                                   temperature=args.temperature, maxLoss=args.max_loss)
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
                          moves=moves, decisions=decisions, finalBoard=board, seed=game_seed,
                          openingPolicy=dict(plies=args.opening_plies, temperature=args.temperature, max_loss=args.max_loss),
                          focusDiscDifference=difference, outcome=outcome)
            replay(record)
            records.append(record)
            totals.update([outcome])
            totals['discDifference'] += difference
            current = None
            save(full=True)
            print(json.dumps(dict(completed=len(records), **totals)), flush=True)
        assert sha(args.before) == provenance['librarySha256']['v1.3']
        assert sha(args.after) == provenance['librarySha256']['v1.4']
        assert sha(args.before_model) == provenance['modelSha256']['v1.3']
        assert sha(args.after_model) == provenance['modelSha256']['v1.4']
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
    parser.add_argument('--before', type=Path, default=ROOT / f'.build/v1.4-evaluation/v1.3-native/libothello.{suffix}')
    parser.add_argument('--before-source', type=Path, default=ROOT / '.build/v1.4-evaluation/v1.3-source')
    parser.add_argument('--after', type=Path, default=ROOT / f'.build/native-v1.4/libothello.{suffix}')
    parser.add_argument('--before-model', type=Path, default=ROOT / 'models/nnue256-v1.2.otnnue')
    parser.add_argument('--after-model', type=Path, default=ROOT / 'models/cnn8x8-v1.4.otcnn')
    parser.add_argument('--output', type=Path, default=ROOT / 'models/v1.4-cnn-vs-v1.3.json')
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--milliseconds', type=int, default=50)
    parser.add_argument('--depth', type=int, default=64)
    parser.add_argument('--opening-plies', type=int, default=12)
    parser.add_argument('--temperature', type=int, default=32)
    parser.add_argument('--max-loss', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--overwrite', action='store_true')
    options = parser.parse_args()
    if options.games < 2 or options.games % 2:
        parser.error('--games must be positive and even')
    if not 1 <= options.milliseconds <= 60000 or not 1 <= options.depth <= 64:
        parser.error('Invalid search budget')
    if not 0 <= options.opening_plies <= 60 or not 0 <= options.temperature <= 1000 or not 0 <= options.max_loss <= 1000:
        parser.error('Invalid opening policy')
    compare(options)
