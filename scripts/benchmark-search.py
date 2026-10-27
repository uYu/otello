#!/usr/bin/env python3
"""Compare fixed-depth searches, including every root candidate score.

Uses only Python's standard library and native C ABIs. Cold searches clear v1.1's
per-thread cache. Sequence searches retain it across consecutive game positions;
repeat searches show the best-case benefit on an identical request.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
Cells = ctypes.c_int * 64
Output = ctypes.c_int * 136


class Engine:
    def __init__(self, path, model):
        self.lib = ctypes.CDLL(str(path.resolve()))
        self.lib.ot_nnue_load.argtypes = [ctypes.c_char_p]
        self.lib.ot_nnue_load.restype = ctypes.c_void_p
        self.lib.ot_nnue_destroy.argtypes = [ctypes.c_void_p]
        self.lib.ot_nnue_search.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)] + [ctypes.c_int] * 4 + [ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]
        self.model = self.lib.ot_nnue_load(str(model.resolve()).encode())
        if not self.model:
            raise RuntimeError('Cannot load model')

    def clear(self):
        if hasattr(self.lib, 'ot_search_clear_cache'):
            self.lib.ot_search_clear_cache()

    def search(self, position, depth, neural):
        board, turn = position
        output = Output()
        start = time.perf_counter_ns()
        args = (Cells(*board), turn, 0, depth, 2, 1, output)
        success = self.lib.ot_nnue_search(self.model, *args) if neural else self.lib.ot_search(*args)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if success != 1 or output[2] != min(depth, board.count(0)):
            raise RuntimeError('Search failed or did not complete requested depth')
        return dict(score=output[1], move=output[0], nodes=output[4], ttHits=output[135], ms=elapsed,
                    candidates={output[7 + i * 2]: output[8 + i * 2] for i in range(output[6])})


def positions(lib, seed, plies):
    rng = random.Random(seed)
    board = Cells()
    board[28] = board[35] = 1
    board[27] = board[36] = -1
    turn = 1
    found = []
    for ply in range(max(plies) + 1):
        if ply in plies:
            found.append((list(board), turn))
        moves = Cells()
        count = lib.ot_legal(board, turn, moves)
        out = Output()
        move = moves[rng.randrange(count)] if count else -1
        if lib.ot_play(board, turn, move, out) != 1:
            raise RuntimeError('Fixture game ended unexpectedly')
        board, turn = Cells(*out[:64]), out[64]
    return found


def run(args):
    engines = [Engine(args.before, args.model), Engine(args.after, args.model)]
    fixtures = {mode: [p for seed in (91, 2026) for p in positions(engines[0].lib, seed, plies)]
                for mode, plies in [('cold', [12, 24, 36, 48]), ('sequence', [24, 25, 26, 27])]}
    fixtures['repeat'] = fixtures['cold']
    report = dict(version='v1.1-search', depth=args.depth, rounds=args.rounds, seed=[91, 2026],
                  beforeLibrarySha256=hashlib.sha256(args.before.read_bytes()).hexdigest(),
                  afterLibrarySha256=hashlib.sha256(args.after.read_bytes()).hexdigest(),
                  modelSha256=hashlib.sha256(args.model.read_bytes()).hexdigest(), results=[])
    evaluators = {'both': (False, True), 'heuristic': (False,), 'nnue': (True,)}
    for neural in evaluators[args.evaluator]:
        for mode, samples in fixtures.items():
            totals = [[], []]
            for round_index in range(args.rounds):
                results = [[], []]
                # Alternate execution order to reduce systematic timing bias.
                for index in ((0, 1) if round_index % 2 == 0 else (1, 0)):
                    engine = engines[index]
                    engine.clear()
                    for p in samples:
                        if mode in ('cold', 'repeat'):
                            engine.clear()
                        if mode == 'repeat':
                            engine.search(p, args.depth, neural)
                        results[index].append(engine.search(p, args.depth, neural))
                    totals[index].append(dict(ms=sum(r['ms'] for r in results[index]),
                                              nodes=sum(r['nodes'] for r in results[index]),
                                              ttHits=sum(r['ttHits'] for r in results[index])))
                for old, new in zip(*results):
                    assert old['score'] == new['score'] and old['candidates'] == new['candidates'], 'Fixed-depth scores changed'
            row = dict(evaluator='nnue256-v1' if neural else 'heuristic', mode=mode, positions=len(samples),
                       before={k: statistics.median(r[k] for r in totals[0]) for k in totals[0][0]},
                       after={k: statistics.median(r[k] for r in totals[1]) for k in totals[1][0]},
                       allCandidateScoresEqual=True)
            row['speedup'] = row['before']['ms'] / row['after']['ms']
            report['results'].append(row)
            print(json.dumps(row), flush=True)
    for engine in engines:
        engine.lib.ot_nnue_destroy(engine.model)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/nnue256-v1.otnnue')
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--evaluator', choices=('both', 'heuristic', 'nnue'), default='both')
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    if not 1 <= options.depth <= 64 or options.rounds < 1:
        parser.error('depth must be 1..64 and rounds must be positive')
    run(options)
