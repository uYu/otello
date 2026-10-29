#!/usr/bin/env python3
"""Paired-color v1.8 MCTS experiment and policy ablation, with independent replay."""
import argparse
import ctypes
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pattern_match_base', ROOT/'scripts/compare-nnue-search.py')
base = importlib.util.module_from_spec(spec); sys.modules[spec.name] = base; spec.loader.exec_module(base)
rules = base.rules
CONFIGS = [dict(id='nnue', name='v1.3 NNUE + uniform MCTS', search='mcts', model='models/nnue256-v1.2.otnnue', policy=False),
           dict(id='pattern_value', name='v1.8 pattern + uniform MCTS', search='mcts', model='models/pattern32-v1.8.otpat', policy=False),
           dict(id='pattern_policy', name='v1.8 pattern + policy MCTS', search='mcts', model='models/pattern32-v1.8.otpat', policy=True)]
base.CONFIGS = CONFIGS; base.BY_ID = {c['id']: c for c in CONFIGS}


def worker(connection, library, config):
    handle = None
    try:
        lib = ctypes.CDLL(library); ptr = ctypes.POINTER(ctypes.c_int)
        lib.ot_nnue_load.argtypes = [ctypes.c_char_p]; lib.ot_nnue_load.restype = ctypes.c_void_p
        lib.ot_nnue_destroy.argtypes = [ctypes.c_void_p]; lib.ot_nnue_error.restype = ctypes.c_char_p
        lib.ot_nnue_mcts_play_ex.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]+[ctypes.c_int]*4+[ptr, ptr, ptr, ctypes.POINTER(ctypes.c_double)]
        handle = lib.ot_nnue_load(str((ROOT/config['model']).resolve()).encode())
        if not handle:
            raise RuntimeError(lib.ot_nnue_error().decode())
        connection.send(dict(ready=True))
        while True:
            command = connection.recv()
            if command['kind'] == 'quit':
                break
            if command['kind'] == 'clear':
                lib.ot_search_clear_cache(); lib.ot_mcts_clear_cache(); connection.send(dict(cleared=True)); continue
            cells = (ctypes.c_int*64)(*command['board'])
            output, selection, visits, stats = (ctypes.c_int*136)(), (ctypes.c_int*6)(), (ctypes.c_int*64)(), (ctypes.c_double*72)()
            tick = time.perf_counter_ns(); seed = command['seed']
            ok = lib.ot_nnue_mcts_play_ex(handle, cells, command['turn'], command['ms'], 0, seed,
                                         base.POLICY['plies'], base.POLICY['temperature'], base.POLICY['max_loss'], int(config['policy']),
                                         output, selection, visits, stats)
            wall = (time.perf_counter_ns()-tick)/1e6
            if ok != 1:
                raise RuntimeError(lib.ot_nnue_error().decode())
            candidates = [dict(move=output[7+2*i], score=output[8+2*i], visits=visits[i], value=stats[8+i]) for i in range(output[6])]
            connection.send(dict(move=output[0], score=output[1], depth=output[2], elapsedMs=output[3], nodes=output[4], exact=bool(output[5]), ttHits=output[135], wallMs=wall,
                                 candidates=candidates, search='mcts',
                                 mcts=dict(simulations=int(stats[0]), evaluations=int(stats[1]), meanDepth=stats[2]/1000, reusedNodes=int(stats[3]), treeNodes=int(stats[4]), memoryLimited=bool(stats[7])),
                                 selection=dict(seed=seed, eligible=selection[0], temperatureMilli=selection[1], lossLimit=selection[2], bestScore=selection[3], scoreLoss=selection[4], sampled=bool(selection[5]))))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        connection.send(dict(error=str(error)))
    finally:
        if handle:
            lib.ot_nnue_destroy(handle)
        connection.close()


base.worker = worker


def compare(args):
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing experiment')
    configs = [dict(c, modelSha256=base.sha(ROOT/c['model'])) for c in CONFIGS]
    provenance = dict(version='v1.8-pattern-pilot', millisecondsPerMove=args.milliseconds, gamesPerPair=args.games_per_pair,
                      plannedGames=3*args.games_per_pair, baseSeed=args.seed, start='initial', opening=[], useBook=False,
                      openingPolicy=base.POLICY, configs=configs, librarySha256=base.sha(args.library),
                      sourceSha256={str(p.relative_to(ROOT)): base.sha(p) for p in sorted((ROOT/'cpp').rglob('*')) if p.suffix in ('.cpp', '.hpp', '.h')},
                      scriptSha256={str(p.relative_to(ROOT)): base.sha(p) for p in [Path(__file__), ROOT/'scripts/compare-nnue-search.py', ROOT/'scripts/compare-resnet.py', ROOT/'scripts/opening_policy.py']},
                      platform=platform.platform(), mcts=dict(exploration=1.5, maxNodes=2097152, learnedPolicyFpuReduction=0.15,
                                                              value='expected final disc difference /64; WDL trained as auxiliary output, unused by search', rollout=False),
                      limits='Exploratory paired-color sample from the initial board, no prescribed openings/book. Same time and library. Uniform-versus-learned ablation changes priors and finite FPU together. No Elo or statistical significance claim.')
    records, engines = [], {}; started = time.perf_counter(); current = None
    def save(complete=False):
        totals, matrix = base.aggregate(records)
        value = dict(**provenance, complete=complete, games=len(records), seconds=time.perf_counter()-started,
                     results=totals, matrix=matrix, records=records, replayedGames=len(records), replayedActions=sum(len(r['moves']) for r in records),
                     uniqueTranscripts=len({tuple(r['moves']) for r in records}))
        rules.write_json(args.output, value)
        value.pop('records'); value['currentGame'] = current; rules.write_json(args.output.with_suffix('.live.json'), value)
    try:
        for c in CONFIGS:
            engines[c['id']] = base.Engine(args.library, c)
        save()
        for pair, (left, right) in enumerate(itertools.combinations(CONFIGS, 2)):
            for round_index in range(args.games_per_pair):
                black, white = (left, right) if round_index % 2 == 0 else (right, left)
                for c in (black, white):
                    engines[c['id']].request(kind='clear')
                seed = (args.seed ^ ((round_index//2+1)*0x9e3779b9)) & 0x7fffffff
                board, turn = rules.initial(), 1; moves, decisions = [], []
                current = dict(number=len(records)+1, pair=pair, black=black['id'], white=white['id'], ply=0)
                while True:
                    if not rules.legal(board, turn):
                        if not rules.legal(board, -turn):
                            break
                        move = -1
                    else:
                        name = black['id'] if turn == 1 else white['id']
                        r = engines[name].request(kind='search', board=board, turn=turn, seed=seed, ms=args.milliseconds)
                        decision = dict(ply=len(moves), turn=turn, engine=name, **r)
                        base.audit_decision(board, turn, decision); decisions.append(decision); move = r['move']
                    board, turn = rules.play(board, turn, move); moves.append(move)
                    assert len(moves) <= 120
                    current['ply'] = len(moves)
                difference = sum(board)
                record = dict(number=len(records)+1, pair=pair, round=round_index, black=black['id'], white=white['id'], seed=seed,
                              opening=[], openingPolicy=base.POLICY, moves=moves, decisions=decisions, finalBoard=board,
                              blackDiscDifference=difference, winner=black['id'] if difference > 0 else white['id'] if difference < 0 else None)
                base.replay(record); records.append(record); current = None; save()
                print(json.dumps(dict(completed=len(records), planned=provenance['plannedGames'], black=black['id'], white=white['id'], difference=difference, winner=record['winner'])), flush=True)
        assert base.sha(args.library) == provenance['librarySha256']
        assert all(base.sha(ROOT/c['model']) == c['modelSha256'] for c in configs)
        save(complete=True)
    finally:
        for engine in engines.values():
            engine.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    suffix = 'dylib' if sys.platform == 'darwin' else 'so'
    p.add_argument('--library', type=Path, default=ROOT/f'.build/native-v1.8/libothello.{suffix}')
    p.add_argument('--output', type=Path, default=ROOT/'models/v1.8-pattern-matches.json')
    p.add_argument('--milliseconds', type=int, default=100)
    p.add_argument('--games-per-pair', type=int, default=20)
    p.add_argument('--seed', type=int, default=20261029)
    args = p.parse_args()
    if args.games_per_pair < 2 or args.games_per_pair % 2 or not 1 <= args.milliseconds <= 60000:
        p.error('Even paired-color games >=2 and time 1..60000 required')
    compare(args)
