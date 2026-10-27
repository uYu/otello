#!/usr/bin/env python3
"""Direct value-evaluation latency on the same boards; excludes search.

NNUE calls here include a full accumulator refresh. During search NNUE uses
incremental updates, so this benchmark does not replace equal-time matches.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    lib = ctypes.CDLL(str(args.library.resolve()))
    ptr = ctypes.POINTER(ctypes.c_int)
    lib.ot_nnue_load.argtypes = [ctypes.c_char_p]
    lib.ot_nnue_load.restype = ctypes.c_void_p
    lib.ot_nnue_destroy.argtypes = [ctypes.c_void_p]
    lib.ot_nnue_evaluate.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ptr]
    lib.ot_play.argtypes = [ptr, ctypes.c_int, ctypes.c_int, ptr]
    Cells, Out = ctypes.c_int * 64, ctypes.c_int * 67
    report = json.loads((ROOT / 'models/v1.3-vs-v1.1-nnue.json').read_text())
    fixtures = []
    for game in report['records'][:4]:
        board = Cells()
        board[28] = board[35] = 1
        board[27] = board[36] = -1
        turn, placed = 1, 0
        for move in game['moves']:
            if placed in (8, 20, 32, 44) and move >= 0:
                fixtures.append((Cells(*board), turn))
            out = Out()
            assert lib.ot_play(board, turn, move, out) == 1
            board, turn = Cells(*out[:64]), out[64]
            placed += move >= 0
    handles = []
    paths = [('v1.3 NNUE full refresh', args.before_model), ('ResNet 32 channels / 2 blocks', args.after_model)]
    try:
        for _, path in paths:
            handle = lib.ot_nnue_load(str(path.resolve()).encode())
            if not handle:
                raise RuntimeError('Cannot load model')
            handles.append(handle)
        results = [[], []]
        score = ctypes.c_int()
        for round_index in range(args.rounds):
            for i in ((0, 1) if round_index % 2 == 0 else (1, 0)):
                for board, turn in fixtures:
                    lib.ot_nnue_evaluate(handles[i], board, turn, ctypes.byref(score))
                start = time.perf_counter_ns()
                checksum = 0
                for j in range(args.evaluations):
                    board, turn = fixtures[j % len(fixtures)]
                    assert lib.ot_nnue_evaluate(handles[i], board, turn, ctypes.byref(score)) == 1
                    checksum += score.value
                results[i].append(dict(microsecondsPerEvaluation=(time.perf_counter_ns() - start) / 1000 / args.evaluations,
                                       checksum=checksum))
        value = dict(positions=len(fixtures),evaluationsPerRound=args.evaluations,rounds=args.rounds,
                     librarySha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                     scriptSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=[],
                     limits='Same boards, direct C ABI scores, NNUE includes full refresh; actual NNUE search uses incremental updates. Includes Python/C ABI call overhead.')
        for (name, path), samples in zip(paths, results):
            value['results'].append(dict(model=name,modelSha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                         medianMicroseconds=statistics.median(s['microsecondsPerEvaluation'] for s in samples),samples=samples))
        value['resnetToNnueLatencyRatio']=value['results'][1]['medianMicroseconds']/value['results'][0]['medianMicroseconds']
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps(value,indent=2))
    finally:
        for handle in handles:
            lib.ot_nnue_destroy(handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    suffix = 'dylib' if sys.platform == 'darwin' else 'so'
    parser.add_argument('--library',type=Path,default=ROOT/f'.build/native-v1.5/libothello.{suffix}')
    parser.add_argument('--before-model',type=Path,default=ROOT/'models/nnue256-v1.2.otnnue')
    parser.add_argument('--after-model',type=Path,default=ROOT/'models/resnet32x2-v1.5.otres')
    parser.add_argument('--output',type=Path,default=ROOT/'models/v1.5-inference-benchmark.json')
    parser.add_argument('--rounds',type=int,default=7)
    parser.add_argument('--evaluations',type=int,default=4096)
    options = parser.parse_args()
    if min(options.rounds,options.evaluations)<1:
        parser.error('Rounds and evaluations must be positive')
    run(options)
