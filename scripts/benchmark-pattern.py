#!/usr/bin/env python3
"""Measure full value inference and combined policy/value through the native ABI."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np
from pattern_model import native_library

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    if args.output.exists():
        raise ValueError('Refusing to overwrite a benchmark')
    data = np.load(ROOT/'.build/training/v1/positions.npz')
    boards = [(ctypes.c_int*64)(*map(int, b)) for b in data['test_board'][:64]]
    turns = data['test_turn'][:64]
    lib = native_library(args.library); handles = []
    paths = [ROOT/'models/nnue256-v1.2.otnnue', ROOT/'models/pattern32-v1.8.otpat']
    try:
        for path in paths:
            handle = lib.ot_nnue_load(str(path).encode())
            if not handle:
                raise RuntimeError(lib.ot_nnue_error().decode())
            handles.append(handle)
        cases = [('NNUE value full refresh', handles[0], False), ('Pattern value full refresh', handles[1], False),
                 ('Pattern value and policy', handles[1], True)]
        samples = [[] for _ in cases]
        score, prediction = ctypes.c_int(), (ctypes.c_double*68)()
        for round_index in range(7):
            for index in [(round_index+j) % len(cases) for j in range(len(cases))]:
                _, handle, combined = cases[index]
                tick = time.perf_counter_ns(); checksum = 0
                for j in range(4096):
                    k = j % len(boards)
                    if combined:
                        assert lib.ot_pattern_predict(handle, boards[k], int(turns[k]), prediction) == 1
                        checksum += prediction[0]
                    else:
                        assert lib.ot_nnue_evaluate(handle, boards[k], int(turns[k]), ctypes.byref(score)) == 1
                        checksum += score.value
                samples[index].append(dict(microseconds=(time.perf_counter_ns()-tick)/4096000, checksum=checksum))
        report = dict(positions=len(boards), rounds=7, evaluationsPerRound=4096,
                      librarySha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                      scriptSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      modelSha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                      limits='Includes Python/C ABI overhead and full refresh. Search may use incremental evaluation. No concurrent training or matches during measurement.',
                      results=[dict(name=case[0], medianMicroseconds=statistics.median(s['microseconds'] for s in rows), samples=rows)
                               for case, rows in zip(cases, samples)])
        args.output.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=2))
    finally:
        for handle in handles:
            lib.ot_nnue_destroy(handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    suffix = 'dylib' if sys.platform == 'darwin' else 'so'
    parser.add_argument('--library', type=Path, default=ROOT/f'.build/native-v1.8/libothello.{suffix}')
    parser.add_argument('--output', type=Path, default=ROOT/'models/v1.8-inference-benchmark.json')
    run(parser.parse_args())
