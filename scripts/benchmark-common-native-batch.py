#!/usr/bin/env python3
"""Fair two-model diagnostic: same batch scheduler/backend/thread/time budgets."""
import ctypes
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('engineering', ROOT/'scripts/check-engineering.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def main():
    assert os.environ.get('VECLIB_MAXIMUM_THREADS') == '1'
    output = ROOT/'.build/training/common-native-batch-v1.18'
    output.mkdir(exist_ok=True)
    original = e.load('.build/native-cnnpolicy/libothello.dylib')
    batch = e.load('.build/native-common-batch-v1.18/libothello.dylib')
    batch.ot_cnn_set_native_batch_evaluator.argtypes = [ctypes.c_int]
    batch.ot_cnn_native_batch_validate.argtypes = [ctypes.c_int]
    batch.ot_cnn_native_batch_validation_errors.argtypes = [ctypes.POINTER(ctypes.c_double)]*2
    batch.ot_cnn_batch_stats.argtypes = [ctypes.POINTER(ctypes.c_uint)]*2
    for lib in (original, batch):
        lib.ot_nnue_error.restype = ctypes.c_char_p
    with np.load(ROOT/'.build/training/v1/positions.npz') as data:
        board, turn = data['validation_board'], data['validation_turn']
    indices = json.loads((ROOT/'models/v1.16-cnn-batch-diagnostic.json').read_text())['positions'][:4]
    models = [('CNN', 'models/cnn2-c32-cosine-joint-v1.13.otcnn'), ('v1.9', 'models/nnue256-outcome-v1.9.otnnue')]
    variants = [0, 1, 8, 16, 32, 64]
    records, checks = [], []
    for label, path in models:
        handles = {id(lib): lib.ot_nnue_load(str(ROOT/path).encode()) for lib in (original, batch)}
        assert all(handles.values())

        def search(lib, index, simulations, ms):
            cells = (ctypes.c_int*64)(*map(int, board[index]))
            out, policy, visits, stats = (ctypes.c_int*136)(), (ctypes.c_int*6)(), (ctypes.c_int*64)(), (ctypes.c_double*72)()
            lib.ot_mcts_clear_cache()
            tick = time.perf_counter()
            success = lib.ot_nnue_mcts_play(handles[id(lib)], cells, int(turn[index]), ms, simulations, 20261028,
                                            0, 0, 0, out, policy, visits, stats)
            if success != 1:
                raise ValueError(lib.ot_nnue_error())
            seconds = time.perf_counter()-tick
            if simulations:
                assert stats[0] <= simulations
            calls, leaves = ctypes.c_uint(), ctypes.c_uint()
            if lib is batch:
                batch.ot_cnn_batch_stats(ctypes.byref(calls), ctypes.byref(leaves))
            return dict(wallSeconds=seconds, simulations=int(stats[0]), evaluations=int(stats[1]), meanDepth=stats[2]/1000,
                        maxDepth=out[2], move=out[0], score=out[1], exact=bool(out[5]),
                        actualBatchMean=leaves.value/calls.value if calls.value else 1)

        batch.ot_cnn_native_batch_validate(1)
        for size in variants[1:]:
            batch.ot_cnn_set_native_batch_evaluator(size)
            for index in indices[:2]:
                search(batch, index, 256, 0)
        value, policy = ctypes.c_double(), ctypes.c_double()
        batch.ot_cnn_native_batch_validation_errors(ctypes.byref(value), ctypes.byref(policy))
        checks.append(dict(model=label, maxValueIntegerError=value.value, maxPolicyError=policy.value))
        batch.ot_cnn_native_batch_validate(0)
        for i, index in enumerate(indices):
            for size in variants[i:]+variants[:i]:
                lib = original if size == 0 else batch
                if size:
                    batch.ot_cnn_set_native_batch_evaluator(size)
                r = dict(model=label, index=index, requestedBatch=size, budgetSimulations=8192, milliseconds=0, **search(lib, index, 8192, 0))
                records.append(r)
                print(json.dumps(r), flush=True)
        # Same 2s budget on both models and identical two positions.
        for i, index in enumerate(indices[:2]):
            for size in ([0, 1, 32] if i == 0 else [32, 1, 0]):
                lib = original if size == 0 else batch
                if size:
                    batch.ot_cnn_set_native_batch_evaluator(size)
                r = dict(model=label, index=index, requestedBatch=size, budgetSimulations=0, milliseconds=2000, **search(lib, index, 0, 2000))
                records.append(r)
                print(json.dumps(r), flush=True)
        for lib in (original, batch):
            lib.ot_nnue_destroy(handles[id(lib)])
    summary = []
    for label, path in models:
        for ms in (0, 2000):
            for size in variants:
                rows = [r for r in records if r['model'] == label and r['milliseconds'] == ms and r['requestedBatch'] == size]
                if not rows:
                    continue
                summary.append(dict(model=label, milliseconds=ms, requestedBatch=size,
                                    simulationsPerSecond=sum(r['simulations'] for r in rows)/sum(r['wallSeconds'] for r in rows),
                                    meanSimulations=float(np.mean([r['simulations'] for r in rows])),
                                    meanDepth=float(np.mean([r['meanDepth'] for r in rows])),
                                    actualBatchMean=float(np.mean([r['actualBatchMean'] for r in rows]))))
    best = {label:max([r for r in summary if r['model'] == label and r['milliseconds'] == 0 and r['requestedBatch'] != 0],
                     key=lambda r:r['simulationsPerSecond'])['requestedBatch'] for label, _ in models}
    report = dict(version='v1.18-common-native-batch-diagnostic', status='candidate', defaultPromoted=False,
                  backend='same C++ virtual-visit MCTS and single-thread Accelerate SGEMM for both models',
                  nnuePrecision='integer-valued transform exact in FP32; bound-proven exact policy matrix or int64 fallback; int64 value and double output policy',
                  checks=checks, bestFixedBudgetBatch=best, positions=indices, split='original frozen validation',
                  blasThreads=1, summary=summary, records=records,
                  models={label:dict(path=path,sha256=e.sha(path)) for label,path in models},
                  librarySha256=e.sha('.build/native-common-batch-v1.18/libothello.dylib'),
                  sourceSha256={str(p.relative_to(ROOT)):e.sha(str(p.relative_to(ROOT))) for p in
                                list((ROOT/'experiments/common-native-batch-v1.18').glob('*'))+[ROOT/'scripts/benchmark-common-native-batch.py'] if p.is_file()},
                  limits='Few fixed positions with concurrent old formal matches; engineering diagnosis, not a win-rate comparison; batch 0 is old scalar backend')
    target = ROOT/'models/v1.18-common-native-batch-diagnostic.json'
    assert not target.exists()
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(complete=True, summary=summary)), flush=True)


if __name__ == '__main__':
    main()
