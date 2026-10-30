#!/usr/bin/env python3
"""Single-thread Accelerate batching: fixed-budget diagnostic, no match changes."""
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
    assert os.environ.get('VECLIB_MAXIMUM_THREADS') == '1', 'Use one BLAS thread while formal matches run'
    original = e.load('.build/native-cnnpolicy/libothello.dylib')
    batch = e.load('.build/native-cnn-native-batch-v1.17/libothello.dylib')
    batch.ot_cnn_set_native_batch_evaluator.argtypes = [ctypes.c_int]
    batch.ot_cnn_native_batch_validate.argtypes = [ctypes.c_int]
    batch.ot_cnn_native_batch_validation_errors.argtypes = [ctypes.POINTER(ctypes.c_double)]*2
    batch.ot_cnn_batch_stats.argtypes = [ctypes.POINTER(ctypes.c_uint)]*2
    for lib in (original, batch):
        lib.ot_nnue_error.restype = ctypes.c_char_p
    model = ROOT/'models/cnn2-c32-cosine-joint-v1.13.otcnn'
    handles = {id(lib): lib.ot_nnue_load(str(model).encode()) for lib in (original, batch)}
    assert all(handles.values())
    with np.load(ROOT/'.build/training/v1/positions.npz') as data:
        board, turn = data['validation_board'], data['validation_turn']
    indices = json.loads((ROOT/'models/v1.16-cnn-batch-diagnostic.json').read_text())['positions']
    cells = [(ctypes.c_int*64)(*map(int, board[i])) for i in indices]

    def search(lib, index, simulations):
        out, policy, visits, stats = (ctypes.c_int*136)(), (ctypes.c_int*6)(), (ctypes.c_int*64)(), (ctypes.c_double*72)()
        lib.ot_mcts_clear_cache()
        tick = time.perf_counter()
        ok = lib.ot_nnue_mcts_play(handles[id(lib)], cells[index], int(turn[indices[index]]), 0, simulations,
                                   20261028, 0, 0, 0, out, policy, visits, stats)
        if ok != 1:
            raise ValueError(lib.ot_nnue_error())
        elapsed = time.perf_counter()-tick
        assert int(stats[0]) <= simulations and sum(visits) <= stats[0]
        return dict(move=out[0], score=out[1], depth=out[2], simulations=int(stats[0]), evaluations=int(stats[1]),
                    meanDepth=stats[2]/1000, wallSeconds=elapsed, exact=bool(out[5]))

    batch.ot_cnn_native_batch_validate(1)
    for size in (1, 8, 16, 32, 64, 128):
        batch.ot_cnn_set_native_batch_evaluator(size)
        for i in range(2):
            search(batch, i, 256)
    verr, perr = ctypes.c_double(), ctypes.c_double()
    batch.ot_cnn_native_batch_validation_errors(ctypes.byref(verr), ctypes.byref(perr))
    assert verr.value <= 1 and perr.value < 1e-5
    batch.ot_cnn_native_batch_validate(0)
    variants = [0, 1, 8, 16, 32, 64, 128]
    records = []
    for budget in (1024, 8192):
        for i in range(8):
            order = variants[i % len(variants):]+variants[:i % len(variants)]
            for size in order:
                lib = original if size == 0 else batch
                if size:
                    batch.ot_cnn_set_native_batch_evaluator(size)
                row = search(lib, i, budget)
                calls, leaves = ctypes.c_uint(), ctypes.c_uint()
                if size:
                    batch.ot_cnn_batch_stats(ctypes.byref(calls), ctypes.byref(leaves))
                records.append(dict(index=indices[i], budget=budget, requestedBatch=size, **row,
                                    actualBatchMean=leaves.value/calls.value if calls.value else 1.))
                print(json.dumps(records[-1]), flush=True)
    summary = []
    for budget in (1024, 8192):
        for size in variants:
            rows = [r for r in records if r['budget'] == budget and r['requestedBatch'] == size]
            summary.append(dict(budget=budget, requestedBatch=size,
                                simulationsPerSecond=sum(r['simulations'] for r in rows)/sum(r['wallSeconds'] for r in rows),
                                meanDepth=float(np.mean([r['meanDepth'] for r in rows])),
                                actualBatchMean=float(np.mean([r['actualBatchMean'] for r in rows]))))
    report = dict(version='v1.17-native-cnn-batch-diagnostic', status='candidate', defaultPromoted=False,
                  modelSha256=e.sha(str(model.relative_to(ROOT))), positions=indices, split='frozen validation',
                  blasThreads=1, maxValueIntegerError=verr.value, maxPolicyError=perr.value,
                  backend='C++ virtual-visit MCTS + single-thread Apple Accelerate FP32 SGEMM',
                  summary=summary, records=records,
                  limits='Fixed-budget diagnosis while formal matches continue; no playing-strength conclusion; large batches alter search allocation',
                  sourceSha256={str(p.relative_to(ROOT)):e.sha(str(p.relative_to(ROOT))) for p in
                                list((ROOT/'experiments/cnn-native-batch-v1.17').glob('*'))+[ROOT/'scripts/benchmark-cnn-native-batch.py'] if p.is_file()},
                  librarySha256=e.sha('.build/native-cnn-native-batch-v1.17/libothello.dylib'))
    target = ROOT/'models/v1.17-cnn-native-batch-diagnostic.json'
    assert not target.exists()
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(complete=True, summary=summary)), flush=True)
    for lib in (original, batch):
        lib.ot_nnue_destroy(handles[id(lib)])


if __name__ == '__main__':
    main()
