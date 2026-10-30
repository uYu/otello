#!/usr/bin/env python3
"""Two-position 2s diagnostic; deliberately separate from the formal matches."""
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
    old = e.load('.build/native-cnnpolicy/libothello.dylib')
    new = e.load('.build/native-cnn-native-batch-v1.17/libothello.dylib')
    new.ot_cnn_set_native_batch_evaluator.argtypes = [ctypes.c_int]
    new.ot_cnn_batch_stats.argtypes = [ctypes.POINTER(ctypes.c_uint)]*2
    model = str(ROOT/'models/cnn2-c32-cosine-joint-v1.13.otcnn').encode()
    handles = {id(lib): lib.ot_nnue_load(model) for lib in (old, new)}
    indices = json.loads((ROOT/'models/v1.17-cnn-native-batch-diagnostic.json').read_text())['positions'][:2]
    records = []
    with np.load(ROOT/'.build/training/v1/positions.npz') as data:
        for i, index in enumerate(indices):
            for size in ([0, 1, 32] if i == 0 else [32, 1, 0]):
                lib = old if size == 0 else new
                if size:
                    assert new.ot_cnn_set_native_batch_evaluator(size) == 1
                cells = (ctypes.c_int*64)(*map(int, data['validation_board'][index]))
                out, policy, visits, stats = (ctypes.c_int*136)(), (ctypes.c_int*6)(), (ctypes.c_int*64)(), (ctypes.c_double*72)()
                lib.ot_mcts_clear_cache()
                tick = time.perf_counter()
                assert lib.ot_nnue_mcts_play(handles[id(lib)], cells, int(data['validation_turn'][index]),
                                            2000, 0, 20261028, 0, 0, 0, out, policy, visits, stats) == 1
                elapsed = time.perf_counter()-tick
                calls, leaves = ctypes.c_uint(), ctypes.c_uint()
                if size:
                    new.ot_cnn_batch_stats(ctypes.byref(calls), ctypes.byref(leaves))
                records.append(dict(index=index, requestedBatch=size, wallSeconds=elapsed, move=out[0], score=out[1],
                                    exact=bool(out[5]), simulations=int(stats[0]), evaluations=int(stats[1]), maxDepth=out[2], meanDepth=stats[2]/1000,
                                    actualBatchMean=leaves.value/calls.value if calls.value else 1))
                print(json.dumps(records[-1]), flush=True)
    target = ROOT/'models/v1.17-cnn-native-batch-2s-diagnostic.json'
    assert not target.exists()
    target.write_text(json.dumps(dict(records=records, milliseconds=2000, positions=indices, blasThreads=1,
        sourceSha256=e.sha('scripts/benchmark-cnn-batch-timed.py'), librarySha256=e.sha('.build/native-cnn-native-batch-v1.17/libothello.dylib'),
        limits='Two fixed validation positions; concurrent formal matches; no win-rate or strength conclusion'), indent=2)+'\n')
    for lib in (old, new):
        lib.ot_nnue_destroy(handles[id(lib)])


if __name__ == '__main__':
    main()
