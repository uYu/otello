#!/usr/bin/env python3
"""Measure compact CNN search with the same native batch32 backend and positions."""
import argparse
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

def main(output):
    assert os.environ.get('VECLIB_MAXIMUM_THREADS') == '1'
    library = '.build/native-light-cnn-v1.20/libothello.dylib'
    lib = e.load(library)
    lib.ot_cnn_set_native_batch_evaluator.argtypes = [ctypes.c_int]
    lib.ot_cnn_batch_stats.argtypes = [ctypes.POINTER(ctypes.c_uint)]*2
    assert lib.ot_cnn_set_native_batch_evaluator(32) == 1
    with np.load(ROOT/'.build/training/v1/positions.npz') as data:
        board, turn = data['validation_board'], data['validation_turn']
    indices = json.loads((ROOT/'models/v1.16-cnn-batch-diagnostic.json').read_text())['positions'][:4]
    models = [('v1.9','models/nnue256-outcome-v1.9.otnnue'),
              ('C32H64-reference','models/cnn2-c32-cosine-v4-value-v1.13.otcnn')]
    models += [(f'C{c}H32-{criterion}',f'models/cnn2-c{c}-h32-{criterion}-v1.20.otcnn')
               for c in (32,16,8) for criterion in ('joint','value')]
    handles = {label:lib.ot_nnue_load(str(ROOT/path).encode()) for label,path in models}
    assert all(handles.values())
    rows = []
    try:
        # Interleave models across positions to reduce thermal/order effects.
        for round_index,index in enumerate(indices):
            order = models[round_index:]+models[:round_index]
            for label,path in order:
                for ms,sims in [(0,8192)]+([(2000,0)] if round_index < 2 else []):
                    cells = (ctypes.c_int*64)(*map(int,board[index]))
                    out, policy, visits = (ctypes.c_int*136)(), (ctypes.c_int*6)(), (ctypes.c_int*64)()
                    stats = (ctypes.c_double*72)()
                    lib.ot_mcts_clear_cache()
                    tick = time.perf_counter()
                    assert lib.ot_nnue_mcts_play(handles[label], cells, int(turn[index]), ms, sims,
                        20261028, 0, 0, 0, out, policy, visits, stats) == 1
                    seconds = time.perf_counter()-tick
                    calls, leaves = ctypes.c_uint(),ctypes.c_uint()
                    lib.ot_cnn_batch_stats(ctypes.byref(calls),ctypes.byref(leaves))
                    row = dict(model=label,index=index,milliseconds=ms,requestedSimulations=sims,
                        simulations=int(stats[0]),seconds=seconds,meanDepth=stats[2]/1000,
                        exact=bool(out[5]),actualBatchMean=leaves.value/calls.value if calls.value else 1)
                    rows.append(row)
                    print(json.dumps(row),flush=True)
    finally:
        for handle in handles.values():
            lib.ot_nnue_destroy(handle)
    summary = []
    for label,_ in models:
        for ms in (0,2000):
            selected = [r for r in rows if r['model']==label and r['milliseconds']==ms]
            summary.append(dict(model=label,milliseconds=ms,
                simulationsPerSecond=sum(r['simulations'] for r in selected)/sum(r['seconds'] for r in selected),
                meanDepth=float(np.mean([r['meanDepth'] for r in selected]))))
    report = dict(complete=True,split='validation',positions=indices,batchSize=32,blasThreads=1,
        backend='identical virtual-visit scheduler and Accelerate matrix backend; cold search',
        limits='Four positions, two timed positions; speed diagnosis, no strength claim',
        librarySha256=e.sha(library), models={label:dict(path=p,sha256=e.sha(p)) for label,p in models},
        records=rows,summary=summary)
    assert not output.exists()
    output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
