#!/usr/bin/env python3
"""Validate batched MCTS and measure an isolated GPU batching diagnostic."""
import ctypes
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('engineering', ROOT/'scripts/check-engineering.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, e.PTR, e.PTR,
                          ctypes.POINTER(ctypes.c_uint64), e.PTR, ctypes.POINTER(ctypes.c_double))


def main():
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise ValueError('GPU batch benchmark requires MPS')
    output = ROOT/'.build/training/cnn-batch-v1.16'
    output.mkdir(exist_ok=True)
    old = e.load('.build/native-cnnpolicy/libothello.dylib')
    batch = e.load('.build/native-cnn-batch-v1.16/libothello.dylib')
    widening = e.load('.build/native-cnn-widening-v1.15/libothello.dylib')
    for lib in (old, batch, widening):
        lib.ot_nnue_error.restype = ctypes.c_char_p
        lib.ot_nnue_infer.argtypes = [ctypes.c_void_p, e.PTR, ctypes.c_int, e.PTR, ctypes.POINTER(ctypes.c_double)]
    batch.ot_cnn_set_batch_evaluator.argtypes = [CALLBACK, ctypes.c_int]
    model = ROOT/'models/cnn2-c32-cosine-joint-v1.13.otcnn'
    handles = {id(lib): lib.ot_nnue_load(str(model).encode()) for lib in (old, batch, widening)}
    assert all(handles.values())
    with np.load(ROOT/'.build/training/v1/positions.npz') as data:
        board, turn = data['validation_board'], data['validation_turn']
    candidates = np.flatnonzero(((board != 0).sum(1) >= 20) & ((board != 0).sum(1) <= 40))
    indices = np.random.default_rng(20261028).choice(candidates, 8, replace=False)
    boards, turns = board[indices], turn[indices]
    cells = [(ctypes.c_int*64)(*map(int, b)) for b in boards]
    net = torch.jit.load(str(ROOT/'models/cnn2-c32-cosine-joint-v1.13.torchscript.pt')).to('mps').eval()
    errors, callback_sizes, callback_ms = [], [], []

    @CALLBACK
    def cpu_callback(n, raw_board, raw_turn, masks, raw_scores, raw_policy):
        try:
            for i in range(n):
                b = ctypes.cast(ctypes.byref(raw_board.contents, i*64*ctypes.sizeof(ctypes.c_int)), e.PTR)
                v = ctypes.c_int()
                p = (ctypes.c_double*65)()
                assert batch.ot_nnue_infer(handles[id(batch)], b, raw_turn[i], ctypes.byref(v), p) == 1
                raw_scores[i] = v.value
                for m in range(65):
                    raw_policy[i*65+m] = p[m]
            return 1
        except BaseException as error:
            errors.append(str(error))
            return -1

    @CALLBACK
    def gpu_callback(n, raw_board, raw_turn, masks, raw_scores, raw_policy):
        try:
            tick = time.perf_counter()
            b = np.ctypeslib.as_array(raw_board, shape=(n*64,)).reshape(n, 64)
            t = np.ctypeslib.as_array(raw_turn, shape=(n,))
            bitmask = np.ctypeslib.as_array(masks, shape=(n,))
            legal = np.zeros((n, 65), bool)
            legal[:, :64] = ((bitmask[:, None] >> np.arange(64, dtype=np.uint64)) & 1) != 0
            legal[:, 64] = bitmask == 0
            with torch.inference_mode():
                v, logits = net(torch.from_numpy(b.astype(np.int8)).to('mps'), torch.from_numpy(t.astype(np.float32)).to('mps'))
                p = logits.masked_fill(~torch.from_numpy(legal).to('mps'), -torch.inf).softmax(1)
                joined = torch.cat((v[:, None], p), 1).cpu().numpy()
            np.ctypeslib.as_array(raw_scores, shape=(n,))[:] = np.rint(joined[:, 0]*1000).astype(np.int32)
            np.ctypeslib.as_array(raw_policy, shape=(n*65,))[:] = joined[:, 1:].reshape(-1)
            callback_sizes.append(n)
            callback_ms.append((time.perf_counter()-tick)*1000)
            return 1
        except BaseException as error:
            errors.append(repr(error))
            return -1

    def search(lib, index, simulations=512):
        out, policy, visits, stats = (ctypes.c_int*136)(), (ctypes.c_int*6)(), (ctypes.c_int*64)(), (ctypes.c_double*72)()
        lib.ot_mcts_clear_cache()
        tick = time.perf_counter()
        ok = lib.ot_nnue_mcts_play(handles[id(lib)], cells[index], int(turns[index]), 0, simulations,
                                   20261028, 0, 0, 0, out, policy, visits, stats)
        if ok != 1:
            raise ValueError((lib.ot_nnue_error() or b'unknown').decode()+str(errors))
        elapsed = time.perf_counter()-tick
        assert int(stats[0]) <= simulations and sum(visits) <= stats[0]
        return dict(move=out[0], score=out[1], depth=out[2], simulations=int(stats[0]), evaluations=int(stats[1]),
                    meanDepth=stats[2]/1000, wallSeconds=elapsed, visits=list(visits), exact=bool(out[5]))

    # A one-leaf batch using the exact original CPU evaluator must match bit-for-bit.
    batch.ot_cnn_set_batch_evaluator(cpu_callback, 1)
    for i in range(8):
        a, b = search(old, i), search(batch, i)
        assert {k: v for k, v in a.items() if k != 'wallSeconds'} == {k: v for k, v in b.items() if k != 'wallSeconds'}, (i, a, b)
    # Failure must roll back reservations and leave a usable tree.
    @CALLBACK
    def bad_callback(*args):
        return -1
    batch.ot_cnn_set_batch_evaluator(bad_callback, 16)
    try:
        search(batch, 0)
        raise AssertionError('Failed evaluator accepted')
    except ValueError:
        pass
    batch.ot_cnn_set_batch_evaluator(cpu_callback, 1)
    assert search(batch, 0)['move'] == search(old, 0)['move']
    # Warm all GPU batch shapes before timing, and validate numerical export parity.
    max_value_error = max_policy_error = 0.
    for size in (1, 8, 16, 32):
        b = np.tile(boards, (4, 1))[:size].astype(np.int32)
        t = np.tile(turns, 4)[:size].astype(np.int32)
        masks = np.zeros(size, np.uint64)
        old.ot_legal.argtypes = [e.PTR, ctypes.c_int, e.PTR]
        for i in range(size):
            moves = (ctypes.c_int*64)()
            count = old.ot_legal(b[i].ctypes.data_as(e.PTR), int(t[i]), moves)
            masks[i] = sum(1 << int(m) for m in moves[:count])
        scores, probs = np.zeros(size, np.int32), np.zeros((size, 65), np.float64)
        for _ in range(3):
            assert gpu_callback(size, b.ctypes.data_as(e.PTR), t.ctypes.data_as(e.PTR),
                                masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), scores.ctypes.data_as(e.PTR),
                                probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))) == 1, errors
        for i in range(size):
            score, p = ctypes.c_int(), (ctypes.c_double*65)()
            assert old.ot_nnue_infer(handles[id(old)], b[i].ctypes.data_as(e.PTR), int(t[i]), ctypes.byref(score), p) == 1
            max_value_error = max(max_value_error, abs(scores[i]-score.value))
            max_policy_error = max(max_policy_error, float(abs(probs[i]-np.array(p)).max()))
    assert max_value_error <= 1 and max_policy_error < 1e-5, (max_value_error, max_policy_error)
    records = []
    variants = ['cpu-original', 'cpu-widening', 'gpu-batch-1', 'gpu-batch-8', 'gpu-batch-16', 'gpu-batch-32']
    for index in range(8):
        # Rotate timing order to limit systematic warm-up/temperature bias.
        order = variants[index % len(variants):]+variants[:index % len(variants)]
        for variant in order:
            callback_sizes.clear()
            callback_ms.clear()
            if variant.startswith('gpu'):
                requested = int(variant.split('-')[-1])
                batch.ot_cnn_set_batch_evaluator(gpu_callback, requested)
                lib = batch
            else:
                requested = 1
                lib = old if variant == 'cpu-original' else widening
            r = search(lib, index, 1024)
            records.append(dict(variant=variant, index=int(indices[index]), requestedBatch=requested, **r,
                                actualBatchMean=float(np.mean(callback_sizes)) if callback_sizes else 1,
                                callbackCalls=len(callback_sizes), callbackTotalMs=sum(callback_ms)))
            print(json.dumps({k: v for k, v in records[-1].items() if k != 'visits'}), flush=True)
            (output/'progress.json').write_text(json.dumps(dict(completed=len(records), total=48, variant=variant))+'\n')
    summary = []
    for variant in variants:
        rows = [r for r in records if r['variant'] == variant]
        total_seconds = sum(r['wallSeconds'] for r in rows)
        summary.append(dict(variant=variant, simulationsPerSecond=sum(r['simulations'] for r in rows)/total_seconds,
                            meanDepth=float(np.mean([r['meanDepth'] for r in rows])), meanMaxDepth=float(np.mean([r['depth'] for r in rows])),
                            actualBatchMean=float(np.mean([r['actualBatchMean'] for r in rows])), totalSeconds=total_seconds))
    report = dict(version='v1.16-cnn-batch-diagnostic', defaultPromoted=False, modelSha256=e.sha(str(model.relative_to(ROOT))),
                  batchOneExactCpuParityPositions=8, failedCallbackRecovery=True, maxGpuValueIntegerError=max_value_error,
                  maxGpuPolicyError=max_policy_error, positions=list(map(int, indices)), split='original frozen validation, 20..40 occupied cells',
                  budget=1024, summary=summary, records=records, torchVersion=torch.__version__,
                  method='C++ tree, virtual visit reservations, no duplicate pending tree leaves; synchronous Python/MPS joint inference; drain before returning',
                  limits='Short fixed-simulation diagnostic while formal CPU matches continue; wall times are not isolated release benchmarks; strength unmeasured',
                  sourceSha256={p: e.sha(p) for p in ['experiments/cnn-batch-v1.16/mcts.cpp', 'experiments/cnn-batch-v1.16/CMakeLists.txt',
                                                   'experiments/cnn-widening-v1.15/mcts.cpp', 'scripts/benchmark-cnn-batch.py']},
                  libraries={p: e.sha(p) for p in ['.build/native-cnnpolicy/libothello.dylib', '.build/native-cnn-batch-v1.16/libothello.dylib',
                                                  '.build/native-cnn-widening-v1.15/libothello.dylib']})
    target = ROOT/'models/v1.16-cnn-batch-diagnostic.json'
    assert not target.exists()
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(complete=True, summary=summary)), flush=True)
    batch.ot_cnn_set_batch_evaluator(CALLBACK(), 1)
    for lib in (old, batch, widening):
        lib.ot_nnue_destroy(handles[id(lib)])


if __name__ == '__main__':
    main()
