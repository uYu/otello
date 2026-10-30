#!/usr/bin/env python3
"""Check malformed CNNs, fused inference and unchanged v1.9 fixed-work search."""
import ctypes
import importlib.util
import json
import struct
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('engineering', ROOT/'scripts/check-engineering.py')
engineering = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engineering)


def main():
    new_path = '.build/native-cnnpolicy/libothello.dylib'
    old_path = '.build/native-engineering/libothello.dylib'
    old, new = engineering.load(old_path), engineering.load(new_path)
    new.ot_nnue_infer.argtypes = [ctypes.c_void_p, engineering.PTR, ctypes.c_int, engineering.PTR, ctypes.POINTER(ctypes.c_double)]
    model = ROOT/'models/nnue256-outcome-v1.9.otnnue'
    a, b = old.ot_nnue_load(str(model).encode()), new.ot_nnue_load(str(model).encode())
    assert a and b
    with np.load(ROOT/'.build/training/v1/positions.npz') as data:
        boards, turns = data['test_board'][::250], data['test_turn'][::250]
    try:
        for lib in (old, new):
            lib.ot_mcts_clear_cache()
        for board, turn in zip(boards, turns):
            cells = (ctypes.c_int*64)(*map(int, board))
            assert engineering.search(old, a, cells, int(turn)) == engineering.search(new, b, cells, int(turn))
        for lib in (old, new):
            lib.ot_mcts_clear_cache()
        new.ot_legal.argtypes = [engineering.PTR, ctypes.c_int, engineering.PTR]
        new.ot_play.argtypes = [engineering.PTR, ctypes.c_int, ctypes.c_int, engineering.PTR]
        cells = (ctypes.c_int*64)()
        cells[28] = cells[35] = 1
        cells[27] = cells[36] = -1
        turn = 1
        for i in range(16):
            assert engineering.search(old, a, cells, turn) == engineering.search(new, b, cells, turn)
            moves, out = (ctypes.c_int*64)(), (ctypes.c_int*136)()
            n = new.ot_legal(cells, turn, moves)
            assert new.ot_play(cells, turn, moves[i % n] if n else -1, out) == 1
            cells, turn = (ctypes.c_int*64)(*list(out)[:64]), int(out[64])
    finally:
        old.ot_nnue_destroy(a)
        new.ot_nnue_destroy(b)
    checked = []
    for name in ('cosine-joint', 'cosine-v4-value'):
        path = ROOT/f'models/cnn2-c32-{name}-v1.13.otcnn'
        binary = path.read_bytes()
        handle = new.ot_nnue_load(str(path).encode())
        assert handle
        try:
            for board, turn in zip(boards, turns):
                cells = (ctypes.c_int*64)(*map(int, board))
                score, fused_score = ctypes.c_int(), ctypes.c_int()
                p, fused_p = (ctypes.c_double*65)(), (ctypes.c_double*65)()
                assert new.ot_nnue_evaluate(handle, cells, int(turn), ctypes.byref(score)) == 1
                assert new.ot_nnue_policy(handle, cells, int(turn), p) == 1
                assert new.ot_nnue_infer(handle, cells, int(turn), ctypes.byref(fused_score), fused_p) == 1
                assert score.value == fused_score.value and list(p) == list(fused_p)
                renamed = (ctypes.c_int*64)(*map(int, -board))
                assert new.ot_nnue_infer(handle, renamed, -int(turn), ctypes.byref(fused_score), fused_p) == 1
                assert score.value == fused_score.value and list(p) == list(fused_p)
        finally:
            new.ot_nnue_destroy(handle)
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory)/'bad.otcnn'
            bad_channels = bytearray(binary)
            bad_channels[16:20] = struct.pack('<I', 1000000)
            bad_head = bytearray(binary)
            bad_head[20:24] = struct.pack('<I', 32)
            nan = bytearray(binary)
            nan[28:32] = struct.pack('<f', float('nan'))
            for content in (binary[:27], binary[:-1], binary+b'extra', bytes(bad_channels), bytes(bad_head), bytes(nan)):
                bad.write_bytes(content)
                assert not new.ot_nnue_load(str(bad).encode())
        export = json.loads(path.with_suffix('.export.json').read_text())
        assert export['sha256'] == engineering.sha(str(path.relative_to(ROOT)))
        assert all(s['positions'] == 6000 and s['maxValueError'] < .00051 and s['maxPolicyError'] < .00001 for s in export['metrics'].values())
        checked.append(dict(name=name, malformedRejected=6, fusedAndColorPositions=len(boards), export=export))
    result = dict(verified=True, baselineFixedWorkParity=True, baselineColdPositions=len(boards), baselineWarmPositions=16, simulations=512,
                  librarySha256=engineering.sha(new_path), previousLibrarySha256=engineering.sha(old_path), candidates=checked)
    (ROOT/'models/v1.14-cnn-policy-native-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PASS: v1.9 fixed-work parity, fused/color inference, corrupt model rejection and full holdout export parity')


if __name__ == '__main__':
    main()
