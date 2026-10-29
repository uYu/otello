"""Othello-specific ternary pattern network and reproducible integer export."""
import ctypes
import math
from pathlib import Path
import struct
import numpy as np
import torch
from torch import nn

DIM, HIDDEN, ROWS, INPUTS, OUTPUTS = 32, 32, 55728, 97, 46


def shapes():
    result = []
    for axis in range(2):
        for i in range(8):
            distance = min(i, 7-i)
            result.append(([j*8+i if axis else i*8+j for j in range(8)], distance*6561, int(distance != 0)))
    for direction in range(2):
        for delta in range(-4, 5):
            length = 8-abs(delta)
            squares = [y*8+(7-(y+delta) if direction else y+delta) for y in range(8) if 0 <= y+delta < 8]
            result.append((squares, 26244+sum(3**k for k in range(4, length)), 1))
    for corner in range(4):
        result.append(([(7-y if corner & 2 else y)*8+(7-x if corner & 1 else x)
                        for y in range(3) for x in range(3)], 36045, 2))
    assert len(result) == 38
    return result


SHAPES = shapes()
POOL = np.zeros((38, 3), np.float32)
MEMBERS = np.zeros((38, 64), np.float32)
for i, (squares, _, group) in enumerate(SHAPES):
    POOL[i, group] = 1
    MEMBERS[i, squares] = 1
COUNTS = POOL.sum(axis=0)
MEMBERS /= MEMBERS.sum(axis=0)
BUCKET = []
for square in range(64):
    x, y = sorted((min(square % 8, 7-square % 8), min(square // 8, 7-square // 8)))
    BUCKET.append(x*4-x*(x-1)//2+y-x)


def encode(board, turn):
    relative = board.astype(np.int16)*turn.astype(np.int16)[:, None]
    digits = np.where(relative == 1, 1, np.where(relative == -1, 2, 0))
    return np.stack([offset+digits[:, squares] @ (3**np.arange(len(squares)))
                     for squares, offset, _ in SHAPES], axis=1).astype(np.uint16)


def legal_mask(board, turn):
    relative = board*turn[:, None]
    own = np.packbits(relative == 1, axis=1, bitorder='little').copy().view('<u8').reshape(-1)
    opp = np.packbits(relative == -1, axis=1, bitorder='little').copy().view('<u8').reshape(-1)
    empty, moves = ~(own | opp), np.zeros(len(board), np.uint64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not dx and not dy:
                continue
            delta = dy*8+dx
            mask = np.uint64(0x7f7f7f7f7f7f7f7f if dx == 1 else 0xfefefefefefefefe if dx == -1 else 0xffffffffffffffff)
            def shift(value):
                return (value & mask) << np.uint64(delta) if delta > 0 else (value & mask) >> np.uint64(-delta)
            ray = shift(own) & opp
            for _ in range(5):
                ray |= shift(ray) & opp
            moves |= shift(ray) & empty
    return np.unpackbits(moves.astype('<u8').view(np.uint8).reshape(-1, 8), axis=1, bitorder='little').astype(bool)


def rounded(x):
    return torch.sign(x)*torch.floor(torch.abs(x)+0.5)


def quant(x, scale=256):
    # Straight-through quantization; C++ uses the same away-from-zero tie rule.
    return x+(rounded(x*scale)/scale-x).detach()


class PatternNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(ROWS, DIM)
        nn.init.normal_(self.embedding.weight, std=0.05)
        self.trunk = nn.Linear(INPUTS, HIDDEN)
        self.head = nn.Linear(HIDDEN, OUTPUTS)
        self.register_buffer('pool', torch.from_numpy(POOL))
        self.register_buffer('counts', torch.from_numpy(COUNTS))
        self.register_buffer('members', torch.from_numpy(MEMBERS))
        self.register_buffer('bucket', torch.tensor(BUCKET))

    def forward(self, codes, phase):
        e = quant(self.embedding(codes))
        pooled = torch.einsum('bpd,pg->bgd', e, self.pool)/self.counts[None, :, None]
        x = torch.cat((quant(pooled.flatten(1).clamp(-4, 4)), quant(phase[:, None])), dim=1)
        h = nn.functional.linear(x, quant(self.trunk.weight), quant(self.trunk.bias, 65536))
        h = quant(h.clamp(0, 1))
        out = nn.functional.linear(h, quant(self.head.weight), quant(self.head.bias, 65536))
        out = quant(out.clamp(-4, 4))
        projection = torch.einsum('bpd,bd->bp', e, out[:, 4:36])/math.sqrt(DIM)
        policy = projection @ self.members + out[:, 36:46][:, self.bucket]
        return torch.tanh(out[:, 0]), out[:, 1:4], policy


def export(net, path):
    def array(tensor, dtype, scale):
        value = tensor.detach().cpu().numpy()*scale
        value = np.sign(value)*np.floor(np.abs(value)+0.5)
        limits = np.iinfo(dtype)
        if not np.isfinite(value).all() or value.min() < limits.min or value.max() > limits.max:
            raise ValueError('Pattern parameters outside integer format')
        return value.astype(dtype)
    arrays = [array(net.embedding.weight, '<i2', 256), array(net.trunk.weight, '<i2', 256),
              array(net.trunk.bias, '<i4', 65536), array(net.head.weight, '<i2', 256),
              array(net.head.bias, '<i4', 65536)]
    binary = struct.pack('<8s6I', b'OTPAT1\0\0', ROWS, DIM, HIDDEN, INPUTS, OUTPUTS, 256)
    binary += b''.join(x.tobytes() for x in arrays)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(binary)
    return binary


def native_library(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    ptr = ctypes.POINTER(ctypes.c_int)
    lib.ot_nnue_load.argtypes = [ctypes.c_char_p]
    lib.ot_nnue_load.restype = ctypes.c_void_p
    lib.ot_nnue_destroy.argtypes = [ctypes.c_void_p]
    lib.ot_nnue_error.restype = ctypes.c_char_p
    lib.ot_pattern_predict.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    lib.ot_nnue_evaluate.argtypes = [ctypes.c_void_p, ptr, ctypes.c_int, ptr]
    return lib


def parity(library, model, board, turn, expected):
    lib = native_library(library)
    handle = lib.ot_nnue_load(str(Path(model).resolve()).encode())
    if not handle:
        raise ValueError(lib.ot_nnue_error().decode())
    actual, scores = [], []
    try:
        for row, actor in zip(board, turn):
            cells, out, score = (ctypes.c_int*64)(*map(int, row)), (ctypes.c_double*68)(), ctypes.c_int()
            if lib.ot_pattern_predict(handle, cells, int(actor), out) != 1:
                raise ValueError(lib.ot_nnue_error().decode())
            assert lib.ot_nnue_evaluate(handle, cells, int(actor), ctypes.byref(score)) == 1
            actual.append(list(out)); scores.append(score.value)
    finally:
        lib.ot_nnue_destroy(handle)
    value, wdl, policy = expected
    wanted = np.column_stack((value, torch.softmax(torch.from_numpy(wdl), dim=1).numpy(), policy))
    errors = np.max(np.abs(np.asarray(actual)-wanted), axis=0)
    if errors.max() > 2e-5:
        raise ValueError(f'Pattern Python/C++ mismatch: {errors.max()}')
    return dict(positions=len(board), maxDiscError=float(errors[0]), maxWdlError=float(errors[1:4].max()),
                maxPolicyLogitError=float(errors[4:].max()), maxScoreRoundingError=float(np.max(np.abs(np.array(scores)-value*1000))))
