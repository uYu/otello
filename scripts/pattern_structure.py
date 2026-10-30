"""Independent v1.37 ablations: tied D4 table or ten geometric mean pools."""
import math
import struct
from pathlib import Path
import numpy as np
import torch
from torch import nn
from pattern_model import PatternNet, SHAPES, ROWS, DIM, quant, export as legacy_export


def canonical_rows():
    mapping = np.empty(ROWS, np.int64)
    representatives = []
    offset = 0
    for length, tables in [(8, 4), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1)]:
        size = 3**length
        ids = np.arange(size)
        digits = ids[:, None] // (3**np.arange(length)) % 3
        transformed = (digits.reshape(-1, 3, 3).transpose(0, 2, 1).reshape(-1, 9)
                       if length == 9 else digits[:, ::-1])
        canonical = np.minimum(ids, transformed @ (3**np.arange(length)))
        unique, inverse = np.unique(canonical, return_inverse=True)
        for _ in range(tables):
            mapping[offset:offset+size] = len(representatives) + inverse
            representatives.extend((offset+unique).tolist())
            offset += size
    assert offset == ROWS and len(representatives) == 28503
    return mapping, np.array(representatives)


class StructureNet(PatternNet):
    def __init__(self, variant):
        super().__init__()
        if variant not in ('control', 'symmetry', 'pool10'):
            raise ValueError(variant)
        self.variant = variant
        if variant == 'symmetry':
            mapping, representatives = canonical_rows()
            # Same original random table as control; select orbit representatives.
            self.embedding = nn.Embedding.from_pretrained(
                self.embedding.weight.detach()[representatives].clone(), freeze=False)
            self.register_buffer('canonical', torch.from_numpy(mapping))
        elif variant == 'pool10':
            pool = np.zeros((38, 10), np.float32)
            for i, (squares, offset, _) in enumerate(SHAPES):
                group = offset // 6561 if i < 16 else 4+len(squares)-4 if i < 34 else 9
                pool[i, group] = 1
            self.pool = torch.from_numpy(pool)
            self.counts = self.pool.sum(0)
            self.trunk = nn.Linear(321, 32)

    def forward(self, codes, phase):
        if self.variant == 'symmetry':
            codes = self.canonical[codes]
        e = quant(self.embedding(codes))
        pooled = torch.einsum('bpd,pg->bgd', e, self.pool)/self.counts[None, :, None]
        x = torch.cat((quant(pooled.flatten(1).clamp(-4, 4)), quant(phase[:, None])), dim=1)
        h = quant(nn.functional.linear(x, quant(self.trunk.weight), quant(self.trunk.bias, 65536)).clamp(0, 1))
        out = quant(nn.functional.linear(h, quant(self.head.weight), quant(self.head.bias, 65536)).clamp(-4, 4))
        projection = torch.einsum('bpd,bd->bp', e, out[:, 4:36])/math.sqrt(DIM)
        return torch.tanh(out[:, 0]), out[:, 1:4], projection @ self.members + out[:, 36:46][:, self.bucket]


def export(net, path):
    if net.variant == 'control':
        return legacy_export(net, path, outcome=True)
    # OTPAT4: legacy 32-byte header + uint32 independent-ablation flag.
    def array(tensor, dtype, scale):
        value = tensor.detach().cpu().numpy()*scale
        value = np.sign(value)*np.floor(np.abs(value)+.5)
        limits = np.iinfo(dtype)
        if not np.isfinite(value).all() or value.min() < limits.min or value.max() > limits.max:
            raise ValueError('Pattern parameters outside integer format')
        return value.astype(dtype).tobytes()
    binary = struct.pack('<8s7I', b'OTPAT4\0\0', net.embedding.num_embeddings, DIM, 32,
                         net.trunk.in_features, 46, 256, 1 if net.variant == 'symmetry' else 2)
    for tensor, dtype, scale in [(net.embedding.weight, '<i2', 256), (net.trunk.weight, '<i2', 256),
        (net.trunk.bias, '<i4', 65536), (net.head.weight, '<i2', 256), (net.head.bias, '<i4', 65536)]:
        binary += array(tensor, dtype, scale)
    Path(path).write_bytes(binary)
    return binary
