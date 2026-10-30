"""Two stage conditional policy readouts; a single original outcome head."""
import math
import struct
from pathlib import Path
import numpy as np
import torch
from torch import nn
from pattern_model import DIM, quant
from pattern_structure import StructureNet


class PhasePolicyNet(StructureNet):
    def __init__(self, low_empty=36, high_empty=48):
        super().__init__('symmetry')
        if not 0 <= low_empty < high_empty <= 60:
            raise ValueError('Invalid phase interval')
        self.low_empty, self.high_empty = low_empty, high_empty
        self.early_policy = nn.Linear(32, 42)
        with torch.no_grad():
            self.early_policy.weight.copy_(self.head.weight[4:])
            self.early_policy.bias.copy_(self.head.bias[4:])

    def blend(self, early, late, phase):
        denominator = self.high_empty-self.low_empty
        numerator = (torch.round(phase*60)-self.low_empty).clamp(0, denominator)[:, None]
        return quant((numerator*early+(denominator-numerator)*late)/denominator)

    def forward(self, codes, phase):
        e = quant(self.embedding(self.canonical[codes]))
        pooled = torch.einsum('bpd,pg->bgd', e, self.pool)/self.counts[None, :, None]
        x = torch.cat((quant(pooled.flatten(1).clamp(-4, 4)), quant(phase[:, None])), dim=1)
        h = quant(nn.functional.linear(x, quant(self.trunk.weight), quant(self.trunk.bias, 65536)).clamp(0, 1))
        out = quant(nn.functional.linear(h, quant(self.head.weight), quant(self.head.bias, 65536)).clamp(-4, 4))
        early = quant(nn.functional.linear(h, quant(self.early_policy.weight), quant(self.early_policy.bias, 65536)).clamp(-4, 4))
        policy = self.blend(early, out[:, 4:], phase)
        projection = torch.einsum('bpd,bd->bp', e, policy[:, :32])/math.sqrt(DIM)
        return torch.tanh(out[:, 0]), out[:, 1:4], projection @ self.members + policy[:, 32:][:, self.bucket]


def export(net, path):
    def array(tensor, dtype, scale):
        value = tensor.detach().cpu().numpy()*scale
        value = np.sign(value)*np.floor(np.abs(value)+.5)
        limits = np.iinfo(dtype)
        if not np.isfinite(value).all() or value.min() < limits.min or value.max() > limits.max:
            raise ValueError('Pattern parameters outside integer format')
        return value.astype(dtype).tobytes()
    binary = struct.pack('<8s9I', b'OTPAT7\0\0', 28503, DIM, 32, 97, 46, 256,
                         1, net.low_empty, net.high_empty)
    for layer in [net.embedding, net.trunk, net.head, net.early_policy]:
        binary += array(layer.weight, '<i2', 256)
        if hasattr(layer, 'bias') and layer.bias is not None:
            binary += array(layer.bias, '<i4', 65536)
    Path(path).write_bytes(binary)
    return binary
